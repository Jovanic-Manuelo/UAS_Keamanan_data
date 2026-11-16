"""Aplikasi web Flask untuk memperkirakan risiko phishing dari teks URL.

Jalankan ``pip install -r requirements.txt`` lalu ``python app.py`` dan buka
``http://127.0.0.1:5000``. Pipeline dan metadata model dimuat saat modul
diimpor, sehingga kedua berkas model harus tersedia di folder ``model``.

Alur pengecekan:
    input -> validasi -> pipeline fitur/model -> penjelasan dan saran

URL hanya dianalisis sebagai teks; aplikasi tidak mengunjungi alamat tersebut.
"""
import csv
import io
import ipaddress
import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

# Pipeline joblib merujuk ke URLFeatureExtractor yang didefinisikan di features.py.
from features import (BRANDS, FEATURE_CONTRACT_VERSION, SUSPICIOUS_WORDS,
                      extract_features, normalize_url, registered_domain,
                      split_url)

# Lokasi aplikasi dan artefak model yang dibuat oleh proses training.
BASE = Path(__file__).resolve().parent
PIPELINE = joblib.load(BASE / "model" / "phishing_url_pipeline.joblib")
META = json.loads((BASE / "model" / "metadata.json").read_text())
THRESHOLD = float(META["threshold"])  # Ambang klasifikasi dari data validasi.

# Batas input untuk mencegah permintaan individual atau batch terlalu besar.
MAX_URL_LEN = 2000
MAX_BATCH = 200
LOW_CONF_MARGIN = 0.15  # Selisih probabilitas dari ambang yang dianggap kurang meyakinkan.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # Batas unggahan Flask: 1 MiB.
HISTORY = deque(maxlen=100)  # Riwayat sementara; hilang saat proses aplikasi berhenti.

# Regex untuk pemeriksaan skema, host, dan oktet alamat IPv4.
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9\-_.]*[a-z0-9])?$")
_IPV4_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")


# --------------------------------------------------------------------------
# Validasi input
# --------------------------------------------------------------------------
def validate_url(raw):
    """Validasi bentuk URL tanpa mengirim permintaan jaringan.

    Pemeriksaan meliputi input kosong/panjang/spasi, skema HTTP(S), host,
    format domain, dan rentang oktet IPv4. URL tanpa skema juga dapat diterima
    jika host memenuhi pemeriksaan.

    Args:
        raw: Nilai yang dikirim pengguna.

    Returns:
        Pasangan ``(url, None)`` jika diterima, atau ``(None, pesan)`` jika
        ditolak. Nilai URL yang diterima adalah teks input yang telah di-trim.
    """
    if raw is None:
        return None, "Input kosong. Ketik atau tempel sebuah URL."
    url = str(raw).strip()
    if not url:
        return None, "Input kosong. Ketik atau tempel sebuah URL."
    if len(url) > MAX_URL_LEN:
        return None, f"URL terlalu panjang (maksimal {MAX_URL_LEN} karakter)."
    if re.search(r"\s", url):
        return None, "URL tidak boleh mengandung spasi atau baris baru."

    m = _SCHEME_RE.match(url)
    if m and "://" in url[: len(m.group(0)) + 2]:
        scheme = m.group(1).lower()
        if scheme not in ("http", "https"):
            return None, f"Skema '{scheme}://' tidak didukung. Hanya http dan https."
    elif m and not re.match(r"^[^/]+:\d{1,5}(/|$)", url):
        # contoh: javascript:alert(1), mailto:a@b.com
        return None, f"'{m.group(1)}:' bukan alamat web. Masukkan URL http/https."

    try:
        host, _, _, _ = split_url(normalize_url(url))
    except ValueError:
        return None, "Host atau nomor port pada URL tidak valid."
    if not host:
        return None, "Tidak ditemukan nama domain pada input."

    # urlsplit (melalui split_url) menghapus tanda kurung siku dari hostname.
    # Karena itu validasi alamat IPv6 dilakukan terhadap host yang sudah diparsing.
    if ":" in host:
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return None, "Alamat IPv6 tidak valid. Gunakan format IPv6 dalam tanda kurung siku."
        return url, None

    ip = _IPV4_RE.match(host)
    if ip:
        if any(int(o) > 255 for o in ip.groups()):
            return None, "Alamat IP tidak valid (tiap angka maksimal 255)."
        return url, None
    if "." not in host:
        return None, "Ini tidak terlihat seperti URL. Contoh yang benar: contoh.com/halaman"
    if not _HOST_RE.match(host) or ".." in host:
        return None, "Nama domain mengandung karakter yang tidak valid."
    tld = host.rsplit(".", 1)[1]
    if not (re.fullmatch(r"[a-z]{2,24}", tld) or tld.startswith("xn--")):
        return None, "Akhiran domain (TLD) tidak valid."
    return url, None


# --------------------------------------------------------------------------
# Penjelasan hasil
# --------------------------------------------------------------------------
def explain(url, feats):
    """Susun sinyal risiko/aman yang mudah dipahami dari fitur URL.

    Ini adalah penjelasan berbasis aturan atas ciri teks URL, bukan penjelasan
    atas bobot atau keputusan internal model.

    Args:
        url: URL yang sudah lolos validasi.
        feats: Dictionary fitur dari :func:`features.extract_features`.

    Returns:
        Tuple ``(risk_signals, safe_signals, note)``. Dua nilai pertama adalah
        daftar pesan; ``note`` berisi catatan tambahan atau ``None``.
    """
    norm = normalize_url(url).lower()
    host, _, path, _ = split_url(normalize_url(url))
    risk, safe = [], []

    if feats["has_ip"]:
        risk.append("Memakai alamat IP, bukan nama domain. Situs resmi hampir selalu memakai nama domain.")
    words = [w for w in SUSPICIOUS_WORDS if w in norm]
    if words:
        risk.append("Mengandung kata yang sering dipakai halaman phishing: " + ", ".join(words[:6]) + ".")
    if feats["brand_outside_domain"]:
        domain = registered_domain(url)
        subdomain = host[:-len(domain)].rstrip(".") if host.endswith(domain) else ""
        domain_name = domain.split(".", 1)[0]
        outside = subdomain + " " + path.lower()
        brands = [b for b in BRANDS if b in outside and b not in domain_name.lower()]
        nama = ", ".join(brands[:3]) if brands else "merek terkenal"
        risk.append(f"Nama merek ({nama}) muncul di subdomain atau path, padahal domain utamanya bukan milik merek itu.")
    if feats["is_shortener"]:
        risk.append("Memakai layanan pemendek tautan, sehingga tujuan aslinya tersembunyi.")
    if not feats["is_common_tld"] and not feats["has_ip"]:
        risk.append("Akhiran domain (TLD) jarang muncul pada situs legitimate di data latih.")
    if feats["n_digits_host"] >= 3 and not feats["has_ip"]:
        risk.append(f"Nama domain mengandung banyak angka ({feats['n_digits_host']}).")
    if feats["n_hyphens_host"] >= 2:
        risk.append(f"Nama domain mengandung banyak tanda hubung ({feats['n_hyphens_host']}).")
    if feats["n_subdomains"] >= 3:
        risk.append(f"Subdomain bertingkat banyak ({feats['n_subdomains']} tingkat).")
    if feats["url_length"] > 75:
        risk.append(f"URL sangat panjang ({feats['url_length']} karakter).")
    if feats["has_php"]:
        risk.append("Mengarah ke file .php, pola yang sering ada pada halaman login palsu.")
    if feats["has_exe"]:
        risk.append("Mengarah ke file yang bisa diunduh/dijalankan (exe, apk, zip, dan sejenisnya).")
    if feats["n_at"]:
        risk.append("Mengandung karakter '@', yang bisa dipakai untuk menyamarkan domain tujuan.")
    if feats["has_port"]:
        risk.append("Memakai nomor port khusus.")
    if feats["has_double_slash"]:
        risk.append("Ada '//' di tengah path, kadang dipakai untuk pengalihan.")
    if feats["n_percent"] >= 3:
        risk.append("Banyak karakter ter-encode (%), bisa dipakai untuk menyembunyikan isi URL.")

    if feats["is_common_tld"]:
        safe.append("Akhiran domain umum dipakai situs legitimate.")
    if not words:
        safe.append("Tidak ada kata bernuansa login/verifikasi.")
    if feats["url_length"] <= 40 and feats["n_special_total"] == 0:
        safe.append("URL pendek dan sederhana.")
    if not feats["has_ip"] and feats["n_digits_host"] == 0 and feats["n_hyphens_host"] == 0:
        safe.append("Nama domain bersih, tanpa angka atau tanda hubung.")

    note = None
    if feats["path_length"] + feats["query_length"] == 0:
        note = ("URL ini hanya berisi nama domain. Informasinya sedikit, sehingga "
                "prediksi untuk bentuk seperti ini kurang akurat dibanding URL lengkap.")
    return risk, safe, note


# Saran ditampilkan sesuai label; pada keyakinan rendah, saran uncertain
# ditambahkan sebelum saran berdasarkan label prediksi.
ADVICE = {
    "phishing": [
        "Jangan memasukkan kata sandi, OTP, PIN, atau data kartu di halaman ini.",
        "Jangan mengunduh atau menjalankan file dari tautan ini.",
        "Buka situs resminya dengan mengetik alamatnya sendiri di browser, bukan lewat tautan ini.",
        "Laporkan tautan ke tim IT/admin atau ke layanan yang ditiru.",
        "Jika sudah terlanjur mengisi data: segera ganti kata sandi, aktifkan verifikasi dua langkah, dan hubungi bank bila menyangkut rekening.",
    ],
    "legitimate": [
        "Tetap periksa ejaan nama domain sebelum login (misalnya huruf yang ditukar atau ditambah).",
        "Jangan pernah membagikan OTP atau PIN, walaupun situsnya terlihat resmi.",
        "Hasil ini hanya dari teks URL. Isi halaman tidak diperiksa, jadi tetap waspada.",
    ],
    "uncertain": [
        "Model tidak cukup yakin. Perlakukan tautan ini sebagai mencurigakan sampai terbukti aman.",
        "Cek alamat resmi lewat sumber lain (aplikasi resmi, kontak resmi) sebelum memasukkan data apa pun.",
    ],
}


def analyze(raw, save=True):
    """Jalankan validasi dan analisis lengkap atas satu URL.

    URL yang lolos validasi dikirim ke pipeline untuk memperoleh probabilitas,
    lalu dipetakan ke label memakai :data:`THRESHOLD`. Penjelasan dan saran
    dihitung dari fitur yang sama. Secara default ringkasan hasil disimpan
    pada riwayat dalam memori.

    Args:
        raw: Input URL mentah.
        save: Jika ``True``, tambahkan ringkasan hasil ke :data:`HISTORY`.

    Returns:
        Dictionary hasil dengan ``ok=False`` dan ``error`` jika input ditolak,
        atau ``ok=True`` beserta label, probabilitas, sinyal, dan saran.
    """
    url, err = validate_url(raw)
    shown = "" if raw is None else str(raw).strip()[:300]
    if err:
        return {"ok": False, "input": shown, "error": err}

    prob = float(PIPELINE.predict_proba([url])[0, 1])
    label = "phishing" if prob >= THRESHOLD else "legitimate"
    low_conf = abs(prob - THRESHOLD) < LOW_CONF_MARGIN
    feats = extract_features(url)
    risk, safe, note = explain(url, feats)

    advice = list(ADVICE[label])
    if low_conf:
        advice = ADVICE["uncertain"] + advice

    result = {
        "ok": True,
        "input": url,
        "normalized": normalize_url(url),
        "label": label,
        "label_text": "Terindikasi PHISHING" if label == "phishing" else "Kemungkinan LEGITIMATE",
        "probability": round(prob, 4),
        "percent": round(prob * 100, 1),
        "threshold": THRESHOLD,
        "confidence": "rendah" if low_conf else "tinggi",
        "risk_signals": risk,
        "safe_signals": safe,
        "note": note,
        "advice": advice,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    if save:
        HISTORY.appendleft({k: result[k] for k in ("time", "input", "label", "percent", "confidence")})
    return result


def read_batch_inputs(text, file_storage):
    """Gabungkan URL dari textarea dan berkas CSV/TXT opsional.

    Untuk CSV dengan kolom bernama ``url``, kolom itu yang dibaca; jika tidak
    ada, kolom pertama digunakan. Berkas dibaca sebagai UTF-8 dengan dukungan
    BOM. Header dibuang hanya jika kolom ``url`` ditemukan.

    Args:
        text: Isi textarea; setiap baris nonkosong dianggap satu URL.
        file_storage: Objek berkas Flask opsional.

    Returns:
        Daftar URL yang tidak kosong, dengan urutan textarea lalu isi berkas.

    Raises:
        ValueError: Jika ekstensi berkas bukan CSV/TXT atau isinya bukan UTF-8.
    """
    items = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if file_storage and file_storage.filename:
        name = file_storage.filename.lower()
        if not name.endswith((".csv", ".txt")):
            raise ValueError("File harus berformat .csv atau .txt.")
        try:
            content = file_storage.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("File tidak bisa dibaca sebagai teks (gunakan UTF-8).")
        rows = [r for r in csv.reader(io.StringIO(content)) if r and any(c.strip() for c in r)]
        if rows:
            header = [c.strip().lower() for c in rows[0]]
            col = header.index("url") if "url" in header else 0
            if "url" in header:
                rows = rows[1:]
            items += [r[col].strip() for r in rows if len(r) > col and r[col].strip()]
    return items


# --------------------------------------------------------------------------
# Halaman
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Tampilkan halaman utama; proses URL hanya pada permintaan POST."""
    result = analyze(request.form.get("url")) if request.method == "POST" else None
    return render_template("index.html", result=result, active="index")


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Tampilkan form pemeriksaan batch dan proses input saat POST.

    Input gabungan dibatasi oleh :data:`MAX_BATCH`. Hasil tiap baris dapat
    berhasil diprediksi atau ditolak secara individual; halaman juga menerima
    ringkasan jumlah phishing, legitimate, dan input tidak valid.
    """
    results, error, summary = None, None, None
    text = request.form.get("urls", "")
    if request.method == "POST":
        try:
            items = read_batch_inputs(text, request.files.get("file"))
        except ValueError as e:
            items, error = [], str(e)
        if not error and not items:
            error = "Tidak ada URL yang dimasukkan. Isi kotak teks atau unggah file."
        elif len(items) > MAX_BATCH:
            error = f"Terlalu banyak URL ({len(items)}). Maksimal {MAX_BATCH} per pengecekan."
        elif not error:
            results = [analyze(u) for u in items]
            summary = {
                "total": len(results),
                "phishing": sum(r["ok"] and r["label"] == "phishing" for r in results),
                "legitimate": sum(r["ok"] and r["label"] == "legitimate" for r in results),
                "invalid": sum(not r["ok"] for r in results),
            }
    return render_template("batch.html", results=results, error=error, summary=summary,
                           text=text, max_batch=MAX_BATCH, active="batch")


@app.route("/history")
def history():
    """Tampilkan ringkasan pemeriksaan yang tersimpan di memori."""
    return render_template("history.html", history=list(HISTORY), active="history")


@app.route("/history/clear", methods=["POST"])
def history_clear():
    """Hapus seluruh riwayat sementara dan tampilkan halaman riwayat kosong."""
    HISTORY.clear()
    return render_template("history.html", history=[], active="history")


@app.route("/model")
def model_info():
    """Tampilkan metadata model yang dibaca dari berkas metadata."""
    return render_template("model.html", meta=META, active="model")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Prediksi satu URL dari JSON dan kembalikan hasil dalam format JSON.

    Input yang tidak lolos validasi menghasilkan status HTTP 400. Endpoint ini
    tidak menambahkan hasil ke riwayat agar permintaan pengujian tidak
    mengubah state aplikasi.
    """
    data = request.get_json(silent=True) or {}
    res = analyze(data.get("url"), save=False)
    return jsonify(res), (200 if res["ok"] else 400)


@app.errorhandler(413)
def too_large(_):
    """Tampilkan pesan unggahan terlalu besar untuk HTTP 413."""
    return render_template("batch.html", results=None, summary=None, text="", max_batch=MAX_BATCH,
                           error="File terlalu besar (maksimal 1 MB).", active="batch"), 413


if __name__ == "__main__":
    # Batasi server pengembangan ke loopback; debug dinonaktifkan.
    app.run(host="127.0.0.1", port=5000, debug=False)
