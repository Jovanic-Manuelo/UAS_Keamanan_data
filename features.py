"""Normalisasi URL dan ekstraksi fitur leksikal untuk model deteksi phishing.

Modul ini digunakan saat menyiapkan data training maupun saat aplikasi
memprediksi URL. Memakai implementasi yang sama pada kedua tahap menjaga
bentuk dan urutan fitur tetap konsisten. Pipeline scikit-learn yang disimpan
sebagai berkas joblib juga merujuk ke :class:`URLFeatureExtractor` di modul ini,
sehingga modul ini harus tersedia ketika pipeline dimuat.

Fungsi utama:
    normalize_url: menormalkan ejaan URL untuk ekstraksi fitur.
    split_url: memisahkan host, port, path, dan query.
    extract_features: mengubah satu URL menjadi pemetaan fitur numerik.
"""
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

# Gunakan daftar public suffix bawaan paket agar ekstraksi tidak mengakses internet.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Kosakata yang dicari di seluruh URL; hitungan fitur menghitung tiap kemunculan.
SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "password", "passwd", "banking",
    "bank", "wallet", "webscr", "auth", "billing", "invoice", "payment",
    "recover", "unlock", "suspend", "support", "service", "webmail", "admin",
]

# Merek yang bisa disalahgunakan pada subdomain atau path di luar domain merek.
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "facebook", "instagram", "whatsapp", "netflix", "amazon", "ebay",
    "dropbox", "linkedin", "twitter", "yahoo", "chase", "wellsfargo", "dhl",
    "fedex", "usps", "steam", "binance", "coinbase", "metamask", "bca", "bri",
    "mandiri", "bni", "dana", "gopay", "shopee", "tokopedia",
]

# Domain layanan pemendek tautan yang ditandai sebagai fitur tersendiri.
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "s.id", "t.ly", "tiny.cc",
    "bitly.com", "lnkd.in", "qr.ae", "v.gd", "shorte.st", "adf.ly",
}

# Akhiran domain yang dianggap umum pada kumpulan data proyek ini.
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co.uk", "ac.id", "co.id",
               "go.id", "id", "de", "fr", "jp", "ca", "au", "com.au", "us"}

# Urutan kolom fitur yang menjadi kontrak dengan pipeline/model tersimpan.
FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length",
    "n_dots_host", "n_subdomains", "n_hyphens_host", "n_digits_host",
    "digit_ratio_host", "longest_host_token", "host_entropy",
    "path_depth", "n_digits_path", "n_hyphens_path", "n_underscores",
    "n_percent", "n_equals", "n_ampersand", "n_question", "n_at",
    "n_tilde", "n_special_total", "path_entropy", "longest_path_token",
    "has_ip", "has_port", "has_double_slash", "has_php", "has_html",
    "has_exe", "is_shortener", "is_common_tld", "tld_length",
    "n_suspicious_words", "brand_outside_domain", "n_query_params",
]

# Naikkan saat makna nilai fitur berubah, meskipun nama dan urutannya tetap.
FEATURE_CONTRACT_VERSION = 2

# Pola internal untuk mengenali skema URL, alamat IPv4/hex, dan token alfanumerik.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$")
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_url(raw):
    """Normalisasikan URL tanpa mengubah isi path atau query.

    Skema yang berakhiran ``://``, garis miring di awal, awalan ``www.`` pada
    host, dan garis miring di ujung dibuang. Host dibuat huruf kecil; bagian
    setelah host selain garis miring di ujung dipertahankan.

    Args:
        raw: Nilai URL apa pun yang dapat diubah menjadi string.

    Returns:
        URL sebagai string ternormalisasi. Fungsi ini tidak memvalidasi URL.
    """
    u = str(raw).strip()
    u = _SCHEME_RE.sub("", u)
    u = u.lstrip("/")
    # pisahkan host dari sisanya supaya hanya host yang dijadikan huruf kecil
    m = re.search(r"[/?#]", u)
    host, rest = (u, "") if m is None else (u[: m.start()], u[m.start():])
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    rest = rest.rstrip("/")
    return host + rest


def split_url(norm):
    """Pisahkan URL ternormalisasi menjadi komponen utamanya.

    ``urlsplit`` menerima URL tanpa skema ketika diawali ``//``. Informasi
    user/password tidak dimasukkan ke host; fragment juga tidak dikembalikan.

    Args:
        norm: URL yang sudah diproses oleh :func:`normalize_url`.

    Returns:
        Tuple ``(host, port, path, query)``. Host berupa nama host tanpa
        kurung siku IPv6; port berupa string kosong jika tidak ada. Port yang
        tidak valid dapat menyebabkan ``urlsplit`` melempar ``ValueError``.
    """
    parsed = urlsplit("//" + norm)
    host = parsed.hostname or ""
    port = str(parsed.port) if parsed.port is not None else ""
    return host, port, parsed.path, parsed.query


def registered_domain(raw):
    """Ambil domain terdaftar untuk mengelompokkan URL berdasarkan situs.

    Contohnya, ``a.b.contoh.co.id`` dikelompokkan sebagai
    ``contoh.co.id``. Hasil ini dapat dipakai untuk mencegah URL dari situs
    yang sama tersebar ke partisi training dan pengujian.

    Args:
        raw: URL mentah yang akan diproses.

    Returns:
        Domain terdaftar jika domain dan suffix dikenali; selain itu host,
        atau ``"unknown"`` jika host kosong.
    """
    host, _, _, _ = split_url(normalize_url(raw))
    ext = _TLD(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return host or "unknown"


def _entropy(s):
    """Hitung entropi Shannon karakter dalam string.

    Nilai yang lebih tinggi menandakan distribusi karakter yang lebih merata.
    String kosong menghasilkan ``0.0``.
    """
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _longest_token(s):
    """Kembalikan panjang token alfanumerik terpanjang dalam string.

    Karakter selain huruf ASCII dan angka menjadi pemisah. Jika tidak ada
    token, hasilnya ``0``.
    """
    toks = [t for t in _TOKEN_RE.split(s.lower()) if t]
    return max((len(t) for t in toks), default=0)


def extract_features(raw):
    """Ekstrak semua fitur leksikal untuk satu URL.

    Ukuran yang bernama ``*_path`` hanya dihitung dari path; query diukur
    terpisah melalui panjang dan jumlah parameternya. Fitur dengan nama umum,
    seperti jumlah karakter khusus dan kata mencurigakan, dihitung pada URL
    ternormalisasi secara keseluruhan. ``n_suspicious_words`` menjumlahkan
    semua kemunculan setiap istilah pada :data:`SUSPICIOUS_WORDS`.

    Args:
        raw: URL mentah yang akan dianalisis; URL tidak diakses atau dibuka.

    Returns:
        Dictionary berisi satu nilai numerik untuk setiap nama pada
        :data:`FEATURE_NAMES`. Urutan kolom DataFrame ditetapkan oleh daftar
        tersebut, bukan oleh urutan akses dictionary.
    """
    norm = normalize_url(raw)
    host, port, path, query = split_url(norm)
    ext = _TLD(host)
    suffix = ext.suffix
    sub = ext.subdomain
    low_path = path.lower()
    low_query = query.lower()
    low_all = norm.lower()

    is_ip = bool(_IP_RE.match(host)) or bool(_HEX_IP_RE.match(host))
    n_digits_host = sum(ch.isdigit() for ch in host)
    reg_dom = f"{ext.domain}.{suffix}" if ext.domain and suffix else host
    # Merek di subdomain/path, bukan pada domain utama atau query.
    outside = sub + " " + low_path
    brand_outside = int(any(b in outside and b not in ext.domain.lower() for b in BRANDS))

    f = {
        "url_length": len(norm),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "n_dots_host": host.count("."),
        "n_subdomains": 0 if (is_ip or not sub) else sub.count(".") + 1,
        "n_hyphens_host": host.count("-"),
        "n_digits_host": n_digits_host,
        "digit_ratio_host": n_digits_host / len(host) if host else 0.0,
        "longest_host_token": _longest_token(host),
        "host_entropy": _entropy(host),
        "path_depth": len([p for p in path.split("/") if p]),
        "n_digits_path": sum(ch.isdigit() for ch in low_path),
        "n_hyphens_path": low_path.count("-"),
        "n_underscores": low_all.count("_"),
        "n_percent": low_all.count("%"),
        "n_equals": low_all.count("="),
        "n_ampersand": low_all.count("&"),
        "n_question": low_all.count("?"),
        "n_at": low_all.count("@"),
        "n_tilde": low_all.count("~"),
        "n_special_total": sum(low_all.count(c) for c in "_%=&?@~!$*,;+"),
        "path_entropy": _entropy(low_path),
        "longest_path_token": _longest_token(low_path),
        "has_ip": int(is_ip),
        "has_port": int(bool(port)),
        "has_double_slash": int("//" in path),
        "has_php": int(bool(re.search(r"\.php(?:/|$)", low_path))),
        "has_html": int(bool(re.search(r"\.html?", low_path))),
        "has_exe": int(bool(re.search(r"\.(exe|apk|scr|bat|msi|zip|rar)\b", low_path))),
        "is_shortener": int(reg_dom in SHORTENERS or host in SHORTENERS),
        "is_common_tld": int(suffix in COMMON_TLDS),
        "tld_length": len(suffix),
        "n_suspicious_words": sum(low_all.count(w) for w in SUSPICIOUS_WORDS),
        "brand_outside_domain": brand_outside,
        "n_query_params": len([q for q in low_query.split("&") if q]) if low_query else 0,
    }
    return f


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer scikit-learn dari URL mentah ke DataFrame fitur.

    Transformer ini stateless: tidak mempelajari parameter dari data. Dengan
    menaruhnya sebagai langkah pertama pipeline, pipeline dapat menerima URL
    mentah dan meneruskannya sebagai kolom fitur dalam urutan
    :data:`FEATURE_NAMES`.
    """

    def fit(self, X, y=None):
        """Penuhi antarmuka scikit-learn tanpa mempelajari parameter.

        Args:
            X: Data input; tidak dibaca karena transformer stateless.
            y: Label target opsional; tidak digunakan.

        Returns:
            Instance transformer ini agar dapat dirangkai dalam pipeline.
        """
        return self

    def transform(self, X):
        """Ubah setiap URL dalam ``X`` menjadi satu baris fitur.

        Args:
            X: Iterable URL mentah.

        Returns:
            DataFrame pandas dengan kolom sesuai urutan :data:`FEATURE_NAMES`.
        """
        rows = [extract_features(u) for u in X]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    def get_feature_names_out(self, input_features=None):
        """Kembalikan nama fitur keluaran dalam urutan tetap.

        ``input_features`` diterima untuk kompatibilitas antarmuka
        scikit-learn, tetapi tidak mengubah nama fitur yang dihasilkan.
        """
        return list(FEATURE_NAMES)
