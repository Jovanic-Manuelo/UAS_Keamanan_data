"""Uji integrasi fungsi web Flask melalui test client lokal.

Jalankan ``python uji_fungsi.py`` setelah artefak model dan dependensi aplikasi
tersedia. Skrip menguji endpoint dan alur UI, mencetak status setiap kasus,
serta menulis hasilnya ke ``hasil_uji_fungsi.csv`` di direktori kerja saat ini.
Skrip ini bukan evaluasi akurasi ilmiah model dan tidak membuka URL: alamat
simulasi hanya dikirim sebagai teks ke test client Flask.

Kasus dibagi menjadi input valid, input yang harus ditolak, konsistensi
normalisasi, serta halaman dan fitur aplikasi. Kasus dijalankan pada tingkat
modul, sehingga mengimpor skrip ini juga menjalankan seluruh rangkaian tes dan
menulis berkas CSV.
"""
import csv
import io

from app import MAX_BATCH, app

# Test client menjalankan route Flask tanpa membuka server atau port jaringan.
client = app.test_client()
# Baris laporan yang nantinya ditulis ke CSV.
rows = []


def catat(no, kelompok, masukan, harapan, aktual, ket=""):
    """Tambahkan satu hasil perbandingan ke laporan pengujian.

    ``harapan`` dan ``aktual`` dibandingkan langsung; hasil kesamaan dicatat
    sebagai ``LULUS``, selain itu ``GAGAL``.
    """
    status = "LULUS" if harapan == aktual else "GAGAL"
    rows.append([no, kelompok, masukan, harapan, aktual, status, ket])


def prediksi(url):
    """Kirim URL ke endpoint prediksi dan sederhanakan hasil respons.

    Returns:
        ``(label, persen)`` untuk prediksi berhasil atau
        ``("ditolak", pesan_error)`` untuk input yang tidak valid.
    """
    r = client.post("/api/predict", json={"url": url})
    d = r.get_json()
    return (d["label"], d["percent"]) if d["ok"] else ("ditolak", d["error"])


# A. Contoh URL simulasi yang diharapkan masuk kelas legitimate atau phishing.
VALID = [
    ("https://www.wikipedia.org/", "legitimate"),
    ("https://en.wikipedia.org/wiki/Phishing", "legitimate"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "legitimate"),
    ("https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do", "legitimate"),
    ("https://github.com/pallets/flask", "legitimate"),
    ("https://www.kemdikbud.go.id", "legitimate"),
    ("https://www.instagram.com/accounts/login/", "legitimate"),
    ("http://klikbca-verifikasi.xyz/login.php", "phishing"),
    ("http://secure-paypal.com.account-update.info/webscr/cmd=_login", "phishing"),
    ("http://203.0.113.45/~user/bank/signin.html", "phishing"),
    ("http://appleid.apple.com.verify-account.support/login", "phishing"),
    ("https://netflix-billing-update.com/account/payment", "phishing"),
    ("http://example.com/wp-content/plugins/x/office365/login.php", "phishing"),
    ("http://dana-kaget-gratis.top/claim?id=88213", "phishing"),
]
for i, (url, harapan) in enumerate(VALID, 1):
    label, info = prediksi(url)
    catat(f"A{i}", "input valid", url, harapan, label, f"prob. phishing {info}%" if label != "ditolak" else info)

# B. Input dengan skema, host, atau isi tidak valid yang harus ditolak.
INVALID = [
    ("", "kosong"),
    ("     ", "hanya spasi"),
    ("halo ini bukan url", "kalimat biasa"),
    ("contoh", "satu kata tanpa titik"),
    ("ftp://files.example.com/data.zip", "skema ftp"),
    ("javascript:alert(1)", "skema javascript"),
    ("http://", "skema tanpa domain"),
    ("http://999.300.1.1/login", "alamat IP tidak valid"),
    ("http://example.com:abc/login", "port bukan angka"),
    ("http://example.com:65536/login", "port di luar rentang"),
    ("http://exa mple.com/login", "spasi di tengah"),
    ("http://" + "a" * 2100 + ".com", "lebih dari 2000 karakter"),
    ("http://situs.c0m1", "TLD tidak valid"),
]
for i, (url, jenis) in enumerate(INVALID, 1):
    label, info = prediksi(url)
    tampil = url if len(url) < 60 else url[:40] + f"... ({len(url)} karakter)"
    catat(f"B{i}", f"tidak valid: {jenis}", repr(tampil), "ditolak", label, info if label == "ditolak" else f"prob. {info}%")

# C. Variasi skema, www, dan kapitalisasi harus menghasilkan probabilitas sama.
varian = ["https://www.wikipedia.org/", "http://wikipedia.org", "wikipedia.org", "WWW.Wikipedia.org/"]
probs = {prediksi(v)[1] for v in varian}
catat("C1", "konsistensi", "4 penulisan wikipedia.org (http/https/www/huruf besar)", "probabilitas sama",
      "probabilitas sama" if len(probs) == 1 else "berbeda", f"nilai: {sorted(probs)}")

# D. Periksa halaman, riwayat, unggahan, batas batch, dan escaping input.
r = client.get("/")
catat("D1", "halaman", "GET /", 200, r.status_code)
r = client.post("/", data={"url": "http://klikbca-verifikasi.xyz/login.php"})
catat("D2", "halaman", "POST / (form cek satu URL)", True, "Terindikasi PHISHING" in r.get_data(as_text=True), "hasil tampil di halaman")
r = client.post("/", data={"url": "<script>alert(1)</script>"})
catat("D3", "halaman", "POST / dengan tag <script>", True,
      "<script>alert(1)</script>" not in r.get_data(as_text=True), "input di-escape, tidak dijalankan")

teks = "wikipedia.org\nhttp://klikbca-verifikasi.xyz/login.php\nbukan url"
html = client.post("/batch", data={"urls": teks}).get_data(as_text=True)
catat("D4", "cek banyak", "3 baris teks (1 aman, 1 phishing, 1 tidak valid)", True,
      all(s in html for s in ["<b>3</b>", "tidak valid"]), "ringkasan 3 diperiksa + baris tidak valid")

berkas = io.BytesIO(b"url,catatan\nhttps://github.com/pallets/flask,a\nhttp://203.0.113.45/bank/signin.html,b\n")
html = client.post("/batch", data={"file": (berkas, "daftar.csv")}, content_type="multipart/form-data").get_data(as_text=True)
catat("D5", "cek banyak", "unggah CSV 2 baris dengan kolom 'url'", True, "<b>2</b>" in html, "kolom url terbaca")

berkas = io.BytesIO(b"abc")
html = client.post("/batch", data={"file": (berkas, "gambar.png")}, content_type="multipart/form-data").get_data(as_text=True)
catat("D6", "cek banyak", "unggah file .png", True, "harus berformat" in html, "ditolak dengan pesan")

banyak = "\n".join(f"situs{i}.com" for i in range(MAX_BATCH + 1))
html = client.post("/batch", data={"urls": banyak}).get_data(as_text=True)
catat("D7", "cek banyak", f"{MAX_BATCH + 1} URL (melebihi batas)", True, "Terlalu banyak URL" in html, "ditolak dengan pesan")

html = client.get("/history").get_data(as_text=True)
catat("D8", "riwayat", "GET /history setelah pengecekan", True, "klikbca-verifikasi.xyz" in html, "pengecekan sebelumnya tercatat")
r = client.get("/model")
catat("D9", "info model", "GET /model", 200, r.status_code)

# Tulis laporan lengkap agar hasil dapat ditinjau atau dipakai dalam tabel.
with open("hasil_uji_fungsi.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["no", "kelompok", "input", "hasil_diharapkan", "hasil_aktual", "status", "keterangan"])
    w.writerows(rows)

# Cetak ringkasan per kasus dan jumlah keseluruhan ke terminal.
for r in rows:
    print(f"{r[0]:<4}{r[5]:<6} {str(r[2])[:62]:<63} harap={r[3]} | aktual={r[4]} | {r[6]}")
lulus = sum(r[5] == "LULUS" for r in rows)
print(f"\n{lulus} dari {len(rows)} uji lulus. Tersimpan di hasil_uji_fungsi.csv")
