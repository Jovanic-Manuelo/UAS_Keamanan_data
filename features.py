
"""
features.py - Normalisasi URL dan ekstraksi fitur leksikal untuk deteksi phishing.

Modul ini digunakan saat training dan prediksi agar ekstraksi fitur konsisten.
URL tidak diakses atau dibuka; seluruh fitur diekstrak dari teks URL.

Catatan:
- FEATURE_NAMES mempertahankan 36 fitur dan urutannya.
- FEATURE_CONTRACT_VERSION dinaikkan menjadi 3 karena beberapa definisi fitur berubah.
- Model lama harus dilatih ulang menggunakan versi modul ini.
"""

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# KONFIGURASI
# ============================================================

# Menggunakan public suffix list bawaan paket, tanpa akses internet.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "security", "account", "update", "confirm", "password",
    "passwd", "banking", "bank", "wallet", "webscr", "auth", "billing",
    "invoice", "payment", "recover", "unlock", "suspend", "support",
    "service", "webmail", "admin",
]

BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook",
    "google", "gmail", "facebook", "instagram", "whatsapp", "netflix",
    "amazon", "ebay", "dropbox", "linkedin", "twitter", "yahoo",
    "chase", "wellsfargo", "dhl", "fedex", "usps", "steam", "binance",
    "coinbase", "metamask", "bca", "bri", "mandiri", "bni", "dana",
    "gopay", "shopee", "tokopedia",
]

# Domain resmi yang dikenali untuk mengurangi false positive.
# Daftar ini bukan daftar lengkap seluruh domain resmi setiap merek.
BRAND_OFFICIAL_DOMAINS = {
    "paypal": {"paypal.com", "paypalobjects.com"},
    "apple": {"apple.com", "icloud.com"},
    "icloud": {"icloud.com", "apple.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com",
                  "microsoftonline.com", "outlook.com"},
    "office365": {"office.com", "microsoft.com", "microsoftonline.com"},
    "outlook": {"outlook.com", "live.com", "microsoft.com"},
    "google": {"google.com", "google.co.id", "google.co.uk"},
    "gmail": {"gmail.com", "google.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "netflix": {"netflix.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.co.id"},
    "ebay": {"ebay.com"},
    "dropbox": {"dropbox.com"},
    "linkedin": {"linkedin.com"},
    "twitter": {"twitter.com", "x.com"},
    "yahoo": {"yahoo.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "dhl": {"dhl.com"},
    "fedex": {"fedex.com"},
    "usps": {"usps.com"},
    "steam": {"steampowered.com", "steamcommunity.com"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "metamask": {"metamask.io"},
    "bca": {"bca.co.id"},
    "bri": {"bri.co.id"},
    "mandiri": {"bankmandiri.co.id", "mandiri.co.id"},
    "bni": {"bni.co.id"},
    "dana": {"dana.id"},
    "gopay": {"gopay.co.id", "gojek.com"},
    "shopee": {"shopee.co.id", "shopee.com"},
    "tokopedia": {"tokopedia.com"},
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy",
    "s.id", "t.ly", "tiny.cc", "bitly.com", "lnkd.in", "qr.ae",
    "v.gd", "shorte.st", "adf.ly",
}

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "co.uk", "ac.id", "co.id",
    "go.id", "id", "de", "fr", "jp", "ca", "au", "com.au", "us",
}

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

# Definisi beberapa fitur berubah. Model lama tidak boleh langsung digunakan.
FEATURE_CONTRACT_VERSION = 3

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


# ============================================================
# NORMALISASI DAN PARSING URL
# ============================================================

def normalize_url(raw):
    """
    Normalisasi URL:
    - Menghapus skema seperti http:// dan https://.
    - Menghapus garis miring di awal.
    - Mengubah host menjadi huruf kecil.
    - Menghapus awalan www. pada host.
    - Menghapus garis miring di akhir URL.

    Path, query, dan fragment tetap dipertahankan.
    Fungsi ini tidak melakukan validasi keamanan URL.
    """
    if raw is None:
        raise ValueError("URL tidak boleh kosong.")

    # Hindari mengubah NaN menjadi string 'nan'.
    if isinstance(raw, float) and math.isnan(raw):
        raise ValueError("URL tidak boleh bernilai NaN.")

    u = str(raw).strip()

    if not u:
        raise ValueError("URL tidak boleh kosong.")

    u = _SCHEME_RE.sub("", u)
    u = u.lstrip("/")

    # Pisahkan host dari path, query, atau fragment.
    match = re.search(r"[/?#]", u)

    if match is None:
        host, rest = u, ""
    else:
        host, rest = u[:match.start()], u[match.start():]

    host = host.lower()

    if host.startswith("www."):
        host = host[4:]

    rest = rest.rstrip("/")

    return host + rest


def split_url(norm):
    """
    Memisahkan URL menjadi (host, port, path, query).

    urlsplit menangani userinfo, port, dan IPv6 dengan lebih baik
    daripada pemisahan menggunakan regex.

    Fragment sengaja tidak dimasukkan ke path atau query.
    Fragment tetap dipertahankan pada URL ternormalisasi untuk fitur
    yang menghitung keseluruhan URL.
    """
    try:
        parsed = urlsplit("//" + norm)

        host = parsed.hostname or ""

        # Akses .port dapat menghasilkan ValueError jika port tidak valid.
        port_value = parsed.port
        port = str(port_value) if port_value is not None else ""

    except ValueError as exc:
        raise ValueError(f"Format URL atau port tidak valid: {exc}") from exc

    if not host:
        raise ValueError("URL tidak memiliki hostname.")

    return host, port, parsed.path, parsed.query


def registered_domain(raw):
    """
    Mengambil domain terdaftar untuk kebutuhan grouping.

    Contoh:
    a.b.example.co.id -> example.co.id
    """
    host, _, _, _ = split_url(normalize_url(raw))

    # tldextract tidak dimaksudkan untuk mengolah alamat IP.
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass

    ext = _TLD(host)

    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"

    return host or "unknown"


# ============================================================
# FUNGSI BANTU EKSTRAKSI FITUR
# ============================================================

def _entropy(s):
    """Menghitung Shannon entropy karakter dalam string."""
    if not s:
        return 0.0

    n = len(s)

    return -sum(
        (count / n) * math.log2(count / n)
        for count in Counter(s).values()
    )


def _longest_token(s):
    """Mengembalikan panjang token alfanumerik terpanjang."""
    tokens = _TOKEN_RE.findall(s.lower())

    return max((len(token) for token in tokens), default=0)


def _is_ip_address(host):
    """
    Memeriksa apakah host merupakan alamat IP yang valid.

    Mendukung IPv4 dan IPv6 menggunakan ipaddress.
    Format integer hexadecimal juga diperiksa jika nilainya
    berada dalam rentang integer IPv4.
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Beberapa URL menggunakan representasi integer hexadecimal IPv4.
    if re.fullmatch(r"0x[0-9a-fA-F]+", host):
        try:
            value = int(host, 16)
            return 0 <= value <= 0xFFFFFFFF
        except ValueError:
            return False

    return False


def _count_suspicious_words(text):
    """
    Menghitung total kemunculan kata mencurigakan sebagai token utuh.

    Contoh:
    - 'login' dihitung.
    - 'secure-login' menghitung dua istilah jika keduanya ada di daftar.
    - 'myloginpage' tidak dihitung sebagai 'login'.
    - Kemunculan berulang dihitung berulang.
    """
    total = 0

    for word in SUSPICIOUS_WORDS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(word)
            + r"(?![a-z0-9])"
        )

        total += len(re.findall(pattern, text))

    return total


def _brand_outside_domain(host, path, query, fragment, registered):
    """
    Mendeteksi penyebutan merek pada URL yang bukan domain resmi merek.

    Pencocokan dilakukan berdasarkan token, bukan substring.
    Dengan demikian:
    - paypal-secure-example.com dapat terdeteksi.
    - paypal.evil.com dapat terdeteksi.
    - pineapple.com tidak dianggap mengandung token 'apple'.

    Daftar domain resmi bersifat terbatas dan perlu diperbarui
    jika cakupan merek proyek bertambah.
    """
    text = " ".join([host, path, query, fragment]).lower()
    url_tokens = set(_TOKEN_RE.findall(text))

    for brand in BRANDS:
        if brand not in url_tokens:
            continue

        official_domains = BRAND_OFFICIAL_DOMAINS.get(brand, set())

        if registered.lower() in official_domains:
            continue

        # Jika merek muncul pada URL tetapi domain terdaftarnya
        # bukan domain resmi merek tersebut, tandai sebagai indikasi.
        return 1

    return 0


# ============================================================
# EKSTRAKSI FITUR UTAMA
# ============================================================

def extract_features(raw):
    """
    Mengekstrak 36 fitur numerik dari satu URL.

    Konsistensi cakupan:
    - Fitur *_path hanya menggunakan path.
    - query_length dan n_query_params hanya menggunakan query.
    - n_suspicious_words dan fitur karakter global menggunakan
      URL ternormalisasi, termasuk query dan fragment.
    - Fragment tidak dimasukkan ke path atau query.
    """
    norm = normalize_url(raw)

    host, port, path, query = split_url(norm)

    # Ambil fragment secara terpisah.
    try:
        parsed = urlsplit("//" + norm)
        fragment = parsed.fragment
    except ValueError as exc:
        raise ValueError(f"Fragment URL tidak dapat diparsing: {exc}") from exc

    ext = _TLD(host)

    suffix = ext.suffix
    subdomain = ext.subdomain

    low_host = host.lower()
    low_path = path.lower()
    low_query = query.lower()
    low_fragment = fragment.lower()
    low_all = norm.lower()

    is_ip = _is_ip_address(host)

    n_digits_host = sum(ch.isdigit() for ch in host)

    registered = (
        f"{ext.domain}.{suffix}"
        if ext.domain and suffix
        else host
    )

    brand_outside = _brand_outside_domain(
        host=low_host,
        path=low_path,
        query=low_query,
        fragment=low_fragment,
        registered=registered,
    )

    # Pemeriksaan ekstensi file menggunakan batas ekstensi.
    # Query dan fragment tidak ikut diperiksa sebagai bagian path.
    has_php = bool(
        re.search(r"\.php(?:$|/)", low_path)
    )

    has_html = bool(
        re.search(r"\.(?:html?|xhtml)(?:$|/)", low_path)
    )

    has_exe = bool(
        re.search(
            r"\.(?:exe|apk|scr|bat|msi|zip|rar)(?:$|/)",
            low_path,
        )
    )

    f = {
        "url_length": len(norm),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),

        "n_dots_host": host.count("."),
        "n_subdomains": (
            0 if (is_ip or not subdomain)
            else subdomain.count(".") + 1
        ),
        "n_hyphens_host": host.count("-"),
        "n_digits_host": n_digits_host,
        "digit_ratio_host": (
            n_digits_host / len(host) if host else 0.0
        ),
        "longest_host_token": _longest_token(host),
        "host_entropy": _entropy(host),

        "path_depth": len([
            part for part in path.split("/") if part
        ]),
        "n_digits_path": sum(ch.isdigit() for ch in low_path),
        "n_hyphens_path": low_path.count("-"),

        "n_underscores": low_all.count("_"),
        "n_percent": low_all.count("%"),
        "n_equals": low_all.count("="),
        "n_ampersand": low_all.count("&"),
        "n_question": low_all.count("?"),
        "n_at": low_all.count("@"),
        "n_tilde": low_all.count("~"),

        "n_special_total": sum(
            low_all.count(char)
            for char in "_%=&?@~!$*,;+"
        ),

        "path_entropy": _entropy(low_path),
        "longest_path_token": _longest_token(low_path),

        "has_ip": int(is_ip),
        "has_port": int(bool(port)),
        "has_double_slash": int("//" in path),

        "has_php": int(has_php),
        "has_html": int(has_html),
        "has_exe": int(has_exe),

        "is_shortener": int(
            registered in SHORTENERS or host in SHORTENERS
        ),
        "is_common_tld": int(suffix in COMMON_TLDS),
        "tld_length": len(suffix),

        "n_suspicious_words": _count_suspicious_words(low_all),

        "brand_outside_domain": brand_outside,

        "n_query_params": (
            len([item for item in low_query.split("&") if item])
            if low_query
            else 0
        ),
    }

    return f


# ============================================================
# SCIKIT-LEARN TRANSFORMER
# ============================================================

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer stateless dari URL mentah menjadi DataFrame fitur.

    Tidak ada parameter yang dipelajari dari data training.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_features(url) for url in X]

        return pd.DataFrame(
            rows,
            columns=FEATURE_NAMES,
        )

    def get_feature_names_out(self, input_features=None):
        return list(FEATURE_NAMES)
