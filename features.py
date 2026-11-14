"""
features.py - Normalisasi URL dan ekstraksi fitur leksikal.

Modul ini dipakai oleh DUA tempat:
  1. Notebook training  -> membuat fitur dari dataset.
  2. Aplikasi Flask     -> membuat fitur dari URL yang diketik pengguna.

Karena kodenya sama persis, preprocessing saat training dan saat prediksi
dijamin konsisten. Pipeline yang disimpan ke .joblib juga memakai kelas
URLFeatureExtractor dari file ini, jadi file ini HARUS ada di folder yang sama
dengan app.py saat model dimuat.
"""
import math
import re
from collections import Counter

import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

# Pakai daftar public suffix bawaan paket (tanpa akses internet).
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Kata yang sering muncul di URL phishing (halaman login / verifikasi palsu).
SUSPICIOUS_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "password", "passwd", "banking",
    "bank", "wallet", "webscr", "auth", "billing", "invoice", "payment",
    "recover", "unlock", "suspend", "support", "service", "webmail", "admin",
]

# Merek yang sering ditiru pelaku phishing.
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "gmail", "facebook", "instagram", "whatsapp", "netflix", "amazon", "ebay",
    "dropbox", "linkedin", "twitter", "yahoo", "chase", "wellsfargo", "dhl",
    "fedex", "usps", "steam", "binance", "coinbase", "metamask", "bca", "bri",
    "mandiri", "bni", "dana", "gopay", "shopee", "tokopedia",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "s.id", "t.ly", "tiny.cc",
    "bitly.com", "lnkd.in", "qr.ae", "v.gd", "shorte.st", "adf.ly",
}

COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co.uk", "ac.id", "co.id",
               "go.id", "id", "de", "fr", "jp", "ca", "au", "com.au", "us"}

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

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$")
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_url(raw):
    """Samakan bentuk URL: buang skema (http/https), 'www.', dan '/' di ujung.

    Alasan: di dataset, hampir semua URL legitimate berawalan 'https://www.'
    sementara phishing tidak. Kalau bagian itu dibiarkan, model hanya menghafal
    awalan tersebut dan bukan pola phishing yang sebenarnya.
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
    """Pecah URL yang sudah dinormalisasi menjadi (host, port, path, query)."""
    m = re.search(r"[/?#]", norm)
    hostport, rest = (norm, "") if m is None else (norm[: m.start()], norm[m.start():])
    if "@" in hostport:                     # user:pass@host
        hostport = hostport.rsplit("@", 1)[1]
    port = ""
    pm = re.match(r"^(.*):(\d{1,5})$", hostport)
    if pm:
        hostport, port = pm.group(1), pm.group(2)
    rest = rest.split("#", 1)[0]
    path, _, query = rest.partition("?")
    return hostport, port, path, query


def registered_domain(raw):
    """Domain utama, mis. 'a.b.contoh.co.id' -> 'contoh.co.id'.

    Dipakai sebagai 'grup' saat membagi data, supaya URL dari domain yang sama
    tidak tersebar ke data latih dan data uji sekaligus.
    """
    host, _, _, _ = split_url(normalize_url(raw))
    ext = _TLD(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return host or "unknown"


def _entropy(s):
    """Entropi Shannon: makin acak susunan karakternya, makin tinggi nilainya."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _longest_token(s):
    toks = [t for t in _TOKEN_RE.split(s.lower()) if t]
    return max((len(t) for t in toks), default=0)


def extract_features(raw):
    """Ubah satu URL menjadi dict fitur numerik (urutannya = FEATURE_NAMES)."""
    norm = normalize_url(raw)
    host, port, path, query = split_url(norm)
    ext = _TLD(host)
    suffix = ext.suffix
    sub = ext.subdomain
    low_path = (path + "?" + query).lower() if query else path.lower()
    low_all = norm.lower()

    is_ip = bool(_IP_RE.match(host)) or bool(_HEX_IP_RE.match(host))
    n_digits_host = sum(ch.isdigit() for ch in host)
    reg_dom = f"{ext.domain}.{suffix}" if ext.domain and suffix else host
    # merek disebut di subdomain / path, padahal domain utamanya bukan merek itu
    outside = (sub + " " + low_path).lower()
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
        "has_php": int(".php" in low_path),
        "has_html": int(".htm" in low_path),
        "has_exe": int(bool(re.search(r"\.(exe|apk|scr|bat|msi|zip|rar)\b", low_path))),
        "is_shortener": int(reg_dom in SHORTENERS or host in SHORTENERS),
        "is_common_tld": int(suffix in COMMON_TLDS),
        "tld_length": len(suffix),
        "n_suspicious_words": sum(w in low_all for w in SUSPICIOUS_WORDS),
        "brand_outside_domain": brand_outside,
        "n_query_params": len([q for q in query.split("&") if q]) if query else 0,
    }
    return f


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer scikit-learn: daftar URL mentah -> DataFrame fitur.

    Tidak punya parameter yang dipelajari (stateless), jadi fit() tidak
    melakukan apa-apa. Diletakkan sebagai langkah pertama Pipeline supaya
    file .joblib menerima URL mentah secara langsung.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_features(u) for u in X]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    def get_feature_names_out(self, input_features=None):
        return list(FEATURE_NAMES)
