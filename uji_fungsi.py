
from features import (
    extract_features,
    registered_domain,
    FEATURE_NAMES,
    FEATURE_CONTRACT_VERSION,
)

test_urls = {
    "valid_ipv4": "http://192.168.1.1/login",
    "invalid_ipv4": "http://999.999.999.999/login",
    "valid_ipv6": "http://[2001:db8::1]/login",
    "suspicious_domain": "https://paypal-secure-example.com/login",
    "official_paypal": "https://www.paypal.com/login",
    "substring_only": "https://example.com/myloginpage",
    "repeated_words": "https://example.com/login/login/login",
    "html_extension": "https://example.com/index.html",
    "html_false_match": "https://example.com/index.htmlbackup",
    "php_extension": "https://example.com/login.php",
    "php_query_only": "https://example.com/login?file=.php",
    "fragment_brand": "https://example.com/#paypal-login",
}

for name, url in test_urls.items():
    try:
        features = extract_features(url)

        print(f"\n{name}: {url}")
        print("has_ip:", features["has_ip"])
        print("n_suspicious_words:", features["n_suspicious_words"])
        print("brand_outside_domain:", features["brand_outside_domain"])
        print("has_html:", features["has_html"])
        print("has_php:", features["has_php"])

    except ValueError as e:
        print(f"\n{name}: URL tidak dapat diproses: {e}")

# Pastikan jumlah dan urutan fitur tetap sama.
sample = extract_features("https://example.com/login")

assert len(sample) == 36
assert list(sample.keys()) == FEATURE_NAMES

print("\nFeature contract version:", FEATURE_CONTRACT_VERSION)
print("Jumlah fitur:", len(FEATURE_NAMES))
print("Semua pemeriksaan struktur fitur berhasil.")
