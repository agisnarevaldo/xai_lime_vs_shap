import re
import unicodedata
import sys

try:
    import emoji
except ImportError:
    print("WARNING: emoji library not available. Mocking emoji demojize.")
    class MockEmoji:
        def demojize(self, text, delimiters=(":", ":")):
            # mock replacements for testing
            text = text.replace("😂", ":face_with_tears_of_joy:")
            text = text.replace("👍", ":thumbs_up:")
            return text
    emoji = MockEmoji()

sys.stdout.reconfigure(encoding='utf-8')

# Slang Dictionary
SLANG_MAP = {
    "yg": "yang",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "dgn": "dengan",
    "kalo": "kalau",
    "jgn": "jangan",
    "tp": "tapi",
    "sm": "sama",
    "sdh": "sudah",
    "krn": "karena",
    "utk": "untuk",
    "sy": "saya",
    "td": "tadi",
    "smg": "semoga",
    "blm": "belum",
    "jd": "jadi",
    "aja": "saja",
    "dah": "sudah",
    "udh": "sudah",
    "gw": "saya",
    "gua": "saya",
    "lu": "kamu",
    "elo": "kamu",
    "msh": "masih",
    "skrg": "sekarang",
    "dpt": "dapat",
    "dr": "dari",
    "km": "kamu",
    "pake": "pakai",
    "karna": "karena",
    "g": "tidak",
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode Normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Safe removal of corrupted character U+FFFD
    text = text.replace('\ufffd', ' ')
    
    # 3. Emoji Demojization (e.g. 😂 -> :face_with_tears_of_joy:)
    try:
        text = emoji.demojize(text)
    except Exception:
        pass
    
    text = text.lower()
    
    # 4. Remove URL and mentions
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[\w_]+', ' ', text)
    
    # 5. Clean punctuation but preserve spaces, !?., and emoji delimiters (: and _)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,:_]', ' ', text)
    
    # 6. Slang normalization (word-by-word replacement)
    words = text.split()
    normalized_words = []
    for w in words:
        clean_w = re.sub(r'[^a-zA-Z0-9]', '', w)
        if clean_w in SLANG_MAP:
            if w.startswith(':') and w.endswith(':'):
                normalized_words.append(w)  # Keep emoji text intact
            else:
                normalized_words.append(SLANG_MAP[clean_w])
        else:
            normalized_words.append(w)
    text = " ".join(normalized_words)
    
    # 7. Convert emoji text separators to spaces (e.g. :face_with_tears_of_joy: -> face with tears of joy)
    text = text.replace(':', ' ').replace('_', ' ')
    
    # 8. Clean up whitespace and repeated punctuation
    text = re.sub(r'([!?.,]){2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test cases
test_cases = [
    "kecewa penjual ga tanggung jawab komplen berhari\ufffd gada solusi",
    "bener\ufffd mantap deh 👍",
    "Barang bagus bgt, gk nyesel beli disini. mantap!",
    "Td nya sy request utk cancel mo ganti p\ufffdbyran nya pake gopay",
    "sangat mengecewakan 😂😂😂!!!",
    "pas beli nanya dulu tipe untuk macbook pro 13inch M1 yang mana di variasi nya, kata penjualnya yang \ufffdnew pro 13\ufffd."
]

print("=== VERIFY CLEANING FUNCTION ===")
for tc in test_cases:
    print(f"Original : {repr(tc)}")
    print(f"Cleaned  : {repr(clean_text(tc))}")
    print("-" * 50)
