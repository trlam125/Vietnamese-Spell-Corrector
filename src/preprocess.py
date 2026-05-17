import re
import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def remove_accents(text: str) -> str:
    # Replace đ/Đ first (they don't decompose in NFD)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", text)

def is_bad_line(line: str) -> bool:
    lower_line = line.lower()
    bad_keywords = ["dịch phụ đề", "phụ đề", "subtitle", "subtitles", "subteam", "facebook", "fanpage", "website", "www", "http", ".com", ".net", ".org"]
    for kw in bad_keywords:
        if kw in lower_line:
            return True
    return False

def clean_wiki_markup(text: str) -> str:
    """Làm sạch markup cơ bản trong Wikipedia XML. Không cần hoàn hảo 100%, chỉ cần đủ tốt để lấy từ vựng."""
    return text

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    if is_bad_line(text):
        return ""
    text = text.lower()
    return text.strip()

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return text.split()

def preprocess_line(line: str) -> list[str]:
    clean = clean_text(line)
    return tokenize(clean)