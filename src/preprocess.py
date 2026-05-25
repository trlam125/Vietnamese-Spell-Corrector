import re
import unicodedata


WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+", re.UNICODE)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", text)


def is_bad_line(line: str) -> bool:
    lower_line = line.lower()
    bad_keywords = [
        "dịch phụ đề",
        "phụ đề",
        "subtitle",
        "subtitles",
        "subteam",
        "facebook",
        "fanpage",
        "website",
        "www",
        "http",
        ".com",
        ".net",
        ".org",
    ]
    return any(kw in lower_line for kw in bad_keywords)


def clean_wiki_markup(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(?is)<ref\b[^>/]*(?:/>|>.*?</ref>)", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"(?is)\{\|.*?\|\}", " ", text)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?s)\{\{[^{}]*\}\}", " ", text)

    text = re.sub(r"\[\[(?:tập tin|file|image|hình|category|thể loại):[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"'{2,}", "", text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    text = clean_wiki_markup(text)
    if is_bad_line(text):
        return ""
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return WORD_RE.findall(text)


def preprocess_line(line: str) -> list[str]:
    clean = clean_text(line)
    return tokenize(clean)
