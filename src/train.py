import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

try:
    from .preprocess import clean_text, tokenize, remove_accents
except ImportError:
    from preprocess import clean_text, tokenize, remove_accents


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILES = [
    BASE_DIR / "data" / "vi.txt",
    BASE_DIR / "data" / "viwiki-latest-pages-articles.xml",
]

MODEL_DIR = BASE_DIR / "models"
WORD_FREQ_PATH = MODEL_DIR / "word_freq.pkl"
ACCENT_MAP_PATH = MODEL_DIR / "accent_map.pkl"

BIGRAM_FREQ_PATH = MODEL_DIR / "bigram_freq.pkl"
WORD_POS_MAP_PATH = MODEL_DIR / "word_pos_map.pkl"

MIN_WORD_LEN = 1
MAX_WORD_LEN = 30


def is_valid_word(word: str) -> bool:
    if not word:
        return False

    if len(word) < MIN_WORD_LEN:
        return False

    if len(word) > MAX_WORD_LEN:
        return False

    return True


def update_models_from_words(
    words: list[str],
    word_freq: Counter,
    accent_map: dict[str, set[str]],
    bigram_freq: Counter = None,
) -> int:
    valid_count = 0

    for word in words:
        if not is_valid_word(word):
            continue

        word_freq[word] += 1

        no_accent = remove_accents(word)
        if no_accent:
            accent_map[no_accent].add(word)

        valid_count += 1

    # Build bigrams from consecutive valid words
    if bigram_freq is not None:
        valid_words = [w for w in words if is_valid_word(w)]
        for i in range(len(valid_words) - 1):
            bigram = (valid_words[i], valid_words[i + 1])
            bigram_freq[bigram] += 1

    return valid_count


def train_text_file(
    file_path: Path,
    word_freq: Counter,
    accent_map: dict[str, set[str]],
    bigram_freq: Counter = None,
) -> tuple[int, int]:
    print(f"\nĐang đọc text file: {file_path}")

    total_lines = 0
    total_words = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1

            clean = clean_text(line)
            words = tokenize(clean)

            total_words += update_models_from_words(
                words,
                word_freq,
                accent_map,
                bigram_freq,
            )

            if total_lines % 100_000 == 0:
                print(
                    f"  Đã xử lý {total_lines:,} dòng | "
                    f"{total_words:,} từ hợp lệ"
                )

    return total_lines, total_words


def strip_namespace(tag: str) -> str:
    """
    XML Wikipedia có namespace kiểu:
    {http://www.mediawiki.org/xml/export-0.11/}text

    Hàm này lấy phần cuối:
    text
    """
    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def iter_wiki_texts(xml_path: Path):
    """
    Đọc Wikipedia XML theo kiểu streaming.
    Không load toàn bộ file 6GB vào RAM.
    """
    context = ET.iterparse(xml_path, events=("end",))

    for event, elem in context:
        tag = strip_namespace(elem.tag)

        if tag == "page":
            title = ""
            article_text = ""

            for child in elem:
                child_tag = strip_namespace(child.tag)

                if child_tag == "title" and child.text:
                    title = child.text

                elif child_tag == "text" and child.text:
                    article_text = child.text

            yield title
            yield article_text

            elem.clear()


def train_wiki_xml_file(
    file_path: Path,
    word_freq: Counter,
    accent_map: dict[str, set[str]],
    bigram_freq: Counter = None,
) -> tuple[int, int]:
    print(f"\nĐang đọc Wikipedia XML file: {file_path}")
    print("File XML lớn nên bước này có thể chạy khá lâu.")

    total_pages = 0
    total_words = 0

    for text in iter_wiki_texts(file_path):
        if not text:
            continue

        clean = clean_text(text)
        words = tokenize(clean)

        total_words += update_models_from_words(
            words,
            word_freq,
            accent_map,
            bigram_freq,
        )

        total_pages += 1

        if total_pages % 10_000 == 0:
            print(
                f"  Đã xử lý {total_pages:,} block wiki | "
                f"{total_words:,} từ hợp lệ"
            )

    return total_pages, total_words


def save_models(
    word_freq: Counter,
    accent_map: dict[str, set[str]],
    bigram_freq: Counter = None,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Convert set -> sorted list để pickle ổn định, nhẹ hơn khi load
    accent_map_to_save = {
        key: sorted(values, key=lambda word: word_freq.get(word, 0), reverse=True)
        for key, values in accent_map.items()
    }

    with open(WORD_FREQ_PATH, "wb") as f:
        pickle.dump(word_freq, f)

    with open(ACCENT_MAP_PATH, "wb") as f:
        pickle.dump(accent_map_to_save, f)

    if bigram_freq is not None:
        # Convert tuple keys to string "w1\tw2" for pickle compatibility
        bigram_to_save = {
            f"{w1}\t{w2}": count
            for (w1, w2), count in bigram_freq.items()
        }
        with open(BIGRAM_FREQ_PATH, "wb") as f:
            pickle.dump(bigram_to_save, f)


def train() -> None:
    print("Bắt đầu train spell corrector...\n")

    word_freq = Counter()
    accent_map = defaultdict(set)
    bigram_freq = Counter()

    total_sources = 0
    total_units = 0
    total_words = 0

    for file_path in DATA_FILES:
        if not file_path.exists():
            print(f"Bỏ qua vì không tìm thấy file: {file_path}")
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".xml":
            # Wikipedia: only count word_freq and accent_map, skip bigrams
            # (bigram memory explodes with 7GB XML)
            units, words = train_wiki_xml_file(
                file_path,
                word_freq,
                accent_map,
                bigram_freq=None,
            )
        else:
            # OpenSubtitles: count everything including bigrams
            units, words = train_text_file(
                file_path,
                word_freq,
                accent_map,
                bigram_freq,
            )

        total_sources += 1
        total_units += units
        total_words += words

    if total_sources == 0:
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu nào trong DATA_FILES."
        )

    print("\nĐang lưu model...")
    save_models(word_freq, accent_map, bigram_freq)

    print("\nTrain hoàn tất.")
    print(f"Số nguồn dữ liệu đã đọc: {total_sources}")
    print(f"Tổng số dòng/block đã xử lý: {total_units:,}")
    print(f"Tổng số từ hợp lệ: {total_words:,}")
    print(f"Số từ khác nhau: {len(word_freq):,}")
    print(f"Số key không dấu trong accent_map: {len(accent_map):,}")
    print(f"Số bigram khác nhau: {len(bigram_freq):,}")

    print("\nModel đã lưu:")
    print(f"- {WORD_FREQ_PATH}")
    print(f"- {ACCENT_MAP_PATH}")
    print(f"- {BIGRAM_FREQ_PATH}")

    print("\nTop 30 từ phổ biến:")
    for word, count in word_freq.most_common(30):
        print(f"{word}: {count}")


if __name__ == "__main__":
    train()
