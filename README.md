# Vietnamese Spell Corrector

A spell corrector and accent restoration tool for Vietnamese, built with OpenSubtitles v2018 and Wikipedia Vietnamese.

## Key Features

- **Vietnamese Accent Restoration** — automatically adds diacritics to unmarked text (e.g. `toi thich` → `tôi thích`)
- **Typo Correction** — detects and fixes spelling errors based on corpus word frequency
- **Top Candidate Suggestions** — returns a ranked list of suggestions with frequency counts
- **Sentence-level Correction** — processes each word in a sentence while preserving structure
- **Multi-source Corpus** — combines OpenSubtitles (conversational) and Wikipedia (formal)
- **Fully Offline** — runs without internet after training

## Tech Stack

- **Python 3.10+** — primary programming language
- **Pickle** — model serialization (word_freq, accent_map)
- **unicodedata** — Unicode normalization (NFD/NFC)
- **xml.etree.ElementTree** — streaming Wikipedia XML parser
- **collections.Counter / defaultdict** — word frequency statistics

## Project Structure

```
project/
├── data/                          # Raw dataset files (not committed)
│   ├── vi.txt                     # OpenSubtitles Vietnamese
│   └── viwiki-latest-pages-articles.xml  # Wikipedia dump (optional)
├── models/                        # Trained model artifacts
│   ├── word_freq.pkl              # Word frequency dictionary
│   └── accent_map.pkl             # Unaccented -> accented word mapping
├── src/
│   ├── spell_corrector.py         # Core correction algorithm
│   ├── preprocess.py              # Text normalization and tokenization
│   ├── train.py                   # Training pipeline
│   └── demo.py                    # Interactive CLI demo
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Download dataset

**OpenSubtitles v2018 Vietnamese (required)**

- Visit: https://opus.nlpl.eu/OpenSubtitles-v2018/mono/
- Download `OpenSubtitles.raw.vi.gz` (~50 MB)
- Extract and place as `data/vi.txt`

> Alternative link: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz

**Wikipedia Vietnamese XML (optional)**

- Visit: https://dumps.wikimedia.org/viwiki/latest/
- Download `viwiki-latest-pages-articles.xml.bz2` (~600 MB), extract
- Place as `data/viwiki-latest-pages-articles.xml`

> The file is ~6 GB after extraction. If missing, the training script skips it automatically.

### 3. Train the model

```bash
python src/train.py
```

The trained models are saved to `models/word_freq.pkl` and `models/accent_map.pkl`.

### 4. Run the demo

```bash
python src/demo.py
```

```
Vietnamese Spell Corrector
Type 'exit' to quit.

Enter incorrect word/sentence: toi thich xem phim hanh dong
Suggestion: tôi thích xem phim hành động
```

## Usage in code

```python
from spell_corrector import SpellCorrector

corrector = SpellCorrector()

corrector.correct_sentence("toi thich xem phim hanh dong")
# -> "tôi thích xem phim hành động"

corrector.correct_word("thich")
# -> "thích"

corrector.suggest_words("thich", top_k=5)
# -> [("thích", 12345), ("thịch", 23), ...]
```

## How the Algorithm Works

Based on Peter Norvig's spell corrector, customized for Vietnamese:

- **Accent Restoration** — lookup `accent_map` to find accented words matching an unaccented form
- **Edits1** — generate candidates via delete, transpose, replace, insert using the full Vietnamese character set (29 letters + diacritics)
- **Frequency Ranking** — pick the candidate with the highest corpus frequency
- **Common Typos** — handle frequently confused character pairs: `i/h`, `u/ư`, `o/ô`, `e/ê`, `a/ă/â`, `n/nh`, `c/k`, `g/gh`
- **Performance Heuristic** — if direct candidates have frequency >= 1000, skip edits to speed up

## License

OpenSubtitles dataset is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The project code is released under the MIT license, for research and educational purposes.

---

# Công cụ Sửa Chính Tả Tiếng Việt

Công cụ sửa lỗi chính tả và khôi phục dấu tiếng Việt, kết hợp OpenSubtitles v2018 và Wikipedia Vietnamese.

## Tính năng chính

- **Khôi phục dấu tiếng Việt** — tự động thêm dấu cho text không dấu (ví dụ: `toi thich` → `tôi thích`)
- **Sửa lỗi gõ sai** — phát hiện và sửa typo dựa trên tần suất từ trong corpus
- **Gợi ý top candidate** — trả về danh sách các từ gợi ý kèm tần suất
- **Sửa cả câu** — xử lý từng từ trong câu, giữ nguyên cấu trúc
- **Corpus đa nguồn** — kết hợp OpenSubtitles (hội thoại) và Wikipedia (văn bản chuẩn)
- **Không cần internet** — chạy offline hoàn toàn sau khi train

## Công nghệ sử dụng

- **Python 3.10+** — ngôn ngữ chính
- **Pickle** — lưu/load model (word_freq, accent_map)
- **unicodedata** — chuẩn hóa Unicode (NFD/NFC)
- **xml.etree.ElementTree** — đọc Wikipedia XML theo streaming
- **collections.Counter / defaultdict** — thống kê tần suất từ

## Cấu trúc project

```
project/
├── data/                          # Raw dữ liệu (không commit)
│   ├── vi.txt                     # OpenSubtitles Vietnamese
│   └── viwiki-latest-pages-articles.xml  # Wikipedia dump (optional)
├── models/                        # Model đã train
│   ├── word_freq.pkl              # Từ điển tần suất từ
│   └── accent_map.pkl             # Mapping: không dấu -> có dấu
├── src/
│   ├── spell_corrector.py         # Core: thuật toán sửa lỗi
│   ├── preprocess.py              # Chuẩn hóa text, tokenize
│   ├── train.py                   # Pipeline training
│   └── demo.py                    # Interactive CLI demo
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Cài đặt

### 1. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Tải dataset

**OpenSubtitles v2018 Vietnamese (bắt buộc)**

- Truy cập: https://opus.nlpl.eu/OpenSubtitles-v2018/mono/
- Tải `OpenSubtitles.raw.vi.gz` (~50 MB)
- Giải nén và đặt vào `data/vi.txt`

> Alternative link: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz

**Wikipedia Vietnamese XML (tùy chọn)**

- Truy cập: https://dumps.wikimedia.org/viwiki/latest/
- Tải `viwiki-latest-pages-articles.xml.bz2` (~600 MB), giải nén
- Đặt vào `data/viwiki-latest-pages-articles.xml`

> File ~6 GB sau giải nén. Nếu không có, script train tự bỏ qua.

### 3. Train model

```bash
python src/train.py
```

Model sẽ được lưu vào `models/word_freq.pkl` và `models/accent_map.pkl`.

### 4. Chạy demo

```bash
python src/demo.py
```

```
Vietnamese Spell Corrector
Nhập 'exit' để thoát.

Nhập từ/câu sai: toi thich xem phim hanh dong
Gợi ý: tôi thích xem phim hành động
```

## Sử dụng trong code

```python
from spell_corrector import SpellCorrector

corrector = SpellCorrector()

corrector.correct_sentence("toi thich xem phim hanh dong")
# -> "tôi thích xem phim hành động"

corrector.correct_word("thich")
# -> "thích"

corrector.suggest_words("thich", top_k=5)
# -> [("thích", 12345), ("thịch", 23), ...]
```

## Thuật toán

Dựa trên Peter Norvig's spell corrector, tùy biến cho tiếng Việt:

- **Khôi phục dấu** — lookup `accent_map` để tìm các từ có dấu tương ứng với dạng không dấu
- **Edits1** — sinh candidates qua delete, transpose, replace, insert với bộ ký tự tiếng Việt (29 chữ cái + dấu)
- **Frequency ranking** — chọn candidate có tần suất cao nhất trong corpus
- **Typo phổ biến** — xử lý các cặp dễ gõ nhầm: `i/h`, `u/ư`, `o/ô`, `e/ê`, `a/ă/â`, `n/nh`, `c/k`, `g/gh`
- **Heuristic tối ưu** — nếu direct candidates có tần suất >= 1000, bỏ qua edits để tăng tốc

## Giấy phép

OpenSubtitles dataset thuộc [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code trong project thuộc MIT license, dùng cho nghiên cứu và học tập.
