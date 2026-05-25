# Vietnamese Spell Corrector

Vietnamese spell correction and accent restoration tool with sentence-level context scoring. The project combines word frequency, accent mapping, bigram context, Vietnamese POS/word segmentation, and optional PhoBERT reranking.

## Key Features

- **Vietnamese accent restoration** - restores diacritics for unmarked text, for example `toi thich` -> `tôi thích`
- **Typo correction** - handles common typing mistakes, missing characters, and similar Vietnamese character variants
- **Sentence-level correction** - corrects whole sentences while preserving punctuation and spacing
- **Candidate suggestions** - returns ranked word suggestions with frequency counts
- **Context scoring** - uses word frequency, edit distance, bigrams, POS tags, word segmentation, and semantic reranking
- **Offline-first runtime** - runs with local model files; PhoBERT is loaded from `models/phobert-base-v2` when available

## Tech Stack

- **Python 3.10+**
- **underthesea** - Vietnamese tokenization, word segmentation, and POS tagging
- **torch / transformers / sentencepiece** - optional PhoBERT sentence scoring
- **pickle** - local model artifact serialization
- **xml.etree.ElementTree** - streaming parser for large Wikipedia XML dumps
- **collections.Counter / defaultdict** - frequency and bigram statistics

## Project Structure

```text
project/
|-- data/                                  # Raw training data, ignored by Git
|   |-- README.md
|   |-- vi.txt                             # OpenSubtitles Vietnamese
|   `-- viwiki-latest-pages-articles.xml   # Wikipedia dump, optional
|-- models/                                # Local model artifacts, ignored by Git
|   |-- word_freq.pkl
|   |-- accent_map.pkl
|   |-- bigram_freq.pkl
|   `-- phobert-base-v2/                   # Optional local PhoBERT files
|-- src/
|   |-- demo.py                            # Interactive CLI demo
|   |-- pos_tagger.py                      # underthesea helpers
|   |-- preprocess.py                      # Text cleanup, tokenization, accent removal
|   |-- semantic_reranker.py               # POS/segmentation/PhoBERT scoring
|   |-- spell_corrector.py                 # Core correction logic
|   `-- train.py                           # Training pipeline
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- .gitattributes
```

## Local Assets

The runtime expects these files when using the trained corrector:

```text
models/word_freq.pkl
models/accent_map.pkl
models/bigram_freq.pkl
```

For transformer reranking, place PhoBERT files here:

```text
models/phobert-base-v2/
|-- bpe.codes
|-- config.json
|-- pytorch_model.bin
|-- tokenizer.json
`-- vocab.txt
```

`data/` and `models/` are intentionally ignored by Git because they contain large raw datasets and binary model files. Share or download them separately when setting up another machine.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. Prepare data or model files

If trained model files already exist in `models/`, you can run the demo directly.

To train from raw data, prepare:

```text
data/vi.txt
data/viwiki-latest-pages-articles.xml   # optional
```

OpenSubtitles v2018 Vietnamese:

- Download `OpenSubtitles.raw.vi.gz` from https://opus.nlpl.eu/OpenSubtitles-v2018/mono/
- Extract it and place the result at `data/vi.txt`
- Alternative link: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz

Wikipedia Vietnamese XML, optional:

- Download `viwiki-latest-pages-articles.xml.bz2` from https://dumps.wikimedia.org/viwiki/latest/
- Extract it and place the result at `data/viwiki-latest-pages-articles.xml`
- The extracted file is large. If it is missing, `src/train.py` skips it automatically.

### 3. Train model files

```powershell
python src/train.py
```

Training outputs:

```text
models/word_freq.pkl
models/accent_map.pkl
models/bigram_freq.pkl
```

### 4. Run the demo

```powershell
python src/demo.py
```

Example:

```text
Vietnamese Spell Corrector
Nhập 'exit' để thoát.

Nhập từ/câu sai: hom nay toi khon di danh cau long
Gợi ý: hôm nay tôi không đi đánh cầu lông
```

## Usage in Code

From the project root:

```python
from src.spell_corrector import SpellCorrector

corrector = SpellCorrector()

print(corrector.correct_sentence("toi thich xem phim hanh dong"))
# tôi thích xem phim hành động

print(corrector.correct_sentence("hom nay toi khon di danh cau long"))
# hôm nay tôi không đi đánh cầu lông

print(corrector.suggest_words("thich", top_k=5))
```

If running a script from inside `src/`, this also works:

```python
from spell_corrector import SpellCorrector
```

## How It Works

The correction pipeline uses multiple layers:

1. Normalize and tokenize the input.
2. Generate candidates from accent mapping and common typo edits.
3. Score candidates with word frequency and edit distance.
4. Score sentence context with bigram probabilities.
5. Add POS and word-segmentation signals through `underthesea`.
6. Rerank whole sentences with PhoBERT if `models/phobert-base-v2` and transformer dependencies are available.

If PhoBERT is unavailable, the corrector still runs with the lighter word frequency, edit-distance, bigram, POS, and segmentation scoring path.

## Git Notes

- Raw data files are ignored: `data/vi.txt`, `data/*.xml`, `data/*.bz2`, `data/*.gz`
- Model files are ignored: `models/`
- Large files are configured for Git LFS in `.gitattributes` if you intentionally force-track them.
- Do not commit `venv/`, `__pycache__/`, local caches, or `.env` files.

## License

OpenSubtitles data is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Wikipedia text is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Project code is intended for research and educational use.

---

# Công cụ Sửa Chính Tả Tiếng Việt

Công cụ sửa lỗi chính tả và khôi phục dấu tiếng Việt với chấm điểm theo ngữ cảnh toàn câu. Dự án kết hợp tần suất từ, bản đồ dấu, ngữ cảnh bigram, POS/tách từ tiếng Việt và tùy chọn xếp hạng lại bằng PhoBERT.

## Tính năng chính

- **Khôi phục dấu tiếng Việt** - thêm dấu cho văn bản không dấu, ví dụ `toi thich` -> `tôi thích`
- **Sửa lỗi gõ sai** - xử lý lỗi gõ phổ biến, thiếu ký tự và các biến thể ký tự tiếng Việt dễ nhầm
- **Sửa theo ngữ cảnh câu** - sửa cả câu và giữ nguyên dấu câu, khoảng trắng
- **Gợi ý ứng viên** - trả về danh sách từ gợi ý đã xếp hạng kèm tần suất
- **Chấm điểm ngữ cảnh** - dùng tần suất từ, khoảng cách chỉnh sửa, bigram, POS, tách từ và semantic reranking
- **Ưu tiên chạy offline** - chạy bằng model cục bộ; PhoBERT được tải từ `models/phobert-base-v2` nếu có

## Công nghệ sử dụng

- **Python 3.10+**
- **underthesea** - tokenize, tách từ và POS tagging tiếng Việt
- **torch / transformers / sentencepiece** - chấm điểm câu bằng PhoBERT nếu có model cục bộ
- **pickle** - lưu các file model cục bộ
- **xml.etree.ElementTree** - đọc streaming file Wikipedia XML lớn
- **collections.Counter / defaultdict** - thống kê tần suất từ và bigram

## Cấu trúc project

```text
project/
|-- data/                                  # Dữ liệu train thô, không commit
|   |-- README.md
|   |-- vi.txt                             # OpenSubtitles Vietnamese
|   `-- viwiki-latest-pages-articles.xml   # Wikipedia dump, tùy chọn
|-- models/                                # Model cục bộ, không commit
|   |-- word_freq.pkl
|   |-- accent_map.pkl
|   |-- bigram_freq.pkl
|   `-- phobert-base-v2/                   # PhoBERT cục bộ, tùy chọn
|-- src/
|   |-- demo.py                            # CLI demo
|   |-- pos_tagger.py                      # Helper cho underthesea
|   |-- preprocess.py                      # Làm sạch text, tokenize, bỏ dấu
|   |-- semantic_reranker.py               # Chấm điểm POS/tách từ/PhoBERT
|   |-- spell_corrector.py                 # Logic sửa lỗi chính
|   `-- train.py                           # Pipeline train
|-- README.md
|-- requirements.txt
|-- .gitignore
`-- .gitattributes
```

## Tài nguyên cục bộ

Khi chạy corrector đã train, cần các file sau:

```text
models/word_freq.pkl
models/accent_map.pkl
models/bigram_freq.pkl
```

Nếu muốn dùng transformer reranking, đặt các file PhoBERT tại:

```text
models/phobert-base-v2/
|-- bpe.codes
|-- config.json
|-- pytorch_model.bin
|-- tokenizer.json
`-- vocab.txt
```

`data/` và `models/` được cố ý bỏ qua khỏi Git vì chứa dữ liệu thô và file model lớn. Khi setup máy khác, hãy sao chép hoặc tải riêng các thư mục này.

## Cài đặt

### 1. Tạo môi trường ảo

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu hoặc model

Nếu đã có sẵn model trong `models/`, có thể chạy demo trực tiếp.

Nếu muốn train lại từ dữ liệu thô, chuẩn bị:

```text
data/vi.txt
data/viwiki-latest-pages-articles.xml   # tùy chọn
```

OpenSubtitles v2018 Vietnamese:

- Tải `OpenSubtitles.raw.vi.gz` từ https://opus.nlpl.eu/OpenSubtitles-v2018/mono/
- Giải nén và đặt kết quả tại `data/vi.txt`
- Link thay thế: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz

Wikipedia Vietnamese XML, tùy chọn:

- Tải `viwiki-latest-pages-articles.xml.bz2` từ https://dumps.wikimedia.org/viwiki/latest/
- Giải nén và đặt kết quả tại `data/viwiki-latest-pages-articles.xml`
- File sau giải nén rất lớn. Nếu không có, `src/train.py` sẽ tự bỏ qua.

### 3. Train model

```powershell
python src/train.py
```

Kết quả train:

```text
models/word_freq.pkl
models/accent_map.pkl
models/bigram_freq.pkl
```

### 4. Chạy demo

```powershell
python src/demo.py
```

Ví dụ:

```text
Vietnamese Spell Corrector
Nhập 'exit' để thoát.

Nhập từ/câu sai: hom nay toi khon di danh cau long
Gợi ý: hôm nay tôi không đi đánh cầu lông
```

## Sử dụng trong code

Khi import từ thư mục gốc project:

```python
from src.spell_corrector import SpellCorrector

corrector = SpellCorrector()

print(corrector.correct_sentence("toi thich xem phim hanh dong"))
# tôi thích xem phim hành động

print(corrector.correct_sentence("hom nay toi khon di danh cau long"))
# hôm nay tôi không đi đánh cầu lông

print(corrector.suggest_words("thich", top_k=5))
```

Nếu chạy script bên trong thư mục `src/`, có thể import như sau:

```python
from spell_corrector import SpellCorrector
```

## Cách hoạt động

Pipeline sửa lỗi gồm nhiều lớp:

1. Chuẩn hóa và tokenize input.
2. Sinh ứng viên từ bản đồ dấu và các biến thể lỗi gõ phổ biến.
3. Chấm điểm ứng viên bằng tần suất từ và khoảng cách chỉnh sửa.
4. Chấm điểm ngữ cảnh câu bằng xác suất bigram.
5. Bổ sung tín hiệu POS và tách từ thông qua `underthesea`.
6. Xếp hạng lại toàn câu bằng PhoBERT nếu có `models/phobert-base-v2` và các thư viện transformer.

Nếu không có PhoBERT, corrector vẫn chạy bằng luồng nhẹ hơn gồm tần suất từ, edit distance, bigram, POS và tách từ.

## Ghi chú Git

- Dữ liệu thô được ignore: `data/vi.txt`, `data/*.xml`, `data/*.bz2`, `data/*.gz`
- Model được ignore: `models/`
- `.gitattributes` đã cấu hình Git LFS cho file lớn nếu bạn cố ý force-track.
- Không commit `venv/`, `__pycache__/`, cache cục bộ hoặc file `.env`.

## Giấy phép

Dữ liệu OpenSubtitles thuộc [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Nội dung Wikipedia thuộc [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Code trong project dùng cho mục đích nghiên cứu và học tập.
