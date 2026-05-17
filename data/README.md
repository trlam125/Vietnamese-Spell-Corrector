# Dataset

Thư mục chứa dữ liệu thô (raw data) dùng để train spell corrector. Các file này không được commit lên git (đã có trong `.gitignore`).

## OpenSubtitles v2018 Vietnamese

|          |                          |
|----------|--------------------------|
| **File** | `vi.txt`                 |
| **Dung lượng** | ~50 MB (nén), ~250 MB (giải nén) |
| **Nội dung** | ~205,000 câu hội thoại tiếng Việt từ phụ đề phim |
| **Giấy phép** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

**Tải:** https://opus.nlpl.eu/datasets/OpenSubtitles?pair=vi&vi

Tìm file `OpenSubtitles.raw.vi.gz` → giải nén → đặt kết quả vào đây với tên `vi.txt`.

> Alternative link: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz

## Wikipedia Vietnamese Dump (tùy chọn)

|          |                          |
|----------|--------------------------|
| **File** | `viwiki-latest-pages-articles.xml` |
| **Dung lượng** | ~600 MB (nén), ~6 GB (giải nén) |
| **Nội dung** | Toàn bộ bài viết Wikipedia tiếng Việt, định dạng XML |
| **Giấy phép** | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

**Tải:** https://dumps.wikimedia.org/viwiki/latest/

Tìm file `viwiki-latest-pages-articles.xml.bz2` → `bzip2 -d` để giải nén → đặt kết quả vào đây.

> File rất lớn. Nếu không có, script train sẽ tự bỏ qua và chỉ dùng OpenSubtitles.
