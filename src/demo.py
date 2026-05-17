import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from spell_corrector import SpellCorrector


def main() -> None:
    corrector = SpellCorrector()

    print("Vietnamese Spell Corrector")
    print("Nhập 'exit' để thoát.")
    print("Ví dụ: tôi thích xem phim hành động")

    while True:
        text = input("\nNhập từ/câu sai: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break

        if not text:
            continue

        print("Gợi ý:", corrector.correct_sentence(text))

        if " " not in text:
            print("Top gợi ý:")
            for word, count in corrector.suggest_words(text, top_k=5):
                print(f"- {word} ({count})")

if __name__ == "__main__":
    main()