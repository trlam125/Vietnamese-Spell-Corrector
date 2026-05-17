import pickle
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

WORD_FREQ_PATH = BASE_DIR / "models" / "word_freq.pkl"
ACCENT_MAP_PATH = BASE_DIR / "models" / "accent_map.pkl"
BIGRAM_FREQ_PATH = BASE_DIR / "models" / "bigram_freq.pkl"

LETTERS = (
    "abcdefghijklmnopqrstuvwxyz"
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
)

# Smaller set for edit operations (base chars only — accents restored via accent_map)
EDIT_LETTERS = "abcdefghijklmnopqrstuvwxyzđ"

MAX_CANDIDATES = 200
DIRECT_EARLY_RETURN_FREQ = 10_000


def _has_vietnamese_accents(text: str) -> bool:
    """Check if text contains Vietnamese diacritics (has been accented)."""
    return text != remove_accents(text)


def remove_accents(text: str) -> str:
    # Replace đ/Đ first (they don't decompose in NFD)
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return unicodedata.normalize("NFC", text)


class SpellCorrector:
    def __init__(
        self,
        word_freq_path: str | Path = WORD_FREQ_PATH,
        accent_map_path: str | Path = ACCENT_MAP_PATH,
        bigram_freq_path: str | Path = BIGRAM_FREQ_PATH,
    ) -> None:
        word_freq_path = Path(word_freq_path)
        accent_map_path = Path(accent_map_path)
        bigram_freq_path = Path(bigram_freq_path)

        if not word_freq_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {word_freq_path}. "
                f"Hãy chạy: python src/train.py trước."
            )

        with open(word_freq_path, "rb") as f:
            self.word_freq = pickle.load(f)

        self.total_words = sum(self.word_freq.values())
        self.vocab = set(self.word_freq.keys())

        if accent_map_path.exists():
            with open(accent_map_path, "rb") as f:
                self.accent_map = pickle.load(f)
        else:
            self.accent_map = {}

        # Load bigram frequency for context scoring with indexed lookups
        if bigram_freq_path.exists():
            with open(bigram_freq_path, "rb") as f:
                raw = pickle.load(f)

            # Index: word -> {next_word: count}  (forward lookups)
            self._bigram_forward: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            # Index: word -> total outgoing bigram count
            self._bigram_out_total: dict[str, int] = defaultdict(int)
            # Index: word -> total incoming bigram count
            self._bigram_in_total: dict[str, int] = defaultdict(int)

            punct_str = string.punctuation + ' "'
            for k, v in raw.items():
                w1, w2 = k.split("\t")
                w1_clean = w1.strip(punct_str)
                w2_clean = w2.strip(punct_str)
                if w1_clean and w2_clean:
                    self._bigram_forward[w1_clean][w2_clean] += v
                    self._bigram_out_total[w1_clean] += v
                    self._bigram_in_total[w2_clean] += v

            # Freeze defaultdicts to regular dicts for faster lookups
            self._bigram_forward = {k: dict(v) for k, v in self._bigram_forward.items()}
            self._bigram_out_total = dict(self._bigram_out_total)
            self._bigram_in_total = dict(self._bigram_in_total)

            # Also keep bigram_freq as a dict for backward compat and quick exists-check
            self.bigram_freq = self._bigram_forward
        else:
            self.bigram_freq = {}
            self._bigram_forward = {}
            self._bigram_out_total = {}
            self._bigram_in_total = {}

        # Candidate cache
        self._candidate_cache: dict[str, set[str]] = {}

    # Các cặp ký tự dễ gõ nhầm trong tiếng Việt
    SIMILAR_CHARS = {
        'i': 'h', 'h': 'i',
        'u': 'ư', 'ư': 'u',
        'o': 'ô', 'ô': 'o', 'ơ': 'o',
        'e': 'ê', 'ê': 'e',
        'a': 'ă', 'ă': 'a', 'â': 'a',
        'n': 'nh', 'nh': 'n',
        'c': 'k', 'k': 'c',
        'g': 'gh', 'gh': 'g',
    }

    def probability(self, word: str) -> float:
        return self.word_freq.get(word, 0) / self.total_words

    def normalize_input_word(self, word: str) -> str:
        word = word.strip().lower()
        word = clean_text(word)
        return word

    def known(self, words: set[str] | list[str]) -> set[str]:
        return {word for word in words if word in self.vocab}

    def edits1(self, word: str) -> set[str]:
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [left + right[1:] for left, right in splits if right]

        transposes = [
            left + right[1] + right[0] + right[2:]
            for left, right in splits
            if len(right) > 1
        ]

        replaces = [
            left + char + right[1:]
            for left, right in splits
            if right
            for char in LETTERS
        ]

        inserts = [
            left + char + right
            for left, right in splits
            for char in LETTERS
        ]

        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set[str]:
        """
        Sinh candidate cách 2 lỗi.
        Chỉ nên dùng với từ ngắn để tránh chậm.
        """
        return {
            edit2
            for edit1 in self.edits1(word)
            for edit2 in self.edits1(edit1)
        }

    def accent_candidates(self, word: str) -> set[str]:
        """
        Tìm các từ có dấu tương ứng với dạng không dấu.
        Ví dụ:
        toi -> tôi, tới
        thich -> thích
        """
        no_accent = remove_accents(word)
        return set(self.accent_map.get(no_accent, []))

    def candidates(self, word: str) -> set[str]:
        normalized = self.normalize_input_word(word)
        if not normalized:
            return set()
        if normalized in self._candidate_cache:
            return self._candidate_cache[normalized]
        result = self._generate_candidates(normalized)
        if len(result) > MAX_CANDIDATES:
            ranked = sorted(result, key=lambda c: self.word_freq.get(c, 0), reverse=True)
            result = set(ranked[:MAX_CANDIDATES])
        self._candidate_cache[normalized] = result
        return result

    def _generate_candidates(self, word: str) -> set[str]:
        """Internal candidate generation — called by cached candidates()."""
        no_accent = remove_accents(word)

        # 1. Khôi phục dấu trực tiếp
        direct_candidates = self.accent_candidates(word)

        if direct_candidates:
            max_direct_freq = max(self.word_freq.get(c, 0) for c in direct_candidates)
            if max_direct_freq >= DIRECT_EARLY_RETURN_FREQ:
                return direct_candidates

        result = set()
        if direct_candidates:
            result.update(direct_candidates)

        # 2. Sửa lỗi 1 ký tự
        if 2 <= len(no_accent) <= 5:
            for i in range(len(no_accent)):
                edit = no_accent[:i] + no_accent[i + 1:]
                if edit:
                    result.update(self.accent_candidates(edit))

            for i in range(len(no_accent) + 1):
                for char in EDIT_LETTERS:
                    edit = no_accent[:i] + char + no_accent[i:]
                    result.update(self.accent_candidates(edit))

            for i, char in enumerate(no_accent):
                if char in self.SIMILAR_CHARS:
                    for similar in self.SIMILAR_CHARS[char]:
                        edit = no_accent[:i] + similar + no_accent[i + 1:]
                        result.update(self.accent_candidates(edit))

        # 3. Nếu từ gốc có tần suất rất cao -> dùng từ gốc
        if word in self.vocab and len(result) > 1:
            word_freq = self.word_freq.get(word, 0)
            max_candidate_freq = max(self.word_freq.get(c, 0) for c in result if c != word)
            if word_freq > max_candidate_freq * 3:
                return {word}

        if result:
            return result

        return {word}

    def correct_word(self, word: str, prev_word: str = None, next_word: str = None) -> str:
        cands = self.candidates(word)

        if not cands:
            return word

        if len(cands) == 1:
            return next(iter(cands))

        has_context = prev_word is not None or next_word is not None
        if has_context and self.bigram_freq:
            best_score = -1.0
            best_candidate = max(cands, key=lambda c: self.word_freq.get(c, 0))

            # Fast lookups using pre-built indexes (no vocab scan!)
            prev_total = self._bigram_out_total.get(prev_word, 0) if prev_word else 0
            next_total = self._bigram_in_total.get(next_word, 0) if next_word else 0

            # Get next_word variants for accented matching
            next_word_variants = {next_word} if next_word else set()
            if next_word and not _has_vietnamese_accents(next_word):
                next_variants = self.accent_candidates(next_word)
                if next_variants:
                    next_word_variants.update(next_variants)

            # Also accumulate next_total from all variants
            if next_word and not _has_vietnamese_accents(next_word) and next_word_variants:
                for nw in next_word_variants:
                    next_total += self._bigram_in_total.get(nw, 0)

            for candidate in cands:
                score = 0.0

                if prev_word and prev_total > 0:
                    bigram_count = self._bigram_forward.get(prev_word, {}).get(candidate, 0)
                    score += (bigram_count / prev_total) * 100_000

                if next_word and next_total > 0:
                    for nw in next_word_variants:
                        bigram_count = self._bigram_forward.get(candidate, {}).get(nw, 0)
                        score += (bigram_count / next_total) * 100_000

                score += self.word_freq.get(candidate, 0) * 0.0001

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            return best_candidate

        return max(cands, key=lambda c: self.word_freq.get(c, 0))

    def suggest_words(self, word: str, top_k: int = 5) -> list[tuple[str, int]]:
        candidates = self.candidates(word)

        ranked = sorted(
            candidates,
            key=lambda candidate: self.word_freq.get(candidate, 0),
            reverse=True
        )

        return [
            (candidate, self.word_freq.get(candidate, 0))
            for candidate in ranked[:top_k]
        ]

    def correct_sentence(self, sentence: str) -> str:
        """
        Sửa từng từ trong câu.
        Không clean cả câu trước, vì clean cả câu có thể làm mất typo user nhập.
        """
        raw_words = sentence.lower().split()
        corrected_words = []

        for i, raw_word in enumerate(raw_words):
            prev_word = corrected_words[i - 1] if i > 0 else None
            next_word = raw_words[i + 1] if i < len(raw_words) - 1 else None
            corrected_word = self.correct_word(raw_word, prev_word, next_word)
            corrected_words.append(corrected_word)

        return " ".join(corrected_words)