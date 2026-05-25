import pickle
import math
import re
import string
from collections import defaultdict
from pathlib import Path

try:
    from .preprocess import clean_text, tokenize, remove_accents as preprocess_remove_accents
    from .semantic_reranker import SemanticReranker
except ImportError:
    from preprocess import clean_text, tokenize, remove_accents as preprocess_remove_accents
    from semantic_reranker import SemanticReranker


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
SENTENCE_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+|\s+|[^\w\s]", re.UNICODE)

MAX_CANDIDATES = 200
DIRECT_EARLY_RETURN_FREQ = 10_000
BEAM_SIZE = 8


def _has_vietnamese_accents(text: str) -> bool:
    """Check if text contains Vietnamese diacritics (has been accented)."""
    return text != remove_accents(text)


def remove_accents(text: str) -> str:
    return preprocess_remove_accents(text)


def edit_distance(left: str, right: str, max_distance: int = 3) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)

        if row_min > max_distance:
            return max_distance + 1
        previous = current

    return previous[-1]


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
        self._sentence_cache: dict[tuple[str, ...], list[str]] = {}
        self.semantic_reranker = SemanticReranker()

    # Các cặp ký tự dễ gõ nhầm trong tiếng Việt
    SIMILAR_CHARS = {
        'i': 'h', 'h': 'i',
        'u': 'ư', 'ư': 'u',
        'o': 'ô', 'ô': 'o', 'ơ': 'o',
        'e': 'ê', 'ê': 'e',
        'a': 'ă', 'ă': 'a', 'â': 'a',
        'n': ('nh',), 'nh': ('n',),
        'c': ('k',), 'k': ('c',),
        'g': ('gh',), 'gh': ('g',),
    }

    def probability(self, word: str) -> float:
        return self.word_freq.get(word, 0) / self.total_words

    def normalize_input_word(self, word: str) -> str:
        word = word.strip().lower()
        word = clean_text(word)
        words = tokenize(word)
        return words[0] if words else ""

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
            if max_direct_freq >= DIRECT_EARLY_RETURN_FREQ and word in self.vocab and _has_vietnamese_accents(word):
                return direct_candidates

        result = set()
        if direct_candidates:
            result.update(direct_candidates)

        # 2. Sửa lỗi 1 ký tự
        if 2 <= len(no_accent) <= 5:
            if not direct_candidates:
                for i in range(len(no_accent)):
                    edit = no_accent[:i] + no_accent[i + 1:]
                    if edit:
                        result.update(self.accent_candidates(edit))

            for i in range(len(no_accent) + 1):
                for char in EDIT_LETTERS:
                    edit = no_accent[:i] + char + no_accent[i:]
                    result.update(self.accent_candidates(edit))

            for source, replacements in self.SIMILAR_CHARS.items():
                if isinstance(replacements, str):
                    replacements = (replacements,)
                start = 0
                while True:
                    i = no_accent.find(source, start)
                    if i == -1:
                        break
                    for replacement in replacements:
                        edit = no_accent[:i] + replacement + no_accent[i + len(source):]
                        result.update(self.accent_candidates(edit))
                    start = i + 1

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

    def _candidate_score(self, original: str, candidate: str) -> float:
        freq = self.word_freq.get(candidate, 0)
        score = math.log1p(freq)

        original_no_accent = remove_accents(original)
        candidate_no_accent = remove_accents(candidate)
        distance = edit_distance(original_no_accent, candidate_no_accent)
        if distance == 0:
            score += 4.0
        else:
            score -= 4.5 * distance

        if original in self.vocab and candidate_no_accent != original_no_accent:
            original_freq = self.word_freq.get(original, 0)
            if len(original_no_accent) <= 3 and original_freq >= 100:
                score -= 7.0
            elif original_freq >= 1000:
                score -= 4.0

        return score

    def _bigram_score(self, previous: str, current: str) -> float:
        count = self._bigram_forward.get(previous, {}).get(current, 0)
        if count <= 0:
            return self.semantic_reranker.transition_score(previous, current)

        prev_freq = max(self.word_freq.get(previous, 0), 1)
        current_freq = max(self.word_freq.get(current, 0), 1)
        pmi = math.log((count * max(self.total_words, 1)) / (prev_freq * current_freq))
        score = 8.0 * math.log1p(count) + 4.0 * max(pmi, 0.0)
        if count >= 5 and pmi >= 1.0:
            score += 35.0
        return score + self.semantic_reranker.transition_score(previous, current)

    def _ranked_candidates_for_sentence(self, word: str, limit: int = 30) -> list[str]:
        candidates = self.candidates(word)
        if not candidates:
            return [word]

        normalized = self.normalize_input_word(word)
        ranked = sorted(
            candidates,
            key=lambda candidate: self._candidate_score(normalized, candidate),
            reverse=True,
        )
        return ranked[:limit]

    def _correct_word_sequence(self, words: list[str]) -> list[str]:
        if not words:
            return []

        cache_key = tuple(words)
        if cache_key in self._sentence_cache:
            return self._sentence_cache[cache_key][:]

        candidate_lists = [
            self._ranked_candidates_for_sentence(word)
            for word in words
        ]

        beam: list[tuple[float, list[str]]] = [
            (2.0 * self._candidate_score(words[0], candidate), [candidate])
            for candidate in candidate_lists[0]
        ]
        beam.sort(key=lambda item: item[0], reverse=True)
        beam = beam[:BEAM_SIZE]

        for i in range(1, len(words)):
            expanded: list[tuple[float, list[str]]] = []
            for previous_score, sequence in beam:
                previous = sequence[-1]
                for candidate in candidate_lists[i]:
                    score = (
                        previous_score
                        + 2.0 * self._candidate_score(words[i], candidate)
                        + self._bigram_score(previous, candidate)
                    )
                    expanded.append((score, sequence + [candidate]))

            expanded.sort(key=lambda item: item[0], reverse=True)
            beam = expanded[:BEAM_SIZE]

        corrected = max(
            beam,
            key=lambda item: (
                item[0]
                + self.semantic_reranker.sequence_score(item[1])
                + self.semantic_reranker.transformer_score(item[1])
            )
        )[1]
        self._sentence_cache[cache_key] = corrected[:]
        return corrected

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
        tokens = SENTENCE_TOKEN_RE.findall(sentence.lower())
        corrected_tokens = []
        corrected_words = []

        word_tokens = [
            token for token in tokens
            if token and not token.isspace() and re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+", token)
        ]
        corrected_word_tokens = self._correct_word_sequence(word_tokens)
        word_index = 0

        for token in tokens:
            if not token or token.isspace() or not re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+", token):
                corrected_tokens.append(token)
                continue

            corrected_word = corrected_word_tokens[word_index]
            corrected_tokens.append(corrected_word)
            corrected_words.append(corrected_word)
            word_index += 1

        return "".join(corrected_tokens)
