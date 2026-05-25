"""POS tagging module for Vietnamese text.

Uses underthesea for POS tagging. If underthesea is not installed,
POS tagging is gracefully disabled.
"""

try:
    from underthesea import pos_tag as _underthesea_pos_tag
    from underthesea import word_tokenize as _underthesea_word_tokenize

    _POS_AVAILABLE = True
except ImportError:
    _underthesea_pos_tag = None
    _underthesea_word_tokenize = None
    _POS_AVAILABLE = False


def pos_tag(words: list[str]) -> list[tuple[str, str]]:
    """Return POS tags for a list of words.

    Returns list of (word, tag) tuples.
    Common Vietnamese POS tags:
        N  - noun
        V  - verb
        A  - adjective
        P  - pronoun
        M  - numeral
        L  - location
        E  - preposition
        C  - conjunction
        T  - adverb
        I  - interjection

    If underthesea is not installed, returns empty tags for all words.
    """
    if not _POS_AVAILABLE:
        return [(word, "") for word in words]

    if not words:
        return []

    tagged = _underthesea_pos_tag(" ".join(words))
    # underthesea returns list of (word, tag) tuples
    return tagged


def pos_tag_text(text: str) -> list[tuple[str, str]]:
    if not _POS_AVAILABLE or not text:
        return []

    return _underthesea_pos_tag(text)


def word_tokenize_text(text: str) -> str:
    if not _POS_AVAILABLE or not text:
        return text

    return _underthesea_word_tokenize(text, format="text")


def is_pos_available() -> bool:
    return _POS_AVAILABLE
