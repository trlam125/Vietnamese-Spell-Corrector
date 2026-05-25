"""Semantic scoring helpers for sentence-level correction."""

from __future__ import annotations

import math
import os
import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

try:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except Exception:
    AutoModelForMaskedLM = None
    AutoTokenizer = None

try:
    from .pos_tagger import is_pos_available, pos_tag_text, word_tokenize_text
except ImportError:
    from pos_tagger import is_pos_available, pos_tag_text, word_tokenize_text


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_MODEL_DIR = BASE_DIR / "models" / "phobert-base-v2"


@dataclass
class TransformerScorer:
    model_path: str = "vinai/phobert-base-v2"
    local_dir: Path | None = DEFAULT_LOCAL_MODEL_DIR
    tokenizer: object | None = None
    model: object | None = None
    enabled: bool = False

    def __post_init__(self) -> None:
        if AutoTokenizer is None or AutoModelForMaskedLM is None:
            return

        source = str(self.local_dir if self.local_dir and self.local_dir.exists() else self.model_path)
        for local_only in (True, False):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        source,
                        use_fast=False,
                        local_files_only=local_only,
                    )
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        source,
                        local_files_only=local_only,
                    )
                self.model.eval()
                self.enabled = True
                return
            except Exception:
                continue

        self.tokenizer = None
        self.model = None
        self.enabled = False

    def score_sentence(self, sentence: str) -> float:
        if not self.enabled or not sentence:
            return 0.0

        try:
            import torch
        except Exception:
            return 0.0

        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = float(outputs.loss)

        return -loss * len(sentence.split())


@dataclass
class SemanticReranker:
    pos_cache: dict[str, str] = field(default_factory=dict)
    segment_cache: dict[str, str] = field(default_factory=dict)
    transformer: TransformerScorer = field(default_factory=TransformerScorer)

    def is_available(self) -> bool:
        return is_pos_available() or self.transformer.enabled

    def pos_tag_for_word(self, word: str) -> str:
        if word in self.pos_cache:
            return self.pos_cache[word]
        if not is_pos_available():
            self.pos_cache[word] = ""
            return ""

        try:
            tagged = pos_tag_text(word)
        except Exception:
            tagged = []

        tag = tagged[0][1] if tagged else ""
        self.pos_cache[word] = tag
        return tag

    def segmented_text(self, text: str) -> str:
        if text in self.segment_cache:
            return self.segment_cache[text]
        if not is_pos_available():
            self.segment_cache[text] = text
            return text

        try:
            segmented = word_tokenize_text(text)
        except Exception:
            segmented = text

        self.segment_cache[text] = segmented
        return segmented

    def transition_score(self, previous: str, current: str) -> float:
        if not is_pos_available():
            return 0.0

        prev_tag = self.pos_tag_for_word(previous)
        current_tag = self.pos_tag_for_word(current)
        score = 0.0

        if prev_tag == "P" and current_tag in {"R", "V", "A"}:
            score += 6.0
        if prev_tag == "R" and current_tag == "V":
            score += 8.0
        if prev_tag == "V" and current_tag in {"V", "N", "A"}:
            score += 5.0
        if prev_tag == "N" and current_tag == "N":
            score += 4.0
        if prev_tag == "M" and current_tag in {"N", "Nu"}:
            score += 3.0
        if prev_tag == "CH" or current_tag == "CH":
            score -= 6.0

        if "_" in self.segmented_text(f"{previous} {current}"):
            score += 6.0

        return score

    def sequence_score(self, words: list[str]) -> float:
        if len(words) < 2:
            return 0.0

        score = 0.0
        sentence = " ".join(words)

        segmented = self.segmented_text(sentence)
        score += segmented.count("_") * 2.0

        if is_pos_available():
            try:
                tags = pos_tag_text(sentence)
            except Exception:
                tags = []

            for _, tag in tags:
                if tag == "P":
                    score += 1.2
                elif tag == "R":
                    score += 1.4
                elif tag == "V":
                    score += 1.5
                elif tag in {"N", "Np", "Nu"}:
                    score += 1.0

        return score

    def transformer_score(self, words: list[str]) -> float:
        if not self.transformer.enabled:
            return 0.0

        sentence = " ".join(words)
        return self.transformer.score_sentence(sentence)
