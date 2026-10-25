"""Resolve an explicitly named scheme before semantic retrieval.

Matching follows a safe order: exact normalized scheme name, exact
abbreviation, then high-confidence fuzzy title matching. A fuzzy result is
rejected when another scheme is too close, so callers can ask the user to
clarify instead of routing to the wrong scheme.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

FUZZY_THRESHOLD = 0.86
AMBIGUITY_MARGIN = 0.04
WORD_PATTERN = re.compile(r"[a-z0-9]+")
CARD_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "vector_store"
    / "bge_base_en_v1_5"
    / "scheme_cards.json"
)


@dataclass(frozen=True)
class SchemeMatch:
    source_file: str
    scheme_name: str
    match_type: str
    score: float
    matched_text: str


def normalize(value: str) -> str:
    """Normalize punctuation, casing, and accented text for comparison."""
    decomposed = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(WORD_PATTERN.findall(decomposed.casefold()))


def load_scheme_cards(path: Path = CARD_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Scheme-card index is missing. Build it with: python -m src.embeddings.build_scheme_card_index ({path})"
        )
    cards = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError(f"Scheme-card file is empty or invalid: {path}")
    return cards


def title_windows(query: str, title: str) -> Iterable[str]:
    """Yield query phrases likely to contain a possibly misspelled title."""
    query_words = query.split()
    title_word_count = len(title.split())
    minimum = max(2, title_word_count - 2)
    maximum = min(len(query_words), title_word_count + 2)
    for window_size in range(minimum, maximum + 1):
        for start in range(len(query_words) - window_size + 1):
            yield " ".join(query_words[start : start + window_size])


def fuzzy_title_score(query: str, title: str) -> tuple[float, str]:
    """Find the closest title-sized phrase in a complete user question."""
    best_score, best_text = 0.0, ""
    for candidate in title_windows(query, title):
        score = SequenceMatcher(None, candidate, title).ratio()
        if score > best_score:
            best_score, best_text = score, candidate
    return best_score, best_text


def exact_match(question: str, cards: Iterable[dict]) -> SchemeMatch | None:
    normalized_question = normalize(question)
    question_words = set(normalized_question.split())
    title_matches: list[SchemeMatch] = []
    abbreviation_matches: list[SchemeMatch] = []

    for card in cards:
        name = card.get("scheme_name")
        source_file = card.get("source_file")
        if not isinstance(name, str) or not isinstance(source_file, str):
            continue
        normalized_name = normalize(name)
        if normalized_name and normalized_name in normalized_question:
            title_matches.append(SchemeMatch(source_file, name, "exact_title", 1.0, name))

        abbreviation = card.get("abbreviation")
        normalized_abbreviation = normalize(abbreviation) if isinstance(abbreviation, str) else ""
        if len(normalized_abbreviation) >= 3 and normalized_abbreviation in question_words:
            abbreviation_matches.append(
                SchemeMatch(source_file, name, "exact_abbreviation", 1.0, abbreviation)
            )

    candidates = title_matches or abbreviation_matches
    return candidates[0] if len(candidates) == 1 else None


def match_scheme(
    question: str,
    cards: Iterable[dict] | None = None,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> SchemeMatch | None:
    """Return a confident title/abbreviation match, otherwise ``None``.

    ``None`` means the caller should continue with broad scheme-card retrieval
    or ask a clarification question; it never means that no relevant scheme
    exists.
    """
    cards = list(load_scheme_cards() if cards is None else cards)
    direct = exact_match(question, cards)
    if direct:
        return direct

    normalized_question = normalize(question)
    candidates: list[SchemeMatch] = []
    for card in cards:
        name = card.get("scheme_name")
        source_file = card.get("source_file")
        if not isinstance(name, str) or not isinstance(source_file, str):
            continue
        score, matched_text = fuzzy_title_score(normalized_question, normalize(name))
        candidates.append(SchemeMatch(source_file, name, "fuzzy_title", score, matched_text))

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates or candidates[0].score < fuzzy_threshold:
        return None
    if len(candidates) > 1 and candidates[0].score - candidates[1].score < ambiguity_margin:
        return None
    return candidates[0]
