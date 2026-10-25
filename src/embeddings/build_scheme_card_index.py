"""Build one FAISS card per scheme for broad scheme discovery.

The chunk index answers questions *about* an already-selected scheme. This
index instead finds candidate schemes for broad questions such as "I am a
farmer; what financial help is available?". Each card carries the scheme's
metadata plus a short summary of its description, because metadata alone
cannot match questions phrased in a beneficiary's own words.

Run from the repository root after building the chunk index:
    python -m src.embeddings.build_scheme_card_index
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .build_faiss_index import (
    BATCH_SIZE,
    MODEL_DIR,
    MODEL_ID,
    VECTOR_STORE_DIR,
    load_chunks,
)


CARD_FILE = VECTOR_STORE_DIR / "scheme_cards.json"
INDEX_FILE = VECTOR_STORE_DIR / "scheme_index.faiss"
MANIFEST_FILE = VECTOR_STORE_DIR / "scheme_manifest.json"
TEXT_FIELDS = (
    "scheme_name",
    "abbreviation",
    "state",
    "ministry",
    "department",
    "scheme_type",
    "sponsorship",
    "application_mode",
)
SUMMARY_SECTION = "General"
SUMMARY_FALLBACK_SECTION = "Details"
SUMMARY_SENTENCES = 4
SUMMARY_MAX_CHARS = 1400
MIN_SUMMARY_SENTENCE_CHARS = 35
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
UI_NOISE = re.compile(
    r"Are you sure you want to|CancelSign|Sign\s*(In|Out)|Something went wrong|"
    r"Please try again later|You need to sign in|already initiated your application|"
    r"To know more please visit|Was this helpful|Apply Now|Check Eligibility",
    re.IGNORECASE,
)


def first_nonempty(values: list[object]) -> str | None:
    """Return one stable, non-empty value shared by a scheme's chunks."""
    normalized = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not normalized:
        return None
    if len(set(normalized)) > 1:
        raise ValueError(f"Inconsistent metadata values: {sorted(set(normalized))}")
    return normalized[0]


def clean_chunk_text(text: str) -> str:
    """Flatten a chunk and drop residual MyScheme navigation noise."""
    text = " ".join(text.split())
    text = re.sub(r"^Section:\s+[A-Za-z ]+\s*", "", text)
    text = re.sub(r"^.*?Are you sure you want to\s*\??\s*", "", text)
    # Leading category/tag runs such as "EducationDetails" or
    # "SecuritySocial WelfareWidowDetails" are scraped UI, not prose.
    text = re.sub(r"^[A-Z][A-Za-z&/\s]{0,80}?Details", "", text)
    return text.strip()


def summary_sentences(chunks: list[dict]) -> list[str]:
    """Extract clean, informative sentences from a chunk's prose."""
    sentences: list[str] = []
    for chunk in chunks:
        for sentence in SENTENCE_SPLIT.split(clean_chunk_text(chunk["text"])):
            sentence = sentence.strip()
            if len(sentence) < MIN_SUMMARY_SENTENCE_CHARS or UI_NOISE.search(sentence):
                continue
            if sentence not in sentences:
                sentences.append(sentence)
    return sentences


def section_chunks(scheme_chunks: list[dict], section: str) -> list[dict]:
    return [chunk for chunk in scheme_chunks if chunk["metadata"].get("section") == section]


def build_summary(scheme_chunks: list[dict]) -> str:
    """Summarize a scheme using its opening General-section sentences."""
    candidates = summary_sentences(section_chunks(scheme_chunks, SUMMARY_SECTION))
    if not candidates:
        candidates = summary_sentences(section_chunks(scheme_chunks, SUMMARY_FALLBACK_SECTION))

    summary: list[str] = []
    budget = 0
    for sentence in candidates:
        if len(summary) >= SUMMARY_SENTENCES or budget + len(sentence) + 1 > SUMMARY_MAX_CHARS:
            break
        summary.append(sentence)
        budget += len(sentence) + 1

    return " ".join(summary)


def make_card(source_file: str, scheme_chunks: list[dict]) -> dict:
    """Combine a scheme's shared metadata and opening prose into one card."""
    metadata = [chunk["metadata"] for chunk in scheme_chunks]
    card = {"source_file": source_file}
    for field in TEXT_FIELDS:
        card[field] = first_nonempty([item.get(field) for item in metadata])

    categories: list[str] = []
    for item in metadata:
        for category in item.get("beneficiary_category") or []:
            if isinstance(category, str) and category.strip() and category not in categories:
                categories.append(category)
    card["beneficiary_category"] = categories

    lines = [f"Scheme: {card['scheme_name'] or source_file}"]
    field_labels = {
        "abbreviation": "Abbreviation",
        "state": "State or territory",
        "ministry": "Ministry",
        "department": "Department",
        "scheme_type": "Scheme type",
        "sponsorship": "Sponsorship",
        "application_mode": "Application mode",
    }
    for field, label in field_labels.items():
        if card[field]:
            lines.append(f"{label}: {card[field]}")
    if categories:
        lines.append("Beneficiary categories: " + "; ".join(categories))

    summary = build_summary(scheme_chunks)
    card["summary"] = summary or None
    if summary:
        lines.append(f"Summary: {summary}")

    card["card_text"] = "\n".join(lines)
    return card


def build_cards(chunks: list[dict]) -> list[dict]:
    """Create cards in stable source-file order."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        source_file = chunk["metadata"].get("source_file")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError(f"Chunk {chunk.get('chunk_id')} has no source_file")
        by_source[source_file].append(chunk)
    return [make_card(source_file, by_source[source_file]) for source_file in sorted(by_source)]


def cards_fingerprint(cards: list[dict]) -> str:
    payload = "\n".join(f"{card['source_file']}\t{card['card_text']}" for card in cards)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    cards = build_cards(load_chunks())
    model = SentenceTransformer(MODEL_ID, cache_folder=str(MODEL_DIR), local_files_only=True)
    vectors = model.encode(
        [card["card_text"] for card in cards],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CARD_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "distance": "cosine similarity (normalized vectors + inner product)",
                "scheme_count": len(cards),
                "vector_dimensions": int(vectors.shape[1]),
                "metadata_fields": [*TEXT_FIELDS, "beneficiary_category", "summary"],
                "summary_section": SUMMARY_SECTION,
                "summary_sentences": SUMMARY_SENTENCES,
                "summary_max_chars": SUMMARY_MAX_CHARS,
                "cards_sha256": cards_fingerprint(cards),
                "created_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {index.ntotal} scheme cards to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
