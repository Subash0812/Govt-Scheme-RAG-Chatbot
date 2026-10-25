"""Two-stage local retrieval for named and broad scheme questions.

Named-scheme questions use a title/abbreviation match to restrict chunk
retrieval to one source document. Broad questions first search scheme cards
(metadata plus a short description), then retrieve evidence chunks inside each
selected scheme.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .scheme_matching import CARD_FILE, SchemeMatch, load_scheme_cards, match_scheme


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store" / "bge_base_en_v1_5"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_ID = "BAAI/bge-base-en-v1.5"
CHUNK_INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
SCHEME_INDEX_FILE = VECTOR_STORE_DIR / "scheme_index.faiss"
CHUNK_FILE = VECTOR_STORE_DIR / "chunks.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class SchemeHit:
    source_file: str
    scheme_name: str
    score: float


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    source_file: str
    scheme_name: str
    score: float
    rank_within_scheme: int
    text: str
    metadata: dict


@dataclass(frozen=True)
class RetrievalResult:
    route: str
    exact_match: SchemeMatch | None
    schemes: list[SchemeHit]
    chunks_by_source: dict[str, list[ChunkHit]]


class SchemeRetrievalRouter:
    """Route a question to the right scheme(s), then retrieve answer evidence."""

    def __init__(
        self,
        chunks: list[dict],
        chunk_index: faiss.Index,
        cards: list[dict],
        scheme_index: faiss.Index,
        model: Any,
    ) -> None:
        if chunk_index.ntotal != len(chunks):
            raise ValueError("Chunk index count does not match chunks.json.")
        if scheme_index.ntotal != len(cards):
            raise ValueError("Scheme index count does not match scheme_cards.json.")
        self.chunks = chunks
        self.chunk_index = chunk_index
        self.cards = cards
        self.scheme_index = scheme_index
        self.model = model
        self.card_by_source = {card["source_file"]: card for card in cards}
        self.chunk_indices_by_source: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            source_file = chunk["metadata"]["source_file"]
            self.chunk_indices_by_source.setdefault(source_file, []).append(index)

    @classmethod
    def from_local_store(cls) -> "SchemeRetrievalRouter":
        if not SCHEME_INDEX_FILE.exists() or not CARD_FILE.exists():
            raise FileNotFoundError(
                "Scheme-card index is missing. Run: python -m src.embeddings.build_scheme_card_index"
            )
        chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
        cards = load_scheme_cards()
        from sentence_transformers import SentenceTransformer

        return cls(
            chunks=chunks,
            chunk_index=faiss.read_index(str(CHUNK_INDEX_FILE)),
            cards=cards,
            scheme_index=faiss.read_index(str(SCHEME_INDEX_FILE)),
            model=SentenceTransformer(MODEL_ID, cache_folder=str(MODEL_DIR), local_files_only=True),
        )

    def encode_query(self, question: str) -> np.ndarray:
        vector = self.model.encode(
            [QUERY_PREFIX + question], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        return vector[0]

    def rank_source_chunks(
        self, source_file: str, query_vector: np.ndarray, limit: int
    ) -> list[ChunkHit]:
        indices = self.chunk_indices_by_source.get(source_file, [])
        if not indices:
            return []
        vectors = np.vstack([self.chunk_index.reconstruct(index) for index in indices])
        scores = vectors @ query_vector
        ranked_positions = np.argsort(-scores)[:limit]
        hits = []
        for rank, position in enumerate(ranked_positions, start=1):
            chunk = self.chunks[indices[int(position)]]
            metadata = chunk["metadata"]
            hits.append(
                ChunkHit(
                    chunk_id=chunk["chunk_id"],
                    source_file=source_file,
                    scheme_name=metadata.get("scheme_name") or source_file,
                    score=float(scores[int(position)]),
                    rank_within_scheme=rank,
                    text=chunk["text"],
                    metadata=metadata,
                )
            )
        return hits

    def broad_scheme_hits(self, query_vector: np.ndarray, limit: int) -> list[SchemeHit]:
        scores, indices = self.scheme_index.search(query_vector.reshape(1, -1), limit)
        hits = []
        for score, card_index in zip(scores[0], indices[0], strict=True):
            card = self.cards[int(card_index)]
            hits.append(
                SchemeHit(
                    source_file=card["source_file"],
                    scheme_name=card.get("scheme_name") or card["source_file"],
                    score=float(score),
                )
            )
        return hits

    def retrieve(
        self, question: str, scheme_limit: int = 3, chunks_per_scheme: int = 3
    ) -> RetrievalResult:
        if scheme_limit < 1 or chunks_per_scheme < 1:
            raise ValueError("scheme_limit and chunks_per_scheme must both be at least 1.")
        query_vector = self.encode_query(question)
        exact = match_scheme(question, self.cards)
        if exact:
            card = self.card_by_source[exact.source_file]
            scheme = SchemeHit(exact.source_file, card["scheme_name"], exact.score)
            return RetrievalResult(
                route="exact_scheme",
                exact_match=exact,
                schemes=[scheme],
                chunks_by_source={
                    exact.source_file: self.rank_source_chunks(
                        exact.source_file, query_vector, chunks_per_scheme
                    )
                },
            )

        schemes = self.broad_scheme_hits(query_vector, min(scheme_limit, self.scheme_index.ntotal))
        return RetrievalResult(
            route="broad_scheme_discovery",
            exact_match=None,
            schemes=schemes,
            chunks_by_source={
                scheme.source_file: self.rank_source_chunks(
                    scheme.source_file, query_vector, chunks_per_scheme
                )
                for scheme in schemes
            },
        )


def format_result(result: RetrievalResult) -> str:
    """Format compact diagnostic output for notebooks and command-line checks."""
    lines = [f"route={result.route}"]
    if result.exact_match:
        lines.append(
            f"exact_match={result.exact_match.scheme_name} ({result.exact_match.match_type}, "
            f"score={result.exact_match.score:.3f})"
        )
    for scheme in result.schemes:
        lines.append(f"scheme={scheme.scheme_name} score={scheme.score:.3f}")
        for chunk in result.chunks_by_source[scheme.source_file]:
            lines.append(
                f"  chunk={chunk.chunk_id} score={chunk.score:.3f} "
                f"rank_within_scheme={chunk.rank_within_scheme}"
            )
    return "\n".join(lines)
