"""Evaluate scheme and evidence-chunk retrieval for common user intents.

Run from the repository root after building the FAISS index:
    python -m src.embeddings.evaluate_intent_retrieval
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .build_faiss_index import MODEL_DIR, MODEL_ID, VECTOR_STORE_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = PROJECT_ROOT / "data" / "evaluation" / "intent_retrieval_cases.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "results" / "intent_retrieval_report.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_VALUES = (1, 3, 5)


def load_cases() -> list[dict]:
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    for case in cases:
        required = {"id", "question", "gold_source_file", "gold_chunk_ids"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')} is missing: {sorted(missing)}")
    return cases


def first_source_rank(indices: np.ndarray, chunks: list[dict], source_file: str) -> int | None:
    seen_sources: set[str] = set()
    for chunk_index in indices.tolist():
        source = chunks[chunk_index]["metadata"]["source_file"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        if source == source_file:
            return len(seen_sources)
    return None


def first_chunk_rank(indices: np.ndarray, chunks: list[dict], gold_chunk_ids: set[str]) -> int | None:
    return next(
        (
            rank
            for rank, chunk_index in enumerate(indices.tolist(), start=1)
            if chunks[chunk_index]["chunk_id"] in gold_chunk_ids
        ),
        None,
    )


def first_chunk_for_source(indices: np.ndarray, chunks: list[dict], source_file: str) -> dict | None:
    return next(
        (
            chunks[chunk_index]
            for chunk_index in indices.tolist()
            if chunks[chunk_index]["metadata"]["source_file"] == source_file
        ),
        None,
    )


def at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def main() -> None:
    cases = load_cases()
    chunks = json.loads((VECTOR_STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
    if index.ntotal != len(chunks):
        raise ValueError("FAISS vector count and chunk payload length do not match.")

    known_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    for case in cases:
        unknown = set(case["gold_chunk_ids"]) - known_chunk_ids
        if unknown:
            raise ValueError(f"{case['id']} references missing chunks: {sorted(unknown)}")

    model = SentenceTransformer(MODEL_ID, cache_folder=str(MODEL_DIR), local_files_only=True)
    vectors = model.encode(
        [QUERY_PREFIX + case["question"] for case in cases],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, all_indices = index.search(vectors, index.ntotal)

    results = []
    for case, row_scores, row_indices in zip(cases, scores, all_indices, strict=True):
        gold_chunks = set(case["gold_chunk_ids"])
        source_rank = first_source_rank(row_indices, chunks, case["gold_source_file"])
        chunk_rank = first_chunk_rank(row_indices, chunks, gold_chunks)
        selected_chunk = first_chunk_for_source(row_indices, chunks, case["gold_source_file"])
        selected_chunk_id = selected_chunk["chunk_id"] if selected_chunk else None
        results.append(
            {
                **case,
                "scheme_rank": source_rank,
                "evidence_chunk_rank": chunk_rank,
                "current_one_chunk_context": selected_chunk_id,
                "current_context_contains_evidence": selected_chunk_id in gold_chunks,
                "top_chunk_id": chunks[int(row_indices[0])]["chunk_id"],
                "top_chunk_score": round(float(row_scores[0]), 4),
            }
        )

    total = len(results)
    metrics = {
        "scheme_recall": {
            f"at_{k}": round(sum(at_k(row["scheme_rank"], k) for row in results) / total, 4)
            for k in K_VALUES
        },
        "evidence_chunk_recall": {
            f"at_{k}": round(sum(at_k(row["evidence_chunk_rank"], k) for row in results) / total, 4)
            for k in K_VALUES
        },
        "current_one_chunk_context_evidence_rate": round(
            sum(row["current_context_contains_evidence"] for row in results) / total,
            4,
        ),
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "case_count": total,
        "metrics": metrics,
        "cases": results,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated {total} common user-intent questions")
    for label, values in metrics.items():
        print(f"{label}: " + ", ".join(f"{key}={value:.2%}" for key, value in values.items()) if isinstance(values, dict) else f"{label}: {values:.2%}")
    print(f"Detailed report: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
