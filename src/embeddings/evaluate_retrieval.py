"""Evaluate document-level retrieval quality for the local FAISS scheme store.

Run from the repository root after building the index:
    python -m src.embeddings.evaluate_retrieval
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .build_faiss_index import MODEL_DIR, MODEL_ID, VECTOR_STORE_DIR


RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "results"
CASE_FILE = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "retrieval_cases.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K_VALUES = (1, 3, 5)
CHUNK_CANDIDATE_COUNT = 30

def load_cases() -> list[dict]:
    """Load the version-controlled, manually curated question set."""
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    if len(cases) != 100:
        raise ValueError(f"Expected exactly 100 evaluation cases, found {len(cases)}.")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique.")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case.get('id')} has no question.")
        if not case.get("gold_source_files"):
            raise ValueError(f"Case {case.get('id')} has no gold documents.")
    return cases


def unique_documents(indices: np.ndarray, chunks: list[dict], limit: int) -> list[dict]:
    """Collapse chunk hits to schemes to avoid repeated chunks inflating a ranking."""
    results: list[dict] = []
    seen_files: set[str] = set()
    for rank, chunk_index in enumerate(indices.tolist(), start=1):
        chunk = chunks[chunk_index]
        source_file = chunk["metadata"]["source_file"]
        if source_file in seen_files:
            continue
        seen_files.add(source_file)
        results.append(
            {
                "document_rank": len(results) + 1,
                "first_chunk_rank": rank,
                "source_file": source_file,
                "scheme_name": chunk["metadata"].get("scheme_name"),
                "chunk_id": chunk["chunk_id"],
            }
        )
        if len(results) == limit:
            break
    return results


def first_relevant_rank(results: list[dict], gold_files: set[str]) -> int | None:
    return next(
        (item["document_rank"] for item in results if item["source_file"] in gold_files),
        None,
    )


def relevance_counts(results: list[dict], gold_files: set[str], k: int) -> dict[str, int]:
    top_k = results[:k]
    relevant_retrieved = sum(item["source_file"] in gold_files for item in top_k)
    return {
        "relevant_documents": len(gold_files),
        "retrieved_documents": len(top_k),
        "relevant_retrieved_documents": relevant_retrieved,
    }


def main() -> None:
    evaluation_cases = load_cases()
    manifest = json.loads((VECTOR_STORE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID:
        raise ValueError("The vector store was built with a different embedding model.")

    chunks = json.loads((VECTOR_STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
    if index.ntotal != len(chunks):
        raise ValueError("FAISS vector count and chunk payload length do not match.")

    # Evaluation must be reproducible and independent of HF Hub availability.
    # The builder is responsible for the first download.
    model = SentenceTransformer(
        MODEL_ID,
        cache_folder=str(MODEL_DIR),
        local_files_only=True,
    )
    query_vectors = model.encode(
        [QUERY_PREFIX + case["question"] for case in evaluation_cases],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    _, indices = index.search(query_vectors, CHUNK_CANDIDATE_COUNT)
    evaluated_cases = []
    max_k = max(TOP_K_VALUES)
    for case, row in zip(evaluation_cases, indices, strict=True):
        retrieved = unique_documents(row, chunks, max_k)
        gold_files = set(case["gold_source_files"])
        relevant_rank = first_relevant_rank(retrieved, gold_files)
        evaluated_cases.append(
            {
                **case,
                "relevant_document_rank": relevant_rank,
                "counts": {
                    f"at_{k}": relevance_counts(retrieved, gold_files, k)
                    for k in TOP_K_VALUES
                },
                "retrieved_documents": retrieved,
            }
        )

    metrics = {}
    for k in TOP_K_VALUES:
        relevant_retrieved = sum(case["counts"][f"at_{k}"]["relevant_retrieved_documents"] for case in evaluated_cases)
        retrieved = sum(case["counts"][f"at_{k}"]["retrieved_documents"] for case in evaluated_cases)
        relevant = sum(case["counts"][f"at_{k}"]["relevant_documents"] for case in evaluated_cases)
        metrics[f"precision_at_{k}"] = round(relevant_retrieved / retrieved, 4)
        metrics[f"recall_at_{k}"] = round(relevant_retrieved / relevant, 4)
    metrics["mrr_at_5"] = round(
        sum(
            1 / case["relevant_document_rank"]
            if case["relevant_document_rank"] is not None
            else 0
            for case in evaluated_cases
        )
        / len(evaluated_cases),
        4,
    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "corpus_sha256": manifest["corpus_sha256"],
        "evaluation_unit": "scheme document (deduplicated from chunk hits)",
        "case_count": len(evaluated_cases),
        "metrics": metrics,
        "cases": evaluated_cases,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "retrieval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated {len(evaluated_cases)} difficult retrieval cases")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Detailed report: {report_path}")


if __name__ == "__main__":
    main()
