"""Evaluate multi-scheme retrieval for broad, user-style questions.

The labels are manually curated high-confidence matches from the current
45-scheme corpus. They are intentionally not a claim that a person qualifies;
the answer layer must still collect location and eligibility facts.

Run from the repository root after building the FAISS index:
    python -m src.embeddings.evaluate_broad_retrieval
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .build_faiss_index import MODEL_DIR, MODEL_ID, VECTOR_STORE_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = PROJECT_ROOT / "data" / "evaluation" / "broad_retrieval_cases.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "results" / "broad_retrieval_report.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_VALUES = (1, 3, 5)


def load_cases() -> list[dict]:
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    for case in cases:
        required = {"id", "question", "gold_source_files"}
        missing = required - case.keys()
        if missing or not case["gold_source_files"]:
            raise ValueError(f"Invalid case {case.get('id', '<unknown>')}: missing {sorted(missing)}")
    return cases


def unique_documents(indices: np.ndarray, chunks: list[dict]) -> list[str]:
    documents: list[str] = []
    seen: set[str] = set()
    for chunk_index in indices.tolist():
        source_file = chunks[chunk_index]["metadata"]["source_file"]
        if source_file not in seen:
            seen.add(source_file)
            documents.append(source_file)
    return documents


def dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))


def evaluate_case(retrieved: list[str], gold: set[str]) -> dict:
    first_relevant_rank = next((rank for rank, source in enumerate(retrieved, start=1) if source in gold), None)
    metrics: dict[str, dict[str, float]] = {}
    for k in K_VALUES:
        top_k = retrieved[:k]
        relevant = [source for source in top_k if source in gold]
        relevance = [int(source in gold) for source in top_k]
        ideal = [1] * min(k, len(gold))
        metrics[f"at_{k}"] = {
            "precision": len(relevant) / k,
            "recall": len(relevant) / len(gold),
            "hit": float(bool(relevant)),
            "ndcg": dcg(relevance) / dcg(ideal),
        }
    return {
        "first_relevant_rank": first_relevant_rank,
        "mrr_at_5": 1 / first_relevant_rank if first_relevant_rank and first_relevant_rank <= 5 else 0.0,
        "metrics": metrics,
    }


def main() -> None:
    cases = load_cases()
    chunks = json.loads((VECTOR_STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
    indexed_sources = {chunk["metadata"]["source_file"] for chunk in chunks}
    for case in cases:
        unknown = set(case["gold_source_files"]) - indexed_sources
        if unknown:
            raise ValueError(f"{case['id']} references missing sources: {sorted(unknown)}")

    index = faiss.read_index(str(VECTOR_STORE_DIR / "index.faiss"))
    if index.ntotal != len(chunks):
        raise ValueError("FAISS vector count and chunk payload length do not match.")

    model = SentenceTransformer(MODEL_ID, cache_folder=str(MODEL_DIR), local_files_only=True)
    vectors = model.encode(
        [QUERY_PREFIX + case["question"] for case in cases],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    _, indices = index.search(vectors, index.ntotal)

    results = []
    for case, row in zip(cases, indices, strict=True):
        retrieved = unique_documents(row, chunks)
        assessment = evaluate_case(retrieved, set(case["gold_source_files"]))
        results.append(
            {
                **case,
                "retrieved_source_files_top_5": retrieved[:5],
                "relevant_source_files_top_5": [source for source in retrieved[:5] if source in case["gold_source_files"]],
                **assessment,
            }
        )

    summary = {
        f"at_{k}": {
            metric: round(sum(row["metrics"][f"at_{k}"][metric] for row in results) / len(results), 4)
            for metric in ("precision", "recall", "hit", "ndcg")
        }
        for k in K_VALUES
    }
    summary["mrr_at_5"] = round(sum(row["mrr_at_5"] for row in results) / len(results), 4)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "case_count": len(results),
        "evaluation_unit": "deduplicated scheme documents",
        "labeling_scope": "Manually curated, high-confidence relevant schemes from the indexed corpus; not eligibility determinations.",
        "metrics": summary,
        "cases": results,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated {len(results)} broad multi-scheme questions")
    for k in K_VALUES:
        values = summary[f"at_{k}"]
        print(
            f"@{k}: precision={values['precision']:.2%}, recall={values['recall']:.2%}, "
            f"hit-rate={values['hit']:.2%}, nDCG={values['ndcg']:.2%}"
        )
    print(f"MRR@5: {summary['mrr_at_5']:.2%}")
    print(f"Detailed report: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
