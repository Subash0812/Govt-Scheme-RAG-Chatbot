"""Evaluate the retrieval router against the previous flat chunk retrieval.

Three checks are reported:
- broad multi-scheme questions: flat chunks vs scheme-card discovery;
- named-scheme questions: flat chunks vs exact-scheme routing;
- a 100-case single-scheme suite used only as a route/regression diagnostic.

Run from the repository root:
    python -m src.embeddings.evaluate_router_retrieval
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .build_faiss_index import MODEL_ID
from .retrieval_router import ChunkHit, RetrievalResult, SchemeRetrievalRouter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BROAD_CASE_FILE = PROJECT_ROOT / "data" / "evaluation" / "broad_retrieval_cases.json"
INTENT_CASE_FILE = PROJECT_ROOT / "data" / "evaluation" / "intent_retrieval_cases.json"
RETRIEVAL_CASE_FILE = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "results" / "router_retrieval_report.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
K_VALUES = (1, 3, 5)
BASELINE_CANDIDATES = 30
DEFAULT_SCHEMES = 3
DEFAULT_CHUNKS = 3
EXTENDED_SCHEMES = 5
EVIDENCE_CHUNKS = 5
TEXT_SNIPPET = 180


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError(f"Duplicate case ids in {path}")
    return cases


def encode_queries(model, questions: list[str]) -> np.ndarray:
    return model.encode(
        [QUERY_PREFIX + question for question in questions],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def unique_sources(indices: np.ndarray, chunks: list[dict], limit: int) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for chunk_index in indices.tolist():
        source_file = chunks[chunk_index]["metadata"]["source_file"]
        if source_file in seen:
            continue
        seen.add(source_file)
        sources.append(source_file)
        if len(sources) == limit:
            break
    return sources


def ranking_metrics(retrieved: list[str], gold: set[str], k_values=K_VALUES) -> dict:
    first_relevant_rank = next(
        (rank for rank, source in enumerate(retrieved, start=1) if source in gold),
        None,
    )
    metrics: dict[str, dict[str, float]] = {}
    for k in k_values:
        top_k = retrieved[:k]
        relevance = [int(source in gold) for source in top_k]
        relevant = sum(relevance)
        ideal = [1] * min(k, len(gold))
        dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))
        idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
        metrics[f"at_{k}"] = {
            "precision": relevant / k,
            "recall": relevant / len(gold),
            "hit": float(relevant > 0),
            "ndcg": dcg / idcg if idcg else 0.0,
        }
    return {
        "first_relevant_rank": first_relevant_rank,
        "mrr_at_5": 1 / first_relevant_rank
        if first_relevant_rank is not None and first_relevant_rank <= 5
        else 0.0,
        "metrics": metrics,
    }


def chunk_context_stats(chunk_sources: list[str], gold: set[str]) -> dict:
    relevant = [source for source in chunk_sources if source in gold]
    return {
        "chunks": len(chunk_sources),
        "relevant_chunks": len(relevant),
        "chunk_precision": len(relevant) / len(chunk_sources) if chunk_sources else 0.0,
        "gold_schemes_covered": len(set(relevant)),
        "gold_scheme_coverage": len(set(relevant)) / len(gold),
        "distinct_schemes": len(set(chunk_sources)),
        "irrelevant_scheme_count": len(set(chunk_sources) - gold),
    }


def aggregate_metrics(rows: list[dict], key: str, k_values: tuple[int, ...]) -> dict:
    metrics: dict[str, dict[str, float] | float] = {}
    for k in k_values:
        metrics[f"at_{k}"] = {
            name: round(statistics.mean(row[key]["metrics"][f"at_{k}"][name] for row in rows), 4)
            for name in ("precision", "recall", "hit", "ndcg")
        }
    metrics["mrr_at_5"] = round(statistics.mean(row[key]["mrr_at_5"] for row in rows), 4)
    return metrics


def metric_delta(before: dict, after: dict) -> dict:
    deltas: dict[str, float] = {}
    for key, value in before.items():
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                deltas[f"{metric}_{key}"] = round(after[key][metric] - metric_value, 4)
        else:
            deltas[key] = round(after[key] - value, 4)
    return deltas


def first_rank_outcomes(
    rows: list[dict], baseline_key: str, router_key: str
) -> dict[str, list[str]]:
    improved: list[str] = []
    worsened: list[str] = []
    unchanged: list[str] = []
    for row in rows:
        baseline_rank = row[baseline_key]["first_relevant_rank"]
        router_rank = row[router_key]["first_relevant_rank"]
        if baseline_rank is None and router_rank is None:
            unchanged.append(row["id"])
        elif baseline_rank is None or (router_rank is not None and router_rank < baseline_rank):
            improved.append(row["id"])
        elif router_rank is None or router_rank > baseline_rank:
            worsened.append(row["id"])
        else:
            unchanged.append(row["id"])
    return {"improved": improved, "worsened": worsened, "unchanged": unchanged}


def chunk_hit_details(hits: list[ChunkHit]) -> list[dict]:
    return [
        {
            "chunk_id": hit.chunk_id,
            "source_file": hit.source_file,
            "scheme_name": hit.scheme_name,
            "score": round(hit.score, 4),
            "rank_within_scheme": hit.rank_within_scheme,
            "text": " ".join(hit.text.split())[:TEXT_SNIPPET],
        }
        for hit in hits
    ]


def flatten_hits(result: RetrievalResult) -> list[ChunkHit]:
    return [hit for source in result.schemes for hit in result.chunks_by_source[source.source_file]]


def scheme_records(result: RetrievalResult) -> list[dict]:
    return [
        {
            "source_file": scheme.source_file,
            "scheme_name": scheme.scheme_name,
            "score": round(scheme.score, 4),
        }
        for scheme in result.schemes
    ]


def exact_record(result: RetrievalResult) -> dict | None:
    if result.exact_match is None:
        return None
    return {
        "source_file": result.exact_match.source_file,
        "scheme_name": result.exact_match.scheme_name,
        "match_type": result.exact_match.match_type,
        "score": round(result.exact_match.score, 4),
        "matched_text": result.exact_match.matched_text,
    }


def evaluate_broad(router: SchemeRetrievalRouter, cases: list[dict]) -> dict:
    vectors = encode_queries(router.model, [case["question"] for case in cases])
    _, baseline_indices = router.chunk_index.search(vectors, BASELINE_CANDIDATES)

    evaluated = []
    for case, row in zip(cases, baseline_indices, strict=True):
        gold = set(case["gold_source_files"])
        baseline_sources = unique_sources(row, router.chunks, limit=5)
        baseline_chunks = [router.chunks[int(index)] for index in row]
        baseline_record = {
            "schemes": baseline_sources,
            **ranking_metrics(baseline_sources, gold, K_VALUES),
            "chunk_context_at_9": chunk_context_stats(
                [chunk["metadata"]["source_file"] for chunk in baseline_chunks[:9]], gold
            ),
            "chunk_context_at_15": chunk_context_stats(
                [chunk["metadata"]["source_file"] for chunk in baseline_chunks[:15]], gold
            ),
        }

        default = router.retrieve(
            case["question"], scheme_limit=DEFAULT_SCHEMES, chunks_per_scheme=DEFAULT_CHUNKS
        )
        extended = router.retrieve(
            case["question"], scheme_limit=EXTENDED_SCHEMES, chunks_per_scheme=DEFAULT_CHUNKS
        )
        default_hits = flatten_hits(default)
        extended_hits = flatten_hits(extended)
        default_sources = [scheme.source_file for scheme in default.schemes]
        extended_sources = [scheme.source_file for scheme in extended.schemes]

        query_vector = router.encode_query(case["question"])
        card_hits = router.broad_scheme_hits(query_vector, router.scheme_index.ntotal)
        card_ranks = {scheme.source_file: rank for rank, scheme in enumerate(card_hits, start=1)}

        evaluated.append(
            {
                **case,
                "route": default.route,
                "exact_match": exact_record(default),
                "gold_scheme_card_ranks": {
                    source: card_ranks.get(source) for source in case["gold_source_files"]
                },
                "baseline_top5": baseline_record,
                "router_default_3x3": {
                    "schemes": scheme_records(default),
                    **ranking_metrics(default_sources, gold, (1, 3)),
                    "chunk_context": chunk_context_stats(
                        [hit.source_file for hit in default_hits], gold
                    ),
                    "chunks": chunk_hit_details(default_hits),
                },
                "router_extended_5x3": {
                    "schemes": scheme_records(extended),
                    **ranking_metrics(extended_sources, gold, K_VALUES),
                    "chunk_context": chunk_context_stats(
                        [hit.source_file for hit in extended_hits], gold
                    ),
                },
            }
        )

    summary = {
        "case_count": len(evaluated),
        "route_counts": dict(Counter(row["route"] for row in evaluated)),
        "exact_route_count": sum(row["exact_match"] is not None for row in evaluated),
        "baseline_top5": aggregate_metrics(evaluated, "baseline_top5", K_VALUES),
        "router_default_3x3": aggregate_metrics(evaluated, "router_default_3x3", (1, 3)),
        "router_extended_5x3": aggregate_metrics(evaluated, "router_extended_5x3", K_VALUES),
        "chunk_context": {
            "baseline_at_9": round(
                statistics.mean(
                    row["baseline_top5"]["chunk_context_at_9"]["chunk_precision"]
                    for row in evaluated
                ),
                4,
            ),
            "router_default_at_9": round(
                statistics.mean(
                    row["router_default_3x3"]["chunk_context"]["chunk_precision"]
                    for row in evaluated
                ),
                4,
            ),
            "baseline_at_15": round(
                statistics.mean(
                    row["baseline_top5"]["chunk_context_at_15"]["chunk_precision"]
                    for row in evaluated
                ),
                4,
            ),
            "router_extended_at_15": round(
                statistics.mean(
                    row["router_extended_5x3"]["chunk_context"]["chunk_precision"]
                    for row in evaluated
                ),
                4,
            ),
        },
        "first_rank_outcomes": {
            "router_default_vs_baseline": first_rank_outcomes(
                evaluated, "baseline_top5", "router_default_3x3"
            ),
            "router_extended_vs_baseline": first_rank_outcomes(
                evaluated, "baseline_top5", "router_extended_5x3"
            ),
        },
    }
    summary["deltas"] = {
        "router_default_vs_baseline": {
            "precision_at_1": round(
                summary["router_default_3x3"]["at_1"]["precision"]
                - summary["baseline_top5"]["at_1"]["precision"],
                4,
            ),
            "precision_at_3": round(
                summary["router_default_3x3"]["at_3"]["precision"]
                - summary["baseline_top5"]["at_3"]["precision"],
                4,
            ),
            "recall_at_1": round(
                summary["router_default_3x3"]["at_1"]["recall"]
                - summary["baseline_top5"]["at_1"]["recall"],
                4,
            ),
            "recall_at_3": round(
                summary["router_default_3x3"]["at_3"]["recall"]
                - summary["baseline_top5"]["at_3"]["recall"],
                4,
            ),
            "chunk_precision_at_9": round(
                summary["chunk_context"]["router_default_at_9"]
                - summary["chunk_context"]["baseline_at_9"],
                4,
            ),
        },
        "router_extended_vs_baseline": {
            **metric_delta(summary["baseline_top5"], summary["router_extended_5x3"]),
            "chunk_precision_at_15": round(
                summary["chunk_context"]["router_extended_at_15"]
                - summary["chunk_context"]["baseline_at_15"],
                4,
            ),
        },
    }
    return {"summary": summary, "cases": evaluated}


def first_gold_chunk_rank(hits: list[ChunkHit], gold_chunk_ids: set[str]) -> int | None:
    return next(
        (rank for rank, hit in enumerate(hits, start=1) if hit.chunk_id in gold_chunk_ids),
        None,
    )


def evaluate_named(router: SchemeRetrievalRouter, cases: list[dict]) -> dict:
    vectors = encode_queries(router.model, [case["question"] for case in cases])
    _, baseline_indices = router.chunk_index.search(vectors, BASELINE_CANDIDATES)

    evaluated = []
    for case, row in zip(cases, baseline_indices, strict=True):
        gold_source = case["gold_source_file"]
        gold_chunks = set(case["gold_chunk_ids"])
        baseline_sources = unique_sources(row, router.chunks, limit=5)
        baseline_rank = next(
            (rank for rank, source in enumerate(baseline_sources, start=1) if source == gold_source),
            None,
        )
        baseline_raw = [router.chunks[int(index)] for index in row]
        baseline_evidence_rank = next(
            (
                rank
                for rank, chunk in enumerate(baseline_raw, start=1)
                if chunk["chunk_id"] in gold_chunks
            ),
            None,
        )
        baseline_record = {
            "scheme_rank": baseline_rank,
            "evidence_chunk_rank": baseline_evidence_rank,
            "top_chunk_id": baseline_raw[0]["chunk_id"],
            "top_chunk_contains_evidence": baseline_raw[0]["chunk_id"] in gold_chunks,
            "chunk_context_at_5": chunk_context_stats(
                [chunk["metadata"]["source_file"] for chunk in baseline_raw[:5]], {gold_source}
            ),
        }

        default = router.retrieve(case["question"])
        extended = router.retrieve(
            case["question"], scheme_limit=DEFAULT_SCHEMES, chunks_per_scheme=EVIDENCE_CHUNKS
        )
        default_hits = flatten_hits(default)
        extended_hits = flatten_hits(extended)
        default_sources = [scheme.source_file for scheme in default.schemes]
        extended_sources = [scheme.source_file for scheme in extended.schemes]
        default_scheme_rank = next(
            (rank for rank, source in enumerate(default_sources, start=1) if source == gold_source),
            None,
        )
        extended_scheme_rank = next(
            (rank for rank, source in enumerate(extended_sources, start=1) if source == gold_source),
            None,
        )
        default_evidence_rank = first_gold_chunk_rank(default_hits, gold_chunks)
        extended_evidence_rank = first_gold_chunk_rank(extended_hits, gold_chunks)

        evaluated.append(
            {
                **case,
                "baseline_top5": baseline_record,
                "router_default": {
                    "route": default.route,
                    "exact_match": exact_record(default),
                    "scheme_rank": default_scheme_rank,
                    "evidence_chunk_rank": default_evidence_rank,
                    "top_chunk_id": default_hits[0].chunk_id,
                    "top_chunk_contains_evidence": default_hits[0].chunk_id in gold_chunks,
                    "chunks": chunk_hit_details(default_hits),
                },
                "router_extended_5": {
                    "route": extended.route,
                    "exact_match": exact_record(extended),
                    "scheme_rank": extended_scheme_rank,
                    "evidence_chunk_rank": extended_evidence_rank,
                    "top_chunk_contains_evidence": extended_hits[0].chunk_id in gold_chunks,
                    "chunks": chunk_hit_details(extended_hits),
                },
            }
        )

    def recall_at(record_key: str, k: int) -> float:
        return round(
            sum(
                row[record_key]["evidence_chunk_rank"] is not None
                and row[record_key]["evidence_chunk_rank"] <= k
                for row in evaluated
            )
            / len(evaluated),
            4,
        )

    summary = {
        "case_count": len(evaluated),
        "expected_route": "exact_scheme",
        "route_counts": dict(Counter(row["router_default"]["route"] for row in evaluated)),
        "exact_route_accuracy": round(
            sum(
                row["router_default"]["route"] == "exact_scheme"
                and row["router_default"]["exact_match"] is not None
                and row["router_default"]["exact_match"]["source_file"] == row["gold_source_file"]
                for row in evaluated
            )
            / len(evaluated),
            4,
        ),
        "first_routed_scheme_accuracy": round(
            sum(row["router_default"]["scheme_rank"] == 1 for row in evaluated) / len(evaluated),
            4,
        ),
        "gold_scheme_in_router_context": round(
            sum(row["router_default"]["scheme_rank"] is not None for row in evaluated)
            / len(evaluated),
            4,
        ),
        "baseline": {
            "scheme_recall_at_1": round(
                sum(row["baseline_top5"]["scheme_rank"] == 1 for row in evaluated) / len(evaluated),
                4,
            ),
            "evidence_recall": {
                f"at_{k}": round(
                    sum(
                        row["baseline_top5"]["evidence_chunk_rank"] is not None
                        and row["baseline_top5"]["evidence_chunk_rank"] <= k
                        for row in evaluated
                    )
                    / len(evaluated),
                    4,
                )
                for k in K_VALUES
            },
            "top1_evidence_rate": round(
                sum(row["baseline_top5"]["top_chunk_contains_evidence"] for row in evaluated)
                / len(evaluated),
                4,
            ),
            "chunk_scheme_precision_at_5": round(
                statistics.mean(
                    row["baseline_top5"]["chunk_context_at_5"]["chunk_precision"]
                    for row in evaluated
                ),
                4,
            ),
        },
        "router_default": {
            "evidence_recall": {f"at_{k}": recall_at("router_default", k) for k in K_VALUES},
            "top1_evidence_rate": round(
                sum(row["router_default"]["top_chunk_contains_evidence"] for row in evaluated)
                / len(evaluated),
                4,
            ),
        },
        "router_extended_5": {
            "evidence_recall": {f"at_{k}": recall_at("router_extended_5", k) for k in K_VALUES},
            "top1_evidence_rate": round(
                sum(row["router_extended_5"]["top_chunk_contains_evidence"] for row in evaluated)
                / len(evaluated),
                4,
            ),
        },
    }
    return {"summary": summary, "cases": evaluated}


def evaluate_routing_diagnostic(router: SchemeRetrievalRouter, cases: list[dict]) -> dict:
    vectors = encode_queries(router.model, [case["question"] for case in cases])
    _, baseline_indices = router.chunk_index.search(vectors, BASELINE_CANDIDATES)

    evaluated = []
    for case, row in zip(cases, baseline_indices, strict=True):
        gold = set(case["gold_source_files"])
        baseline_sources = unique_sources(row, router.chunks, limit=5)
        baseline_metrics = ranking_metrics(baseline_sources, gold, K_VALUES)
        result = router.retrieve(
            case["question"], scheme_limit=EXTENDED_SCHEMES, chunks_per_scheme=1
        )
        router_sources = [scheme.source_file for scheme in result.schemes]
        router_metrics = ranking_metrics(router_sources, gold, K_VALUES)
        evaluated.append(
            {
                "id": case["id"],
                "question": case["question"],
                "gold_source_files": case["gold_source_files"],
                "route": result.route,
                "exact_match": exact_record(result),
                "selected_schemes": router_sources,
                "baseline_first_relevant_rank": baseline_metrics["first_relevant_rank"],
                "router_first_relevant_rank": router_metrics["first_relevant_rank"],
                "baseline_metrics": baseline_metrics,
                "router_metrics": router_metrics,
            }
        )

    exact_false_positives = [
        {
            "id": row["id"],
            "question": row["question"],
            "routed_scheme": row["exact_match"]["scheme_name"],
            "routed_source_file": row["exact_match"]["source_file"],
            "gold_source_files": row["gold_source_files"],
        }
        for row in evaluated
        if row["route"] == "exact_scheme"
        and row["exact_match"] is not None
        and row["exact_match"]["source_file"] not in set(row["gold_source_files"])
    ]
    exact_routes = sum(row["route"] == "exact_scheme" for row in evaluated)
    summary = {
        "case_count": len(evaluated),
        "evaluation_target": "single-scheme descriptive questions, not the router's broad-query target",
        "route_counts": dict(Counter(row["route"] for row in evaluated)),
        "exact_route_count": exact_routes,
        "exact_route_precision": round(
            (exact_routes - len(exact_false_positives)) / exact_routes if exact_routes else 0.0,
            4,
        ),
        "exact_route_false_positives": exact_false_positives,
        "baseline_top5": aggregate_metrics(evaluated, "baseline_metrics", K_VALUES),
        "router_top5": aggregate_metrics(evaluated, "router_metrics", K_VALUES),
        "first_routed_scheme_accuracy": round(
            sum(row["router_first_relevant_rank"] == 1 for row in evaluated) / len(evaluated),
            4,
        ),
        "first_rank_outcomes": first_rank_outcomes(
            evaluated, "baseline_metrics", "router_metrics"
        ),
        "deltas": metric_delta(
            aggregate_metrics(evaluated, "baseline_metrics", K_VALUES),
            aggregate_metrics(evaluated, "router_metrics", K_VALUES),
        ),
    }
    return {"summary": summary, "cases": evaluated}


def flat_retrieve(router: SchemeRetrievalRouter, question: str, limit: int = 5) -> list[str]:
    vector = router.encode_query(question)
    _, indices = router.chunk_index.search(vector.reshape(1, -1), BASELINE_CANDIDATES)
    return unique_sources(indices[0], router.chunks, limit=limit)


def measure_latency(
    router: SchemeRetrievalRouter, questions: list[str], repeats: int = 3
) -> dict:
    flat_retrieve(router, questions[0])
    router.retrieve(questions[0])

    baseline_times: list[float] = []
    router_times: list[float] = []
    for question in questions:
        for _ in range(repeats):
            started = time.perf_counter()
            flat_retrieve(router, question)
            baseline_times.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            router.retrieve(question)
            router_times.append((time.perf_counter() - started) * 1000)

    def summarize(times: list[float]) -> dict:
        ordered = sorted(times)
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
        return {
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(p95, 2),
        }

    return {
        "measurement": "single-query end-to-end including CPU query encoding",
        "repeats_per_question": repeats,
        "question_count": len(questions),
        "baseline_flat_chunks": summarize(baseline_times),
        "router_default_3x3": summarize(router_times),
    }


def main() -> None:
    broad_cases = load_cases(BROAD_CASE_FILE)
    named_cases = load_cases(INTENT_CASE_FILE)
    diagnostic_cases = load_cases(RETRIEVAL_CASE_FILE)
    router = SchemeRetrievalRouter.from_local_store()

    indexed_sources = {chunk["metadata"]["source_file"] for chunk in router.chunks}
    known_chunks = {chunk["chunk_id"] for chunk in router.chunks}
    for case in broad_cases:
        missing = set(case["gold_source_files"]) - indexed_sources
        if missing:
            raise ValueError(f"{case['id']} references missing sources: {sorted(missing)}")
    for case in named_cases:
        if case["gold_source_file"] not in indexed_sources:
            raise ValueError(f"{case['id']} references missing source: {case['gold_source_file']}")
        missing = set(case["gold_chunk_ids"]) - known_chunks
        if missing:
            raise ValueError(f"{case['id']} references missing chunks: {sorted(missing)}")
    for case in diagnostic_cases:
        missing = set(case["gold_source_files"]) - indexed_sources
        if missing:
            raise ValueError(f"{case['id']} references missing sources: {sorted(missing)}")

    broad = evaluate_broad(router, broad_cases)
    named = evaluate_named(router, named_cases)
    diagnostic = evaluate_routing_diagnostic(router, diagnostic_cases)
    latency = measure_latency(
        router, [case["question"] for case in broad_cases + named_cases]
    )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "corpus": {
            "scheme_cards": len(router.cards),
            "chunks": len(router.chunks),
        },
        "configuration": {
            "baseline": "flat FAISS chunk index, top 30 candidates",
            "broad_scheme_metrics_baseline": "unique scheme documents from chunks, top 5",
            "broad_router_default": f"{DEFAULT_SCHEMES} schemes x {DEFAULT_CHUNKS} chunks",
            "broad_router_extended": f"{EXTENDED_SCHEMES} schemes x {DEFAULT_CHUNKS} chunks",
            "named_router_default": f"routed scheme(s), {DEFAULT_CHUNKS} chunks",
            "named_router_extended": f"routed scheme(s), {EVIDENCE_CHUNKS} chunks",
            "chunk_context_baseline": "raw flat chunks at equal 9 and 15 chunk budgets",
        },
        "broad_multi_scheme": broad,
        "named_intent": named,
        "routing_diagnostic_100": diagnostic,
        "latency_ms": latency,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    broad_summary = broad["summary"]
    named_summary = named["summary"]
    diagnostic_summary = diagnostic["summary"]
    print(f"Broad suite: {broad_summary['case_count']} cases, routes={broad_summary['route_counts']}")
    print(
        "  baseline top5: "
        f"P@1={broad_summary['baseline_top5']['at_1']['precision']:.2%}, "
        f"R@5={broad_summary['baseline_top5']['at_5']['recall']:.2%}, "
        f"MRR@5={broad_summary['baseline_top5']['mrr_at_5']:.2%}, "
        f"chunk precision@9={broad_summary['chunk_context']['baseline_at_9']:.2%}"
    )
    print(
        "  router 3x3: "
        f"P@1={broad_summary['router_default_3x3']['at_1']['precision']:.2%}, "
        f"R@3={broad_summary['router_default_3x3']['at_3']['recall']:.2%}, "
        f"MRR@5={broad_summary['router_default_3x3']['mrr_at_5']:.2%}, "
        f"chunk precision@9={broad_summary['chunk_context']['router_default_at_9']:.2%}"
    )
    print(
        "  router 5x3: "
        f"P@1={broad_summary['router_extended_5x3']['at_1']['precision']:.2%}, "
        f"R@5={broad_summary['router_extended_5x3']['at_5']['recall']:.2%}, "
        f"MRR@5={broad_summary['router_extended_5x3']['mrr_at_5']:.2%}, "
        f"chunk precision@15={broad_summary['chunk_context']['router_extended_at_15']:.2%}"
    )
    print(
        f"Named suite: route accuracy={named_summary['exact_route_accuracy']:.2%}, "
        f"first-scheme accuracy={named_summary['first_routed_scheme_accuracy']:.2%}"
    )
    print(
        "  evidence recall baseline="
        f"{named_summary['baseline']['evidence_recall']}, "
        f"router top3={named_summary['router_default']['evidence_recall']}, "
        f"router top5={named_summary['router_extended_5']['evidence_recall']}"
    )
    print(
        f"Diagnostic 100: routes={diagnostic_summary['route_counts']}, "
        f"exact-route precision={diagnostic_summary['exact_route_precision']:.2%}, "
        f"first-scheme accuracy={diagnostic_summary['first_routed_scheme_accuracy']:.2%}"
    )
    print(
        "  baseline P@1="
        f"{diagnostic_summary['baseline_top5']['at_1']['precision']:.2%}, "
        f"router P@1={diagnostic_summary['router_top5']['at_1']['precision']:.2%}"
    )
    print(f"Latency: {latency}")
    print(f"Detailed report: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
