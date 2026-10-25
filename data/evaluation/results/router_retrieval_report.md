# Retrieval Router Evaluation Report

**Date:** 2026-10-25 (re-run after scheme-card enrichment)  
**Model:** `BAAI/bge-base-en-v1.5`  
**Corpus:** 46 scheme cards / 274 chunks  
**Detailed JSON:** [router_retrieval_report.json](router_retrieval_report.json)

> **Integration note:** this evaluation calls `SchemeRetrievalRouter` directly. The Streamlit app still uses the keyword-based `SchemeRetriever` in `rag.py`, so the running chatbot will not see any of these results until the router is wired in.

## What changed in this revision

Recommendation #1 from the previous revision is now implemented: scheme cards are no longer metadata-only. Each card appends a `Summary:` line built from up to 4 clean sentences of the scheme's `General` section (falling back to `Details` when `General` holds only scraped UI noise), capped at 1400 characters. UI fragments, sub-35-character sentences and duplicates are filtered out.

| Card text | Before | After |
|---|---:|---:|
| Mean characters | 294 | 1,034 |
| Max characters | ~700 | 1,683 |
| Max tokens (`cl100k_base`) | ~150 | 420 |
| Cards with a summary | 0 / 46 | 46 / 46 |

All cards stay under BGE's 512-token window, and the metadata lines are preserved verbatim as a prefix, so `scheme_matching` (which reads `scheme_name` / `abbreviation` fields, not `card_text`) is unaffected. New `cards_sha256`: `e2785e62…d7549c74`.

The pre-fix artifacts are kept alongside the new ones for rollback: `scheme_cards.metadata_only.bak.json`, `scheme_index.metadata_only.bak.faiss`, `scheme_manifest.metadata_only.bak.json`. Delete them once the enriched index is confirmed in production use.

## Verdict

Card enrichment is a clear net win and removes the main broad-query failure mode.

- **Broad questions:** first-scheme precision 37.5% → **50.0%** (flat baseline 25.0%), MRR@5 50.4% → **64.6%** (baseline 45.2%), recall@5 44.8% → **57.3%** (baseline 54.2%). The router now beats the flat baseline on every scheme-level metric.
- **No regressions:** against the flat baseline, 3 broad cases improved, 5 unchanged, **0 worsened**. The previous revision had 3 worsened.
- **Context purity recovered:** relevant-chunk share at a 9-chunk budget 37.5% → **41.7%** (baseline 45.8%), and at 15 chunks 32.5% → **40.0%** (baseline 38.3%). The 15-chunk budget now beats flat retrieval.
- **Named-scheme routing:** unchanged, as expected — the exact route does not consult card vectors. Evidence recall@5 with 5 chunks/scheme remains **100%** vs 60% for flat retrieval.
- **Descriptive single-scheme questions (100-case diagnostic):** the router still trails flat retrieval, but the gap narrowed — P@1 74% → **76%** (baseline 79%), recall@5 87% → **92%** (baseline 97%), MRR 79.6% → **83.3%** (baseline 86.5%). Flat retrieval remains the better path for this query shape.

## 1. Broad, multi-scheme questions (8 labeled cases)

The baseline reproduces the previous `evaluate_broad_retrieval.py` result exactly. Router default is 3 schemes × 3 chunks; extended is 5 schemes × 3 chunks.

| Approach | P@1 | P@3 | R@3 | R@5 | Hit@5 | nDCG@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat baseline (top-5 unique schemes) | 25.0% | 33.3% | 36.5% | 54.2% | 87.5% | 47.1% | 45.2% |
| Router, metadata-only cards, 3×3 | 37.5% | 37.5% | 35.4% | — | 62.5% (Hit@3) | 42.2% (nDCG@3) | 47.9% |
| Router, metadata-only cards, 5×3 | 37.5% | 37.5% | 35.4% | 44.8% | 75.0% | 43.1% | 50.4% |
| **Router, enriched cards, 3×3** | **50.0%** | **41.7%** | **39.6%** | — | **87.5%** (Hit@3) | **49.0%** (nDCG@3) | **64.6%** |
| **Router, enriched cards, 5×3** | **50.0%** | **41.7%** | **39.6%** | **57.3%** | **87.5%** | **54.8%** | **64.6%** |

Context relevance (fraction of retrieved chunks whose source scheme is in the gold set):

| Context budget | Flat baseline | Router, metadata cards | Router, enriched cards |
|---|---:|---:|---:|
| 9 chunks | 45.8% | 37.5% | **41.7%** |
| 15 chunks | 38.3% | 32.5% | **40.0%** |

### Per-case outcome

`rank` = rank of the first gold scheme; `chunk` = fraction of chunks from gold schemes.

| Case | Baseline rank / chunk@9 | Metadata cards rank / chunk@9 | Enriched cards rank / chunk@9 | Enriched 5×3 rank / chunk@15 |
|---|---|---|---|---|
| `farmer-financial-assistance` | 4 / 44% | 1 / 67% | **1 / 67%** | 1 / 80% |
| `farmer-personalised` | 3 / 44% | 1 / 100% | **1 / 67%** | 1 / 80% |
| `woman-small-business` | 1 / 78% | 2 / 67% | **1 / 67%** | 1 / 40% |
| `economically-weaker-sections` | 3 / 33% | none / 0% | **3 / 33%** | 3 / 20% |
| `low-income-family-financial-assistance` | 2 / 44% | none / 0% | **2 / 33%** | 2 / 40% |
| `student-scholarships` | 5 / 33% | 3 / 33% | **3 / 33%** | 3 / 20% |
| `women-financial-support` | 1 / 89% | 1 / 33% | **1 / 33%** | 1 / 40% |
| `low-income-student` | none / 0% | none / 0% | **none / 0%** | none / 0% |

Three cases that previously regressed or failed now match or beat the baseline: `economically-weaker-sections` and `low-income-family-financial-assistance` went from *no gold scheme in the top 5* to ranks 3 and 2, and `woman-small-business` moved from rank 2 to rank 1.

## 2. Why the cards now work — and where they still don't

Gold-scheme card ranks out of 46, before vs after enrichment:

| Broad case | Metadata-only card ranks | Enriched card ranks |
|---|---|---|
| `economically-weaker-sections` | `aasra` 6, `aaas-assam` 15, `aan` 22, `aaby` 29 | **`aasra` 3**, `aaas-assam` 12, `aan` 20, `aaby` 22 |
| `low-income-family-financial-assistance` | `aan` 5, `aasra` 6, `aaas-assam` 17, `aaby` 26 | **`aan` 2, `aasra` 4, `aaby` 6**, `aaas-assam` 13 |
| `student-scholarships` | `aamgsiscs` 3, `aaby` 45 | `aamgsiscs` 3, **`aaby` 25** |
| `low-income-student` | `aavy` 19, `aaby` 46 | `aaby` 19, `aavy` 32 |
| `women-financial-support` | `admptpfdap` 1, `abhb` 6 | **`abhb` 1**, `admptpfdap` 5 |
| `woman-small-business` | `aabcs` 1, `25-ciss` 4 | **`aabcs` 1, `25-ciss` 2** |
| `farmer-personalised` | ranks 1–8 | **`abn` 1, `agr2fmsfotscst` 3, `agr50s-apt` 4, `agfiscmpwa` 5**, `agr4fmsscf` 6, `agr3fmsstf` 8 |

The previously cited pathological example is fixed in direction but not in full: `aaby` has "Economically weaker sections" verbatim in its beneficiary metadata yet ranked **29/46**; it now ranks **22/46**. Adding prose lifted it 7 places, but a 4-sentence summary cannot carry every eligible-group phrase. This is the remaining argument for lexical/hybrid card retrieval (recommendation 2).

**Still failing — `low-income-student`** (gold `aaby`, `aavy`). Both documents do contain supporting sentences (`aaby`: "scholarship to the children of the insured"; `aavy`: "free education to poor children of construction workers"), but neither lands in the first four `General`-section sentences that make the card. The query's "student from a low-income family" phrasing matches education-loan and scholarship cards (`aelss`, `afm`, `absy`, `aass`) more strongly. Fixing this needs either a longer/multi-section summary or lexical matching on benefit phrases — not more of the same summary.

**Coverage is still the binding constraint on `economically-weaker-sections`.** Gold schemes sit at card ranks 3, 12, 20 and 22; a 5-scheme cutoff captures only one. Recall on 4-gold cases is capped by `scheme_limit`, not by card quality.

Chunk selection is not the problem. When scheme discovery is right, chunks are on-topic (`farmer-financial-assistance` and `farmer-personalised` both 67% at 9 chunks, 80% at 15).

## 3. Named-scheme questions (10 labeled cases)

Unchanged by this fix — the exact route bypasses card vectors entirely.

| Metric | Flat baseline | Router, 3 chunks/scheme | Router, 5 chunks/scheme |
|---|---:|---:|---:|
| Exact route accuracy | — | 90.0% (9/10) | 90.0% (9/10) |
| First routed scheme correct | 100% | 100% | 100% |
| Gold scheme present in context | 100% | 100% | 100% |
| Evidence recall@1 | 40% | 40% | 40% |
| Evidence recall@3 | 40% | 50% | 50% |
| Evidence recall@5 | 60% | n/a (only 3 chunks) | **100%** |
| Top-1 chunk has gold evidence | 40% | 40% | 40% |
| Context chunks from correct scheme @5 | 68% | 100% (exact routes) | 100% (exact routes) |

- **Evidence recall@5: 60% → 100%** when the router returns 5 chunks from the matched scheme. Still the strongest win.
- The default of 3 chunks/scheme is too shallow; four gold evidence chunks sit at within-scheme rank 4 and one at rank 5 (`aabcs-application`, `aabcs-documents`, `aaby-age-limit`, `arogya-nidhi-benefits`, `capital-subsidy-ineligibility`).
- The one non-exact route (`merit-grant-implementer`) misses the exact title because the question omits "(Special Incentive)". It still routes broad, picks `aamgsiscs` first, and returns its gold evidence at rank 1.
- Top-1 evidence stays at 40%: the first chunk is usually the intro, not the eligibility/benefits section.

## 4. Routing diagnostic on the 100-case single-scheme suite

This suite is not the router's broad-query target; it is a stress test for route behavior.

| Metric | Flat baseline | Router, metadata cards | Router, enriched cards |
|---|---:|---:|---:|
| First-scheme accuracy | 79.0% | 74.0% | **76.0%** |
| Recall@1 | 77.5% | 72.5% | **74.5%** |
| Recall@3 | 95.0% | 86.0% | **90.5%** |
| Recall@5 | 97.0% | 87.0% | **92.0%** |
| MRR@5 | 86.5% | 79.6% | **83.3%** |

Enrichment narrowed the gap on every metric but did not close it. Against the baseline, 8 cases improved, 13 worsened, 79 unchanged.

Route distribution: 91 broad, 9 exact. Exact-route precision is 66.7% (6 correct, 3 false positives) — identical to the previous revision, confirming the card change does not affect title matching:

| Case | Why the exact route is wrong |
|---|---|
| `goa-housing-2` | Question mentions "Assam's Aponar Apon Ghar" as a contrast; router routes to that scheme, gold is the Goa scheme. |
| `agri-business-training-2` | Phrase "agricultural extension service providers" matches the `Agricultural Extension` title; gold is Agri-Clinics. |
| `labour-board-death-gujarat-2` | Generic "accidental-death benefit" fuzzy-matches the Puducherry scheme; gold is the Gujarat Labour Welfare Board scheme. |

Conclusion is unchanged: do not use the router as a drop-in replacement for descriptive single-scheme queries. Use exact routing when a scheme is confidently named, and keep flat chunk retrieval as the fallback.

## 5. Latency

Measured on 18 broad + named questions, single-query end-to-end on CPU including query encoding:

| Pipeline | Mean | Median | p95 |
|---|---:|---:|---:|
| Flat baseline | 39.9 ms | 40.0 ms | 46.8 ms |
| Router 3×3 | 88.8 ms | 58.9 ms | 200.1 ms |

Longer card text is an offline build cost only; query-time latency is effectively unchanged from the previous revision (83.3 / 56.6 / 189.9 ms). The router remains ~2.2× slower on average, acceptable for a chat app.

## 6. Recommendations, in priority order

1. ~~**Enrich scheme cards and rebuild the scheme index.**~~ **Done in this revision.** Broad P@1 37.5% → 50.0%, MRR@5 50.4% → 64.6%, zero regressions vs baseline.
2. **Add lexical/hybrid card retrieval.** BM25 or an exact-phrase boost over card text, or semantic top-20 plus a cross-encoder rerank. `aaby` still ranks 22/46 for a query whose exact phrase appears in its metadata; `low-income-student` still fails outright. This is now the top remaining lever.
3. **Increase candidate coverage.** Default to 5 schemes rather than 3 — extended already matches default on P@1/MRR while lifting recall@5 to 57.3%. For named routes return 5 chunks per scheme (evidence recall@5 60% → 100%).
4. **Add route guardrails.** Do not exact-route when the query contains multiple scheme titles, contrast/negation language ("not", "named differently from"), or only a generic title. The same 3 false positives persist.
5. **Keep a fallback path.** The 100-case diagnostic still favors flat retrieval for descriptive single-scheme questions (79% vs 76% P@1).
6. **Add chunk reranking or section-aware metadata.** Top-1 evidence remains 40% even though recall@5 reaches 100%. A cross-encoder over within-scheme candidates would put the eligibility/benefits chunk first.
7. **Expand the broad test set.** Eight hand-labeled cases drove this tuning decision; the deltas between card strategies were often 1 case wide. A 25–30 question broad suite would make the next round of tuning reliable.

## Reproduction

```bash
python -m src.embeddings.build_scheme_card_index      # rebuild cards + scheme index
python -m src.embeddings.evaluate_router_retrieval    # re-run this evaluation
```

The detailed per-case report is written to `data/evaluation/results/router_retrieval_report.json`. Existing unit tests pass (7/7).
