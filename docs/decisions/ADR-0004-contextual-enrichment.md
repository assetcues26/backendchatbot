# ADR-0004: Enrich chunks at ingest, and ask when a question straddles areas

**Status:** accepted
**Date:** 2026-09-03

## The problem, measured

The assistant answered specific questions well and went wrong on general ones.
The cause was not the model. It was that many chunks are, as text, identical
across different parts of the product.

Every AssetCues document is written to one template. Sampling 250 chunks from
the live index and finding each one's nearest neighbour **in a different
module** gave:

| similarity | what the pair was |
|---|---|
| 1.000 | `ASSETCUES PRODUCT DOCUMENT` — byte-identical mastheads |
| 0.913 | `\| Requirement \| Mapped tests \| Authoritative definition \|` — one row `UAP-FR…`, the other `OSM-FR…` |
| 0.88 | "Retain screenshots or response evidence for every validation…" in two packs |
| 0.85 | `\| Decision record field \| Value to record \|` vs `\| Governance field \| Value to record \|` |

9.6% of sampled chunks had a cross-module twin at 0.80 or above; 2.8% at 0.90
or above.

No retriever can separate those, because as text they are not different. What
distinguishes them is the document they sit in, and the chunk carried none of
that. Turning the embedding model up, or the reranker on, cannot fix a pair
that is the same string.

## Decision

Two changes, plus one guard rail.

### 1. Each chunk carries a passage saying where it lives

At ingest, a model reads the document once, then writes a short situating
passage per chunk. The passage is stored in `chunks.context` and embedded
**with** the text:

```
context + heading_path + text   →   embedded
                         text   →   shown, quoted, cited
```

### 2. Routing on capability, and a question when it is ambiguous

Each document declares a **Capability** in its own header table. That is
parsed deterministically, and retrieval can be scoped to it. After retrieval,
capabilities are scored by their share of the fused RRF score:

- the question names a capability → scope to it;
- one capability holds ≥62% of the score → answer from it;
- otherwise → **return a clarifying question** with the candidates as choices.

The ambiguity test is arithmetic, not another model call. It costs nothing at
query time, it is explainable in the audit log, and it is testable without an
LLM.

### 3. The guard rail: generated text is embedded, never shown

`RetrievedChunk` — the type that reaches the prompt, the citations and the API
response — **has no context field**, and the search query never selects
`chunks.context`. The full-text index is likewise computed from `text` alone,
so a keyword search still matches words that are really in the file.

This is what makes it safe to let a model write into the retrieval path at
all. The worst a hallucinated sentence can do is rank a chunk oddly. It can
never become a claim attributed to a document, because there is no path from
that column to a reader.

## Results

Same 250 chunks, same measurement, before and after:

| | before | after |
|---|---|---|
| max similarity | **1.000** | **0.817** |
| p95 | 0.837 | 0.781 |
| median | 0.707 | 0.672 |
| pairs ≥ 0.90 | 7 (2.8%) | **0** |
| pairs ≥ 0.85 | 9 (3.6%) | **0** |
| pairs ≥ 0.80 | 24 (9.6%) | 7 (2.8%) |

Routing, against the fixed question set:

| question | outcome |
|---|---|
| "What are the open items?" | clarifies — 44 / 23 / 17 / 16 across four capabilities |
| "What evidence must be retained for validation?" | clarifies — 34 / 25 |
| "What are the six Access Categories?" | answers — one capability held everything |
| "What does UAP-FR-045 say?" | answers — 92% User Access |
| "In Approval Workflow, what are the open items?" | answers, scoped — named in the question |
| "Who can approve a request?" | clarifies — 49 / 32 |
| "What are the test cases?" | **answers** from Fields and Screens |

The last one is a known limit rather than a success: all twelve retrieved
chunks came from one capability, so by the evidence it was not ambiguous. A
diversity-aware retrieval pass would be the fix, and is not built.

## Alternatives rejected

**A better embedding model.** Already on `text-embedding-3-large`. It cannot
separate two identical strings; nothing can.

**A cross-encoder reranker.** Reranks the same text and inherits the same
blindness, plus per-query latency and cost. Enrichment moves the work to
ingest, where it is paid once.

**Metadata filtering alone**, without enrichment. Scoping to a capability
helps only once you know which capability — and the whole difficulty is that
the retrieval could not tell. Enrichment is what makes the score-share signal
meaningful in the first place.

**Guessing instead of asking.** The cheapest option, and the one being
removed: a confident, correctly cited answer about the wrong capability is
worse than a question, because nothing in it looks wrong.

**An LLM call to judge ambiguity.** Adds latency and cost to every query, and
produces a decision that cannot be explained or tested. The share-of-score
threshold is a number in the audit log.

## Costs

- Backfilling 23 documents / 806 chunks: about 20 minutes and roughly $1–2.
- Re-uploading one edited document: a few cents. Chunks whose text did not
  change keep their context and their embedding.
- Query time: unchanged. Routing adds no model call.

## What this does not do

- It does not touch access control. `VISIBLE_DOCS_CTE` is unchanged, and the
  capability scope is a separate `scoped_docs` CTE that selects **from**
  `visible_docs` — so it can only ever return a subset. Choosing a capability
  you cannot read returns nothing.
- It does not improve retrieval where a document genuinely lacks the
  information. Enrichment may only restate what the corpus already says.
- It does not diversify retrieval, which is why "What are the test cases?"
  still answers from whichever capability filled the result set.
