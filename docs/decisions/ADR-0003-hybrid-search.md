# ADR-0003: Hybrid search, not vectors alone

**Status:** Accepted · 2026-08-30

## Context

The default RAG design is dense vector search over embedded chunks. This
corpus has a property that makes it a poor default.

## Decision

Hybrid retrieval: pgvector cosine KNN and Postgres full-text search, each
taking 50 candidates inside the ACL restriction, fused with Reciprocal Rank
Fusion (k=60), top 12 to the model.

## Why

**The corpus is dense with exact identifiers.** The AssetCues specifications
are built around stable ids: `UAP-FR-045`, `AWM-TC-001`, `LM-FR-009`,
`UAP-TC-047`. The User Access specification alone contains 85 `UAP-FR-*`
requirements. QA and Engineering will ask "what does UAP-FR-045 say?" on the
first day.

Dense embeddings are weak at exact-token lookup. `UAP-FR-045` and
`UAP-FR-046` embed almost identically, and neither is reliably nearest to a
query that names one of them. Full-text search treats the identifier as a rare
token and ranks it first, which is exactly right.

Conversely, "how does a user switch between profiles?" has no distinctive
tokens and is answered well by vectors and badly by full-text.

Both question shapes are certain to occur, so we need both retrievers.

**RRF avoids score normalisation.** Cosine distance and `ts_rank_cd` are not
comparable, and normalising them requires tuning that drifts with the corpus.
RRF uses only rank position:

```
score(chunk) = Σ 1 / (60 + rank_in_that_list)
```

No tuning, no drift, and a chunk that both retrievers like rises to the top.

**Both halves obey the same ACL.** Each side joins to `visible_docs`, so
adding full-text search does not add a second path that could bypass the
filter. This was a precondition, not an afterthought.

**It is free.** The `tsvector` column is `GENERATED ALWAYS AS ... STORED`, so
Postgres maintains it; there is no second index to populate and nothing that
can fall out of sync.

## Consequences

- One query does both searches. No application-side merging of two backends.
- The generated column and its GIN index cost some storage. Irrelevant here.
- Full-text is configured for English only. A non-English corpus would need
  the configuration changed and the column rebuilt.
- A reranker would likely improve ordering further. Deferred: at 742 chunks
  and top-12, RRF is sufficient, and a reranker adds a per-query API call.

## Verification

`tests/security/test_rbac_leakage.py` queries by requirement id
(`UAP-FR-045`), by test id (`UAP-TC-047`), and by concept ("how do I create an
Asset Category"), and asserts the right roles get the right material for all
three shapes.
