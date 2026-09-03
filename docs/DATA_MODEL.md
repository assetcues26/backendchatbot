# Data model

```
tenants ──< users ──< user_roles >── roles
   │          │                        │
   │          └──< user_document_grants│
   │                     │             │
   └──< documents ──< document_acl >───┘
          │  │
          │  └──< document_versions   (survives deletion)
          │
          └──< chunks   (ON DELETE CASCADE — embeddings die with the document)

audit_log        append-only, survives deletion
access_requests  raised from a refusal
system_state     one row: the global acl_version counter
```

## Tables

### `tenants`
`slug`, `name`, `kind` (`INTERNAL` | `CUSTOMER`). Internal staff live in
`assetcues`; each customer organisation gets its own row. Tenancy is a hard
boundary in both directions, not a hierarchy.

### `roles`
`key`, `clearance` (1-4), `is_internal`. Seeded from `DEFAULT_ROLES` in
`app/db/seed.py`. **Clearance is a ceiling, never a grant.**

### `users`
`id` mirrors `auth.users.id` from Supabase, so the JWT `sub` maps straight
through. `is_active = false` blocks login on the next request.

### `documents`
The important columns:

| Column | Why it exists |
|---|---|
| `status` | `PROCESSING` → `PENDING_REVIEW` → `APPROVED`. Only `APPROVED` is ever retrievable (G1). |
| `sensitivity` | 1-4. Compared against the caller's clearance. |
| `content_sha256` | Re-uploading identical bytes is a no-op. |
| `source_key` | Stable identity across re-uploads; unique per tenant. |
| `declared_audience` | Scraped verbatim from the document's own "Primary audience" field. |
| `suggested_*` | The classifier's proposal, held for an admin to accept or override. Never used for access. |
| `capability` | Which part of the product this document covers. Parsed from the document's own header table, falling back to a sibling's, then the folder. What query routing filters on. |
| `module_declared`, `product_domain` | The coarser groupings the same header table declares. Recorded, not yet routed on. |
| `summary`, `key_terms`, `distinguishing_points` | What the enrichment pass understood the document to be. Feeds chunk contextualisation and the admin panel. |
| `enriched_at` | Null until a model pass has succeeded. What `acues-ingest enrich` looks for. |

### `chunks`
`text`, `heading_path`, `text_sha256`, `token_count`, plus:

- `context` — a short passage written at ingest saying where this chunk
  sits in its document. Embedded together with the text, and **never**
  displayed or cited: nothing selects this column into a query result, and
  `RetrievedChunk` has no field for it. It exists because chunks from
  different capabilities are often identical as text.

- `embedding vector(1536)` — HNSW index with `vector_cosine_ops`, matching the
  `<=>` operator in the retrieval query. An index built with a different
  operator class is silently ignored by the planner.
- `tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED` —
  GIN indexed. Always in sync because Postgres maintains it. Computed from
  `text` only, deliberately: a keyword search must match words that are
  really in the file, not words a model used to describe a neighbour.
- `tenant_id` denormalised. Safe because tenancy is immutable for a document.
  The ACL is deliberately **not** denormalised, so a permission change takes
  effect immediately with no rows to backfill.

### `document_acl`
`(document_id, role_id)`. Absence of a row means no access. Cascades on
document delete.

### `user_document_grants`
Per-user `ALLOW` / `DENY` on one document, optionally expiring. **DENY always
wins.**

### `system_state`
A single row (`CHECK (id = 1)`) holding `acl_version`. Bumped by every ACL,
role or status change; mixed into the answer-cache key so a permission change
invalidates every cached answer at once (G7).

## The retrieval predicate

The one place that decides what a caller may read
(`VISIBLE_DOCS_CTE`, `app/rag/retrieval.py`):

```sql
SELECT d.id FROM documents d
WHERE d.tenant_id = :tenant_id
  AND d.status = 'APPROVED'
  AND d.sensitivity <= :clearance
  AND (
        EXISTS (SELECT 1 FROM document_acl a
                WHERE a.document_id = d.id AND a.role_id = ANY(:role_ids))
     OR EXISTS (SELECT 1 FROM user_document_grants g
                WHERE g.document_id = d.id AND g.user_id = :user_id
                  AND g.effect = 'ALLOW'
                  AND (g.expires_at IS NULL OR g.expires_at > now()))
  )
  AND NOT EXISTS (SELECT 1 FROM user_document_grants g
                  WHERE g.document_id = d.id AND g.user_id = :user_id
                    AND g.effect = 'DENY'
                    AND (g.expires_at IS NULL OR g.expires_at > now()))
```

Every parameter comes from the verified JWT plus a database lookup. Both the
vector search and the full-text search join to this CTE, so neither can return
a chunk the other would have excluded.

## Hybrid search

Vector KNN and full-text each take the top `RETRIEVAL_CANDIDATES` (50) rows
inside the ACL restriction, then fuse with Reciprocal Rank Fusion:

```
score(chunk) = Σ 1 / (RRF_K + rank_in_that_list)
```

RRF needs no score normalisation between two incomparable scales, which is
exactly the situation with cosine distance and `ts_rank_cd`.

Hybrid matters here because the corpus is dense with exact identifiers —
`UAP-FR-045`, `AWM-TC-001`, `LM-FR-009`. Dense embeddings are weak at
exact-token lookup and someone will ask "what does UAP-FR-045 say?" on day
one. Full-text nails those; vectors handle the conceptual questions.

## A note on enum labels

SQLAlchemy stores Python enum **names**, so the Postgres labels are
`'APPROVED'`, `'ALLOW'`, `'DENY'` — not the lowercase values. The raw SQL
matches them literally. Do not convert `DocStatus` or `GrantEffect` to
`StrEnum` without re-checking the migration and the query together;
`UP042` is disabled in `pyproject.toml` for this reason.

## Scale

806 chunks across 23 documents today. Everything above is comfortable to roughly 50,000 chunks on
Supabase's free tier. Beyond that, denormalise `allowed_role_ids` onto `chunks`
with a GIN index and accept the backfill cost on ACL changes. That is a
documented lever, not a rewrite.
