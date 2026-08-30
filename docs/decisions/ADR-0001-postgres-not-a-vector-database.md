# ADR-0001: Vectors live in Postgres, not a dedicated vector database

**Status:** Accepted · 2026-08-30

## Context

We need vector search over ~742 chunks with strict per-role access control,
and the documents are actively edited and deleted. The obvious options were
Qdrant Cloud (1 GB free), Pinecone, and pgvector on Supabase.

## Decision

Postgres 16 with pgvector on Supabase. ACLs, documents, chunks and embeddings
in one transactional store.

## Why

**Deletion is the requirement most likely to break.** The user was explicit
that documents get edited and deleted and must disappear from the database. In
a two-store design that is a distributed transaction, and its failure mode is
silent and dangerous: the document is gone from the admin UI while its
embeddings remain searchable. With one store it is `ON DELETE CASCADE` — a
schema guarantee, not a procedure someone has to remember.

**The ACL and the vectors must be consistent at the instant of the query.**
Access is a join, evaluated inside the same statement as the similarity
search. With a separate vector database you either replicate ACLs into its
payload (and now have a cache-invalidation problem on every permission change)
or you post-filter (which means loading unauthorised content into memory —
unacceptable).

**Hybrid search comes free.** Postgres full-text and pgvector run in one query
under one ACL filter. See ADR-0003.

**Performance is a non-issue at this scale.** 742 chunks. pgvector with HNSW
is competitive with dedicated engines well past this size, and Supabase's own
benchmarks put it ahead of Qdrant on equivalent compute at 99% recall.

**Fewer moving parts on free tiers.** One service to keep alive, one backup
story, one place credentials live. Qdrant's free tier also terminates after a
week of inactivity, which is the same operational chore as Supabase's pause
without the benefit of consolidation.

## Consequences

- ACL changes are instant and atomic. No backfill, no sync worker.
- Supabase's 500 MB free tier is the ceiling. At 742 chunks we use a fraction.
- Beyond ~50k chunks, revisit: denormalise the ACL onto chunks with a GIN
  index, tune HNSW parameters, or move to a dedicated engine with the ACL
  replicated. That is a scaling decision to make with real numbers, not now.
- The free project pauses after 7 days idle; the keep-alive cron handles it.

## Alternatives rejected

**Qdrant Cloud** — good payload filtering, but a second store to keep
consistent with the ACL, and the delete path becomes two-phase.

**Pinecone** — same consistency problem, plus a vendor with no free
persistence guarantee worth relying on for a founder demo.

**In-memory / FAISS** — no persistence, no concurrent access, no filtering.
