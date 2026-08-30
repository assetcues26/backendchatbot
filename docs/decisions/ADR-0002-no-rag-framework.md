# ADR-0002: No LangChain or LlamaIndex

**Status:** Accepted · 2026-08-30

## Context

LangChain and LlamaIndex are the default choice for RAG. Both would have given
us document loaders, chunkers, retrievers and a chain abstraction on day one.

## Decision

Write the pipeline directly against the OpenAI SDK and SQLAlchemy. No RAG
framework.

## Why

**The retrieval call is the security boundary, and it must be readable.** In
this system the ACL predicate lives inside the retrieval query. A reviewer,
an auditor, or a founder asking "how do you know Sales cannot see this" needs
to be able to read that query. Framework retrievers bury it under a
`VectorStoreRetriever` configured three layers up, and a security control
nobody can find is a security control nobody can verify.

**Post-filtering is the framework default, and it is wrong here.** The common
framework pattern is retrieve-then-filter on metadata. That loads unauthorised
content into the process, where it can reach a log line, an exception message,
or a trace. We need pre-filtering in SQL, which means writing the SQL.

**It is not much code.** The entire pipeline — parsers, chunker, hybrid
retrieval, guardrails, orchestration — is about 300 lines of application
logic. The framework would not have saved meaningfully more than it cost in
indirection.

**Dependency surface.** LangChain moves fast and breaks interfaces. On a
system whose core promise is "this cannot leak", an unreviewed transitive
upgrade changing retrieval behaviour is a real risk.

## Consequences

- We own the chunker, the RRF fusion, and the streaming plumbing. All are
  tested directly.
- No community integrations for free. If we later want Drive or Confluence
  connectors we write them, or adopt a framework only at the ingestion edge
  where it cannot touch the ACL path.
- Someone new to the codebase reads `app/rag/retrieval.py` and understands the
  access model in one sitting. That was the goal.

## Alternatives rejected

**LangChain** — see above.

**LlamaIndex** — better retrieval abstractions, same objection: the access
decision ends up inside someone else's query builder.

**A framework for ingestion only** — genuinely tempting for parsers
(`unstructured`, `docling`). Rejected for now because our own parser is 250
lines, already validated against all 21 real documents, and produces markdown
tables that keep requirement ids attached to their rules — which the generic
parsers did not do as well on this corpus.
