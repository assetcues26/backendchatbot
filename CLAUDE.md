# AssetCues Chatbot API

Role-aware Q&A over AssetCues product documentation. **Access is enforced in
the SQL retrieval query, not in the prompt.**

## Commands

```bash
.venv/Scripts/python.exe -m pytest tests/ -q      # tests (integration ones skip without DB)
.venv/Scripts/python.exe -m ruff check . --fix    # lint
.venv/Scripts/python.exe -m mypy app              # typecheck (strict)
.venv/Scripts/python.exe -m alembic upgrade head  # migrate
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
acues-ingest sync "<folder>" [--auto-approve|--dry-run]
acues-ingest enrich [--all] [--force]   # re-read documents already ingested
acues-ingest status
```

Before saying work is done: `ruff check .` && `mypy app` && `pytest -q` all clean.

## The seven rules

1. **Identity comes from the JWT, never from a request.** Only
   `app/core/security.py:current_principal` may produce a `Principal`. Adding
   `role`, `tenant_id`, `clearance` or `user_id`-as-caller to a request schema
   fails `tests/security/test_tampering.py`.
2. **Never post-filter retrieval.** The ACL predicate lives inside the query
   (`VISIBLE_DOCS_CTE` in `app/rag/retrieval.py`). Anything loaded into memory
   can reach a log line or a prompt.
3. **Default-deny.** New and re-uploaded documents land in `PENDING_REVIEW`
   and are readable by nobody until an admin approves. A classifier failure
   falls back to RESTRICTED/admin-only.
4. **Any ACL, role, or document-status change calls
   `audit.bump_acl_version()`.** It invalidates every cached answer. Forgetting
   it lets a revoked user keep reading.
5. **Never widen the refusal message.** One wording for every refusal.
   Confirming that restricted material exists is itself a disclosure.
6. **Generated context is embedded, never shown.** `chunks.context` is written
   by a model at ingest. `RetrievedChunk` has no field for it and the search
   query never selects it, so no generated sentence can be quoted as though a
   document said it. Adding it to either fails `tests/test_enrichment.py`.
7. **A capability scope may only narrow.** It arrives in the request body, so
   `SCOPED_DOCS_CTE` selects FROM `visible_docs` and can only return a subset.
   Never `OR` it into the access predicate.

## Layout

| Path | What lives there |
|---|---|
| `app/rag/retrieval.py` | **The security boundary.** Hybrid search + ACL predicate. Read before changing anything about access. |
| `app/rag/answer.py` | Orchestration: retrieve, generate, verify, audit. Answer cache (G7). |
| `app/rag/guardrails.py` | Citation validation (G6), injection detection. |
| `app/rag/prompts.py` | Prompt assembly, document fencing (G5), refusal text. |
| `app/rag/routing.py` | Which capability a question is about, and when to ask instead. |
| `app/rag/ingest_prompts.py` | Prompts whose output is embedded but never shown. |
| `app/ingest/enrich.py` | Document profiling and per-chunk contextualisation. |
| `app/ingest/backfill.py` | Re-enriching documents already in the database. |
| `app/core/principal.py` | The `Principal` type. Identity rules. |
| `app/core/security.py` | JWT verification, `current_principal`, `require_admin`. |
| `app/db/models.py` | Schema. Cascade rules are documented in the module docstring. |
| `app/db/seed.py` | Roles, clearances, and the starting access matrix. |
| `app/ingest/` | Parsers, chunker, classifier, lifecycle pipeline, CLI. |
| `tests/security/` | The red-team suite. |

## Conventions

- Python 3.12, `from __future__ import annotations` everywhere, full type hints.
- SQLAlchemy 2.0 async. Never a sync session.
- Raw SQL only in `app/rag/retrieval.py`, and every caller value is a bound
  parameter.
- Postgres enum labels are member **NAMES** (`'APPROVED'`, `'ALLOW'`) and the
  raw SQL matches them literally. Do not convert those classes to `StrEnum`.
- Comments explain *why*, never *what*.

## Deeper reference

Read these on demand rather than loading them up front:

- `docs/SECURITY.md` — the seven guardrails, threat model, what is not covered
- `docs/RBAC.md` — roles, clearances, the access matrix and its reasoning
- `docs/DATA_MODEL.md` — schema and the retrieval predicate in full
- `docs/INGESTION.md` — hashing, re-embedding, deletion, re-approval rules
- `docs/RUNBOOK.md` — deploy, key rotation, incident response, known limits
- `docs/decisions/` — why pgvector over a vector DB, why no LangChain, why
  hybrid search, why chunks are enriched at ingest (ADR-0004)

## Environment

Copy `.env.example` to `.env`. `OPENAI_API_KEY` is the only vendor credential;
it is read from the environment and never committed, logged, or sent to the
browser.

`TEST_DATABASE_URL` is a **separate, scratch** database for `tests/security`.
That suite drops every table it finds, and it refuses to run when
`TEST_DATABASE_URL` equals `DATABASE_URL`. Do not point it at the working
database to "just run the tests" -- that deletes the corpus, and an opt-in
flag will not save you, because the flag confirms the suite is destructive and
says nothing about which database is configured.

    TEST_DATABASE_URL=... ALLOW_DESTRUCTIVE_TESTS=1 pytest tests/security
