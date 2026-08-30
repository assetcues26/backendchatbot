# AssetCues RBAC Document Assistant — API

Role-aware question answering over AssetCues product documentation.

**Access is enforced in the SQL retrieval query, not in the prompt.** A
language model instructed to withhold information is not an access control
system.

## What it does

- Ingests `.docx`, `.xlsx`, `.pdf`, `.md` and keeps the database in step as
  documents are edited, replaced and deleted.
- Classifies each document (LLM proposal, human approval) into a sensitivity
  level and a set of readers.
- Answers questions with citations, filtered to what the caller may read.
- Records every query, every access decision and every anomaly.

## Quick start

```bash
cp .env.example .env                              # fill in DATABASE_URL, OPENAI_API_KEY, SUPABASE_*
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m alembic upgrade head
acues-ingest sync "C:/path/to/Product Doc"        # lands in the review queue
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Full setup, including Supabase and the first admin user: `docs/RUNBOOK.md`.

## Stack

| | |
|---|---|
| API | FastAPI, Python 3.12, SQLAlchemy 2.0 async |
| Store | Postgres 16 + pgvector (Supabase) — ACLs and vectors in one transaction |
| Retrieval | Hybrid: pgvector HNSW + Postgres full-text, fused with RRF |
| Models | OpenAI `gpt-5.6-luna` (chat), `text-embedding-3-small` (embeddings) |
| Auth | Supabase Auth (Google SSO + email), JWT verified server-side |

One vendor, one API key. Embedding all 21 documents costs about **$0.004**; a
question costs about **$0.002**.

## The seven guardrails

| | | |
|---|---|---|
| G1 | Ingest quarantine | New documents are readable by nobody until approved |
| G2 | Identity binding | Roles come from the JWT, never from a request |
| G3 | SQL pre-filter | Unauthorised chunks are never fetched |
| G4 | Post-retrieval re-check | Independent second verification before the prompt |
| G5 | Prompt isolation | Document text is fenced and labelled as untrusted data |
| G6 | Citation validation | An answer citing an unsupplied source is retracted |
| G7 | ACL-fingerprinted cache | A permission change invalidates every cached answer |

Detail and threat model: `docs/SECURITY.md`.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Unit tests run anywhere. The red-team suite in `tests/security/` needs a
Postgres with pgvector (`DATABASE_URL`) and skips without one; CI provides
`pgvector/pgvector:pg16` as a service container. It uses deterministic fake
embeddings, so it costs nothing to run.

It asserts, per role, that specific phrases never come back — "Partner
Entitlement Envelope" for Customer, `UAP-TC-047` for Sales, another tenant's
data for anyone.

## Documentation

| | |
|---|---|
| `CLAUDE.md` | Conventions and the rules that must not be broken |
| `docs/SECURITY.md` | Guardrails, threat model, accepted limits |
| `docs/RBAC.md` | Roles, clearances, the access matrix and its reasoning |
| `docs/DATA_MODEL.md` | Schema and the retrieval predicate in full |
| `docs/INGESTION.md` | Hashing, re-embedding, deletion, re-approval |
| `docs/API.md` | Endpoints |
| `docs/RUNBOOK.md` | Deploy, rotate, respond, known limits |
| `docs/decisions/` | Why pgvector, why no framework, why hybrid search |
