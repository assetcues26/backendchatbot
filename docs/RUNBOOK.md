# Runbook

## First-time setup

### 1. Supabase (database + auth)

1. Create a project at supabase.com (free tier).
2. **Settings → Database → Connection string → Session pooler.** Copy the URI
   and change the scheme to `postgresql+asyncpg://`.
3. **Settings → API.** Copy the JWT secret and the service-role key.
4. **Authentication → Providers → Google.** Enable it for staff SSO.
   Add the frontend URL to the redirect allowlist.

The `vector` extension is enabled by the first migration; nothing to click.

### 2. Backend

```bash
cp .env.example .env          # fill in DATABASE_URL, OPENAI_API_KEY, SUPABASE_*
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

### 3. Load the documents

```bash
acues-ingest sync "C:/path/to/Product Doc"
```

They land in `PENDING_REVIEW` and are readable by nobody. Approve them in the
admin panel, or use `--auto-approve` to apply the default matrix for a demo.

Ingest also reads each document and writes a short context per chunk, so that
chunks which are identical as text across different parts of the product can
still be told apart. For documents already loaded before that existed, run
`acues-ingest enrich`. See `docs/INGESTION.md`.

Set `TEST_DATABASE_URL` to a **separate scratch database** while you are here.
`tests/security` drops every table it finds and refuses to run against the
same database as `DATABASE_URL`; that is the only thing standing between a
routine test run and the corpus.

### 4. First admin user

Sign in through the frontend once so Supabase mints the account, then promote
it. Copy the user id from **Supabase → Authentication → Users**:

```sql
insert into users (id, tenant_id, email, display_name, is_active)
select '<supabase-user-uuid>', t.id, '<you@assetcues.com>', 'Your Name', true
from tenants t where t.slug = 'assetcues';

insert into user_roles (user_id, role_id)
select '<supabase-user-uuid>', r.id from roles r where r.key = 'admin';
```

Every later user is created through the admin panel.

## Deployment

### API on Render (free)

| Setting | Value |
|---|---|
| Build | `pip install -e .` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/health` |
| Env vars | `DATABASE_URL`, `OPENAI_API_KEY`, `SUPABASE_*`, `ENVIRONMENT=production`, `CORS_ORIGINS=<vercel url>` |

Run `alembic upgrade head` after any deploy that adds a migration.

### Web on Vercel (free)

Set `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`,
`NEXT_PUBLIC_SUPABASE_ANON_KEY`. Import the repo and deploy.

### Keep-alive

Set the `API_BASE_URL` repository secret. `.github/workflows/keepalive.yml`
pings `/health` every 10 minutes, which keeps Render warm (it sleeps after 15
minutes, with a 30-60s cold start) and keeps Supabase from pausing (it pauses
after 7 days idle). One cron solves both; it stays inside Render's 750
instance-hours per month.

Without it, the first question of a founder demo takes a minute.

## Routine tasks

**Rotate the OpenAI key.** Create a new key, update it in Render, redeploy,
then revoke the old one. It exists only in the environment; nothing to change
in code.

**Add a customer.** Admin panel → Tenants → create. Then create users in that
tenant with the `customer` role. Tenancy isolation is enforced by the
retrieval predicate, so nothing else is needed.

**Revoke someone's access urgently.** Admin panel → Users → set inactive. It
takes effect on their next request: their existing token becomes useless
because the `Principal` is rebuilt from the database every time. Their cached
answers die with the `acl_version` bump.

**Pull a document out of circulation.** Admin → Revoke. Faster than deleting
and reversible.

## Incidents

**A `SECURITY_ANOMALY` event appeared.** This means retrieval (G3) and
re-verification (G4) disagreed, which should be impossible. Treat as a
retrieval bug. The chunk was dropped and never reached the model. Check
`/api/admin/audit?severity=critical`, then diff `app/rag/retrieval.py` against
`git log`.

**Someone reports seeing something they should not.** In this order:
1. `/api/admin/audit` filtered to their email — the log records exactly which
   documents were used.
2. Check their roles and any `user_document_grants` rows.
3. Check the document's `status` and ACL.
4. Reproduce with `/api/compare` for that question.

**Injection findings in the audit log.** A document contains instruction-like
text. It did not change the answer (document text is fenced and labelled as
data), but somebody should look at the source file — the audit entry names it.

**Answers are wrong but not leaking.** Retrieval quality, not security. Try
`RETRIEVAL_TOP_K` up from 12, then a reranker. Do not loosen the ACL.

## Known limits

| Limit | Impact | Fix when it matters |
|---|---|---|
| Answer cache and rate limiter are in-process | Correct for one instance; two instances each keep their own | Move both to Redis |
| Render free sleeps after 15 min | 30-60s cold start | Keep-alive cron, or a paid instance |
| Supabase free: 500 MB, pauses after 7 days | Fine at 742 chunks | Keep-alive cron; upgrade at ~50k chunks |
| No maker-checker on ACL changes | An admin can grant themselves anything | Add approval workflow |
| No MFA | | Enable in Supabase Auth |
| HNSW index tuning is default | Irrelevant at this size | Revisit past ~50k chunks |

## Cost

| Item | Cost |
|---|---|
| Embedding all 21 documents | $0.0042, one time |
| One question (~6k in, ~500 out on `gpt-5.6-luna`) | ~$0.002 |
| 1,000 questions | ~$2 |
| Hosting | $0 |

If answer quality disappoints, change `OPENAI_CHAT_MODEL` to
`gpt-5.6-terra` (about 10x, still cents). No code change.
