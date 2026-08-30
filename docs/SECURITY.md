# Security model

The question this system has to answer is not "does the chatbot work" but
**"can information cross a role boundary, and how do we know"**.

Two principles decide everything else:

1. **Default-deny.** A document nobody has classified is a document nobody can
   read.
2. **The access decision happens in the database query, not in the prompt.** A
   language model instructed to withhold information is not an access control
   system. It is a well-intentioned narrator that can be argued with.

---

## The seven guardrails

Each one is independently testable, and each catches a different failure.

### G1 — Ingest quarantine
**Where:** `app/ingest/pipeline.py`, `documents.status`

New and re-uploaded documents land in `PENDING_REVIEW`. The retrieval
predicate requires `status = 'APPROVED'`, so an unclassified document is
readable by nobody — including admins — regardless of what ACL rows exist.

*Catches:* someone drops a confidential file into the folder.
*Test:* `test_a_pending_document_is_readable_by_nobody` — runs for all seven roles.

### G2 — Identity binding
**Where:** `app/core/security.py`, `app/core/principal.py`

The caller's tenant, roles and clearance are read from the database using the
user id in a verified JWT. Nothing in a request body, query string, or any
other header can influence them.

Because the `Principal` is rebuilt on every request, a token minted before a
role was revoked cannot exercise the revoked access — the revocation applies
on the very next call.

*Catches:* `POST /api/ask {"question": "...", "role": "admin"}`.
*Tests:* `test_tampering.py` scans every request schema and every route
signature; `test_revoking_a_role_takes_effect_on_the_very_next_query`.

### G3 — SQL pre-filter *(the primary control)*
**Where:** `VISIBLE_DOCS_CTE` in `app/rag/retrieval.py`

```
can_read(user, doc) =
      doc.tenant_id   == user.tenant_id
  AND doc.status      == 'APPROVED'
  AND doc.sensitivity <= user.clearance
  AND (role grant OR explicit user allow)
  AND NOT explicit user deny
```

Both vector search and full-text search run *inside* this restriction. An
unauthorised chunk is never fetched, so it cannot reach a log line, an error
message, a trace, or a prompt. Post-filtering is not an acceptable substitute.

Note that clearance and the grant are **independent**: clearance alone never
grants access, and a grant alone never bypasses the ceiling. Two separate
conditions must both fail before a leak occurs.

*Tests:* the bulk of `test_rbac_leakage.py`.

### G4 — Post-retrieval re-verification
**Where:** `verify_chunks` in `app/rag/retrieval.py`

Every retrieved chunk is checked again, by a deliberately different query
path, before it can enter a prompt. In a correct system the two always agree.
A disagreement means a retrieval bug: the chunk is dropped, the answer still
serves, and a `SECURITY_ANOMALY` event is written at `critical`.

*Catches:* a regression in G3.
*Test:* `test_retrieval_and_reverification_never_disagree`.

### G5 — Prompt isolation
**Where:** `app/rag/prompts.py`

Document text is fenced in `<document>` tags and labelled as quoted data. The
system prompt states plainly that instruction-like sentences inside those tags
are words someone typed into a Word file, not commands. Document titles are
escaped so they cannot break out of their own attribute.

*Catches:* prompt injection planted in an ingested document.
*Tests:* `test_document_text_is_fenced_and_labelled_as_data`,
`test_document_title_cannot_break_out_of_its_attribute`.

### G6 — Citation validation
**Where:** `app/rag/guardrails.py`

Every factual claim must carry a citation key, and every key must belong to a
chunk that was in the authorised context. Otherwise the answer is retracted
and replaced with the standard refusal.

**This does not catch unauthorised content** — G3 and G4 make unauthorised
content unreachable. It catches an answer citing a source it was never given,
which is a correctness failure. That distinction is why streaming is safe:
tokens can only be drawn from context the caller was already entitled to.

*Tests:* `test_answer_citing_a_document_it_was_not_given_is_rejected`.

### G7 — ACL-fingerprinted answer cache
**Where:** `cache_key` in `app/rag/answer.py`

```
key = sha256(normalised_question | tenant | sorted(roles) | clearance | acl_version)
```

The classic bug in a system like this is a cache keyed on question text alone,
which cheerfully serves a Sales answer to a Customer. The fingerprint makes
that impossible, and the global `acl_version` counter — bumped by every ACL,
role, or status change — invalidates every entry at once.

*Tests:* `test_same_question_different_roles_gets_different_cache_keys`,
`test_acl_version_bump_invalidates_the_cache_key`.

---

## Supporting controls

- **Append-only audit log.** Every query records the asker, their roles, the
  chunks used, the documents withheld, citations, latency, and any anomaly.
- **Uniform refusal.** One wording for every refusal, with no hint that
  restricted material exists. Otherwise the corpus can be mapped by probing.
  The audit console shows the truth; the end user does not.
- **Per-user rate limiting** blunts "probe with a thousand phrasings".
- **Generic 500s.** The global handler never returns an internal error string,
  which could disclose table names, document titles, or query structure.
- **Hardened XML parsing.** `.docx`/`.xlsx` are zips of XML from an upload
  path; `defusedxml` blocks entity-expansion denial of service.
- **Upload limits.** 25 MB, extension allowlist.

## Threat model

| Threat | Control | Status |
|---|---|---|
| User asks directly for material above their role | G3 | Covered |
| User forges role/tenant in the request | G2 | Covered |
| Retrieval bug returns an out-of-scope chunk | G4 | Covered |
| Prompt injection inside a document | G5 (+G6) | Mitigated |
| Model hallucinates a source | G6 | Covered |
| Stale cached answer after a permission change | G7 | Covered |
| Confidential doc deleted but vectors remain | FK cascade | Covered |
| Unclassified doc becomes readable | G1 | Covered |
| One customer reads another's data | tenant predicate | Covered |
| Corpus mapping via refusal probing | uniform refusal | Covered |
| Compromised admin account | audit log only | **Accepted for POC** |
| Model memorises content across users | no fine-tuning | N/A |

### Known limits — deliberately accepted for this phase

- **The answer cache and rate limiter are in-process.** Correct for a
  single-instance deployment; a second instance needs Redis. See
  `docs/RUNBOOK.md`.
- **No MFA and no SSO enforcement.** Supabase Auth supports both; enabling
  them is a configuration decision, not a code change.
- **An admin can grant themselves anything.** There is no maker-checker on ACL
  changes; every change is logged, but nothing blocks it.
- **Injection detection is heuristic.** It informs the audit log; it never
  blocks an answer, because letting document text censor the assistant would
  be its own vulnerability.
- **No output-side PII scanning.**

## Verifying it yourself

```bash
pytest tests/security -q          # needs DATABASE_URL for the leakage suite
```

The suite plants distinctive strings ("Partner Entitlement Envelope",
`UAP-TC-047`, `GLOBEXONLYTOKEN`) in fixture documents and asserts, per role,
that they do or do not come back. A failure is unambiguous: the words either
appeared or they did not.
