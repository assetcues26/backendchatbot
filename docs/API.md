# API

Base path `/api`. Every endpoint except `GET /health` requires
`Authorization: Bearer <supabase-jwt>`.

**No endpoint accepts an identity claim.** Roles, tenant and clearance are read
from the database using the user id in the verified token.

## Chat

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/me` | Caller's roles, clearance, tenant. |
| `POST` | `/api/ask` | `{question, history, capability}` returns answer plus citations. Rate limited. |
| `POST` | `/api/ask/stream` | Server-sent events. |
| `POST` | `/api/access-request` | Raised from a refusal. Returns nothing about what was withheld. |

### SSE events on `/api/ask/stream`

| Event | Payload |
|---|---|
| `sources` | Chunks in play, sent before generation so the UI can show provenance early. |
| `delta` | One token. |
| `done` | `{turn_id, citations, follow_ups, refused, cached, clarify, capability, latency_ms}` |
| `retracted` | `{answer, reason}` — citation validation failed; replace what was streamed. |
| `error` | `{message, detail}` |

### Clarifying questions

When a question fits several parts of the product equally well, the answer is
a **question** and `clarify` lists the areas to choose from. Re-ask the same
question with one of them as `capability`, or `"*"` to mean "all of them" --
which is not the same as omitting it, because an omitted capability is an
unrouted question that may clarify again.

```
POST /api/ask   {"question": "What are the open items?"}
  -> {"answer": "That question fits more than one part of the product.
                 Did you mean Fields and Screens, Approval Workflow
                 Management, or Reporting Period Management?",
      "clarify": ["Fields and Screens", "Approval Workflow Management",
                  "Reporting Period Management"],
      "citations": []}

POST /api/ask   {"question": "What are the open items?",
                 "capability": "Approval Workflow Management"}
  -> {"answer": "...", "capability": "Approval Workflow Management",
      "clarify": [], "citations": [...]}
```

`capability` is a **search scope, not a permission**. It can only narrow what
the access predicate already allowed, so naming an area you cannot read
returns nothing rather than granting it.

## Admin (requires the `admin` role)

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/admin/documents?status=` | List, with ACL and chunk counts. |
| `POST` | `/api/admin/documents` | Multipart upload. Lands in `PENDING_REVIEW`. |
| `POST` | `/api/admin/documents/{id}/approve` | `{role_keys, sensitivity}`. The only route to `APPROVED`. |
| `POST` | `/api/admin/documents/{id}/revoke` | Back to the queue, ACL cleared. |
| `DELETE` | `/api/admin/documents/{id}` | Cascades to chunks and embeddings. |
| `GET` `POST` | `/api/admin/users` | List, create. |
| `PUT` | `/api/admin/users/{id}/roles` | Bumps `acl_version`. |
| `PUT` | `/api/admin/users/{id}/active` | Disable or re-enable. |
| `POST` | `/api/admin/users/{id}/grants` | Per-user allow/deny on one document. |
| `DELETE` | `/api/admin/users/{id}/grants/{doc_id}` | Clear overrides. |
| `GET` | `/api/admin/roles`, `/api/admin/tenants` | Reference data. |
| `POST` | `/api/admin/tenants` | Create a customer tenant. |
| `GET` | `/api/admin/audit`, `/api/admin/audit/summary` | The audit console. |
| `GET` | `/api/admin/access-requests` | Open requests. |
| `POST` | `/api/compare` | One question, every role, side by side. |
| `POST` | `/api/cache/clear` | Drop the answer cache. |

`/api/compare` is admin-gated because the aggregate view reveals which
documents exist and who can see them.

## Errors

| Code | Meaning |
|---|---|
| 401 | Missing, malformed or expired token. |
| 403 | Authenticated but not permitted, or the account is disabled. |
| 413 / 415 | Upload too large / unsupported type. |
| 422 | Validation failed, or a grant exceeds a role's clearance. |
| 429 | Rate limited. `Retry-After` is set. |
| 500 | Generic. Detail goes to the logs, never to the caller. |

The TypeScript client is generated from `/openapi.json` in CI, so the two
repositories cannot silently drift.
