# Ingestion and the change pipeline

The documents are still being edited, renamed and deleted. The database must
not drift from them, and a deleted document must actually become unanswerable.

## Flow

```
upload / CLI sync
      |
      v
  sha256(bytes) --- unchanged? --> stop, zero cost
      |
      v
  parse (.docx headings + tables, .xlsx sheets, .pdf, .md)
      |
      v
  structure-aware chunk (~450 tokens, split on headings,
      never mid-table-row, carry heading_path)
      |
      v
  sha256(heading_path + text) per chunk
      |                      \
      |                       `--> unchanged chunk: reuse its embedding
      v
  embed only new/changed chunks
      |
      v
  extract "Primary audience"  ->  LLM classifier (strict JSON schema)
      |                           proposes sensitivity + roles + rationale
      v
  status = PENDING_REVIEW          <-- readable by NOBODY
      |
      v
  admin approves  ->  status = APPROVED, ACL written, acl_version bumped
```

## The three hashes

| Hash | Scope | Buys you |
|---|---|---|
| `documents.content_sha256` | whole file | Re-uploading an unchanged file does nothing at all: no parse, no classify, no embed. |
| `chunks.text_sha256` | `heading_path + text` | Editing one section of a 50-page BRD re-embeds 3 chunks, not 300. |
| `source_key` | path relative to the sync root | Stable document identity, so an edit updates rather than duplicating. |

The chunk hash covers the heading path deliberately: moving a paragraph into a
different section changes its meaning, so it must not reuse the old vector.

## Deletion

`chunks.document_id` is `ON DELETE CASCADE`. Deleting a document removes its
chunks and their embeddings in the same transaction.

This is the strongest argument for keeping vectors in Postgres rather than a
separate vector database. With two stores, "delete the document" is a
distributed transaction, and the failure mode (document gone from the UI,
embeddings still searchable) is exactly the leak nobody notices.

What survives a delete, on purpose:

- `document_versions` — what existed, at which hash, and who removed it
- `audit_log` — the delete event with actor and chunk count

`acues-ingest sync` also reconciles deletions: any document whose `source_key`
is no longer present on disk is removed. Use `--no-delete` to disable.

## Replacing a document

The rule for edits:

> A replaced document keeps its existing access **unless the classifier now
> proposes a higher sensitivity than the document currently has**, in which
> case it returns to `PENDING_REVIEW` and its ACL is cleared.

This stops someone pasting confidential content into a customer-visible guide
and having it silently inherit customer access. It also avoids forcing an
administrator to re-approve every typo fix, which is what makes people stop
reading approval prompts.

Every replacement writes a `document_versions` row and a `DOC_UPDATED` audit
event. Escalations are logged at `warning`.

## Chunking rules

Defined in `app/ingest/chunker.py`. Three rules, each because breaking it
produced measurably worse answers on this corpus:

1. **Split on headings first.** A chunk spanning "4.3 Permission Groups" and
   "4.4 Profiles" retrieves for both and answers neither.
2. **Never split a table row.** The AssetCues specs put their substance in
   tables: requirement id, rule, acceptance criterion. Cutting mid-row
   separates `UAP-FR-045` from the rule it names, which is the exact
   association a reader is asking about. Oversized tables split on row
   boundaries with the header repeated, so each piece stays self-describing.
3. **Carry the heading path.** It prefixes the embedded text, so section names
   are searchable, and it makes a citation read "4.3 Permission Groups"
   rather than "chunk 47".

Measured on the real corpus: **21 documents produce 742 chunks and 207,617
tokens, with zero oversized and zero empty chunks**, and all 85 `UAP-FR-*`
identifiers retain their rule text. Embedding cost: **$0.0042**.

## The classifier

`app/ingest/classifier.py` proposes; it never decides. Output lands in
`documents.suggested_*` and the document waits in `PENDING_REVIEW`.

- Constrained by a strict JSON schema, so a malformed response cannot leave a
  document stuck mid-pipeline.
- Any failure returns `FAILSAFE`: sensitivity 4, admin only. **A classifier
  outage must never open a document up.**
- Proposed roles whose clearance is below the proposed sensitivity are
  dropped, so an administrator is never shown a grant that would do nothing.

## CLI

```bash
acues-ingest sync "C:/path/to/Product Doc"     # into the review queue
acues-ingest sync ./docs --auto-approve        # apply the default matrix (demo)
acues-ingest sync ./docs --dry-run             # list actions only
acues-ingest sync ./docs --no-classifier       # no model calls at all
acues-ingest status                            # counts, and an orphan check
```

`--auto-approve` skips the human. It exists to bootstrap the demo from a
known-good folder, it is off by default, and every approval it makes is still
written to the audit log, so there is a record that a machine decided.

Running `sync` twice makes **zero** embedding calls the second time.
