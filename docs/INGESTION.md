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
  read the document once  ->  capability, summary, key terms,
      |                       what distinguishes it from siblings
      v
  write a situating passage per chunk  ->  chunks.context
      |                      \
      |                       `--> unchanged text: carry its context
      |                            forward, no model call
      v
  sha256(context + heading_path + text) per chunk
      |                      \
      |                       `--> unchanged chunk: reuse its embedding
      v
  embed context + heading_path + text  (never shown, never cited)
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

## Contextual enrichment

Every AssetCues document follows one template, so chunks from different parts
of the product can be identical as text -- two mastheads reading
`ASSETCUES PRODUCT DOCUMENT` embedded at cosine 1.000, requirement tables
differing only in the identifier prefix. No retriever separates those. What
distinguishes them is the document they sit in, and the chunk carried none of
it.

So each chunk gets a short passage saying where it sits and what it covers,
stored in `chunks.context`:

```
context + heading_path + text   ->  embedded
                         text   ->  shown, quoted, cited, full-text indexed
```

**That split is the safety story.** `RetrievedChunk` has no context field and
the search query never selects the column, so no generated sentence can reach
a reader as though the document said it. The worst a hallucinated passage can
do is rank a chunk oddly.

Two more properties, both deliberate:

- **Enrichment never blocks ingest.** Every failure path leaves `context`
  empty and the chunk still embeds from its own text: worse ranking, never a
  lost document. `ENRICHMENT_ENABLED=false` turns it off entirely, and the
  capability is still parsed from the document header because that costs
  nothing.
- **Re-uploading is cheap.** A chunk whose *text* is unchanged keeps the
  context it already has, so editing one section of a fifty-page BRD rewrites
  three contexts rather than three hundred. Hence two hashes: `chunk_key`
  covers the source text and carries a context forward, `chunk_digest` covers
  the context too and invalidates the cached embedding.

### Backfilling

`acues-ingest enrich` re-runs the pass over documents already ingested,
working from the database alone -- the chunks hold the text, so no source
folder is needed. That is what makes a prompt improvement deployable without
asking anyone to find and re-upload twenty-three files.

```bash
acues-ingest enrich                # documents never enriched
acues-ingest enrich --all          # re-profile everything
acues-ingest enrich --all --force  # rewrite existing contexts too
acues-ingest enrich --dry-run      # list what would happen
```

Re-running is nearly free: only chunks whose embedded identity actually
changed are re-embedded. Every run bumps `acl_version` at the end, because
cached answers were built from the old vectors.

The 23-document corpus takes about 20 minutes and $1-2.

### What an upload costs now

Enrichment happens inside the upload request, so an admin upload is slower
than it used to be. Measured on the real corpus:

| document | chunks | enrichment |
|---|---|---|
| OSM Lean Product & Functional Specification | 14 | 19s |
| Field & Screen Configuration - Test Cases | 115 | 42s |

Add parsing and classification on top. It is synchronous deliberately -- see
the note on `upload_document` in `app/api/routes/admin.py` -- and the trade
flips if documents get much larger than these.

## What a document declares about itself

Each file carries its own taxonomy in its header table:

```
| Capability     | Approval Workflow Management          |
| Module         | Platform Administration & Controls    |
| Product domain | Platform Foundation & Administration  |
```

`extract_header_fields()` reads it deterministically -- free, and correct by
construction. Capability resolution falls back in order of trust:

1. what the document declares;
2. what a sibling in the same folder declares (a QA workbook beside a
   specification is the same capability whether or not it says so);
3. the folder name.

Only if all three come up empty is a model asked. Query routing filters on
`capability`, so every document has one.

## The four hashes

| Hash | Scope | Buys you |
|---|---|---|
| `documents.content_sha256` | whole file | Re-uploading an unchanged file does nothing at all: no parse, no classify, no embed, no enrichment. |
| `chunks.text_sha256` | `context + heading_path + text` | Editing one section of a 50-page BRD re-embeds 3 chunks, not 300. |
| `enrich.chunk_key` | `heading_path + text` | Carries an existing context across an edit, so unchanged sections cost no model call. |
| `source_key` | path relative to the sync root | Stable document identity, so an edit updates rather than duplicating. |

Both chunk hashes cover the heading path deliberately: moving a paragraph into
a different section changes its meaning, so it must not reuse the old vector.

The two differ in exactly one thing, and it matters. `text_sha256` covers the
context, so **rewriting a context invalidates the embedding** -- correct,
because the context is part of what was embedded. `chunk_key` does not, so
**unchanged text keeps its context** -- also correct, and it is what stops a
one-line edit from re-buying every passage in the document.

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
acues-ingest enrich                            # re-read what is already loaded
acues-ingest enrich --all --force              # rewrite every context
acues-ingest status                            # counts, and an orphan check
```

`--auto-approve` skips the human. It exists to bootstrap the demo from a
known-good folder, it is off by default, and every approval it makes is still
written to the audit log, so there is a record that a machine decided.

Running `sync` twice makes **zero** embedding calls the second time.
