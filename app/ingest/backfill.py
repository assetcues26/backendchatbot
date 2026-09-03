"""Re-enrich documents that are already in the database.

Ingest enriches as it writes, but the corpus was loaded before enrichment
existed, and a prompt improvement should not require re-uploading twenty-two
files. This module re-runs the enrichment pass against documents in place.

It works from the database alone. The chunks already hold the document's text
in order, so joining them back together gives the profile pass everything it
reads -- the masthead, the purpose section, the headings. That matters because
the folder a document was ingested from is often long gone by the time someone
wants to re-enrich it, and because it keeps the backfill honest: it enriches
exactly what is indexed, not what a file on someone's laptop happens to say.

Two properties worth keeping:

* **Nothing about access changes.** No ACL, sensitivity or status is touched.
  The only effect is where a chunk sits in vector space.
* **Re-running is nearly free.** A chunk is re-embedded only when its embedded
  identity actually changes, so a second run costs one profile call per
  document and no embeddings at all.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Row, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.ingest.chunker import Chunk as TextChunk
from app.ingest.enrich import (
    chunk_digest,
    chunk_key,
    embedding_text,
    enrich_document,
)
from app.ingest.parsers import ParsedDocument
from app.ingest.pipeline import apply_profile
from app.rag import llm

# The profile pass reads the first 14k characters; rebuilding much more than
# that is wasted string work on a 30,000-word workbook.
_REBUILD_LIMIT = 60_000


@dataclass(frozen=True, slots=True)
class EnrichReport:
    document_id: uuid.UUID
    title: str
    capability: str
    chunks_total: int
    contexts_written: int
    chunks_embedded: int
    message: str = ""


async def select_documents(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID | None = None,
    only_missing: bool = True,
) -> list[Document]:
    """Documents to enrich, oldest first so a partial run is resumable."""
    query = select(Document).order_by(Document.created_at)
    if tenant_id is not None:
        query = query.where(Document.tenant_id == tenant_id)
    if only_missing:
        query = query.where(Document.enriched_at.is_(None))
    return list((await session.execute(query)).scalars().all())


async def enrich_stored_document(
    session: AsyncSession,
    document: Document,
    *,
    siblings: dict[str, str],
    force: bool = False,
) -> EnrichReport:
    """Run the enrichment pass over one already-ingested document.

    `force` rewrites contexts that are already present. Without it, a chunk
    that already has a context keeps it, which is what makes a resumed or
    repeated run cheap.
    """
    rows = (
        await session.execute(
            select(
                Chunk.id,
                Chunk.ordinal,
                Chunk.heading_path,
                Chunk.text,
                Chunk.context,
                Chunk.text_sha256,
                Chunk.token_count,
                Chunk.embedding,
            )
            .where(Chunk.document_id == document.id)
            .order_by(Chunk.ordinal)
        )
    ).all()

    if not rows:
        return EnrichReport(
            document_id=document.id,
            title=document.title,
            capability=document.capability,
            chunks_total=0,
            contexts_written=0,
            chunks_embedded=0,
            message="no chunks",
        )

    chunks = [
        TextChunk(
            ordinal=row.ordinal,
            heading_path=row.heading_path,
            text=row.text,
            token_count=row.token_count,
        )
        for row in rows
    ]
    parsed = ParsedDocument(
        title=document.title,
        markdown=_rebuild_markdown(rows),
        declared_audience=list(document.declared_audience or []),
        doc_type=document.doc_type,
    )

    reuse = (
        {}
        if force
        else {
            chunk_key(row.heading_path, row.text): row.context
            for row in rows
            if row.context
        }
    )

    profile, contexts = await enrich_document(
        parsed,
        filename=document.source_filename,
        module=document.module,
        chunks=chunks,
        siblings=siblings,
        reuse_contexts=reuse,
        use_llm=True,
    )

    digests = [
        chunk_digest(context, row.heading_path, row.text)
        for context, row in zip(contexts, rows, strict=True)
    ]
    # Re-embed a chunk when what it embeds has changed, and repair any chunk
    # whose embedding is missing while we are here.
    stale = [
        index
        for index, (row, digest) in enumerate(zip(rows, digests, strict=True))
        if digest != row.text_sha256 or row.embedding is None
    ]

    vectors: dict[int, list[float]] = {}
    if stale:
        computed = await llm.embed_texts(
            [
                embedding_text(contexts[i], rows[i].heading_path, rows[i].text)
                for i in stale
            ]
        )
        vectors = dict(zip(stale, computed, strict=True))

    for index, row in enumerate(rows):
        if index not in vectors and digests[index] == row.text_sha256:
            continue
        values: dict[str, Any] = {
            "context": contexts[index],
            "text_sha256": digests[index],
        }
        if index in vectors:
            values["embedding"] = vectors[index]
        await session.execute(
            update(Chunk).where(Chunk.id == row.id).values(**values)
        )

    apply_profile(document, profile)
    await session.flush()

    return EnrichReport(
        document_id=document.id,
        title=document.title,
        capability=document.capability,
        chunks_total=len(rows),
        contexts_written=sum(1 for c in contexts if c),
        chunks_embedded=len(stale),
        message="" if profile.summary else "profile pass returned nothing",
    )


def _rebuild_markdown(rows: Sequence[Row[Any]]) -> str:
    """Reassemble a readable document from its chunks.

    The chunker consumed heading lines into `heading_path`, so they are put
    back as the path changes. Without them the profile pass sees a wall of
    tables and cannot tell which section it is reading.
    """
    parts: list[str] = []
    total = 0
    previous = ""

    for row in rows:
        if row.heading_path and row.heading_path != previous:
            heading = "## " + row.heading_path
            parts.append(heading)
            total += len(heading)
            previous = row.heading_path
        parts.append(row.text)
        total += len(row.text)
        if total >= _REBUILD_LIMIT:
            break

    return "\n\n".join(parts)
