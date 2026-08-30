"""Ingestion lifecycle: create, replace, delete.

The requirement driving this module is that the documents are still being
edited, renamed and deleted, and the database must not drift from them.

How that is achieved:

* **Content hashing at two levels.** A document hash short-circuits a re-upload
  of unchanged bytes. A per-chunk hash means editing one section of a fifty-page
  BRD re-embeds three chunks instead of three hundred.

* **Deletion is a foreign key, not a procedure.** ``chunks.document_id`` is
  ON DELETE CASCADE, so removing a document removes its vectors in the same
  transaction. There is no second datastore that can be left holding stale
  embeddings -- which is the single most likely way a system like this leaks
  after a document is "deleted".

* **Re-ingestion is default-deny by default.** A replaced document returns to
  PENDING_REVIEW *if its classification got stricter*; otherwise it keeps its
  existing ACL. That stops someone widening a document's contents past its
  granted audience, without forcing an administrator to re-approve every typo
  fix.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import audit
from app.core.principal import Principal
from app.db.models import (
    Chunk,
    DocStatus,
    Document,
    DocumentACL,
    DocumentVersion,
    Role,
    Sensitivity,
)
from app.ingest.chunker import Chunk as TextChunk
from app.ingest.chunker import chunk_markdown
from app.ingest.classifier import Classification, classify
from app.ingest.parsers import ParsedDocument, parse_document
from app.rag import llm


class IngestError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: uuid.UUID
    title: str
    status: DocStatus
    action: str  # created | updated | unchanged | failed
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    requires_reapproval: bool
    message: str = ""

    @property
    def embedding_saved(self) -> int:
        return self.chunks_reused


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


async def ingest_bytes(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_key: str,
    filename: str,
    data: bytes,
    module: str = "",
    principal: Principal | None = None,
    run_classifier: bool = True,
    suggested: Classification | None = None,
) -> IngestResult:
    """Ingest or re-ingest one document. Idempotent on content.

    `source_key` is the stable identity across re-uploads: the same key means
    the same document lineage, so uploading an edited file updates it rather
    than creating a duplicate.
    """
    content_hash = sha256_bytes(data)

    existing = (
        await session.execute(
            select(Document).where(
                Document.tenant_id == tenant_id, Document.source_key == source_key
            )
        )
    ).scalar_one_or_none()

    # Nothing changed -- do no work and spend nothing on embeddings.
    if existing is not None and existing.content_sha256 == content_hash:
        return IngestResult(
            document_id=existing.id,
            title=existing.title,
            status=existing.status,
            action="unchanged",
            chunks_total=await _count_chunks(session, existing.id),
            chunks_embedded=0,
            chunks_reused=0,
            requires_reapproval=False,
            message="Content hash unchanged; nothing to do.",
        )

    tmp = Path(filename)
    parsed = await asyncio.to_thread(_parse_bytes, data, tmp.suffix, filename)
    chunks = chunk_markdown(parsed.markdown)
    if not chunks:
        raise IngestError(f"{filename}: parsed to zero chunks")

    if suggested is None and run_classifier:
        suggested = await classify(parsed, filename, module)

    if existing is None:
        return await _create(
            session,
            tenant_id=tenant_id,
            source_key=source_key,
            filename=filename,
            module=module,
            parsed=parsed,
            chunks=chunks,
            content_hash=content_hash,
            byte_size=len(data),
            suggested=suggested,
            principal=principal,
        )

    return await _replace(
        session,
        document=existing,
        filename=filename,
        parsed=parsed,
        chunks=chunks,
        content_hash=content_hash,
        byte_size=len(data),
        suggested=suggested,
        principal=principal,
    )


async def ingest_path(
    session: AsyncSession,
    path: Path,
    *,
    tenant_id: uuid.UUID,
    source_key: str | None = None,
    module: str | None = None,
    principal: Principal | None = None,
    run_classifier: bool = True,
    suggested: Classification | None = None,
) -> IngestResult:
    # Reading and parsing a 30k-word workbook takes long enough to stall the
    # event loop; keep it off the request thread.
    data = await asyncio.to_thread(path.read_bytes)
    return await ingest_bytes(
        session,
        tenant_id=tenant_id,
        source_key=source_key or path.name,
        filename=path.name,
        data=data,
        module=module if module is not None else path.parent.name,
        principal=principal,
        run_classifier=run_classifier,
        suggested=suggested,
    )


def _parse_bytes(data: bytes, suffix: str, filename: str) -> ParsedDocument:
    """Parsers work on paths; stage the upload in a temp file.

    Synchronous and CPU-bound: callers run it via ``asyncio.to_thread``.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        staged = Path(handle.name)
    try:
        parsed = parse_document(staged)
    finally:
        staged.unlink(missing_ok=True)

    # The temp file's random name is useless for titling; recover from the
    # real filename when the document had no usable heading of its own.
    if not parsed.title or staged.stem in parsed.title:
        from app.ingest.parsers import _clean_filename_title

        parsed.title = _clean_filename_title(Path(filename))
    if not parsed.doc_type or parsed.doc_type == "Document":
        from app.ingest.parsers import _derive_doc_type

        parsed.doc_type = _derive_doc_type(filename, parsed.markdown)
    return parsed


async def _create(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    source_key: str,
    filename: str,
    module: str,
    parsed: ParsedDocument,
    chunks: list[TextChunk],
    content_hash: str,
    byte_size: int,
    suggested: Classification | None,
    principal: Principal | None,
) -> IngestResult:
    document = Document(
        tenant_id=tenant_id,
        title=parsed.title,
        source_filename=filename,
        source_key=source_key,
        module=module,
        doc_type=suggested.doc_type if suggested else parsed.doc_type,
        content_sha256=content_hash,
        version=1,
        byte_size=byte_size,
        # Until an administrator approves, the document is readable by nobody.
        sensitivity=int(Sensitivity.RESTRICTED),
        status=DocStatus.PROCESSING,
        declared_audience=parsed.declared_audience,
        suggested_role_keys=suggested.role_keys if suggested else [],
        suggested_sensitivity=suggested.sensitivity if suggested else None,
        classifier_rationale=suggested.rationale if suggested else "",
        uploaded_by=principal.user_id if principal else None,
    )
    session.add(document)
    await session.flush()

    embedded = await _write_chunks(session, document, chunks, reuse={})

    document.status = DocStatus.PENDING_REVIEW
    session.add(
        DocumentVersion(
            document_id=document.id,
            tenant_id=tenant_id,
            version=1,
            content_sha256=content_hash,
            title=document.title,
            sensitivity=document.suggested_sensitivity,
            role_keys=document.suggested_role_keys,
            action="created",
            created_by=principal.user_id if principal else None,
        )
    )
    await audit.record(
        session,
        audit.Event.DOC_UPLOADED,
        principal=principal,
        document_id=document.id,
        title=document.title,
        filename=filename,
        chunks=len(chunks),
        suggested_sensitivity=document.suggested_sensitivity,
        suggested_roles=document.suggested_role_keys,
    )

    return IngestResult(
        document_id=document.id,
        title=document.title,
        status=DocStatus.PENDING_REVIEW,
        action="created",
        chunks_total=len(chunks),
        chunks_embedded=embedded,
        chunks_reused=0,
        requires_reapproval=True,
        message="Awaiting administrator approval before it can be read.",
    )


async def _replace(
    session: AsyncSession,
    *,
    document: Document,
    filename: str,
    parsed: ParsedDocument,
    chunks: list[TextChunk],
    content_hash: str,
    byte_size: int,
    suggested: Classification | None,
    principal: Principal | None,
) -> IngestResult:
    """Update in place, re-embedding only the chunks whose text changed."""
    previous_status = document.status
    previous_sensitivity = document.sensitivity

    # Existing chunk text -> embedding, so unchanged text keeps its vector.
    reuse: dict[str, list[float]] = {}
    for row in (
        await session.execute(
            select(Chunk.text_sha256, Chunk.embedding).where(
                Chunk.document_id == document.id
            )
        )
    ).all():
        if row[1] is not None:
            reuse[row[0]] = list(row[1])

    await session.execute(delete(Chunk).where(Chunk.document_id == document.id))

    document.title = parsed.title
    document.source_filename = filename
    document.content_sha256 = content_hash
    document.byte_size = byte_size
    document.version += 1
    document.declared_audience = parsed.declared_audience
    if suggested is not None:
        document.suggested_role_keys = suggested.role_keys
        document.suggested_sensitivity = suggested.sensitivity
        document.classifier_rationale = suggested.rationale
        document.doc_type = suggested.doc_type

    await session.flush()
    embedded = await _write_chunks(session, document, chunks, reuse=reuse)
    reused = len(chunks) - embedded

    # The safety rule for edits: if the new content classifies as MORE
    # sensitive than the access it currently has, the document goes back into
    # the review queue. Otherwise it keeps its ACL and stays live, so ordinary
    # edits do not create approval busywork.
    proposed = suggested.sensitivity if suggested else previous_sensitivity
    requires_reapproval = (
        previous_status == DocStatus.APPROVED and proposed > previous_sensitivity
    )

    if requires_reapproval:
        document.status = DocStatus.PENDING_REVIEW
        await session.execute(
            delete(DocumentACL).where(DocumentACL.document_id == document.id)
        )
        await audit.bump_acl_version(session)
    elif previous_status != DocStatus.APPROVED:
        document.status = DocStatus.PENDING_REVIEW

    retained_roles = (
        []
        if requires_reapproval
        else sorted(
            (
                await session.execute(
                    select(Role.key)
                    .join(DocumentACL, DocumentACL.role_id == Role.id)
                    .where(DocumentACL.document_id == document.id)
                )
            )
            .scalars()
            .all()
        )
    )

    session.add(
        DocumentVersion(
            document_id=document.id,
            tenant_id=document.tenant_id,
            version=document.version,
            content_sha256=content_hash,
            title=document.title,
            sensitivity=document.sensitivity,
            role_keys=retained_roles,
            action="updated",
            note=(
                "Returned to review: content now classifies as more sensitive."
                if requires_reapproval
                else "Access carried forward; classification unchanged or lower."
            ),
            created_by=principal.user_id if principal else None,
        )
    )
    await audit.record(
        session,
        audit.Event.DOC_UPDATED,
        principal=principal,
        document_id=document.id,
        severity="warning" if requires_reapproval else "info",
        title=document.title,
        version=document.version,
        chunks=len(chunks),
        chunks_embedded=embedded,
        chunks_reused=reused,
        requires_reapproval=requires_reapproval,
    )

    return IngestResult(
        document_id=document.id,
        title=document.title,
        status=document.status,
        action="updated",
        chunks_total=len(chunks),
        chunks_embedded=embedded,
        chunks_reused=reused,
        requires_reapproval=requires_reapproval,
        message=(
            "Content is now more sensitive than its current access; returned "
            "to the review queue and access revoked until re-approved."
            if requires_reapproval
            else "Updated in place; existing access retained."
        ),
    )


async def _write_chunks(
    session: AsyncSession,
    document: Document,
    chunks: list[TextChunk],
    *,
    reuse: dict[str, list[float]],
) -> int:
    """Persist chunks, embedding only those whose text we have not seen.

    Returns the number of chunks that needed a fresh embedding call.
    """
    hashes = [sha256_text(f"{c.heading_path}\n{c.text}") for c in chunks]
    to_embed = [
        index for index, digest in enumerate(hashes) if digest not in reuse
    ]

    vectors: dict[int, list[float]] = {}
    if to_embed:
        texts = [_embedding_text(chunks[i]) for i in to_embed]
        computed = await llm.embed_texts(texts)
        vectors = dict(zip(to_embed, computed, strict=True))

    for index, chunk in enumerate(chunks):
        digest = hashes[index]
        session.add(
            Chunk(
                document_id=document.id,
                tenant_id=document.tenant_id,
                ordinal=chunk.ordinal,
                heading_path=chunk.heading_path,
                text=chunk.text,
                text_sha256=digest,
                token_count=chunk.token_count,
                embedding=vectors.get(index) or reuse.get(digest),
            )
        )

    await session.flush()
    return len(to_embed)


def _embedding_text(chunk: object) -> str:
    """Prepend the heading path so section names are themselves searchable."""
    heading = getattr(chunk, "heading_path", "")
    text = getattr(chunk, "text", "")
    return f"{heading}\n\n{text}".strip() if heading else text


async def _count_chunks(session: AsyncSession, document_id: uuid.UUID) -> int:
    rows = (
        await session.execute(select(Chunk.id).where(Chunk.document_id == document_id))
    ).all()
    return len(rows)


# ---------------------------------------------------------------------------
# Approval and deletion
# ---------------------------------------------------------------------------


async def approve_document(
    session: AsyncSession,
    document: Document,
    role_keys: list[str],
    sensitivity: int,
    principal: Principal,
) -> None:
    """Grant access and make the document readable. The one path to APPROVED."""
    roles = (
        await session.execute(select(Role).where(Role.key.in_(role_keys)))
    ).scalars().all()
    found = {r.key for r in roles}
    missing = set(role_keys) - found
    if missing:
        raise IngestError(f"unknown role keys: {sorted(missing)}")

    # A role may not be granted a document above its clearance. The retrieval
    # predicate would ignore such a grant anyway; refusing here means the
    # admin panel never shows a grant that silently does nothing.
    over = [r.key for r in roles if r.clearance < sensitivity]
    if over:
        raise IngestError(
            f"roles {sorted(over)} have clearance below sensitivity {sensitivity}"
        )

    await session.execute(
        delete(DocumentACL).where(DocumentACL.document_id == document.id)
    )
    for role in roles:
        session.add(
            DocumentACL(
                document_id=document.id,
                role_id=role.id,
                granted_by=principal.user_id,
            )
        )

    document.sensitivity = sensitivity
    document.status = DocStatus.APPROVED
    document.approved_by = principal.user_id
    from datetime import UTC, datetime

    document.approved_at = datetime.now(UTC)

    session.add(
        DocumentVersion(
            document_id=document.id,
            tenant_id=document.tenant_id,
            version=document.version,
            content_sha256=document.content_sha256,
            title=document.title,
            sensitivity=sensitivity,
            role_keys=sorted(found),
            action="approved",
            created_by=principal.user_id,
        )
    )
    await audit.record(
        session,
        audit.Event.DOC_APPROVED,
        principal=principal,
        document_id=document.id,
        title=document.title,
        sensitivity=sensitivity,
        roles=sorted(found),
    )
    await audit.bump_acl_version(session)


async def delete_document(
    session: AsyncSession, document: Document, principal: Principal
) -> int:
    """Hard-delete a document and every chunk and embedding it owns.

    The version history and the audit trail survive, so the record of what
    existed and who removed it is not lost. The retrievable content is gone.
    """
    chunk_count = await _count_chunks(session, document.id)

    session.add(
        DocumentVersion(
            document_id=document.id,
            tenant_id=document.tenant_id,
            version=document.version,
            content_sha256=document.content_sha256,
            title=document.title,
            sensitivity=document.sensitivity,
            action="deleted",
            note=f"{chunk_count} chunks removed",
            created_by=principal.user_id,
        )
    )
    await audit.record(
        session,
        audit.Event.DOC_DELETED,
        principal=principal,
        document_id=document.id,
        severity="warning",
        title=document.title,
        filename=document.source_filename,
        chunks_removed=chunk_count,
    )

    # ON DELETE CASCADE takes the chunks and the ACL rows with it.
    await session.delete(document)
    await audit.bump_acl_version(session)
    return chunk_count
