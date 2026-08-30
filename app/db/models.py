"""SQLAlchemy models.

Design rules that must not be broken:

1. ``chunks.document_id`` is ON DELETE CASCADE. Deleting a document deletes its
   chunks and their embeddings in the same transaction. There is no second
   store to keep in sync -- that is the whole reason the vectors live in
   Postgres.
2. A document is invisible until ``status == APPROVED``. New and re-uploaded
   documents land in PENDING_REVIEW. Default-deny is the schema's job, not the
   application's.
3. Access is never stored on the chunk. It is resolved by joining to
   ``document_acl`` so an ACL edit takes effect immediately and atomically,
   with no rows to backfill and no window where the two disagree.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Must match Settings.embedding_dim. Asserted against the first live embedding
# response during ingestion so the schema can never silently disagree with the
# model that produced the vectors.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Sensitivity(enum.IntEnum):
    """Ceiling levels. A role's clearance must be >= a document's sensitivity."""

    PUBLIC = 1
    CUSTOMER = 2
    INTERNAL = 3
    RESTRICTED = 4


class DocStatus(str, enum.Enum):
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"  # ingested + classified, NOT yet readable
    APPROVED = "approved"  # the only status that is ever retrievable
    ARCHIVED = "archived"
    FAILED = "failed"


class GrantEffect(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"  # always wins over any role-based grant


class TenantKind(str, enum.Enum):
    INTERNAL = "internal"  # AssetCues staff
    CUSTOMER = "customer"  # one row per customer organisation


# --------------------------------------------------------------------------
# Tenancy and identity
# --------------------------------------------------------------------------


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[TenantKind] = mapped_column(
        Enum(TenantKind, name="tenant_kind"), default=TenantKind.CUSTOMER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    # Ceiling only. Holding clearance does NOT grant access to a document --
    # an explicit document_acl row is still required.
    clearance: Mapped[int] = mapped_column(SmallInteger)
    is_internal: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        CheckConstraint("clearance BETWEEN 1 AND 4", name="ck_role_clearance"),
    )


class User(Base):
    __tablename__ = "users"

    # Mirrors Supabase auth.users.id so the JWT `sub` maps straight through.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), index=True
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tenant: Mapped[Tenant] = relationship(lazy="joined")
    roles: Mapped[list[Role]] = relationship(secondary="user_roles", lazy="selectin")


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="RESTRICT"), index=True
    )

    title: Mapped[str] = mapped_column(String(500))
    source_filename: Mapped[str] = mapped_column(String(500))
    # Stable identity across re-uploads: same key == same document lineage.
    source_key: Mapped[str] = mapped_column(String(700), index=True)
    module: Mapped[str] = mapped_column(String(200), default="")
    doc_type: Mapped[str] = mapped_column(String(100), default="")

    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    byte_size: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_path: Mapped[str] = mapped_column(String(1000), default="")

    sensitivity: Mapped[int] = mapped_column(
        SmallInteger, default=int(Sensitivity.RESTRICTED)
    )
    status: Mapped[DocStatus] = mapped_column(
        Enum(DocStatus, name="doc_status"), default=DocStatus.PROCESSING, index=True
    )

    # Scraped verbatim from the document's own "Primary audience" field.
    declared_audience: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    # Classifier proposal, held for the admin to accept or override.
    suggested_role_keys: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    suggested_sensitivity: Mapped[int | None] = mapped_column(SmallInteger)
    classifier_rationale: Mapped[str] = mapped_column(Text, default="")

    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    approved_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    acl: Mapped[list[DocumentACL]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "source_key", name="uq_document_tenant_source"),
        CheckConstraint("sensitivity BETWEEN 1 AND 4", name="ck_doc_sensitivity"),
        Index("ix_documents_status_tenant", "status", "tenant_id"),
    )


class DocumentVersion(Base):
    """Immutable history. Kept even after the document itself is deleted."""

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(500), default="")
    sensitivity: Mapped[int | None] = mapped_column(SmallInteger)
    role_keys: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    action: Mapped[str] = mapped_column(String(32), default="created")
    note: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentACL(Base):
    """Which roles may read a document. Absence of a row means no access."""

    __tablename__ = "document_acl"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserDocumentGrant(Base):
    """Per-user override. DENY always beats any role grant."""

    __tablename__ = "user_document_grants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    effect: Mapped[GrantEffect] = mapped_column(Enum(GrantEffect, name="grant_effect"))
    reason: Mapped[str] = mapped_column(Text, default="")
    granted_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("user_id", "document_id", "effect", name="uq_user_doc_grant"),
    )


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # Denormalised for a cheap first cut of the retrieval filter. Tenancy is
    # immutable for a document, so this cannot drift the way an ACL would.
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)

    ordinal: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64), index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_doc_ordinal"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )


# --------------------------------------------------------------------------
# Audit and operations
# --------------------------------------------------------------------------


class AuditEvent(Base):
    """Append-only. Nothing in the application ever updates or deletes a row."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), index=True
    )
    actor_email: Mapped[str] = mapped_column(String(320), default="")
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    actor_role_keys: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    severity: Mapped[str] = mapped_column(String(16), default="info", index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class AccessRequest(Base):
    """Raised from the chat UI when a user hits a refusal."""

    __tablename__ = "access_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SystemState(Base):
    """Single-row table holding the global ACL generation counter.

    Any ACL, role or document-status change increments ``acl_version``. It is
    mixed into the answer-cache key, so a permission change invalidates every
    cached answer instantly rather than leaving a stale answer readable by
    someone who just lost access.
    """

    __tablename__ = "system_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    acl_version: Mapped[int] = mapped_column(BigInteger, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (CheckConstraint("id = 1", name="ck_system_state_singleton"),)
