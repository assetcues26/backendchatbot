"""Request and response models.

GUARDRAIL G2 -- read before adding a field.

No request model in this file may contain `role`, `roles`, `role_keys`,
`tenant_id`, `clearance`, or `user_id` describing *the caller*. Identity comes
from the verified JWT and a database lookup, never from the request body.
`tests/security/test_tampering.py` inspects every request schema and fails the
build if such a field appears.

Admin models may name a *target* user or role -- "grant engineering to Bob" is
legitimate. The forbidden thing is a request that asserts who the sender is.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    environment: str
    database: str
    documents_approved: int = 0
    chunks_total: int = 0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """A question, plus the caller's own earlier questions for context.

    `history` holds previous USER questions only -- never assistant
    answers. A client can fabricate anything it sends, and fabricated
    "the assistant said X" text would go straight into the prompt. The
    user's own questions grant no capability they did not already have,
    and are enough to resolve "what about for sales?".
    """

    question: str = Field(min_length=1, max_length=2000)
    history: list[str] = Field(default_factory=list, max_length=5)


class CitationOut(BaseModel):
    key: str
    document_id: str
    title: str
    doc_type: str
    module: str
    heading_path: str
    ordinal: int


class AnswerOut(BaseModel):
    answer: str
    citations: list[CitationOut] = []
    follow_ups: list[str] = []
    refused: bool = False
    retracted: bool = False
    cached: bool = False
    latency_ms: int = 0
    chunks_used: int = 0


class RoleComparisonEntry(BaseModel):
    """One column of the founder-facing side-by-side view."""

    role_key: str
    role_name: str
    answer: str
    refused: bool
    citations: list[CitationOut] = []
    chunks_allowed: int
    documents_blocked: int
    blocked_titles: list[str] = []


class RoleComparisonOut(BaseModel):
    question: str
    entries: list[RoleComparisonEntry]
    total_matching_chunks: int


class AccessRequestIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    justification: str = Field(default="", max_length=2000)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class MeOut(BaseModel):
    user_id: uuid.UUID
    email: str
    display_name: str
    tenant_slug: str
    tenant_name: str
    roles: list[str]
    clearance: int
    is_admin: bool


# ---------------------------------------------------------------------------
# Documents (admin)
# ---------------------------------------------------------------------------


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_filename: str
    module: str
    doc_type: str
    status: str
    sensitivity: int
    is_shared: bool = False
    version: int
    byte_size: int
    declared_audience: list[str] = []
    suggested_role_keys: list[str] = []
    suggested_sensitivity: int | None = None
    classifier_rationale: str = ""
    granted_role_keys: list[str] = []
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class ApproveDocumentIn(BaseModel):
    """Roles named here are the *grantees*, not the caller."""

    role_keys: list[str] = Field(min_length=1)
    sensitivity: int = Field(ge=1, le=4)
    note: str = Field(default="", max_length=1000)


class IngestOut(BaseModel):
    document_id: uuid.UUID
    title: str
    status: str
    action: str
    chunks_total: int
    chunks_embedded: int
    chunks_reused: int
    requires_reapproval: bool
    message: str = ""


class BulkIngestOut(BaseModel):
    results: list[IngestOut]
    created: int
    updated: int
    unchanged: int
    failed: int
    embeddings_saved: int


# ---------------------------------------------------------------------------
# Users and roles (admin)
# ---------------------------------------------------------------------------


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    tenant_slug: str
    roles: list[str]
    clearance: int
    is_active: bool
    created_at: datetime


class CreateUserIn(BaseModel):
    email: str = Field(max_length=320)
    display_name: str = Field(default="", max_length=200)
    tenant_slug: str = Field(default="assetcues", max_length=64)
    role_keys: list[str] = []
    user_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Supabase auth user id, when linking an already-registered account. "
            "This identifies the user being created, not the caller."
        ),
    )


class SetUserRolesIn(BaseModel):
    role_keys: list[str]


class SetUserActiveIn(BaseModel):
    is_active: bool


class UserGrantIn(BaseModel):
    document_id: uuid.UUID
    effect: str = Field(pattern="^(allow|deny)$")
    reason: str = Field(default="", max_length=1000)
    expires_at: datetime | None = None


class RoleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    key: str
    name: str
    description: str
    clearance: int
    is_internal: bool


class TenantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    name: str
    kind: str
    is_active: bool


class CreateTenantIn(BaseModel):
    slug: str = Field(min_length=2, max_length=64, pattern="^[a-z0-9-]+$")
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(default="customer", pattern="^(internal|customer)$")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditEventOut(BaseModel):
    id: uuid.UUID
    event_type: str
    actor_email: str
    actor_role_keys: list[str]
    document_id: uuid.UUID | None
    severity: str
    detail: dict[str, Any]
    created_at: datetime


class AuditSummaryOut(BaseModel):
    total_queries: int
    total_refusals: int
    total_anomalies: int
    total_retractions: int
    injections_detected: int
    acl_version: int
