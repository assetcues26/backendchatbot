"""Administrator endpoints: documents, access, users, audit.

Every route here depends on `require_admin`. That check reads the caller's
roles from the database via the verified JWT, so an attacker cannot reach any
of it by asserting a role in a payload.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    ApproveDocumentIn,
    AuditEventOut,
    AuditSummaryOut,
    CreateTenantIn,
    CreateUserIn,
    DocumentOut,
    IngestOut,
    RoleOut,
    SetUserActiveIn,
    SetUserRolesIn,
    TenantOut,
    UserGrantIn,
    UserOut,
)
from app.core import audit
from app.core.principal import Principal
from app.core.security import require_admin
from app.db.models import (
    AuditEvent,
    Chunk,
    DocStatus,
    Document,
    DocumentACL,
    GrantEffect,
    Role,
    Tenant,
    TenantKind,
    User,
    UserDocumentGrant,
    UserRole,
)
from app.db.session import get_session
from app.ingest import pipeline
from app.ingest.parsers import SUPPORTED_SUFFIXES

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def _document_out(session: AsyncSession, doc: Document) -> DocumentOut:
    granted = (
        await session.execute(
            select(Role.key)
            .join(DocumentACL, DocumentACL.role_id == Role.id)
            .where(DocumentACL.document_id == doc.id)
        )
    ).scalars().all()
    count = (
        await session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
    ).scalar_one()
    return DocumentOut(
        id=doc.id,
        title=doc.title,
        source_filename=doc.source_filename,
        module=doc.module,
        doc_type=doc.doc_type,
        status=doc.status.value,
        sensitivity=doc.sensitivity,
        is_shared=doc.is_shared,
        version=doc.version,
        byte_size=doc.byte_size,
        declared_audience=list(doc.declared_audience or []),
        suggested_role_keys=list(doc.suggested_role_keys or []),
        suggested_sensitivity=doc.suggested_sensitivity,
        classifier_rationale=doc.classifier_rationale,
        granted_role_keys=sorted(granted),
        chunk_count=int(count),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    status_filter: str | None = Query(default=None, alias="status"),
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentOut]:
    stmt = select(Document).where(Document.tenant_id == principal.tenant_id)
    if status_filter:
        stmt = stmt.where(Document.status == DocStatus(status_filter))
    stmt = stmt.order_by(Document.status, Document.module, Document.title)
    docs = (await session.execute(stmt)).scalars().all()
    return [await _document_out(session, d) for d in docs]


@router.post("/documents", response_model=IngestOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    module: str = Query(default=""),
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> IngestOut:
    """Upload or replace a document.

    The uploaded file lands in PENDING_REVIEW and is readable by nobody until
    it is approved, whether it is new or a replacement.
    """
    filename = file.filename or "upload"
    suffix = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {suffix!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = await pipeline.ingest_bytes(
            session,
            tenant_id=principal.tenant_id,
            source_key=filename,
            filename=filename,
            data=data,
            module=module,
            principal=principal,
        )
    except pipeline.IngestError as exc:
        await session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await session.commit()
    return IngestOut(
        document_id=result.document_id,
        title=result.title,
        status=result.status.value,
        action=result.action,
        chunks_total=result.chunks_total,
        chunks_embedded=result.chunks_embedded,
        chunks_reused=result.chunks_reused,
        requires_reapproval=result.requires_reapproval,
        message=result.message,
    )


@router.post("/documents/{document_id}/approve", response_model=DocumentOut)
async def approve_document(
    document_id: uuid.UUID,
    payload: ApproveDocumentIn,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    doc = await _get_document(session, document_id, principal)
    try:
        await pipeline.approve_document(
            session, doc, payload.role_keys, payload.sensitivity, principal
        )
    except pipeline.IngestError as exc:
        await session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await session.commit()
    await session.refresh(doc)
    return await _document_out(session, doc)


@router.post("/documents/{document_id}/revoke", response_model=DocumentOut)
async def revoke_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    """Pull a document out of circulation without deleting it."""
    doc = await _get_document(session, document_id, principal)
    await session.execute(
        delete(DocumentACL).where(DocumentACL.document_id == doc.id)
    )
    doc.status = DocStatus.PENDING_REVIEW
    await audit.record(
        session,
        audit.Event.DOC_REJECTED,
        principal=principal,
        document_id=doc.id,
        severity="warning",
        title=doc.title,
    )
    await audit.bump_acl_version(session)
    await session.commit()
    await session.refresh(doc)
    return await _document_out(session, doc)


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Delete a document and every chunk and embedding it owns."""
    doc = await _get_document(session, document_id, principal)
    title = doc.title
    removed = await pipeline.delete_document(session, doc, principal)
    await session.commit()
    return {"deleted": True, "title": title, "chunks_removed": removed}


async def _get_document(
    session: AsyncSession, document_id: uuid.UUID, principal: Principal
) -> Document:
    doc = (
        await session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.tenant_id == principal.tenant_id,
            )
        )
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Roles, tenants, users
# ---------------------------------------------------------------------------


@router.get("/roles", response_model=list[RoleOut])
async def list_roles(session: AsyncSession = Depends(get_session)) -> list[RoleOut]:
    rows = (
        await session.execute(select(Role).order_by(Role.clearance.desc(), Role.key))
    ).scalars().all()
    return [RoleOut.model_validate(r) for r in rows]


@router.get("/tenants", response_model=list[TenantOut])
async def list_tenants(session: AsyncSession = Depends(get_session)) -> list[TenantOut]:
    rows = (await session.execute(select(Tenant).order_by(Tenant.name))).scalars().all()
    return [
        TenantOut(
            id=t.id, slug=t.slug, name=t.name, kind=t.kind.value, is_active=t.is_active
        )
        for t in rows
    ]


@router.post("/tenants", response_model=TenantOut, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    payload: CreateTenantIn, session: AsyncSession = Depends(get_session)
) -> TenantOut:
    existing = (
        await session.execute(select(Tenant).where(Tenant.slug == payload.slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Tenant slug already exists")

    tenant = Tenant(
        slug=payload.slug, name=payload.name, kind=TenantKind(payload.kind)
    )
    session.add(tenant)
    await session.commit()
    await session.refresh(tenant)
    return TenantOut(
        id=tenant.id,
        slug=tenant.slug,
        name=tenant.name,
        kind=tenant.kind.value,
        is_active=tenant.is_active,
    )


async def _user_out(session: AsyncSession, user: User) -> UserOut:
    rows = (
        await session.execute(
            select(Role)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user.id)
        )
    ).scalars().all()
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    ).scalar_one()
    return UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        tenant_slug=tenant.slug,
        roles=sorted(r.key for r in rows),
        clearance=max((r.clearance for r in rows), default=0),
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/users", response_model=list[UserOut])
async def list_users(session: AsyncSession = Depends(get_session)) -> list[UserOut]:
    users = (await session.execute(select(User).order_by(User.email))).scalars().all()
    return [await _user_out(session, u) for u in users]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: CreateUserIn,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    tenant = (
        await session.execute(select(Tenant).where(Tenant.slug == payload.tenant_slug))
    ).scalar_one_or_none()
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    email = payload.email.strip().lower()
    existing = (
        await session.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="A user with that email exists")

    user = User(
        # Supabase mints the real id at first sign-in; a placeholder lets an
        # admin stage roles before the person has ever logged in.
        id=payload.user_id or uuid.uuid4(),
        tenant_id=tenant.id,
        email=email,
        display_name=payload.display_name,
    )
    session.add(user)
    await session.flush()

    await _apply_roles(session, user, payload.role_keys, principal)
    await audit.record(
        session,
        audit.Event.USER_CREATED,
        principal=principal,
        target_email=email,
        tenant=tenant.slug,
        roles=payload.role_keys,
    )
    await session.commit()
    await session.refresh(user)
    return await _user_out(session, user)


@router.put("/users/{user_id}/roles", response_model=UserOut)
async def set_user_roles(
    user_id: uuid.UUID,
    payload: SetUserRolesIn,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    await _apply_roles(session, user, payload.role_keys, principal)
    # A role change alters what this person may read, so every cached answer
    # must go. Cheap, and it makes the revocation test pass.
    await audit.bump_acl_version(session)
    await session.commit()
    await session.refresh(user)
    return await _user_out(session, user)


async def _apply_roles(
    session: AsyncSession, user: User, role_keys: list[str], principal: Principal
) -> None:
    roles = (
        await session.execute(select(Role).where(Role.key.in_(role_keys)))
    ).scalars().all()
    missing = set(role_keys) - {r.key for r in roles}
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Unknown role keys: {sorted(missing)}"
        )

    await session.execute(delete(UserRole).where(UserRole.user_id == user.id))
    for role in roles:
        session.add(
            UserRole(user_id=user.id, role_id=role.id, granted_by=principal.user_id)
        )
    await session.flush()
    await audit.record(
        session,
        audit.Event.ROLE_GRANTED,
        principal=principal,
        target_user=str(user.id),
        target_email=user.email,
        roles=sorted(r.key for r in roles),
    )


@router.put("/users/{user_id}/active", response_model=UserOut)
async def set_user_active(
    user_id: uuid.UUID,
    payload: SetUserActiveIn,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = payload.is_active
    await audit.record(
        session,
        audit.Event.USER_DISABLED if not payload.is_active else audit.Event.USER_CREATED,
        principal=principal,
        severity="warning" if not payload.is_active else "info",
        target_email=user.email,
        is_active=payload.is_active,
    )
    await audit.bump_acl_version(session)
    await session.commit()
    await session.refresh(user)
    return await _user_out(session, user)


@router.post("/users/{user_id}/grants", status_code=status.HTTP_201_CREATED)
async def add_user_grant(
    user_id: uuid.UUID,
    payload: UserGrantIn,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Per-user override on one document. A DENY beats every role grant."""
    effect = GrantEffect(payload.effect.upper()) if payload.effect.isupper() else (
        GrantEffect.ALLOW if payload.effect == "allow" else GrantEffect.DENY
    )
    await session.execute(
        delete(UserDocumentGrant).where(
            UserDocumentGrant.user_id == user_id,
            UserDocumentGrant.document_id == payload.document_id,
            UserDocumentGrant.effect == effect,
        )
    )
    session.add(
        UserDocumentGrant(
            user_id=user_id,
            document_id=payload.document_id,
            effect=effect,
            reason=payload.reason,
            granted_by=principal.user_id,
            expires_at=payload.expires_at,
        )
    )
    await audit.record(
        session,
        audit.Event.ACL_CHANGED,
        principal=principal,
        document_id=payload.document_id,
        target_user=str(user_id),
        effect=effect.value,
        reason=payload.reason,
    )
    await audit.bump_acl_version(session)
    await session.commit()
    return {"status": "granted", "effect": effect.value}


@router.delete("/users/{user_id}/grants/{document_id}", status_code=204)
async def remove_user_grants(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> None:
    await session.execute(
        delete(UserDocumentGrant).where(
            UserDocumentGrant.user_id == user_id,
            UserDocumentGrant.document_id == document_id,
        )
    )
    await audit.record(
        session,
        audit.Event.ACL_CHANGED,
        principal=principal,
        document_id=document_id,
        target_user=str(user_id),
        effect="cleared",
    )
    await audit.bump_acl_version(session)
    await session.commit()


# ---------------------------------------------------------------------------
# Audit console
# ---------------------------------------------------------------------------


@router.get("/audit", response_model=list[AuditEventOut])
async def list_audit(
    limit: int = Query(default=100, le=500),
    event_type: str | None = None,
    severity: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[AuditEventOut]:
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    if event_type:
        stmt = stmt.where(AuditEvent.event_type == event_type)
    if severity:
        stmt = stmt.where(AuditEvent.severity == severity)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        AuditEventOut(
            id=e.id,
            event_type=e.event_type,
            actor_email=e.actor_email,
            actor_role_keys=list(e.actor_role_keys or []),
            document_id=e.document_id,
            severity=e.severity,
            detail=e.detail or {},
            created_at=e.created_at,
        )
        for e in rows
    ]


@router.get("/audit/summary", response_model=AuditSummaryOut)
async def audit_summary(
    session: AsyncSession = Depends(get_session),
) -> AuditSummaryOut:
    return AuditSummaryOut(
        total_queries=await audit.count_events(session, audit.Event.QUERY),
        feedback_up=await audit.count_feedback(session, "up"),
        feedback_down=await audit.count_feedback(session, "down"),
        total_refusals=await audit.count_events(session, audit.Event.QUERY_REFUSED),
        total_anomalies=await audit.count_events(session, audit.Event.SECURITY_ANOMALY),
        total_retractions=await audit.count_events(
            session, audit.Event.CITATION_REJECTED
        ),
        injections_detected=await audit.count_events(
            session, audit.Event.INJECTION_DETECTED
        ),
        acl_version=await audit.get_acl_version(session),
    )


@router.get("/access-requests")
async def list_access_requests(
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
    from app.db.models import AccessRequest

    rows = (
        await session.execute(
            select(AccessRequest, User.email)
            .join(User, User.id == AccessRequest.user_id)
            .order_by(AccessRequest.created_at.desc())
            .limit(200)
        )
    ).all()
    return [
        {
            "id": str(r[0].id),
            "email": r[1],
            "question": r[0].question,
            "justification": r[0].justification,
            "status": r[0].status,
            "created_at": r[0].created_at.isoformat(),
        }
        for r in rows
    ]


@router.post("/access-requests/{request_id}/resolve", status_code=204)
async def resolve_access_request(
    request_id: uuid.UUID,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> None:
    from app.db.models import AccessRequest

    row = (
        await session.execute(
            select(AccessRequest).where(AccessRequest.id == request_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Request not found")
    row.status = "resolved"
    row.resolved_by = principal.user_id
    row.resolved_at = datetime.now(UTC)
    await session.commit()
