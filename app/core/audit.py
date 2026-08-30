"""Append-only audit trail.

Every question asked, every access decision, and every administrative change
lands here. This is what turns "we think the isolation works" into "here is
the log showing it working", which is the thing a founder or an auditor
actually wants to see.

Nothing in the application updates or deletes an audit row.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.principal import Principal
from app.db.models import AuditEvent, SystemState


class Event:
    QUERY = "query"
    QUERY_REFUSED = "query_refused"
    SECURITY_ANOMALY = "security_anomaly"
    CITATION_REJECTED = "citation_rejected"
    INJECTION_DETECTED = "injection_detected"
    DOC_UPLOADED = "document_uploaded"
    DOC_UPDATED = "document_updated"
    DOC_DELETED = "document_deleted"
    DOC_APPROVED = "document_approved"
    DOC_REJECTED = "document_rejected"
    ACL_CHANGED = "acl_changed"
    ROLE_GRANTED = "role_granted"
    ROLE_REVOKED = "role_revoked"
    USER_CREATED = "user_created"
    USER_DISABLED = "user_disabled"
    ACCESS_REQUESTED = "access_requested"
    LOGIN = "login"


async def record(
    session: AsyncSession,
    event_type: str,
    *,
    principal: Principal | None = None,
    document_id: uuid.UUID | None = None,
    severity: str = "info",
    **detail: Any,
) -> None:
    """Write one audit row. Never raises into the caller's happy path."""
    session.add(
        AuditEvent(
            event_type=event_type,
            actor_user_id=principal.user_id if principal else None,
            actor_email=principal.email if principal else "",
            tenant_id=principal.tenant_id if principal else None,
            actor_role_keys=sorted(principal.role_keys) if principal else [],
            document_id=document_id,
            severity=severity,
            detail=detail,
        )
    )


async def get_acl_version(session: AsyncSession) -> int:
    """Current global ACL generation, used in the answer-cache key (G7)."""
    value = (
        await session.execute(select(SystemState.acl_version).where(SystemState.id == 1))
    ).scalar_one_or_none()
    return int(value or 1)


async def bump_acl_version(session: AsyncSession) -> int:
    """Invalidate every cached answer.

    Called after any change to a document's status, its ACL, or a user's
    roles. Cheap, global, and impossible to forget in only one code path
    because it lives next to the audit write that every such change makes.
    """
    state = (
        await session.execute(select(SystemState).where(SystemState.id == 1))
    ).scalar_one_or_none()
    if state is None:
        state = SystemState(id=1, acl_version=2)
        session.add(state)
        await session.flush()
        return 2

    state.acl_version += 1
    await session.flush()
    return int(state.acl_version)


async def count_events(session: AsyncSession, event_type: str) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.event_type == event_type)
            )
        ).scalar_one()
    )
