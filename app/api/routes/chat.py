"""Chat endpoints.

Every handler takes its identity from `current_principal` and nothing else.
The request body carries a question; it never carries a claim about who is
asking.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AccessRequestIn,
    AnswerOut,
    AskRequest,
    CitationOut,
    FeedbackIn,
    MeOut,
    RoleComparisonEntry,
    RoleComparisonOut,
)
from app.core import audit
from app.core.principal import Principal
from app.core.rate_limit import enforce_rate_limit
from app.core.security import current_principal, require_admin
from app.db.models import AccessRequest, AuditEvent, Role, Tenant, User
from app.db.session import get_session
from app.rag import answer as answer_service

logger = logging.getLogger("assetcues.chat")

router = APIRouter(tags=["chat"])


@router.get("/me", response_model=MeOut)
async def me(
    principal: Principal = Depends(current_principal),
    session: AsyncSession = Depends(get_session),
) -> MeOut:
    user = (
        await session.execute(select(User).where(User.id == principal.user_id))
    ).scalar_one()
    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == principal.tenant_id))
    ).scalar_one()
    return MeOut(
        user_id=principal.user_id,
        email=principal.email,
        display_name=user.display_name,
        tenant_slug=tenant.slug,
        tenant_name=tenant.name,
        roles=sorted(principal.role_keys),
        clearance=principal.clearance,
        is_admin=principal.is_admin,
    )


@router.post("/ask", response_model=AnswerOut)
async def ask(
    payload: AskRequest,
    principal: Principal = Depends(current_principal),
    session: AsyncSession = Depends(get_session),
) -> AnswerOut:
    await enforce_rate_limit(principal)
    result = await answer_service.answer_question(
        session, principal, payload.question, payload.history
    )
    return AnswerOut(
        answer=result.answer,
        turn_id=result.turn_id,
        citations=[CitationOut(**asdict(c)) for c in result.citations],
        follow_ups=result.follow_ups,
        refused=result.refused,
        retracted=result.retracted,
        cached=result.cached,
        latency_ms=result.latency_ms,
        chunks_used=result.chunks_used,
    )


@router.post("/ask/stream")
async def ask_stream(
    payload: AskRequest,
    principal: Principal = Depends(current_principal),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    await enforce_rate_limit(principal)

    async def events() -> AsyncIterator[str]:
        try:
            async for name, data in answer_service.stream_answer(
                session, principal, payload.question, payload.history
            ):
                yield f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            # Log it. An error event with no server-side trace is how a
            # streaming bug stays invisible: the client shows one sentence,
            # the access log shows a clean 200, and nothing says what broke.
            logger.exception("streaming answer failed", exc_info=exc)
            payload_out = {
                "message": "The assistant failed to answer.",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            yield f"event: error\ndata: {json.dumps(payload_out)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    payload: FeedbackIn,
    principal: Principal = Depends(current_principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Record a thumbs up or down against an answer.

    The question, the documents used and the roles in play are read back from
    the query's own audit row, not taken from the request. A client could
    otherwise file feedback against a question nobody asked, or against
    someone else's query.
    """
    original = (
        await session.execute(
            select(AuditEvent).where(
                AuditEvent.id == payload.turn_id,
                AuditEvent.event_type.in_(
                    [audit.Event.QUERY, audit.Event.QUERY_REFUSED]
                ),
            )
        )
    ).scalar_one_or_none()

    if original is None:
        raise HTTPException(status_code=404, detail="No such answer")

    # Feedback belongs to the person who asked. Anything else lets one user
    # attach opinions to another user's conversation.
    if original.actor_user_id != principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only rate your own answers",
        )

    detail = original.detail or {}
    await audit.record(
        session,
        audit.Event.FEEDBACK,
        principal=principal,
        severity="warning" if payload.rating == "down" else "info",
        rating=payload.rating,
        comment=payload.comment[:1000],
        turn_id=str(payload.turn_id),
        asked_at=original.created_at.isoformat(),
        # Server-derived, from the original query row.
        question=detail.get("question", ""),
        documents_used=detail.get("documents_used", []),
        chunks_used=detail.get("chunks_used", 0),
        was_refused=original.event_type == audit.Event.QUERY_REFUSED,
        # Client-reported: the server keeps no transcript, so this is the only
        # record of what was actually shown. Stored because the person asked
        # for this answer to be looked at.
        answer_as_shown=payload.answer[:8000],
    )
    await session.commit()
    return {"status": "recorded", "rating": payload.rating}


@router.post("/access-request", status_code=status.HTTP_201_CREATED)
async def request_access(
    payload: AccessRequestIn,
    principal: Principal = Depends(current_principal),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Raised from the refusal card in the chat UI.

    Note what is *not* returned: which documents would have matched. Telling a
    user that restricted material exists is a disclosure in itself, and would
    let someone map the corpus by probing refusals. The administrator sees the
    matching documents in the audit console; the requester does not.
    """
    session.add(
        AccessRequest(
            user_id=principal.user_id,
            question=payload.question[:2000],
            justification=payload.justification[:2000],
        )
    )
    await audit.record(
        session,
        audit.Event.ACCESS_REQUESTED,
        principal=principal,
        question=payload.question[:500],
    )
    await session.commit()
    return {"status": "submitted"}


@router.post("/compare", response_model=RoleComparisonOut)
async def compare_roles(
    payload: AskRequest,
    principal: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> RoleComparisonOut:
    """Ask one question as every role at once. Administrators only.

    This is the demo that answers "how do we protect information across
    profiles" -- not by describing the design, but by running it. Each column
    is a real retrieval under that role's actual permissions, using the same
    code path as a live user.

    It is admin-gated because the aggregate view reveals which documents exist
    and who can see them.
    """
    roles = (
        await session.execute(select(Role).order_by(Role.clearance.desc(), Role.key))
    ).scalars().all()

    entries: list[RoleComparisonEntry] = []
    total_matching = 0

    for role in roles:
        # A synthetic principal carrying exactly one role, inside the caller's
        # own tenant. Built server-side from database rows -- never from input.
        probe = Principal(
            user_id=principal.user_id,
            email=principal.email,
            tenant_id=principal.tenant_id,
            tenant_slug=principal.tenant_slug,
            role_ids=frozenset({role.id}),
            role_keys=frozenset({role.key}),
            clearance=role.clearance,
        )
        context, _ = await answer_service.gather_context(
            session, probe, payload.question
        )
        total_matching = max(
            total_matching, len(context.chunks) + context.blocked_chunk_count
        )

        if context.chunks:
            result = await answer_service.answer_question(
                session, probe, payload.question
            )
            text, refused, citations = (
                result.answer,
                result.refused,
                [CitationOut(**asdict(c)) for c in result.citations],
            )
        else:
            from app.rag.prompts import REFUSAL_TEXT

            text, refused, citations = REFUSAL_TEXT, True, []

        entries.append(
            RoleComparisonEntry(
                role_key=role.key,
                role_name=role.name,
                answer=text,
                refused=refused,
                citations=citations,
                chunks_allowed=len(context.chunks),
                documents_blocked=len(context.blocked),
                blocked_titles=[b.title for b in context.blocked],
            )
        )

    return RoleComparisonOut(
        question=payload.question,
        entries=entries,
        total_matching_chunks=total_matching,
    )


@router.post("/cache/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache(_: Principal = Depends(require_admin)) -> None:
    answer_service.cache_clear()


@router.get("/healthz/deep")
async def deep_health(
    _: Principal = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    version = await audit.get_acl_version(session)
    if version < 1:
        raise HTTPException(status_code=500, detail="system_state not seeded")
    return {"acl_version": version}
