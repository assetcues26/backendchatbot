"""Answer orchestration: retrieve, generate, verify, audit.

The full path a question takes:

    embed -> retrieve under ACL (G3) -> re-verify (G4) -> build fenced
    prompt (G5) -> generate -> validate citations (G6) -> audit

A note on streaming and G6, because it looks like a hole and is not one:
tokens are streamed to the client before citation validation has run. That is
safe, because the only text the model can draw on is the context we gave it,
and that context was already restricted to documents this caller may read.
G6 does not catch unauthorised *content* -- G3 and G4 make unauthorised
content unreachable. G6 catches an answer that cites a source it was not
given, which is a correctness failure, not a disclosure. When it fires we
retract the answer rather than let an untraceable claim stand.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import audit
from app.core.principal import Principal
from app.rag import guardrails, llm, prompts
from app.rag.retrieval import RetrievalResult, RetrievedChunk, retrieve


@dataclass(slots=True)
class Citation:
    key: str
    document_id: str
    title: str
    doc_type: str
    module: str
    heading_path: str
    ordinal: int


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    retracted: bool = False
    latency_ms: int = 0
    chunks_used: int = 0
    blocked_document_count: int = 0
    blocked_chunk_count: int = 0
    anomalies: list[str] = field(default_factory=list)
    injection_findings: list[dict[str, object]] = field(default_factory=list)
    cached: bool = False


# ---------------------------------------------------------------------------
# Answer cache (guardrail G7)
# ---------------------------------------------------------------------------
#
# Keyed by the question AND the caller's permission fingerprint AND the global
# ACL generation. The classic bug in a system like this is a cache keyed on
# question text alone, which happily serves a Sales answer to a Customer. The
# fingerprint makes that impossible; the generation counter means any ACL edit
# invalidates every entry at once.
#
# In-memory is correct for the single-instance free-tier deployment. A
# multi-instance deployment must move this to Redis -- see docs/RUNBOOK.md.

_CACHE: dict[str, tuple[float, AnswerResult]] = {}
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 500


def cache_key(question: str, principal: Principal, acl_version: int) -> str:
    normalised = " ".join(question.lower().split())
    material = f"{normalised}|{principal.acl_fingerprint()}|v{acl_version}"
    return hashlib.sha256(material.encode()).hexdigest()


def cache_get(key: str) -> AnswerResult | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    stored_at, result = entry
    if time.time() - stored_at > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return result


def cache_put(key: str, result: AnswerResult) -> None:
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.time(), result)


def cache_clear() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_citations(chunks: list[RetrievedChunk], cited: set[str]) -> list[Citation]:
    """Only surface sources the answer actually used."""
    return [
        Citation(
            key=c.citation_key,
            document_id=str(c.document_id),
            title=c.title,
            doc_type=c.doc_type,
            module=c.module,
            heading_path=c.heading_path,
            ordinal=c.ordinal,
        )
        for c in chunks
        if c.citation_key.lower() in cited
    ]


async def gather_context(
    session: AsyncSession, principal: Principal, question: str
) -> tuple[RetrievalResult, list[float]]:
    settings = get_settings()
    vector = await llm.embed_query(question)
    result = await retrieve(
        session,
        principal,
        question,
        vector,
        top_k=settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
    )
    return result, vector


async def _finalise(
    session: AsyncSession,
    principal: Principal,
    question: str,
    raw_answer: str,
    context: RetrievalResult,
    started: float,
) -> AnswerResult:
    """Post-generation checks and the audit write. Shared by both paths."""
    check = guardrails.validate_citations(
        raw_answer, context.chunks, refusal_text=prompts.REFUSAL_TEXT
    )
    injection = guardrails.scan_chunks_for_injection(context.chunks)
    latency = int((time.time() - started) * 1000)

    retracted = not check.ok
    answer = prompts.REFUSAL_TEXT if retracted else raw_answer
    is_refusal = answer.strip() == prompts.REFUSAL_TEXT.strip()

    if retracted:
        await audit.record(
            session,
            audit.Event.CITATION_REJECTED,
            principal=principal,
            severity="warning",
            question=question[:500],
            reason=check.failure_reason,
            invalid_keys=sorted(check.invalid),
        )

    if injection:
        await audit.record(
            session,
            audit.Event.INJECTION_DETECTED,
            principal=principal,
            severity="warning",
            question=question[:500],
            findings=injection,
        )

    for anomaly in context.anomalies:
        await audit.record(
            session,
            audit.Event.SECURITY_ANOMALY,
            principal=principal,
            severity="critical",
            question=question[:500],
            detail_text=anomaly,
        )

    await audit.record(
        session,
        audit.Event.QUERY_REFUSED if is_refusal else audit.Event.QUERY,
        principal=principal,
        question=question[:500],
        chunks_used=len(context.chunks),
        documents_used=sorted({str(c.document_id) for c in context.chunks}),
        blocked_documents=[
            {"id": str(b.document_id), "title": b.title, "chunks": b.chunk_count}
            for b in context.blocked
        ],
        blocked_chunk_count=context.blocked_chunk_count,
        citations=sorted(check.cited),
        retracted=retracted,
        latency_ms=latency,
    )
    await session.commit()

    return AnswerResult(
        answer=answer,
        citations=[] if retracted or is_refusal else build_citations(context.chunks, check.cited),
        refused=is_refusal,
        retracted=retracted,
        latency_ms=latency,
        chunks_used=len(context.chunks),
        blocked_document_count=len(context.blocked),
        blocked_chunk_count=context.blocked_chunk_count,
        anomalies=context.anomalies,
        injection_findings=injection,
    )


async def answer_question(
    session: AsyncSession, principal: Principal, question: str
) -> AnswerResult:
    """Non-streaming answer. Used by the role-comparison view and tests."""
    started = time.time()
    version = await audit.get_acl_version(session)
    key = cache_key(question, principal, version)

    hit = cache_get(key)
    if hit is not None:
        cached = AnswerResult(**{**hit.__dict__})
        cached.cached = True
        return cached

    context, _ = await gather_context(session, principal, question)

    if not context.chunks:
        result = await _finalise(
            session, principal, question, prompts.REFUSAL_TEXT, context, started
        )
        cache_put(key, result)
        return result

    raw = await llm.complete(
        prompts.build_system_prompt(principal),
        prompts.build_user_prompt(question, context.chunks),
        max_tokens=1200,
    )
    result = await _finalise(session, principal, question, raw, context, started)
    cache_put(key, result)
    return result


async def stream_answer(
    session: AsyncSession, principal: Principal, question: str
) -> AsyncIterator[tuple[str, object]]:
    """Yield (event_name, payload) pairs for the SSE endpoint.

    Events: `sources`, `delta`, `done`, `retracted`.
    """
    started = time.time()
    version = await audit.get_acl_version(session)
    key = cache_key(question, principal, version)

    hit = cache_get(key)
    if hit is not None:
        yield "delta", hit.answer
        yield "done", {
            "citations": [c.__dict__ for c in hit.citations],
            "refused": hit.refused,
            "cached": True,
            "latency_ms": hit.latency_ms,
        }
        return

    context, _ = await gather_context(session, principal, question)

    if not context.chunks:
        result = await _finalise(
            session, principal, question, prompts.REFUSAL_TEXT, context, started
        )
        cache_put(key, result)
        yield "delta", result.answer
        yield "done", {"citations": [], "refused": True, "cached": False}
        return

    # Announce which sources are in play before the text arrives, so the UI can
    # show provenance while the answer is still being written.
    yield "sources", [
        {
            "key": c.citation_key,
            "title": c.title,
            "doc_type": c.doc_type,
            "heading_path": c.heading_path,
        }
        for c in context.chunks
    ]

    parts: list[str] = []
    async for delta in llm.stream_completion(
        prompts.build_system_prompt(principal),
        prompts.build_user_prompt(question, context.chunks),
        max_tokens=1200,
    ):
        parts.append(delta)
        yield "delta", delta

    result = await _finalise(
        session, principal, question, "".join(parts), context, started
    )
    cache_put(key, result)

    if result.retracted:
        yield "retracted", {"answer": result.answer, "reason": "citation_check_failed"}
        return

    yield "done", {
        "citations": [c.__dict__ for c in result.citations],
        "refused": result.refused,
        "cached": False,
        "latency_ms": result.latency_ms,
    }
