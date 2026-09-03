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
given, which is a correctness failure, not a disclosure. Its response is
graded accordingly: bad citation markers are stripped while any traceable
claim remains, and the answer is withheld only when nothing can be traced.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core import audit
from app.core.principal import Principal
from app.rag import guardrails, llm, prompts, routing
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
    # Id of this query's audit row. Feedback points at it, so the recorded
    # question and sources come from the server rather than the client.
    turn_id: str = ""
    citations: list[Citation] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    refused: bool = False
    retracted: bool = False
    latency_ms: int = 0
    chunks_used: int = 0
    blocked_document_count: int = 0
    blocked_chunk_count: int = 0
    anomalies: list[str] = field(default_factory=list)
    injection_findings: list[dict[str, object]] = field(default_factory=list)
    cached: bool = False
    # Which part of the product this was answered from, and -- when the
    # question fitted several equally well -- the choices offered instead
    # of a guess. A non-empty `clarify` means `answer` is a question.
    capability: str = ""
    clarify: list[str] = field(default_factory=list)


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


def cache_key(
    question: str, principal: Principal, acl_version: int, capability: str = ""
) -> str:
    normalised = " ".join(question.lower().split())
    # The scope is part of the key: the same question answered from two
    # capabilities is two different answers.
    material = (
        f"{normalised}|{principal.acl_fingerprint()}|v{acl_version}|{capability}"
    )
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


# How many earlier questions a follow-up may see. Five is enough to resolve
# "what about for sales?" without letting the prompt grow unbounded.
MAX_HISTORY = 5

# Only the user's own earlier QUESTIONS are accepted as conversation memory --
# never assistant answers. A client could fabricate an "assistant said X" turn,
# and anything we put in the prompt from that is attacker-controlled text. The
# user's own questions add no capability they did not already have, and are
# enough to resolve a reference.
# Word boundaries are load-bearing: without them "Entitlement" contains
# "it", every question looks like a follow-up, and the gate saves nothing.
_FOLLOWUP_HINT = re.compile(
    r"^\s*(and|also|what about|how about|why|so|then|ok|okay)\b"
    r"|\b(it|that|those|these|them|this|the same|there)\b",
    re.IGNORECASE,
)


def looks_like_a_follow_up(question: str) -> bool:
    """Cheap gate before spending a model call on rewriting.

    A long, self-contained question does not need condensing, and most do not.
    Skipping those keeps the common path to one model call.
    """
    return len(question) < 90 or bool(_FOLLOWUP_HINT.search(question))


async def condense_question(question: str, history: list[str]) -> str:
    """Rewrite a follow-up so retrieval sees a standalone question.

    Retrieval matches on the question text, so "what about for sales?" would
    otherwise retrieve nothing useful. Failure is non-fatal: we fall back to
    the original wording rather than lose the turn.
    """
    trimmed = [q.strip()[:500] for q in history if q and q.strip()][-MAX_HISTORY:]
    if not trimmed or not looks_like_a_follow_up(question):
        return question

    try:
        rewritten = await llm.complete(
            prompts.CONDENSE_SYSTEM_PROMPT,
            prompts.build_condense_prompt(question, trimmed),
            max_tokens=120,
        )
    except Exception:  # noqa: BLE001 - a rewrite is an optimisation, not a step
        return question

    rewritten = rewritten.strip().strip('"')
    # Guard against a model that decides to answer instead of rewrite.
    if not rewritten or len(rewritten) > 400:
        return question
    return rewritten


async def suggest_follow_ups(
    question: str, answer: str, chunks: list[RetrievedChunk]
) -> list[str]:
    """Propose next questions, grounded in what this caller could actually read.

    Generated from the chunks already retrieved for them, so a suggestion can
    never point at a document they would then be refused.
    """
    if not chunks or not answer.strip():
        return []

    headings = list(
        dict.fromkeys(
            f"{c.title} - {c.heading_path}" if c.heading_path else c.title
            for c in chunks
        )
    )[:8]

    try:
        raw = await llm.complete_json(
            prompts.FOLLOWUP_SYSTEM_PROMPT,
            prompts.build_followup_prompt(question, answer, headings),
            prompts.FOLLOWUP_SCHEMA,
            "follow_up_questions",
            max_tokens=200,
        )
        parsed = json.loads(raw).get("questions", [])
    except Exception:  # noqa: BLE001 - suggestions are a nicety, never a blocker
        return []

    out: list[str] = []
    for item in parsed:
        text = str(item).strip()
        if text and text.lower() != question.strip().lower() and len(text) < 160:
            out.append(text)
    return out[:3]


async def gather_context(
    session: AsyncSession,
    principal: Principal,
    question: str,
    capability: str = "",
) -> tuple[RetrievalResult, list[float]]:
    """Embed and retrieve. `question` must already be standalone."""
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
        capability=capability,
    )
    return result, vector


def search_scope(capability: str) -> str:
    """The value retrieval should filter on for a requested capability.

    `ALL_CAPABILITIES` means "no filter", not "a capability literally named
    `*`" -- passing it through would match nothing and refuse every question.
    """
    return "" if capability == routing.ALL_CAPABILITIES else capability


async def route_question(
    session: AsyncSession,
    principal: Principal,
    question: str,
    context: RetrievalResult,
    capability: str,
) -> tuple[RetrievalResult, routing.Route]:
    """Work out which part of the product to answer from.

    When the caller already chose one -- they clicked a chip on a clarifying
    question -- that choice stands and the retrieval was scoped to it already.

    Otherwise the routing runs over what came back. If the question named a
    capability outright, it is worth retrieving again scoped to it: the first
    pass spent most of its twelve slots on other capabilities, and the reader
    asked about one.
    """
    if capability == routing.ALL_CAPABILITIES:
        # They have seen the choices and want the lot. Do not ask again.
        return context, routing.Route(reason="the caller asked for every area")

    if capability:
        return context, routing.Route(
            capability=capability, reason="chosen by the caller"
        )

    decision = routing.route(question, context.chunks)

    if decision.named and decision.capability:
        scoped, _ = await gather_context(
            session,
            principal,
            question,
            capability=search_scope(decision.capability),
        )
        # An empty scoped result means the capability name matched something
        # this caller cannot read. Keep what they could.
        if scoped.chunks:
            return scoped, decision

    return context, decision


async def _finalise_clarification(
    session: AsyncSession,
    principal: Principal,
    question: str,
    decision: routing.Route,
    context: RetrievalResult,
    started: float,
) -> AnswerResult:
    """Return a question instead of an answer.

    No model call and no citations: nothing has been answered, and presenting
    it with sources would suggest otherwise. The arithmetic that triggered it
    goes into the audit row, so "why did it ask me that?" is answerable
    afterwards without re-running anything.
    """
    latency = int((time.time() - started) * 1000)
    event = await audit.record(
        session,
        audit.Event.QUERY,
        principal=principal,
        question=question[:500],
        chunks_used=len(context.chunks),
        documents_used=sorted({str(c.document_id) for c in context.chunks}),
        blocked_chunk_count=context.blocked_chunk_count,
        clarified=True,
        routing_reason=decision.reason,
        candidates=[
            {"capability": c.capability, "share": round(c.share, 3)}
            for c in decision.candidates
        ],
        latency_ms=latency,
    )
    await session.commit()

    return AnswerResult(
        answer=routing.clarification_question(decision.candidates),
        turn_id=str(event.id),
        clarify=decision.choices,
        latency_ms=latency,
        chunks_used=len(context.chunks),
        blocked_document_count=len(context.blocked),
        blocked_chunk_count=context.blocked_chunk_count,
    )


async def _finalise(
    session: AsyncSession,
    principal: Principal,
    question: str,
    raw_answer: str,
    context: RetrievalResult,
    started: float,
    follow_ups: list[str] | None = None,
    decision: routing.Route | None = None,
) -> AnswerResult:
    """Post-generation checks and the audit write. Shared by both paths."""
    check = guardrails.validate_citations(
        raw_answer, context.chunks, refusal_text=prompts.REFUSAL_TEXT
    )
    injection = guardrails.scan_chunks_for_injection(context.chunks)
    latency = int((time.time() - started) * 1000)

    # G6, proportionately.
    #
    # An untraceable citation is a sourcing failure, not a disclosure -- the
    # content itself was already bounded to authorised chunks by G3 and G4. The
    # common case is the model naming a real, authorised document but inventing
    # a chunk ordinal within it. Discarding an otherwise correct answer for
    # that, and telling the reader "I don't have information available to you",
    # states something false about their access.
    #
    # So: if any claim is still traceable, drop the bad markers and keep the
    # answer. Retract only when nothing at all can be traced.
    retracted = False
    answer = raw_answer
    if not check.ok:
        if check.has_traceable_support:
            answer = guardrails.strip_citations(raw_answer, check.invalid)
        else:
            retracted = True
            answer = prompts.UNVERIFIED_TEXT

    is_refusal = answer.strip() == prompts.REFUSAL_TEXT.strip()

    # Audit every sourcing failure, repaired or not. A rising count of
    # repaired answers is the signal that the model is drifting on citation
    # format, and it would be invisible if only retractions were recorded.
    if not check.ok:
        await audit.record(
            session,
            audit.Event.CITATION_REJECTED,
            principal=principal,
            severity="warning" if retracted else "info",
            question=question[:500],
            reason=check.failure_reason,
            invalid_keys=sorted(check.invalid),
            valid_keys=sorted(check.valid),
            outcome="retracted" if retracted else "invalid_citations_stripped",
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

    query_event = await audit.record(
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
        capability=decision.capability if decision else "",
        routing_reason=decision.reason if decision else "",
        latency_ms=latency,
    )
    await session.commit()

    return AnswerResult(
        answer=answer,
        turn_id=str(query_event.id),
        citations=(
            []
            if retracted or is_refusal
            else build_citations(context.chunks, check.valid)
        ),
        follow_ups=follow_ups or [],
        refused=is_refusal,
        retracted=retracted,
        latency_ms=latency,
        chunks_used=len(context.chunks),
        blocked_document_count=len(context.blocked),
        blocked_chunk_count=context.blocked_chunk_count,
        anomalies=context.anomalies,
        injection_findings=injection,
        capability=decision.capability if decision else "",
    )


async def answer_question(
    session: AsyncSession,
    principal: Principal,
    question: str,
    history: list[str] | None = None,
    capability: str = "",
) -> AnswerResult:
    """Non-streaming answer. Used by the role-comparison view and tests.

    `capability` scopes the search to one part of the product. It arrives from
    the client -- it is the chip someone clicked on a clarifying question --
    and can only ever narrow what the access predicate already allowed.
    """
    started = time.time()
    version = await audit.get_acl_version(session)

    # Resolve references against the caller's own earlier questions before
    # anything touches retrieval. The cache is keyed on the resolved text, so
    # two people asking the same follow-up from different threads do not
    # collide.
    resolved = await condense_question(question, history or [])
    key = cache_key(resolved, principal, version, capability)

    hit = cache_get(key)
    if hit is not None:
        return replace(hit, cached=True)

    context, _ = await gather_context(
        session, principal, resolved, capability=search_scope(capability)
    )

    if not context.chunks:
        result = await _finalise(
            session, principal, resolved, prompts.REFUSAL_TEXT, context, started
        )
        cache_put(key, result)
        return result

    context, decision = await route_question(
        session, principal, resolved, context, capability
    )
    if decision.needs_clarification:
        result = await _finalise_clarification(
            session, principal, resolved, decision, context, started
        )
        cache_put(key, result)
        return result

    raw = await llm.complete(
        prompts.build_system_prompt(principal),
        prompts.build_user_prompt(resolved, context.chunks),
        max_tokens=1200,
    )
    suggestions = await suggest_follow_ups(resolved, raw, context.chunks)
    result = await _finalise(
        session, principal, resolved, raw, context, started, suggestions, decision
    )
    cache_put(key, result)
    return result


async def stream_answer(
    session: AsyncSession,
    principal: Principal,
    question: str,
    history: list[str] | None = None,
    capability: str = "",
) -> AsyncIterator[tuple[str, object]]:
    """Yield (event_name, payload) pairs for the SSE endpoint.

    Events: `sources`, `delta`, `done`, `retracted`, `follow_ups`.

    A clarifying question arrives as a normal `delta` plus a `clarify` list on
    `done`, so a client that ignores the list still shows the question.
    """
    started = time.time()
    version = await audit.get_acl_version(session)

    resolved = await condense_question(question, history or [])
    key = cache_key(resolved, principal, version, capability)

    hit = cache_get(key)
    if hit is not None:
        yield "delta", hit.answer
        yield "done", {
            "turn_id": hit.turn_id,
            "citations": [asdict(c) for c in hit.citations],
            "follow_ups": hit.follow_ups,
            "refused": hit.refused,
            "clarify": hit.clarify,
            "capability": hit.capability,
            "cached": True,
            "latency_ms": hit.latency_ms,
        }
        return

    context, _ = await gather_context(
        session, principal, resolved, capability=search_scope(capability)
    )

    if not context.chunks:
        result = await _finalise(
            session, principal, resolved, prompts.REFUSAL_TEXT, context, started
        )
        cache_put(key, result)
        yield "delta", result.answer
        yield "done", {
            "turn_id": result.turn_id,
            "citations": [],
            "follow_ups": [],
            "refused": True,
            "clarify": [],
            "capability": "",
            "cached": False,
        }
        return

    context, decision = await route_question(
        session, principal, resolved, context, capability
    )
    if decision.needs_clarification:
        result = await _finalise_clarification(
            session, principal, resolved, decision, context, started
        )
        cache_put(key, result)
        yield "delta", result.answer
        yield "done", {
            "turn_id": result.turn_id,
            "citations": [],
            "follow_ups": [],
            "refused": False,
            "clarify": result.clarify,
            "capability": "",
            "cached": False,
        }
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
        prompts.build_user_prompt(resolved, context.chunks),
        max_tokens=1200,
    ):
        parts.append(delta)
        yield "delta", delta

    raw = "".join(parts)
    # Suggestions are generated after the text has streamed, so the reader is
    # never kept waiting on them.
    suggestions = await suggest_follow_ups(resolved, raw, context.chunks)
    result = await _finalise(
        session, principal, resolved, raw, context, started, suggestions, decision
    )
    cache_put(key, result)

    if result.retracted:
        yield "retracted", {
            "answer": result.answer,
            "reason": "no claim in the answer could be traced to a source",
        }
        return

    yield "done", {
        "turn_id": result.turn_id,
        "citations": [asdict(c) for c in result.citations],
        "follow_ups": result.follow_ups,
        "refused": result.refused,
        "clarify": [],
        "capability": result.capability,
        "cached": False,
        "latency_ms": result.latency_ms,
    }
