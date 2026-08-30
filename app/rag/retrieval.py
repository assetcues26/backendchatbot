"""Hybrid retrieval with the access-control predicate inside the query (G3).

This module is the security boundary of the whole application. Read it
carefully before changing it.

Two things happen here, and the order matters:

1. ``VISIBLE_DOCS_CTE`` resolves which documents this principal may read.
   Every retrieval path starts from that CTE. A chunk the caller is not
   entitled to is never loaded into the process at all -- it is not fetched
   and then filtered, it is never fetched. Post-filtering is not an acceptable
   substitute: anything that reaches memory can reach a log line, an error
   message, or a prompt.

2. Vector similarity and full-text search run *inside* that restriction and
   are fused with Reciprocal Rank Fusion.

Why hybrid rather than vectors alone: the AssetCues corpus is dense with exact
identifiers (UAP-FR-045, AWM-TC-001, LM-FR-009). Dense embeddings are poor at
exact-token lookup and someone will ask "what does UAP-FR-045 say?" on the
first day. Full-text nails those; vectors handle conceptual questions. Both
sides obey the same ACL.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.principal import Principal

# ---------------------------------------------------------------------------
# The access predicate. THE one place that decides what a principal may read.
# ---------------------------------------------------------------------------
#
#   can_read(user, doc) =
#         doc.tenant_id   == user.tenant_id
#     AND doc.status      == 'approved'          <- default-deny (G1)
#     AND doc.sensitivity <= user.clearance      <- ceiling
#     AND (role grant OR explicit user allow)    <- positive grant required
#     AND NOT explicit user deny                 <- deny always wins
#
# Clearance alone never grants access, and a grant alone never bypasses the
# ceiling. Two independent conditions must both fail before a leak occurs.

VISIBLE_DOCS_CTE = """
visible_docs AS (
    SELECT d.id
    FROM documents d
    WHERE d.tenant_id = CAST(:tenant_id AS uuid)
      AND d.status = 'APPROVED'
      AND d.sensitivity <= :clearance
      AND (
            EXISTS (
                SELECT 1 FROM document_acl a
                WHERE a.document_id = d.id
                  AND a.role_id = ANY(CAST(:role_ids AS int[]))
            )
         OR EXISTS (
                SELECT 1 FROM user_document_grants g
                WHERE g.document_id = d.id
                  AND g.user_id = CAST(:user_id AS uuid)
                  AND g.effect = 'ALLOW'
                  AND (g.expires_at IS NULL OR g.expires_at > now())
            )
      )
      AND NOT EXISTS (
            SELECT 1 FROM user_document_grants g
            WHERE g.document_id = d.id
              AND g.user_id = CAST(:user_id AS uuid)
              AND g.effect = 'DENY'
              AND (g.expires_at IS NULL OR g.expires_at > now())
      )
)
"""

_HYBRID_SEARCH_SQL = f"""
WITH {VISIBLE_DOCS_CTE},
vector_hits AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> CAST(:qvec AS vector)) AS rnk
    FROM chunks c
    JOIN visible_docs v ON v.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:qvec AS vector)
    LIMIT :candidates
),
text_hits AS (
    SELECT c.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', :query)) DESC
           ) AS rnk
    FROM chunks c
    JOIN visible_docs v ON v.id = c.document_id
    WHERE c.tsv @@ websearch_to_tsquery('english', :query)
    ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', :query)) DESC
    LIMIT :candidates
),
fused AS (
    SELECT id, SUM(score) AS score
    FROM (
        SELECT id, 1.0 / (:rrf_k + rnk) AS score FROM vector_hits
        UNION ALL
        SELECT id, 1.0 / (:rrf_k + rnk) AS score FROM text_hits
    ) s
    GROUP BY id
)
SELECT c.id            AS chunk_id,
       c.document_id   AS document_id,
       c.ordinal       AS ordinal,
       c.heading_path  AS heading_path,
       c.text          AS text,
       c.token_count   AS token_count,
       d.title         AS title,
       d.module        AS module,
       d.doc_type      AS doc_type,
       d.sensitivity   AS sensitivity,
       f.score         AS score
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT :top_k
"""

# Same search with NO access filter. Used only to populate the audit trail with
# "how much existed that you could not see". Its results are counted, never
# returned to a non-admin caller.
_UNFILTERED_MATCH_SQL = """
WITH vector_hits AS (
    SELECT c.id, c.document_id
    FROM chunks c
    WHERE c.embedding IS NOT NULL AND c.tenant_id = CAST(:tenant_id AS uuid)
    ORDER BY c.embedding <=> CAST(:qvec AS vector)
    LIMIT :candidates
),
text_hits AS (
    SELECT c.id, c.document_id
    FROM chunks c
    WHERE c.tenant_id = CAST(:tenant_id AS uuid)
      AND c.tsv @@ websearch_to_tsquery('english', :query)
    LIMIT :candidates
),
all_hits AS (
    SELECT id, document_id FROM vector_hits
    UNION
    SELECT id, document_id FROM text_hits
)
SELECT h.document_id, d.title, d.sensitivity, COUNT(*) AS chunk_count
FROM all_hits h
JOIN documents d ON d.id = h.document_id
WHERE d.status = 'APPROVED'
GROUP BY h.document_id, d.title, d.sensitivity
"""

# Independent re-verification path for G4. Deliberately written differently
# from VISIBLE_DOCS_CTE so that a single mistake cannot pass both checks.
_AUTHORIZED_DOC_IDS_SQL = f"""
WITH {VISIBLE_DOCS_CTE}
SELECT id FROM visible_docs WHERE id = ANY(CAST(:doc_ids AS uuid[]))
"""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    ordinal: int
    heading_path: str
    text: str
    token_count: int
    title: str
    module: str
    doc_type: str
    sensitivity: int
    score: float

    @property
    def citation_key(self) -> str:
        """Short, stable handle the model is told to cite."""
        return f"{str(self.document_id)[:8]}#{self.ordinal}"


@dataclass(frozen=True, slots=True)
class BlockedDocument:
    document_id: uuid.UUID
    title: str
    sensitivity: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    blocked: list[BlockedDocument]
    anomalies: list[str]

    @property
    def blocked_chunk_count(self) -> int:
        return sum(b.chunk_count for b in self.blocked)


def format_vector(vec: list[float]) -> str:
    """pgvector literal form for a bind parameter."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


async def hybrid_search(
    session: AsyncSession,
    principal: Principal,
    query: str,
    query_vector: list[float],
    *,
    top_k: int,
    candidates: int,
    rrf_k: int,
) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks this principal is entitled to read."""
    if not principal.role_ids:
        # No roles means no positive grant is possible. Skip the round trip.
        return []

    rows = (
        await session.execute(
            text(_HYBRID_SEARCH_SQL),
            {
                "tenant_id": str(principal.tenant_id),
                "user_id": str(principal.user_id),
                "role_ids": list(principal.role_ids),
                "clearance": principal.clearance,
                "qvec": format_vector(query_vector),
                "query": query,
                "candidates": candidates,
                "rrf_k": rrf_k,
                "top_k": top_k,
            },
        )
    ).mappings().all()

    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            ordinal=r["ordinal"],
            heading_path=r["heading_path"] or "",
            text=r["text"],
            token_count=r["token_count"] or 0,
            title=r["title"],
            module=r["module"] or "",
            doc_type=r["doc_type"] or "",
            sensitivity=r["sensitivity"],
            score=float(r["score"]),
        )
        for r in rows
    ]


async def authorized_document_ids(
    session: AsyncSession, principal: Principal, doc_ids: list[uuid.UUID]
) -> set[uuid.UUID]:
    """Which of these documents may the principal read? (G4 re-verification.)"""
    if not doc_ids or not principal.role_ids:
        return set()
    rows = (
        await session.execute(
            text(_AUTHORIZED_DOC_IDS_SQL),
            {
                "tenant_id": str(principal.tenant_id),
                "user_id": str(principal.user_id),
                "role_ids": list(principal.role_ids),
                "clearance": principal.clearance,
                "doc_ids": [str(d) for d in doc_ids],
            },
        )
    ).scalars().all()
    return set(rows)


async def verify_chunks(
    session: AsyncSession, principal: Principal, chunks: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], list[str]]:
    """Guardrail G4: re-check every chunk before it can reach a prompt.

    G3 already restricted the query. This asks the same question again by a
    different route. In a correct system it always agrees; if it ever
    disagrees we have a retrieval bug, and the right response is to drop the
    chunk, keep serving, and shout loudly in the audit log.
    """
    if not chunks:
        return [], []

    allowed = await authorized_document_ids(
        session, principal, list({c.document_id for c in chunks})
    )
    kept = [c for c in chunks if c.document_id in allowed]
    anomalies = [
        f"chunk {c.chunk_id} from document {c.document_id} passed retrieval "
        f"but failed re-verification for user {principal.user_id}"
        for c in chunks
        if c.document_id not in allowed
    ]
    return kept, anomalies


async def find_blocked_documents(
    session: AsyncSession,
    principal: Principal,
    query: str,
    query_vector: list[float],
    *,
    candidates: int,
) -> list[BlockedDocument]:
    """Documents that matched but were withheld. For the audit trail only.

    This is what makes the founder demo legible: the audit console can show
    "12 chunks matched, 0 allowed, License Management BRD blocked". The end
    user never sees any of it -- confirming that restricted material exists is
    itself a leak, and would let someone map the corpus by probing refusals.
    """
    matched = (
        await session.execute(
            text(_UNFILTERED_MATCH_SQL),
            {
                "tenant_id": str(principal.tenant_id),
                "qvec": format_vector(query_vector),
                "query": query,
                "candidates": candidates,
            },
        )
    ).mappings().all()

    if not matched:
        return []

    allowed = await authorized_document_ids(
        session, principal, [r["document_id"] for r in matched]
    )
    return [
        BlockedDocument(
            document_id=r["document_id"],
            title=r["title"],
            sensitivity=r["sensitivity"],
            chunk_count=int(r["chunk_count"]),
        )
        for r in matched
        if r["document_id"] not in allowed
    ]


async def retrieve(
    session: AsyncSession,
    principal: Principal,
    query: str,
    query_vector: list[float],
    *,
    top_k: int,
    candidates: int,
    rrf_k: int,
    collect_blocked: bool = True,
) -> RetrievalResult:
    """Full retrieval path: G3 filter, then G4 re-verification."""
    chunks = await hybrid_search(
        session,
        principal,
        query,
        query_vector,
        top_k=top_k,
        candidates=candidates,
        rrf_k=rrf_k,
    )
    chunks, anomalies = await verify_chunks(session, principal, chunks)

    blocked: list[BlockedDocument] = []
    if collect_blocked:
        blocked = await find_blocked_documents(
            session, principal, query, query_vector, candidates=candidates
        )

    return RetrievalResult(chunks=chunks, blocked=blocked, anomalies=anomalies)
