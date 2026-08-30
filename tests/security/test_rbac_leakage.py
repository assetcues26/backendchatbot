"""The red-team suite: prove information cannot cross a role boundary.

Every test here runs the real retrieval query against a real Postgres. No
mocks, no fakes on the security path -- if the SQL is wrong, these fail.

The assertions are phrased as "this exact phrase must never come back", using
distinctive tokens planted in the fixture documents. That makes a failure
unambiguous: either the words appeared or they did not.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select, text

from app.core.principal import Principal
from app.db.models import Chunk, DocStatus, GrantEffect, UserDocumentGrant
from app.rag.retrieval import authorized_document_ids, hybrid_search, retrieve
from tests.security.conftest import fake_embedding, principal_for

pytestmark = pytest.mark.integration

# Phrases that exist in exactly one fixture document each.
PARTNER_ENVELOPE = "Partner Entitlement Envelope"
BACKEND_RUNBOOK = "controlled backend runbook"
ROADMAP_ID = "UAP-RM-001"
REQUIREMENT_ID = "UAP-FR-045"
TEST_CASE_ID = "UAP-TC-047"
CUSTOMER_SAFE = "Asset Category"
PENDING_TOKEN = "SECRETPENDINGTOKEN"
GLOBEX_TOKEN = "GLOBEXONLYTOKEN"

SEARCH = {
    "envelope": "What is the Partner Entitlement Envelope?",
    "roadmap": "Is there an access-administration audit history?",
    "requirement": "What does UAP-FR-045 say about editing a custom Profile?",
    "testcase": "What does test UAP-TC-047 verify about Legal Entity isolation?",
    "category": "How do I create an Asset Category?",
    "pricing": "Tell me about the draft pricing tier",
}


async def search_as(session, principal: Principal, query: str) -> str:
    """Run the real retrieval path and return the concatenated visible text."""
    result = await retrieve(
        session,
        principal,
        query,
        fake_embedding(query),
        top_k=12,
        candidates=50,
        rrf_k=60,
        collect_blocked=False,
    )
    return "\n".join(c.text for c in result.chunks)


# ---------------------------------------------------------------------------
# The headline case: the License Management BRD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["sales", "product", "engineering", "admin"])
async def test_authorised_roles_can_read_the_licence_commercial_model(
    session, world, role
) -> None:
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["envelope"])
    assert PARTNER_ENVELOPE in body, f"{role} should be able to read the License BRD"


@pytest.mark.parametrize("role", ["qa", "support", "customer", "noroles"])
async def test_unauthorised_roles_never_see_the_licence_commercial_model(
    session, world, role
) -> None:
    """The demo's headline claim, asserted rather than described."""
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["envelope"])
    assert PARTNER_ENVELOPE not in body
    assert BACKEND_RUNBOOK not in body


# ---------------------------------------------------------------------------
# Roadmap material must not reach Sales or Customer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["sales", "customer"])
async def test_roadmap_material_is_withheld_from_sales_and_customers(
    session, world, role
) -> None:
    """The documents themselves forbid presenting roadmap items as shipped."""
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["roadmap"])
    assert ROADMAP_ID not in body


@pytest.mark.parametrize("role", ["engineering", "product", "qa", "support"])
async def test_specifications_reach_the_roles_the_documents_name(
    session, world, role
) -> None:
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["requirement"])
    assert REQUIREMENT_ID in body


async def test_sales_cannot_read_specifications(session, world) -> None:
    principal = await principal_for(session, world["users"]["sales"])
    body = await search_as(session, principal, SEARCH["requirement"])
    assert REQUIREMENT_ID not in body


# ---------------------------------------------------------------------------
# QA material
# ---------------------------------------------------------------------------


async def test_qa_can_read_test_material(session, world) -> None:
    principal = await principal_for(session, world["users"]["qa"])
    body = await search_as(session, principal, SEARCH["testcase"])
    assert TEST_CASE_ID in body


@pytest.mark.parametrize("role", ["sales", "support", "customer"])
async def test_test_material_is_withheld_from_sales_support_and_customers(
    session, world, role
) -> None:
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["testcase"])
    assert TEST_CASE_ID not in body


# ---------------------------------------------------------------------------
# Customers can still be useful
# ---------------------------------------------------------------------------


async def test_customer_can_read_the_user_guide(session, world) -> None:
    """Isolation that blocks everything is easy and useless."""
    principal = await principal_for(session, world["users"]["customer"])
    body = await search_as(session, principal, SEARCH["category"])
    assert CUSTOMER_SAFE in body


# ---------------------------------------------------------------------------
# G1 - default deny
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role", ["admin", "product", "engineering", "qa", "sales", "support", "customer"]
)
async def test_a_pending_document_is_readable_by_nobody(session, world, role) -> None:
    """Even with an ACL granting every role, PENDING_REVIEW reaches no one."""
    principal = await principal_for(session, world["users"][role])
    body = await search_as(session, principal, SEARCH["pricing"])
    assert PENDING_TOKEN not in body


async def test_approving_a_document_is_what_makes_it_readable(session, world) -> None:
    doc = world["docs"]["pending_doc"]
    principal = await principal_for(session, world["users"]["customer"])

    assert PENDING_TOKEN not in await search_as(session, principal, SEARCH["pricing"])

    doc.status = DocStatus.APPROVED
    await session.commit()

    assert PENDING_TOKEN in await search_as(session, principal, SEARCH["pricing"])


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


async def test_one_customer_cannot_read_another_customers_document(
    session, world
) -> None:
    acme = await principal_for(session, world["users"]["customer"])
    body = await search_as(session, acme, "What identifier is used for asset classes?")
    assert GLOBEX_TOKEN not in body


async def test_the_owning_tenant_can_read_its_own_document(session, world) -> None:
    globex = await principal_for(session, world["users"]["globex_customer"])
    body = await search_as(session, globex, "What identifier is used for asset classes?")
    assert GLOBEX_TOKEN in body


async def test_internal_staff_do_not_see_customer_tenant_documents(
    session, world
) -> None:
    """Tenancy is a hard boundary in both directions, not a hierarchy."""
    admin = await principal_for(session, world["users"]["admin"])
    body = await search_as(session, admin, "What identifier is used for asset classes?")
    assert GLOBEX_TOKEN not in body


# ---------------------------------------------------------------------------
# Per-user overrides
# ---------------------------------------------------------------------------


async def test_an_explicit_deny_beats_a_role_grant(session, world) -> None:
    user = world["users"]["sales"]
    principal = await principal_for(session, user)
    assert PARTNER_ENVELOPE in await search_as(session, principal, SEARCH["envelope"])

    session.add(
        UserDocumentGrant(
            user_id=user.id,
            document_id=world["docs"]["license_brd"].id,
            effect=GrantEffect.DENY,
            reason="under investigation",
        )
    )
    await session.commit()

    assert PARTNER_ENVELOPE not in await search_as(
        session, principal, SEARCH["envelope"]
    )


async def test_an_explicit_allow_grants_one_document_without_a_role(
    session, world
) -> None:
    user = world["users"]["support"]
    principal = await principal_for(session, user)
    assert PARTNER_ENVELOPE not in await search_as(
        session, principal, SEARCH["envelope"]
    )

    session.add(
        UserDocumentGrant(
            user_id=user.id,
            document_id=world["docs"]["license_brd"].id,
            effect=GrantEffect.ALLOW,
            reason="temporary escalation",
        )
    )
    await session.commit()

    assert PARTNER_ENVELOPE in await search_as(session, principal, SEARCH["envelope"])


async def test_an_expired_allow_grant_does_not_apply(session, world) -> None:
    from datetime import UTC, datetime, timedelta

    user = world["users"]["support"]
    principal = await principal_for(session, user)
    session.add(
        UserDocumentGrant(
            user_id=user.id,
            document_id=world["docs"]["license_brd"].id,
            effect=GrantEffect.ALLOW,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
    )
    await session.commit()

    assert PARTNER_ENVELOPE not in await search_as(
        session, principal, SEARCH["envelope"]
    )


# ---------------------------------------------------------------------------
# Revocation takes effect immediately
# ---------------------------------------------------------------------------


async def test_revoking_a_role_takes_effect_on_the_very_next_query(
    session, world
) -> None:
    """A token issued before the revocation must not still work.

    This holds because the Principal is rebuilt from the database on every
    request; the JWT supplies only a user id.
    """
    from app.db.models import UserRole

    user = world["users"]["sales"]
    assert PARTNER_ENVELOPE in await search_as(
        session, await principal_for(session, user), SEARCH["envelope"]
    )

    await session.execute(
        UserRole.__table__.delete().where(UserRole.user_id == user.id)
    )
    await session.commit()

    assert PARTNER_ENVELOPE not in await search_as(
        session, await principal_for(session, user), SEARCH["envelope"]
    )


async def test_a_user_with_no_roles_can_read_nothing(session, world) -> None:
    principal = await principal_for(session, world["users"]["noroles"])
    assert principal.clearance == 0
    for query in SEARCH.values():
        assert await search_as(session, principal, query) == ""


# ---------------------------------------------------------------------------
# Clearance ceiling is independent of the grant
# ---------------------------------------------------------------------------


async def test_granting_a_role_a_document_above_its_clearance_has_no_effect(
    session, world
) -> None:
    """Two conditions must both hold. A stray ACL row alone is not enough."""
    from app.db.models import DocumentACL

    session.add(
        DocumentACL(
            document_id=world["docs"]["license_brd"].id,
            role_id=world["roles"]["customer"].id,
        )
    )
    await session.commit()

    principal = await principal_for(session, world["users"]["customer"])
    assert principal.clearance == 2
    assert PARTNER_ENVELOPE not in await search_as(
        session, principal, SEARCH["envelope"]
    )


# ---------------------------------------------------------------------------
# G4 - the independent re-verification agrees with the query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role", ["admin", "product", "engineering", "qa", "sales", "support", "customer"]
)
async def test_retrieval_and_reverification_never_disagree(session, world, role) -> None:
    principal = await principal_for(session, world["users"][role])
    for query in SEARCH.values():
        chunks = await hybrid_search(
            session, principal, query, fake_embedding(query),
            top_k=12, candidates=50, rrf_k=60,
        )
        if not chunks:
            continue
        allowed = await authorized_document_ids(
            session, principal, list({c.document_id for c in chunks})
        )
        assert {c.document_id for c in chunks} <= allowed, (
            f"G3 returned chunks that G4 rejects for role {role!r}"
        )


async def test_blocked_documents_are_reported_for_the_audit_trail(
    session, world
) -> None:
    """The audit console needs to know what was withheld, even though the
    user is never told."""
    principal = await principal_for(session, world["users"]["customer"])
    result = await retrieve(
        session, principal, SEARCH["envelope"], fake_embedding(SEARCH["envelope"]),
        top_k=12, candidates=50, rrf_k=60, collect_blocked=True,
    )
    assert not any(PARTNER_ENVELOPE in c.text for c in result.chunks)
    assert any("License" in b.title for b in result.blocked)
    assert result.blocked_chunk_count > 0


# ---------------------------------------------------------------------------
# Deletion really deletes
# ---------------------------------------------------------------------------


async def test_deleting_a_document_removes_every_chunk_and_embedding(
    session, world
) -> None:
    from app.ingest.pipeline import delete_document

    doc = world["docs"]["license_brd"]
    doc_id = doc.id
    admin = await principal_for(session, world["users"]["admin"])

    before = (
        await session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
        )
    ).scalar_one()
    assert before > 0

    await delete_document(session, doc, admin)
    await session.commit()

    after = (
        await session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
        )
    ).scalar_one()
    assert after == 0

    orphans = (
        await session.execute(
            text(
                "SELECT count(*) FROM chunks c "
                "LEFT JOIN documents d ON d.id = c.document_id WHERE d.id IS NULL"
            )
        )
    ).scalar_one()
    assert orphans == 0

    sales = await principal_for(session, world["users"]["sales"])
    assert PARTNER_ENVELOPE not in await search_as(session, sales, SEARCH["envelope"])


async def test_deleting_a_document_leaves_no_dangling_acl_rows(session, world) -> None:
    from app.db.models import DocumentACL
    from app.ingest.pipeline import delete_document

    doc = world["docs"]["uam_spec"]
    doc_id = doc.id
    admin = await principal_for(session, world["users"]["admin"])
    await delete_document(session, doc, admin)
    await session.commit()

    remaining = (
        await session.execute(
            select(func.count())
            .select_from(DocumentACL)
            .where(DocumentACL.document_id == doc_id)
        )
    ).scalar_one()
    assert remaining == 0


async def test_the_audit_trail_survives_deletion(session, world) -> None:
    """What was deleted and by whom must remain answerable."""
    from app.db.models import AuditEvent, DocumentVersion
    from app.ingest.pipeline import delete_document

    doc = world["docs"]["uam_tests"]
    doc_id = doc.id
    admin = await principal_for(session, world["users"]["admin"])
    await delete_document(session, doc, admin)
    await session.commit()

    versions = (
        await session.execute(
            select(func.count())
            .select_from(DocumentVersion)
            .where(DocumentVersion.document_id == doc_id)
        )
    ).scalar_one()
    events = (
        await session.execute(
            select(func.count())
            .select_from(AuditEvent)
            .where(AuditEvent.document_id == doc_id)
        )
    ).scalar_one()
    assert versions >= 1
    assert events >= 1


# ---------------------------------------------------------------------------
# Injected identity cannot influence retrieval
# ---------------------------------------------------------------------------


async def test_a_forged_principal_still_cannot_exceed_the_database(
    session, world
) -> None:
    """Even handed a fabricated Principal, the query is bounded by real rows.

    A caller cannot reach this far -- identity comes from the JWT -- but this
    proves the SQL does not trust its inputs either: claiming clearance 4 and
    every role id yields nothing, because no ACL row matches those ids for the
    customer's tenant.
    """
    real = await principal_for(session, world["users"]["customer"])
    forged = Principal(
        user_id=real.user_id,
        email=real.email,
        tenant_id=real.tenant_id,
        tenant_slug=real.tenant_slug,
        role_ids=frozenset(range(1, 50)),
        role_keys=frozenset({"admin"}),
        clearance=4,
    )
    body = await search_as(session, forged, SEARCH["envelope"])
    assert PARTNER_ENVELOPE not in body, (
        "documents in another tenant must stay unreachable regardless of "
        "claimed roles"
    )


async def test_a_nonexistent_tenant_returns_nothing(session, world) -> None:
    ghost = Principal(
        user_id=uuid.uuid4(),
        email="ghost@example.com",
        tenant_id=uuid.uuid4(),
        tenant_slug="ghost",
        role_ids=frozenset({world["roles"]["admin"].id}),
        role_keys=frozenset({"admin"}),
        clearance=4,
    )
    for query in SEARCH.values():
        assert await search_as(session, ghost, query) == ""
