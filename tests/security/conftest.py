"""Fixtures for the database-backed security tests.

These tests need a real Postgres with pgvector, because the thing under test
*is* the SQL. Mocking the database here would test nothing: the access
predicate lives in the query.

They do not need OpenAI. Embeddings are deterministic fakes, so the suite is
free to run and fast enough for every commit.

Set TEST_DATABASE_URL -- deliberately NOT DATABASE_URL -- to a scratch
Postgres. CI provides pgvector/pgvector:pg16 as a service container. Without
it the whole module skips rather than failing.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.principal import Principal
from app.db.models import (
    Base,
    Chunk,
    DocStatus,
    Document,
    DocumentACL,
    Role,
    Tenant,
    TenantKind,
    User,
    UserRole,
)
from app.db.seed import seed_roles, seed_system_state

EMBEDDING_DIM = 1536


def fake_embedding(text_value: str) -> list[float]:
    """Deterministic pseudo-embedding.

    Derived from a hash of the text, so identical text embeds identically and
    different text lands far apart. Good enough to exercise ranking; the point
    of these tests is the ACL filter, not retrieval quality.
    """
    digest = hashlib.sha256(text_value.encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    vector: list[float] = []
    state = seed
    for _ in range(EMBEDDING_DIM):
        state = (state * 6364136223846793005 + 1442695040888963407) % (2**64)
        vector.append(((state >> 33) / float(2**31)) - 1.0)
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def _database_url() -> str | None:
    """The scratch database to test against, or None to skip.

    Read from TEST_DATABASE_URL and never from DATABASE_URL. That separation
    is the whole protection and it is not a style choice -- see the note
    below.
    """
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url:
        return None

    # The placeholder from tests/conftest.py means "nothing real was
    # configured", so treat it as absent rather than failing on connect.
    if "postgres:postgres@localhost:5432/postgres" in url:
        return None

    return url


def _refuses_to_run_against(url: str) -> str:
    """Why this database must not be used, or an empty string if it is fine."""
    working = os.environ.get("DATABASE_URL", "")
    if working and url.strip() == working.strip():
        return (
            "TEST_DATABASE_URL is the same database as DATABASE_URL. This "
            "suite drops every table; pointing it at the working database "
            "destroys the corpus."
        )
    return ""


# THIS SUITE IS DESTRUCTIVE. It drops and recreates every table.
#
# It reads TEST_DATABASE_URL, never DATABASE_URL, and refuses to run when the
# two are equal. That is deliberate, and it is the third design:
#
#   1. A row-count guard could not tell leftover fixtures from real documents.
#   2. A separate Postgres schema was silently defeated by `create_all`
#      finding `public.chunks` through the search_path and creating nothing.
#   3. An opt-in flag -- ALLOW_DESTRUCTIVE_TESTS=1 -- against DATABASE_URL.
#
# The third looked sufficient and was not. The flag says "yes, I accept this
# is destructive"; it says nothing about WHICH database is configured, and the
# configured one is the working database by default. Setting the flag while
# .env pointed at the live project dropped 23 enriched documents and 806
# chunks, and every part of that was working as designed.
#
# A confirmation flag cannot fix that, because the mistake is not "I forgot it
# was destructive" -- it is "I forgot what DATABASE_URL was pointing at". So
# the address is separate now: this suite can only ever reach a database that
# was configured for it and nothing else.
#
#     TEST_DATABASE_URL=postgresql+asyncpg://...
#     ALLOW_DESTRUCTIVE_TESTS=1
#     pytest tests/security
#
# The flag stays as a second hand on the wheel. The separate URL is the brake.
_schema_ready = False

# Every table the fixtures write to, ordered so TRUNCATE ... CASCADE is cheap.
_DATA_TABLES = (
    "chunks",
    "document_acl",
    "user_document_grants",
    "document_versions",
    "documents",
    "user_roles",
    "users",
    "tenants",
    "audit_log",
    "access_requests",
    "roles",
    "system_state",
)


@pytest_asyncio.fixture(scope="function")
async def session() -> AsyncIterator[AsyncSession]:
    url = _database_url()
    if url is None:
        pytest.skip(
            "set TEST_DATABASE_URL to a SCRATCH Postgres with pgvector to run "
            "these. It is deliberately not DATABASE_URL: this suite drops "
            "every table."
        )

    refusal = _refuses_to_run_against(url)
    if refusal:
        # Not a skip. A skip is what someone scrolls past; this is a mistake
        # that costs the corpus, so it stops the run.
        pytest.fail(refusal, pytrace=False)

    if not os.environ.get("ALLOW_DESTRUCTIVE_TESTS"):
        pytest.skip(
            "this suite DROPS EVERY TABLE in TEST_DATABASE_URL. Re-run with "
            "ALLOW_DESTRUCTIVE_TESTS=1 once you are sure that database is "
            "scratch -- never against one holding your documents."
        )

    global _schema_ready
    # A new engine per test. pytest-asyncio runs each test on its own event
    # loop, and asyncpg connections are bound to the loop that opened them, so
    # a shared pool deadlocks. NullPool keeps nothing to leak across loops.
    engine = create_async_engine(
        url, connect_args={"statement_cache_size": 0}, poolclass=NullPool
    )

    try:
        async with engine.begin() as conn:
            if not _schema_ready:
                # Build once per session. Rebuilding per test costs ~70s each
                # against a remote pooler, and a security suite nobody runs is
                # decorative.
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                        "ON chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                )
                _schema_ready = True
            else:
                await conn.execute(
                    text(
                        f"TRUNCATE TABLE {', '.join(_DATA_TABLES)} "
                        f"RESTART IDENTITY CASCADE"
                    )
                )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database unavailable or missing pgvector: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        await seed_system_state(s)
        await seed_roles(s)
        await s.commit()
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def world(session: AsyncSession) -> dict[str, object]:
    """A miniature AssetCues corpus with the real access matrix applied.

    Four documents that mirror the shape of the real set, each carrying a
    distinctive phrase so a leak is unambiguous in an assertion.
    """
    roles = {
        r.key: r for r in (await session.execute(_select_roles())).scalars().all()
    }

    internal = Tenant(slug="assetcues", name="AssetCues", kind=TenantKind.INTERNAL)
    acme = Tenant(slug="acme", name="Acme Corp", kind=TenantKind.CUSTOMER)
    globex = Tenant(slug="globex", name="Globex", kind=TenantKind.CUSTOMER)
    session.add_all([internal, acme, globex])
    await session.flush()

    docs: dict[str, Document] = {}

    async def add_document(
        name: str,
        *,
        tenant: Tenant,
        title: str,
        doc_type: str,
        sensitivity: int,
        role_keys: list[str],
        body: str,
        status: DocStatus = DocStatus.APPROVED,
    ) -> Document:
        doc = Document(
            tenant_id=tenant.id,
            # Mirrors app/ingest/pipeline.py: documentation authored in the
            # AssetCues tenant is product documentation and reaches every
            # customer, subject to sensitivity and ACL. Customer-owned
            # material stays scoped to its tenant.
            is_shared=tenant.kind == TenantKind.INTERNAL,
            title=title,
            source_filename=f"{name}.docx",
            source_key=f"{name}.docx",
            module="Test Module",
            doc_type=doc_type,
            content_sha256=hashlib.sha256(body.encode()).hexdigest(),
            sensitivity=sensitivity,
            status=status,
        )
        session.add(doc)
        await session.flush()

        for index, paragraph in enumerate(body.strip().split("\n\n")):
            session.add(
                Chunk(
                    document_id=doc.id,
                    tenant_id=tenant.id,
                    ordinal=index,
                    heading_path=f"Section {index + 1}",
                    text=paragraph.strip(),
                    text_sha256=hashlib.sha256(paragraph.encode()).hexdigest(),
                    token_count=len(paragraph.split()),
                    embedding=fake_embedding(paragraph),
                )
            )

        for key in role_keys:
            session.add(DocumentACL(document_id=doc.id, role_id=roles[key].id))

        docs[name] = doc
        return doc

    # Sensitivity 4. The demo's crown jewel: Sales needs it, QA and Support
    # must not have it, Customer certainly must not.
    await add_document(
        "license_brd",
        tenant=internal,
        title="License Management BRD",
        doc_type="Business Requirements Document",
        sensitivity=4,
        role_keys=["admin", "product", "engineering", "sales"],
        body=(
            "The Partner Entitlement Envelope is the AssetCues-defined ceiling "
            "for partner-available modules and aggregate allocated quantities.\n\n"
            "An Entitlement Reduction requires a controlled backend runbook "
            "executed by authorised Technical and DevOps personnel."
        ),
    )

    # Sensitivity 3, with roadmap material Sales must not quote as shipped.
    await add_document(
        "uam_spec",
        tenant=internal,
        title="User Access & Permission Management Specification",
        doc_type="Product & Functional Specification",
        sensitivity=3,
        role_keys=["admin", "product", "engineering", "qa", "support"],
        body=(
            "UAP-FR-045 allows an authorized administrator to edit or delete an "
            "eligible custom Profile even when that administrator was not the "
            "original creator.\n\n"
            "Roadmap UAP-RM-001 describes a comprehensive access-administration "
            "audit history. It is not currently available and must not be "
            "presented as shipped."
        ),
    )

    # Sensitivity 3, QA only among the non-admin roles.
    await add_document(
        "uam_tests",
        tenant=internal,
        title="User Access Validation & Governance Pack",
        doc_type="Validation & Governance Pack",
        sensitivity=3,
        role_keys=["admin", "product", "engineering", "qa"],
        body=(
            "UAP-TC-047 creates a Permission Group in Legal Entity A and searches "
            "for it from Legal Entity B. The expected result is that the group is "
            "unavailable in Legal Entity B."
        ),
    )

    # Sensitivity 2. The only thing a customer can read.
    await add_document(
        "taxonomy_guide",
        tenant=internal,
        title="Asset Taxonomy User & Administrator Guide",
        doc_type="User & Administrator Guide",
        sensitivity=2,
        role_keys=[
            "admin", "product", "engineering", "qa", "sales", "support", "customer",
        ],
        body=(
            "To create an Asset Category, open the Asset Category page from an "
            "authorized Group profile and enter the Code and Name. AssetCues "
            "treats the Code and Name pair as the unique combination."
        ),
    )

    # Quarantined: classified but never approved. Nobody may read it (G1).
    await add_document(
        "pending_doc",
        tenant=internal,
        title="Unapproved Draft Pricing Model",
        doc_type="Business Requirements Document",
        sensitivity=2,
        role_keys=["admin", "customer", "sales", "support", "qa", "engineering", "product"],
        status=DocStatus.PENDING_REVIEW,
        body="The unapproved draft mentions a SECRETPENDINGTOKEN pricing tier.",
    )

    # A different customer's document, to prove tenant isolation.
    await add_document(
        "globex_doc",
        tenant=globex,
        title="Globex Onboarding Notes",
        doc_type="User Manual",
        sensitivity=2,
        role_keys=["admin", "customer", "support"],
        body="Globex uses the GLOBEXONLYTOKEN identifier for its asset classes.",
    )

    users: dict[str, User] = {}

    async def add_user(
        key: str, email: str, tenant: Tenant, role_keys: list[str]
    ) -> User:
        user = User(
            id=uuid.uuid4(), tenant_id=tenant.id, email=email, display_name=key
        )
        session.add(user)
        for role_key in role_keys:
            session.add(UserRole(user_id=user.id, role_id=roles[role_key].id))
        users[key] = user
        return user

    await add_user("admin", "admin@assetcues.com", internal, ["admin"])
    await add_user("engineering", "eng@assetcues.com", internal, ["engineering"])
    await add_user("product", "product@assetcues.com", internal, ["product"])
    await add_user("qa", "qa@assetcues.com", internal, ["qa"])
    await add_user("sales", "sales@assetcues.com", internal, ["sales"])
    await add_user("support", "support@assetcues.com", internal, ["support"])
    await add_user("customer", "buyer@acme.example", acme, ["customer"])
    await add_user("globex_customer", "buyer@globex.example", globex, ["customer"])
    await add_user("noroles", "nobody@assetcues.com", internal, [])

    await session.commit()

    return {"docs": docs, "users": users, "roles": roles, "tenants": {
        "internal": internal, "acme": acme, "globex": globex
    }}


def _select_roles():  # noqa: ANN202 - tiny helper
    from sqlalchemy import select

    return select(Role)


async def principal_for(session: AsyncSession, user: User) -> Principal:
    """Build a Principal the same way the auth layer does, from the database."""
    from app.core.security import load_principal

    return await load_principal(user.id, session)
