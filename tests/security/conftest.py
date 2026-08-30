"""Fixtures for the database-backed security tests.

These tests need a real Postgres with pgvector, because the thing under test
*is* the SQL. Mocking the database here would test nothing: the access
predicate lives in the query.

They do not need OpenAI. Embeddings are deterministic fakes, so the suite is
free to run and fast enough for every commit.

Set DATABASE_URL to run them. CI provides pgvector/pgvector:pg16 as a service
container; locally, point it at a scratch Supabase project. Without it the
whole module skips rather than failing.
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
    url = os.environ.get("DATABASE_URL", "")
    if not url or "localhost:5432/db" in url:
        return None
    return url


@pytest_asyncio.fixture(scope="function")
async def session() -> AsyncIterator[AsyncSession]:
    url = _database_url()
    if url is None:
        pytest.skip("set DATABASE_URL to a Postgres with pgvector to run these")

    engine = create_async_engine(url, connect_args={"statement_cache_size": 0})

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON chunks "
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
    except Exception as exc:  # pragma: no cover
        await engine.dispose()
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

        await session.flush()
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
        await session.flush()
        for role_key in role_keys:
            session.add(UserRole(user_id=user.id, role_id=roles[role_key].id))
        await session.flush()
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
