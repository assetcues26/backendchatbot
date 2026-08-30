"""Seed data: roles, the internal tenant, and the starting RBAC matrix.

The role/clearance table below is the authoritative definition. It is
duplicated in `app/ingest/classifier.py` for offline use; a test fails the
build if the two ever drift.

Clearance is a *ceiling*, not a grant. A role with clearance 4 can still read
nothing at all until a document_acl row exists for it. See
`app/rag/retrieval.py` for the full predicate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Role, Sensitivity, SystemState, Tenant, TenantKind

INTERNAL_TENANT_SLUG = "assetcues"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    key: str
    name: str
    clearance: int
    is_internal: bool
    description: str


DEFAULT_ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        "admin",
        "Administrator",
        4,
        True,
        "Manages users, roles and document access. Can read everything.",
    ),
    RoleSpec(
        "product",
        "Product",
        4,
        True,
        "Owns the specifications, including licensing and commercial structure.",
    ),
    RoleSpec(
        "engineering",
        "Engineering",
        4,
        True,
        "Builds the product, including the licensing and entitlement system, "
        "so it needs the commercial specification as well as the functional "
        "ones.",
    ),
    RoleSpec(
        "qa",
        "QA",
        3,
        True,
        "Validates the product. Reads specifications and all test material, "
        "but never commercial or partner material.",
    ),
    RoleSpec(
        "sales",
        "Sales",
        4,
        True,
        "Sells the product. Reads customer-facing guides and the commercial "
        "licensing model, but not specifications containing roadmap items.",
    ),
    RoleSpec(
        "support",
        "Customer Support",
        3,
        True,
        "Answers customer tickets. Reads guides and specifications for product "
        "truth, but not QA material or commercial licensing.",
    ),
    RoleSpec(
        "customer",
        "Customer",
        2,
        False,
        "An AssetCues customer. Reads user and administrator guides only.",
    ),
)


# --------------------------------------------------------------------------
# Starting document-access matrix, keyed by document type.
# --------------------------------------------------------------------------
#
# This is seed data an administrator overrides per document, not a hardcoded
# rule. The reasoning behind the non-obvious rows:
#
#   - Sales does NOT get specifications. They carry roadmap and deferred items,
#     and the documents themselves say roadmap material "is not to be presented
#     as available". A salesperson quoting UAP-RM-001 as shipped is a real
#     commercial risk.
#   - Support DOES get specifications. Every spec names Support in its own
#     "Primary audience" field, and ticket answers need product truth.
#   - The License Management BRD is RESTRICTED: partner entitlement envelopes,
#     commercial structure, and a backend DevOps reduction runbook. QA and
#     Support are excluded; Sales is included because it is their commercial
#     model.
#   - User guides are the only category that reaches Customer.

DOC_TYPE_ACCESS: dict[str, tuple[int, tuple[str, ...]]] = {
    "Product & Functional Specification": (
        int(Sensitivity.INTERNAL),
        ("admin", "product", "engineering", "qa", "support"),
    ),
    "Business Requirements Document": (
        int(Sensitivity.INTERNAL),
        ("admin", "product", "engineering", "qa", "support"),
    ),
    "User & Administrator Guide": (
        int(Sensitivity.CUSTOMER),
        ("admin", "product", "engineering", "qa", "sales", "support", "customer"),
    ),
    "User Manual": (
        int(Sensitivity.CUSTOMER),
        ("admin", "product", "engineering", "qa", "sales", "support", "customer"),
    ),
    "Test Cases": (
        int(Sensitivity.INTERNAL),
        ("admin", "product", "engineering", "qa"),
    ),
    "Validation & Governance Pack": (
        int(Sensitivity.INTERNAL),
        ("admin", "product", "engineering", "qa"),
    ),
    "Document": (int(Sensitivity.RESTRICTED), ("admin",)),
}

# Overrides keyed by a case-insensitive substring of the source filename.
# Checked before DOC_TYPE_ACCESS.
FILENAME_ACCESS_OVERRIDES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    (
        "license_management_brd",
        int(Sensitivity.RESTRICTED),
        ("admin", "product", "engineering", "sales"),
    ),
)


def suggest_access(doc_type: str, filename: str) -> tuple[int, list[str]]:
    """The seeded proposal for a document. Used by the CLI bulk loader."""
    lowered = filename.lower()
    for needle, sensitivity, roles in FILENAME_ACCESS_OVERRIDES:
        if needle in lowered:
            return sensitivity, list(roles)

    sensitivity, roles = DOC_TYPE_ACCESS.get(
        doc_type, (int(Sensitivity.RESTRICTED), ("admin",))
    )
    return sensitivity, list(roles)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


async def seed_roles(session: AsyncSession) -> dict[str, int]:
    """Insert missing roles, update drifted ones. Returns key -> id."""
    existing = {
        r.key: r for r in (await session.execute(select(Role))).scalars().all()
    }

    for spec in DEFAULT_ROLES:
        role = existing.get(spec.key)
        if role is None:
            session.add(
                Role(
                    key=spec.key,
                    name=spec.name,
                    clearance=spec.clearance,
                    is_internal=spec.is_internal,
                    description=spec.description,
                )
            )
        else:
            role.name = spec.name
            role.clearance = spec.clearance
            role.is_internal = spec.is_internal
            role.description = spec.description

    await session.flush()
    rows = (await session.execute(select(Role))).scalars().all()
    return {r.key: r.id for r in rows}


async def seed_internal_tenant(session: AsyncSession) -> uuid.UUID:
    tenant = (
        await session.execute(
            select(Tenant).where(Tenant.slug == INTERNAL_TENANT_SLUG)
        )
    ).scalar_one_or_none()

    if tenant is None:
        tenant = Tenant(
            slug=INTERNAL_TENANT_SLUG,
            name="AssetCues",
            kind=TenantKind.INTERNAL,
        )
        session.add(tenant)
        await session.flush()

    return tenant.id


async def seed_system_state(session: AsyncSession) -> None:
    state = (
        await session.execute(select(SystemState).where(SystemState.id == 1))
    ).scalar_one_or_none()
    if state is None:
        session.add(SystemState(id=1, acl_version=1))
        await session.flush()


async def seed_all(session: AsyncSession) -> tuple[uuid.UUID, dict[str, int]]:
    await seed_system_state(session)
    role_ids = await seed_roles(session)
    tenant_id = await seed_internal_tenant(session)
    await session.commit()
    return tenant_id, role_ids
