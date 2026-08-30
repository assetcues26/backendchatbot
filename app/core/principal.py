"""The Principal: who is asking, resolved server-side.

This is the single source of truth for identity in a request. It is built
*only* from a verified JWT plus a database lookup. Nothing in an HTTP body,
query string, or header other than `Authorization` can influence it.

Guardrail G2 is the rule that every route must obey:

    A request may say what it wants to know.
    It may never say who it is.

If you find yourself adding `role`, `tenant_id`, `clearance`, or `user_id` to
a request schema, that is the bug. `tests/security/test_tampering.py` fails
the build when it happens.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# Role keys. Kept in sync with the `roles` table by
# tests/security/test_role_seed.py, which fails if the two ever diverge.
ROLE_ADMIN = "admin"
ROLE_ENGINEERING = "engineering"
ROLE_PRODUCT = "product"
ROLE_QA = "qa"
ROLE_SALES = "sales"
ROLE_SUPPORT = "support"
ROLE_CUSTOMER = "customer"

ALL_ROLE_KEYS = frozenset(
    {
        ROLE_ADMIN,
        ROLE_ENGINEERING,
        ROLE_PRODUCT,
        ROLE_QA,
        ROLE_SALES,
        ROLE_SUPPORT,
        ROLE_CUSTOMER,
    }
)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller. Immutable by construction."""

    user_id: uuid.UUID
    email: str
    tenant_id: uuid.UUID
    tenant_slug: str
    role_ids: frozenset[int] = field(default_factory=frozenset)
    role_keys: frozenset[str] = field(default_factory=frozenset)
    clearance: int = 0
    is_active: bool = True

    @property
    def is_admin(self) -> bool:
        return ROLE_ADMIN in self.role_keys

    @property
    def is_customer(self) -> bool:
        return ROLE_CUSTOMER in self.role_keys

    def has_any_role(self, *keys: str) -> bool:
        return bool(self.role_keys & set(keys))

    def audit_fields(self) -> dict[str, object]:
        return {
            "user_id": str(self.user_id),
            "email": self.email,
            "tenant_id": str(self.tenant_id),
            "roles": sorted(self.role_keys),
            "clearance": self.clearance,
        }

    def acl_fingerprint(self) -> str:
        """Stable identity of this caller's *permissions*, for cache keying.

        Two callers with the same fingerprint are entitled to byte-identical
        answers. Changing roles changes the fingerprint, so a cached answer
        can never survive a permission change (guardrail G7). The global
        `acl_version` is mixed in separately by the cache layer to cover
        changes to the documents rather than to the user.
        """
        roles = ",".join(sorted(self.role_keys))
        return f"{self.tenant_id}|{roles}|{self.clearance}"
