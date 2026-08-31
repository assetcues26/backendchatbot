"""Guardrail G2: a request may say what it wants, never who it is.

This suite reads the actual API schemas and route signatures rather than
testing a handful of examples, so it keeps working as endpoints are added. If
someone adds `role` to a request body six months from now, this fails.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from app.api import schemas
from app.api.routes import admin, chat
from app.core.security import current_principal, require_admin

# Fields that would let a caller assert their own identity.
FORBIDDEN_CALLER_FIELDS = {
    "role",
    "roles",
    "role_key",
    "role_keys",
    "clearance",
    "tenant_id",
    "tenant",
    "is_admin",
    "principal",
    "acting_as",
    "sensitivity_override",
}

# Request models legitimately naming a *target* of an admin action. The caller
# is still resolved from the JWT; these say who the action is about.
ADMIN_TARGET_MODELS = {
    "ApproveDocumentIn",  # which roles get access to a document
    "CreateUserIn",  # the user being created
    "SetUserRolesIn",  # the roles being assigned to someone else
    "UserGrantIn",  # a per-user override
    "CreateTenantIn",
}


def _request_models() -> list[type[BaseModel]]:
    """Every pydantic model that could be used as a request body."""
    out = []
    for name, obj in vars(schemas).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and (name.endswith("In") or name.endswith("Request"))
        ):
            out.append(obj)
    return out


def test_request_models_exist() -> None:
    assert _request_models(), "no request models found; the scan would be vacuous"


@pytest.mark.parametrize("model", _request_models(), ids=lambda m: m.__name__)
def test_no_request_model_lets_caller_assert_identity(model: type[BaseModel]) -> None:
    if model.__name__ in ADMIN_TARGET_MODELS:
        pytest.skip(f"{model.__name__} names an admin action target, not the caller")

    offending = set(model.model_fields) & FORBIDDEN_CALLER_FIELDS
    assert not offending, (
        f"{model.__name__} exposes {sorted(offending)}. Identity must come from "
        f"the JWT via current_principal, never from the request body (G2). "
        f"See app/core/principal.py."
    )


def test_ask_request_carries_only_a_question_and_prior_questions() -> None:
    """Pin the chat request surface so a new field needs a deliberate change.

    `history` was added for follow-ups and holds the caller's own earlier
    QUESTIONS. It must never grow a slot for assistant answers: a client can
    put anything in a request body, and fabricated "the assistant said X"
    text would be attacker-controlled input going straight into the prompt.
    """
    assert set(schemas.AskRequest.model_fields) == {"question", "history"}

    history = schemas.AskRequest.model_fields["history"]
    assert history.annotation == list[str], (
        "history must be a flat list of question strings; a structured turn "
        "type would let an assistant answer be smuggled in"
    )


def _routes_of(module: object) -> list:
    return [r for r in module.router.routes if hasattr(r, "endpoint")]


@pytest.mark.parametrize("module", [chat, admin], ids=["chat", "admin"])
def test_every_route_resolves_identity_through_the_dependency(module: object) -> None:
    """No handler may take a Principal that did not come from the JWT."""
    router = module.router  # type: ignore[attr-defined]
    router_level = {
        d.dependency for d in getattr(router, "dependencies", []) if d.dependency
    }

    for route in _routes_of(module):
        signature = inspect.signature(route.endpoint)
        dependencies = {
            p.default.dependency
            for p in signature.parameters.values()
            if hasattr(p.default, "dependency")
        }
        # The health check and unauthenticated routes are declared elsewhere;
        # everything reachable from these two routers must be authenticated.
        secured = bool((dependencies | router_level) & {current_principal, require_admin})
        assert secured, (
            f"{route.path} does not depend on current_principal or "
            f"require_admin, so it has no verified caller identity."
        )


def test_admin_router_is_admin_only_at_the_router_level() -> None:
    """A new admin route must not be able to forget its own auth check."""
    deps = {d.dependency for d in admin.router.dependencies if d.dependency}
    assert require_admin in deps, (
        "the admin router must carry require_admin as a router-level dependency "
        "so that adding a route cannot accidentally leave it unprotected"
    )
