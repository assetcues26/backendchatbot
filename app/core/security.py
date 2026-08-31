"""JWT verification and Principal construction (guardrail G2).

Supabase signs access tokens either with an asymmetric project key
(ES256/RS256, published at the project's JWKS endpoint) or, on older projects,
with the shared HS256 secret. Both are verified here; the token's own header
decides which. We never trust unverified claims and we never read identity
from anywhere but this module.

The token gives us a user id. Everything that matters for access control --
tenant, roles, clearance, active status -- is then read from *our* database,
not from the token. A token issued before a role was revoked therefore cannot
grant the revoked access: the revocation is visible on the very next request.
That property is what makes the "revoke mid-conversation" red-team test pass.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.principal import Principal
from app.db.models import Role, Tenant, User, UserRole
from app.db.session import get_session

logger = logging.getLogger("assetcues.security")

_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)
_INACTIVE = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled"
)


# Supabase projects created since the 2025 signing-key change issue tokens
# signed with an asymmetric key (ES256/RS256) published at the project's JWKS
# endpoint. Older projects sign with the shared HS256 secret. Both are
# supported: the token's own header decides which path is used.
_JWKS_TTL_SECONDS = 600
_jwks_cache: list[dict[str, Any]] = []
_jwks_fetched_at = 0.0


async def _get_jwks(settings: Settings, *, force: bool = False) -> list[dict[str, Any]]:
    """Fetch and cache the project's public signing keys."""
    global _jwks_cache, _jwks_fetched_at

    fresh = time.time() - _jwks_fetched_at < _JWKS_TTL_SECONDS
    if _jwks_cache and fresh and not force:
        return _jwks_cache

    if not settings.supabase_url:
        return []

    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
            keys: list[dict[str, Any]] = list(payload.get("keys", []))
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("could not fetch JWKS from %s: %s", url, exc)
        return _jwks_cache  # stale keys beat no keys

    _jwks_cache = keys
    _jwks_fetched_at = time.time()
    return keys


async def decode_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify signature, expiry and audience. Raises 401 on any failure."""
    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise _UNAUTHENTICATED from exc

    algorithm = header.get("alg", "")
    options = {"require_exp": True, "require_sub": True}

    if algorithm.startswith("HS"):
        if not settings.supabase_jwt_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_JWT_SECRET is not configured",
            )
        try:
            return jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                options=options,
            )
        except JWTError as exc:
            raise _UNAUTHENTICATED from exc

    # Asymmetric. Find the key this token names; if we do not have it, the
    # project may have rotated, so refetch once before giving up.
    kid = header.get("kid")
    for force in (False, True):
        keys = await _get_jwks(settings, force=force)
        key = next((k for k in keys if k.get("kid") == kid), None)
        if key is None:
            continue
        try:
            return jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience="authenticated",
                options=options,
            )
        except JWTError as exc:
            raise _UNAUTHENTICATED from exc

    logger.warning("no JWKS key matches kid=%s (alg=%s)", kid, algorithm)
    raise _UNAUTHENTICATED


async def load_principal(user_id: uuid.UUID, session: AsyncSession) -> Principal:
    """Build the Principal from the database. The token is not consulted here."""
    user = (
        await session.execute(
            select(User).where(User.id == user_id).join(Tenant, User.tenant_id == Tenant.id)
        )
    ).scalar_one_or_none()

    if user is None:
        raise _UNAUTHENTICATED
    if not user.is_active:
        raise _INACTIVE

    rows = (
        await session.execute(
            select(Role).join(UserRole, UserRole.role_id == Role.id).where(
                UserRole.user_id == user_id
            )
        )
    ).scalars().all()

    tenant = (
        await session.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    ).scalar_one()

    if not tenant.is_active:
        raise _INACTIVE

    return Principal(
        user_id=user.id,
        email=user.email,
        tenant_id=user.tenant_id,
        tenant_slug=tenant.slug,
        role_ids=frozenset(r.id for r in rows),
        role_keys=frozenset(r.key for r in rows),
        # A user with no roles has clearance 0 and can read nothing at all.
        clearance=max((r.clearance for r in rows), default=0),
        is_active=user.is_active,
    )


async def current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """FastAPI dependency: the only supported way to learn who is calling."""
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    claims = await decode_token(credentials.credentials, settings)
    raw_sub = claims.get("sub")
    try:
        user_id = uuid.UUID(str(raw_sub))
    except (ValueError, TypeError) as exc:
        raise _UNAUTHENTICATED from exc

    principal = await load_principal(user_id, session)
    # Stashed for the audit middleware; never read back as an input.
    request.state.principal = principal
    return principal


async def require_admin(
    principal: Principal = Depends(current_principal),
) -> Principal:
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )
    return principal
