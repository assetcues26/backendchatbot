"""JWT verification and Principal construction (guardrail G2).

Supabase signs access tokens with the project's JWT secret (HS256). We verify
the signature, expiry and audience ourselves; we never trust unverified claims
and we never read identity from anywhere but this module.

The token gives us a user id. Everything that matters for access control --
tenant, roles, clearance, active status -- is then read from *our* database,
not from the token. A token issued before a role was revoked therefore cannot
grant the revoked access: the revocation is visible on the very next request.
That property is what makes the "revoke mid-conversation" red-team test pass.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.principal import Principal
from app.db.models import Role, Tenant, User, UserRole
from app.db.session import get_session

_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)
_INACTIVE = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled"
)


def decode_token(token: str, settings: Settings) -> dict[str, object]:
    """Verify signature, expiry and audience. Raises 401 on any failure."""
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
            options={"require_exp": True, "require_sub": True},
        )
    except JWTError as exc:
        raise _UNAUTHENTICATED from exc


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

    claims = decode_token(credentials.credentials, settings)
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
