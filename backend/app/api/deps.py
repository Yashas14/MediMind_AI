"""
FastAPI dependency injection — authentication, database, common deps.

Provides reusable dependencies that are injected into route handlers
via FastAPI's ``Depends()`` mechanism.

Phase 3: Added WebSocket auth, type aliases, pagination dependency.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.core.security import verify_access_token
from app.models.user import User, UserRole

logger = get_logger(__name__)
settings = get_settings()

# HTTP Bearer token extractor
bearer_scheme = HTTPBearer(auto_error=False)


# ── Pagination ──────────────────────────────────────────────────────

class PaginationParams(BaseModel):
    """Reusable pagination parameters."""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


def pagination(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> PaginationParams:
    """FastAPI dependency for pagination query params."""
    return PaginationParams(limit=limit, offset=offset)


# ── HTTP Auth ───────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the JWT token, returning the authenticated user.

    Raises:
        HTTPException 401: If token is missing, invalid, or expired.
        HTTPException 403: If the user is deactivated.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Require the current user to have admin role.

    Raises:
        HTTPException 403: If the user is not an admin.
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )
    return user


# ── WebSocket Auth ──────────────────────────────────────────────────

async def get_ws_user(
    websocket: WebSocket,
    token: str = Query(default=""),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Authenticate a WebSocket connection via query-string token.

    WebSockets cannot send Authorization headers from browsers,
    so the JWT is passed as ``?token=<jwt>`` query parameter.

    Returns:
        The authenticated User, or None if the token is invalid
        (callers decide whether to allow anonymous access).
    """
    if not token:
        if settings.debug:
            logger.warning("WebSocket connected without auth token (debug mode)")
        return None

    payload = verify_access_token(token)
    if payload is None:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user and not user.is_active:
        return None

    return user


# ── Type aliases for cleaner route signatures ───────────────────────

CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DBSession = Annotated[AsyncSession, Depends(get_db)]
Pagination = Annotated[PaginationParams, Depends(pagination)]
