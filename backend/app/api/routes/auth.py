"""
Authentication endpoints — register, login, token refresh, Google OAuth2.

Implements JWT-based authentication with support for both local (email/password)
and Google OAuth2 sign-in flows.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.models.user import AuthProvider, User

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = get_logger(__name__)
settings = get_settings()


# ── Request / Response Schemas ──────────────────────────────────────

class RegisterRequest(BaseModel):
    """Schema for local user registration."""

    email: str
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="User", max_length=255)
    preferred_language: str = Field(default="en", max_length=10)


class LoginRequest(BaseModel):
    """Schema for email/password login."""

    email: str
    password: str


class TokenResponse(BaseModel):
    """Schema for JWT token pair response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict[str, Any]


class RefreshRequest(BaseModel):
    """Schema for token refresh."""

    refresh_token: str


class GoogleOAuthRequest(BaseModel):
    """Schema for Google OAuth2 code exchange."""

    code: str
    redirect_uri: Optional[str] = None


# ── Endpoints ───────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user with email and password",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Create a new local user account.

    Validates that the email is not already registered, hashes the
    password, and returns a JWT token pair.
    """
    # Check for existing user
    existing = await db.execute(
        select(User).where(User.email == body.email),
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        auth_provider=AuthProvider.LOCAL,
        preferred_language=body.preferred_language,
        is_verified=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("New user registered: %s", user.id)

    return _build_token_response(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate a local user and return JWT tokens.

    Updates ``last_login_at`` on success.
    """
    result = await db.execute(
        select(User).where(User.email == body.email),
    )
    user = result.scalar_one_or_none()

    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact support.",
        )

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)

    logger.info("User logged in: %s", user.id)

    return _build_token_response(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an expired access token",
)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Exchange a valid refresh token for a new access + refresh pair."""
    from jose import JWTError, jwt as jose_jwt

    try:
        payload = jose_jwt.decode(
            body.refresh_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type.",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user_id = payload.get("sub")
    result = await db.execute(
        select(User).where(User.id == user_id),
    )
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    return _build_token_response(user)


@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Authenticate via Google OAuth2",
)
async def google_oauth(
    body: GoogleOAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Handle Google OAuth2 code exchange.

    Exchanges the authorization code for Google user info, then either
    logs in or registers the user.
    """
    # Exchange code for Google user info
    google_user = await _exchange_google_code(
        body.code,
        body.redirect_uri or settings.google_redirect_uri,
    )

    if not google_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate with Google.",
        )

    # Check if user exists by Google ID or email
    result = await db.execute(
        select(User).where(
            (User.google_id == google_user["sub"])
            | (User.email == google_user["email"]),
        ),
    )
    user = result.scalar_one_or_none()

    if user:
        # Update Google ID if not set (user originally registered locally)
        if not user.google_id:
            user.google_id = google_user["sub"]
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
    else:
        # Register new OAuth user
        user = User(
            email=google_user["email"],
            full_name=google_user.get("name", ""),
            auth_provider=AuthProvider.GOOGLE,
            google_id=google_user["sub"],
            avatar_url=google_user.get("picture"),
            is_verified=True,  # Google emails are pre-verified
            last_login_at=datetime.now(timezone.utc),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info("New Google OAuth user: %s", user.id)

    return _build_token_response(user)


# ── Helpers ─────────────────────────────────────────────────────────

def _build_token_response(user: User) -> TokenResponse:
    """Build a JWT token pair response for the given user."""
    token_data = {"sub": str(user.id), "email": user.email, "role": user.role.value}

    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "avatar_url": user.avatar_url,
            "preferred_language": user.preferred_language,
        },
    )


async def _exchange_google_code(
    code: str, redirect_uri: str,
) -> Optional[dict[str, Any]]:
    """Exchange a Google OAuth2 authorisation code for user info.

    Returns the decoded ID token claims or None on failure.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Exchange code for tokens
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            if token_resp.status_code != 200:
                logger.error("Google token exchange failed: %s", token_resp.text)
                return None

            tokens = token_resp.json()

            # Fetch user info
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error("Google userinfo fetch failed: %s", userinfo_resp.text)
                return None

            return userinfo_resp.json()

    except httpx.HTTPError as exc:
        logger.error("Google OAuth HTTP error: %s", exc)
        return None
