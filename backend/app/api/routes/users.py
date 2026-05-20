"""
User profile endpoints — view and update health profile.

All endpoints require JWT auth. Provides CRUD for the user's
health profile, preferences, and account management.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import CurrentUser, DBSession
from app.core.logging import get_logger
from app.models.health_profile import BiologicalSex, BloodGroup, HealthProfile

router = APIRouter(prefix="/users", tags=["Users"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    """Full user profile response."""

    id: str
    email: str
    full_name: str
    role: str
    auth_provider: str
    avatar_url: Optional[str] = None
    preferred_language: str
    is_verified: bool
    created_at: str
    last_login_at: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    """Request to update user preferences."""

    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    preferred_language: Optional[str] = Field(None, max_length=10)
    avatar_url: Optional[str] = None


class HealthProfileResponse(BaseModel):
    """Health profile response."""

    id: str
    date_of_birth: Optional[str] = None
    biological_sex: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    blood_group: Optional[str] = None
    bmi: Optional[float] = None
    allergies: Optional[list[str]] = None
    chronic_conditions: Optional[list[str]] = None
    current_medications: Optional[list[dict[str, Any]]] = None
    past_surgeries: Optional[list[dict[str, Any]]] = None
    family_history: Optional[list[str]] = None
    lifestyle: Optional[dict[str, Any]] = None
    emergency_contact: Optional[dict[str, Any]] = None
    updated_at: Optional[str] = None


class HealthProfileUpsertRequest(BaseModel):
    """Create or update a health profile."""

    date_of_birth: Optional[date] = None
    biological_sex: Optional[str] = Field(
        None, pattern="^(male|female|intersex|prefer_not_to_say)$",
    )
    height_cm: Optional[float] = Field(None, ge=30, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=500)
    blood_group: Optional[str] = None
    allergies: Optional[list[str]] = None
    chronic_conditions: Optional[list[str]] = None
    current_medications: Optional[list[dict[str, Any]]] = None
    past_surgeries: Optional[list[dict[str, Any]]] = None
    family_history: Optional[list[str]] = None
    lifestyle: Optional[dict[str, Any]] = None
    emergency_contact: Optional[dict[str, Any]] = None


class ChangePasswordRequest(BaseModel):
    """Request to change password."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


# ── User Profile Endpoints ─────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
)
async def get_my_profile(
    user: CurrentUser,
) -> UserProfileResponse:
    """Return the authenticated user's profile information."""
    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        auth_provider=user.auth_provider.value,
        avatar_url=user.avatar_url,
        preferred_language=user.preferred_language,
        is_verified=user.is_verified,
        created_at=user.created_at.isoformat(),
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


@router.patch(
    "/me",
    response_model=UserProfileResponse,
    summary="Update current user profile",
)
async def update_my_profile(
    body: UpdateProfileRequest,
    user: CurrentUser,
    db: DBSession,
) -> UserProfileResponse:
    """Update the authenticated user's profile settings."""
    if body.full_name is not None:
        user.full_name = body.full_name
    if body.preferred_language is not None:
        user.preferred_language = body.preferred_language
    if body.avatar_url is not None:
        user.avatar_url = body.avatar_url

    await db.commit()
    await db.refresh(user)

    logger.info("User %s updated profile", user.id)

    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        auth_provider=user.auth_provider.value,
        avatar_url=user.avatar_url,
        preferred_language=user.preferred_language,
        is_verified=user.is_verified,
        created_at=user.created_at.isoformat(),
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


@router.post(
    "/me/change-password",
    summary="Change password",
)
async def change_password(
    body: ChangePasswordRequest,
    user: CurrentUser,
    db: DBSession,
) -> Response:
    """Change the current user's password (local auth only)."""
    from app.core.security import hash_password, verify_password
    from app.models.user import AuthProvider

    if user.auth_provider != AuthProvider.LOCAL:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password change is only available for local accounts.",
        )

    if not user.hashed_password or not verify_password(
        body.current_password, user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    user.hashed_password = hash_password(body.new_password)
    await db.commit()

    logger.info("User %s changed password", user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Health Profile Endpoints ────────────────────────────────────────

@router.get(
    "/me/health-profile",
    response_model=HealthProfileResponse,
    summary="Get health profile",
)
async def get_health_profile(
    user: CurrentUser,
    db: DBSession,
) -> HealthProfileResponse:
    """Return the authenticated user's health profile."""
    result = await db.execute(
        select(HealthProfile).where(HealthProfile.user_id == user.id),
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Health profile not found. Create one first.",
        )

    return _profile_to_response(profile)


@router.put(
    "/me/health-profile",
    response_model=HealthProfileResponse,
    summary="Create or update health profile",
)
async def upsert_health_profile(
    body: HealthProfileUpsertRequest,
    user: CurrentUser,
    db: DBSession,
) -> HealthProfileResponse:
    """Create or update the authenticated user's health profile."""
    result = await db.execute(
        select(HealthProfile).where(HealthProfile.user_id == user.id),
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        profile = HealthProfile(user_id=user.id)
        db.add(profile)

    # Update all provided fields
    if body.date_of_birth is not None:
        profile.date_of_birth = body.date_of_birth
    if body.biological_sex is not None:
        profile.biological_sex = BiologicalSex(body.biological_sex)
    if body.height_cm is not None:
        profile.height_cm = body.height_cm
    if body.weight_kg is not None:
        profile.weight_kg = body.weight_kg
    if body.blood_group is not None:
        try:
            profile.blood_group = BloodGroup(body.blood_group)
        except ValueError:
            profile.blood_group = BloodGroup.UNKNOWN
    if body.allergies is not None:
        profile.allergies = body.allergies
    if body.chronic_conditions is not None:
        profile.chronic_conditions = body.chronic_conditions
    if body.current_medications is not None:
        profile.current_medications = body.current_medications
    if body.past_surgeries is not None:
        profile.past_surgeries = body.past_surgeries
    if body.family_history is not None:
        profile.family_history = body.family_history
    if body.lifestyle is not None:
        profile.lifestyle = body.lifestyle
    if body.emergency_contact is not None:
        profile.emergency_contact = body.emergency_contact

    await db.commit()
    await db.refresh(profile)

    logger.info("Health profile updated for user %s", user.id)

    return _profile_to_response(profile)


# ── Helpers ─────────────────────────────────────────────────────────

def _profile_to_response(profile: HealthProfile) -> HealthProfileResponse:
    """Convert a HealthProfile model to a response schema."""
    bmi: float | None = None
    if profile.height_cm and profile.weight_kg:
        h_m = profile.height_cm / 100.0
        bmi = round(profile.weight_kg / (h_m * h_m), 1)

    return HealthProfileResponse(
        id=str(profile.id),
        date_of_birth=profile.date_of_birth.isoformat() if profile.date_of_birth else None,
        biological_sex=profile.biological_sex.value if profile.biological_sex else None,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        blood_group=profile.blood_group.value if profile.blood_group else None,
        bmi=bmi,
        allergies=profile.allergies,
        chronic_conditions=profile.chronic_conditions,
        current_medications=profile.current_medications,
        past_surgeries=profile.past_surgeries,
        family_history=profile.family_history,
        lifestyle=profile.lifestyle,
        emergency_contact=profile.emergency_contact,
        updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
    )
