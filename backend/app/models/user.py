"""
User model — authentication, profile, and preferences.

Supports both local (email + password) and OAuth2 (Google) registration.
Passwords are stored as bcrypt hashes; raw passwords never touch the DB.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chat import ChatSession
    from app.models.diagnosis import DiagnosisRecord
    from app.models.health_profile import HealthProfile


class AuthProvider(str, enum.Enum):
    """Supported authentication providers."""

    LOCAL = "local"
    GOOGLE = "google"


class UserRole(str, enum.Enum):
    """Application-level roles."""

    PATIENT = "patient"
    ADMIN = "admin"


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Core user account model.

    Attributes:
        email: Unique email address (used as login identifier).
        hashed_password: bcrypt hash — nullable for OAuth-only users.
        full_name: Display name.
        auth_provider: Registration method (local or Google OAuth).
        google_id: Google ``sub`` claim for OAuth users.
        role: Application role (patient or admin).
        is_active: Soft-delete flag. Inactive users cannot log in.
        is_verified: Email verification status.
        avatar_url: Profile picture URL (from Google or uploaded).
        preferred_language: ISO 639-1 language code for UI/responses.
        last_login_at: Timestamp of most recent successful login.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False,
    )
    hashed_password: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    full_name: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    auth_provider: Mapped[AuthProvider] = mapped_column(
        Enum(AuthProvider, name="auth_provider_enum"),
        default=AuthProvider.LOCAL,
        nullable=False,
    )
    google_id: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True,
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum"),
        default=UserRole.PATIENT,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    preferred_language: Mapped[str] = mapped_column(
        String(10), default="en", nullable=False,
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # ── Relationships ───────────────────────────────────────────────
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    diagnosis_records: Mapped[list[DiagnosisRecord]] = relationship(
        "DiagnosisRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    health_profile: Mapped[Optional[HealthProfile]] = relationship(
        "HealthProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role.value}>"
