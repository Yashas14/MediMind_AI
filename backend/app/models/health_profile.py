"""
Health profile model — persistent patient health information.

Stores demographics, medical history, allergies, current medications,
and wearable-device data.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    Date,
    Enum,
    Float,
    ForeignKey,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class BloodGroup(str, enum.Enum):
    """Standard ABO + Rh blood group classification."""

    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"


class BiologicalSex(str, enum.Enum):
    """Biological sex for medical context."""

    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class HealthProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Comprehensive patient health profile.

    Attributes:
        user_id: One-to-one FK to the User table.
        date_of_birth: Used for age-based medical reasoning.
        biological_sex: Required for sex-specific diagnostic logic.
        height_cm: Height in centimetres.
        weight_kg: Weight in kilograms.
        blood_group: ABO + Rh blood type.
        allergies: List of known allergies (drugs, food, environmental).
        chronic_conditions: Pre-existing conditions (e.g., diabetes, hypertension).
        current_medications: Active medications with dosage info.
        past_surgeries: Surgical history.
        family_history: Hereditary conditions in the family.
        lifestyle: Lifestyle factors (smoking, alcohol, exercise frequency).
        vaccination_records: Vaccination history as structured JSON.
        emergency_contact: Name + phone for emergency situations.
        wearable_data: Latest sync from Apple HealthKit / Google Fit.
        notes_encrypted: Free-text notes encrypted at rest.
    """

    __tablename__ = "health_profiles"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # ── Demographics ────────────────────────────────────────────────
    date_of_birth: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True,
    )
    biological_sex: Mapped[Optional[BiologicalSex]] = mapped_column(
        Enum(BiologicalSex, name="biological_sex_enum"),
        nullable=True,
    )
    height_cm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    weight_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    blood_group: Mapped[Optional[BloodGroup]] = mapped_column(
        Enum(BloodGroup, name="blood_group_enum"),
        default=BloodGroup.UNKNOWN,
        nullable=True,
    )

    # ── Medical History ─────────────────────────────────────────────
    allergies: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True, default=list,
    )
    chronic_conditions: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True, default=list,
    )
    current_medications: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, default=list,
    )
    past_surgeries: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, default=list,
    )
    family_history: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True, default=list,
    )

    # ── Lifestyle ───────────────────────────────────────────────────
    lifestyle: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Records ─────────────────────────────────────────────────────
    vaccination_records: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, default=list,
    )

    # ── Emergency Contact ───────────────────────────────────────────
    emergency_contact: Mapped[Optional[dict[str, str]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Wearable / IoT Data ─────────────────────────────────────────
    wearable_data: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Encrypted Notes ─────────────────────────────────────────────
    notes_encrypted: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    # ── Relationships ───────────────────────────────────────────────
    user: Mapped[User] = relationship(
        "User", back_populates="health_profile",
    )

    @property
    def bmi(self) -> Optional[float]:
        """Calculate Body Mass Index if height and weight are available."""
        if self.height_cm and self.weight_kg and self.height_cm > 0:
            height_m = self.height_cm / 100.0
            return round(self.weight_kg / (height_m ** 2), 1)
        return None

    def __repr__(self) -> str:
        return f"<HealthProfile id={self.id} user={self.user_id}>"
