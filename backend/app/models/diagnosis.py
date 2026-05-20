"""
Diagnosis record model — stores AI-generated diagnoses.

Each record captures the full output of a diagnostic interaction:
conditions with probabilities, ICD-10 codes, triage level,
precautions, and the raw model outputs for audit / retraining.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class TriageLevel(str, enum.Enum):
    """Urgency classification for a diagnosis."""

    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    SELF_CARE = "self_care"


class DiagnosisRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Persistent record of an AI-generated diagnosis.

    Attributes:
        user_id: FK to the patient.
        session_id: FK to the chat session that produced this diagnosis.
        input_symptoms: List of symptom strings provided by the user.
        symptom_severity_scores: Per-symptom severity (1–10) as JSON.
        primary_condition: Top predicted condition name.
        primary_confidence: Confidence score for the primary condition (0–1).
        differential_diagnoses: Top 3–5 alternative diagnoses with scores.
        icd10_codes: Mapped ICD-10 codes for identified conditions.
        triage_level: Urgency classification.
        triage_explanation: Plain-language explanation of the triage assessment.
        precautions: Recommended precautions / next steps.
        description: Detailed description of the primary condition.
        model_version: Identifier of the ML model/ensemble that produced this.
        raw_model_output: Full model output JSON for audit / retraining.
        feedback_rating: Optional user feedback (1–5 stars).
        feedback_text: Optional free-text feedback from the user.
    """

    __tablename__ = "diagnosis_records"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Symptom Input ───────────────────────────────────────────────
    input_symptoms: Mapped[list[str]] = mapped_column(
        JSON, nullable=False,
    )
    symptom_severity_scores: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Primary Diagnosis ───────────────────────────────────────────
    primary_condition: Mapped[str] = mapped_column(
        String(500), nullable=False,
    )
    primary_confidence: Mapped[float] = mapped_column(
        Float, nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    # ── Differential Diagnoses ──────────────────────────────────────
    differential_diagnoses: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── ICD-10 Mapping ────────────────────────────────────────────
    icd10_codes: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Triage ──────────────────────────────────────────────────────
    triage_level: Mapped[TriageLevel] = mapped_column(
        Enum(TriageLevel, name="triage_level_enum"),
        default=TriageLevel.ROUTINE,
        nullable=False,
    )
    triage_explanation: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    # ── Precautions & Recommendations ───────────────────────────────
    precautions: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── Model Metadata ──────────────────────────────────────────────
    model_version: Mapped[str] = mapped_column(
        String(100), default="ensemble-v1", nullable=False,
    )
    raw_model_output: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True,
    )

    # ── User Feedback ───────────────────────────────────────────────
    feedback_rating: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    feedback_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    # ── Relationships ───────────────────────────────────────────────
    user: Mapped[User] = relationship(
        "User", back_populates="diagnosis_records",
    )

    def __repr__(self) -> str:
        return (
            f"<DiagnosisRecord id={self.id} "
            f"condition={self.primary_condition!r} "
            f"confidence={self.primary_confidence:.2f} "
            f"triage={self.triage_level.value}>"
        )
