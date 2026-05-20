"""
Diagnosis history and records endpoints.

Provides access to past diagnosis records, summary generation,
and feedback collection. Full agent integration in Phase 2.
"""

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import CurrentUser, DBSession, Pagination
from app.core.logging import get_logger
from app.models.diagnosis import DiagnosisRecord

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class DiagnosisResponse(BaseModel):
    """Schema for a diagnosis record response."""

    id: str
    primary_condition: str
    primary_confidence: float
    differential_diagnoses: Optional[list[dict[str, Any]]] = None
    icd10_codes: Optional[list[str]] = None
    triage_level: str
    triage_explanation: Optional[str] = None
    precautions: Optional[list[str]] = None
    description: Optional[str] = None
    input_symptoms: list[str]
    model_version: str
    created_at: str
    disclaimer: str = (
        "⚠️ This diagnosis is AI-generated and for informational purposes only. "
        "Always consult a qualified healthcare professional."
    )


class FeedbackRequest(BaseModel):
    """Schema for user feedback on a diagnosis."""

    rating: int = Field(..., ge=1, le=5)
    text: Optional[str] = Field(None, max_length=1000)


# ── Endpoints ───────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=list[DiagnosisResponse],
    summary="Get diagnosis history for the current user",
)
async def get_diagnosis_history(
    user: CurrentUser,
    db: DBSession,
    page: Pagination,
) -> list[DiagnosisResponse]:
    """Return paginated diagnosis history for the authenticated user."""
    result = await db.execute(
        select(DiagnosisRecord)
        .where(DiagnosisRecord.user_id == user.id)
        .order_by(DiagnosisRecord.created_at.desc())
        .limit(page.limit)
        .offset(page.offset),
    )
    records = result.scalars().all()

    return [
        DiagnosisResponse(
            id=str(r.id),
            primary_condition=r.primary_condition,
            primary_confidence=r.primary_confidence,
            differential_diagnoses=r.differential_diagnoses,
            icd10_codes=r.icd10_codes,
            triage_level=r.triage_level.value,
            triage_explanation=r.triage_explanation,
            precautions=r.precautions,
            description=r.description,
            input_symptoms=r.input_symptoms,
            model_version=r.model_version,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]


@router.get(
    "/{diagnosis_id}",
    response_model=DiagnosisResponse,
    summary="Get a specific diagnosis record",
)
async def get_diagnosis(
    diagnosis_id: UUID,
    user: CurrentUser,
    db: DBSession,
) -> DiagnosisResponse:
    """Retrieve a single diagnosis record by ID."""
    result = await db.execute(
        select(DiagnosisRecord).where(
            DiagnosisRecord.id == diagnosis_id,
            DiagnosisRecord.user_id == user.id,
        ),
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnosis record not found.",
        )

    return DiagnosisResponse(
        id=str(record.id),
        primary_condition=record.primary_condition,
        primary_confidence=record.primary_confidence,
        differential_diagnoses=record.differential_diagnoses,
        icd10_codes=record.icd10_codes,
        triage_level=record.triage_level.value,
        triage_explanation=record.triage_explanation,
        precautions=record.precautions,
        description=record.description,
        input_symptoms=record.input_symptoms,
        model_version=record.model_version,
        created_at=record.created_at.isoformat(),
    )


@router.post(
    "/{diagnosis_id}/feedback",
    summary="Submit feedback for a diagnosis",
)
async def submit_feedback(
    diagnosis_id: UUID,
    body: FeedbackRequest,
    user: CurrentUser,
    db: DBSession,
) -> Response:
    """Allow users to rate and comment on a diagnosis for model improvement."""
    result = await db.execute(
        select(DiagnosisRecord).where(
            DiagnosisRecord.id == diagnosis_id,
            DiagnosisRecord.user_id == user.id,
        ),
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnosis record not found.",
        )

    record.feedback_rating = body.rating
    record.feedback_text = body.text
    await db.commit()

    logger.info(
        "Feedback submitted for diagnosis %s: rating=%d",
        diagnosis_id,
        body.rating,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
