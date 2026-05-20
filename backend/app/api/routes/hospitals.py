"""
Hospital finder and health summary endpoints.

Integrates with Google Maps Places API for nearest facility search
and provides health summary generation from real diagnosis data.

Phase 3: Real Google Maps service, aggregated health summaries.
"""

from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DBSession
from app.core.logging import get_logger

router = APIRouter(tags=["Health Services"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class NearbyHospitalRequest(BaseModel):
    """Request schema for hospital search."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(default=10.0, ge=1.0, le=100.0)
    urgency: Optional[str] = Field(
        default="routine",
        description="Urgency level: emergency, urgent, routine",
    )


class HospitalResult(BaseModel):
    """Schema for a hospital search result."""

    name: str
    address: str
    latitude: float
    longitude: float
    distance_km: float
    rating: Optional[float] = None
    phone: Optional[str] = None
    open_now: Optional[bool] = None
    place_id: str


class HealthSummaryResponse(BaseModel):
    """Schema for a health summary."""

    user_id: str
    summary_text: str
    total_consultations: int
    common_symptoms: list[str]
    recent_diagnoses: list[dict[str, Any]]
    health_score: Optional[float] = None
    disclaimer: str = (
        "⚠️ This health summary is AI-generated and for informational purposes only."
    )


# ── Endpoints ───────────────────────────────────────────────────────

@router.post(
    "/hospitals/nearby",
    response_model=list[HospitalResult],
    summary="Find nearby hospitals",
)
async def find_nearby_hospitals(
    body: NearbyHospitalRequest,
    user: CurrentUser,
) -> list[HospitalResult]:
    """Search for hospitals and clinics near the user's location.

    Uses Google Maps Places API when configured, falls back to
    mock data in development.
    """
    logger.info(
        "Hospital search by user %s: lat=%f lon=%f radius=%f urgency=%s",
        user.id,
        body.latitude,
        body.longitude,
        body.radius_km,
        body.urgency,
    )

    from app.services.google_maps import get_google_maps_service

    service = get_google_maps_service()
    results = await service.search_nearby_hospitals(
        latitude=body.latitude,
        longitude=body.longitude,
        radius_km=body.radius_km,
        urgency=body.urgency or "routine",
    )

    return [
        HospitalResult(
            name=r["name"],
            address=r["address"],
            latitude=r["latitude"],
            longitude=r["longitude"],
            distance_km=r["distance_km"],
            rating=r.get("rating"),
            phone=r.get("phone"),
            open_now=r.get("open_now"),
            place_id=r["place_id"],
        )
        for r in results
    ]


@router.get(
    "/health/summary",
    response_model=HealthSummaryResponse,
    summary="Generate a health summary for the user",
)
async def get_health_summary(
    user: CurrentUser,
    db: DBSession,
) -> HealthSummaryResponse:
    """Generate a comprehensive health summary based on consultation history.

    Aggregates diagnosis records, symptom patterns, and session counts
    from the database for the authenticated user.
    """
    logger.info("Health summary requested for user %s", user.id)

    from app.models.chat import ChatSession
    from app.models.diagnosis import DiagnosisRecord

    # Count consultations
    total_consultations = await db.scalar(
        select(func.count(ChatSession.id)).where(ChatSession.user_id == user.id),
    ) or 0

    # Get recent diagnoses
    diag_result = await db.execute(
        select(DiagnosisRecord)
        .where(DiagnosisRecord.user_id == user.id)
        .order_by(DiagnosisRecord.created_at.desc())
        .limit(10),
    )
    diagnoses = diag_result.scalars().all()

    recent = [
        {
            "condition": d.primary_condition,
            "confidence": d.primary_confidence,
            "triage_level": d.triage_level.value,
            "date": d.created_at.isoformat(),
        }
        for d in diagnoses
    ]

    # Extract common symptoms across all diagnoses
    all_symptoms: list[str] = []
    for d in diagnoses:
        all_symptoms.extend(d.input_symptoms or [])
    # Count and sort
    symptom_counts = Counter(all_symptoms)
    common = [s for s, _ in symptom_counts.most_common(10)]

    summary_text = (
        f"You have had {total_consultations} consultation(s) on this platform. "
    )
    if diagnoses:
        summary_text += (
            f"Most recent diagnosis: {diagnoses[0].primary_condition} "
            f"(confidence: {diagnoses[0].primary_confidence:.0%})."
        )
    else:
        summary_text += "No diagnoses have been recorded yet."

    return HealthSummaryResponse(
        user_id=str(user.id),
        summary_text=summary_text,
        total_consultations=total_consultations,
        common_symptoms=common,
        recent_diagnoses=recent,
        health_score=None,
    )
