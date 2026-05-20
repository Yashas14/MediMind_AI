"""
Symptom analysis endpoints.

Provides structured symptom analysis, severity scoring, and extraction
from free-text input. Full agent integration in Phase 2.
"""

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.core.logging import get_logger

router = APIRouter(prefix="/symptoms", tags=["Symptoms"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class SymptomAnalysisRequest(BaseModel):
    """Request schema for symptom analysis."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Free-text description of symptoms.",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code of the input.",
    )


class ExtractedSymptom(BaseModel):
    """A single extracted symptom with metadata."""

    name: str
    severity: int = Field(..., ge=1, le=10)
    body_region: Optional[str] = None
    duration: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class SymptomAnalysisResponse(BaseModel):
    """Response schema for symptom analysis."""

    extracted_symptoms: list[ExtractedSymptom]
    raw_text: str
    language_detected: str
    disclaimer: str = (
        "⚠️ Symptom analysis is AI-generated and for informational purposes only. "
        "Consult a healthcare professional for accurate diagnosis."
    )


# ── Endpoints ───────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=SymptomAnalysisResponse,
    summary="Analyse free-text symptoms",
)
async def analyze_symptoms(
    body: SymptomAnalysisRequest,
    user: CurrentUser,
) -> SymptomAnalysisResponse:
    """Extract and score symptoms from free-text input.

    Uses the SymptomExtractorAgent (Phase 2) to parse natural language
    into structured symptom data with severity scores.
    """
    logger.info("Symptom analysis requested for text of length %d", len(body.text))

    try:
        from app.agents.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        result = await orchestrator.symptom_agent.process({
            "text": body.text,
            "language": body.language,
        })

        extracted = [
            ExtractedSymptom(
                name=s.get("canonical_name", "unknown"),
                severity=s.get("severity", 5),
                body_region=s.get("body_region"),
                duration=s.get("duration"),
                confidence=s.get("confidence", 0.5),
            )
            for s in result.get("matched_symptoms", [])
        ]

        return SymptomAnalysisResponse(
            extracted_symptoms=extracted,
            raw_text=body.text,
            language_detected=result.get("language_detected", body.language),
        )

    except Exception as exc:
        logger.error("Symptom extraction failed: %s", exc)
        return SymptomAnalysisResponse(
            extracted_symptoms=[],
            raw_text=body.text,
            language_detected=body.language,
        )


@router.get(
    "/list",
    response_model=list[dict[str, Any]],
    summary="Get all known symptoms",
)
async def list_symptoms() -> list[dict[str, Any]]:
    """Return the master list of recognised symptoms.

    Loaded from the training dataset symptom columns and enriched
    with severity data from the SymptomExtractorAgent.
    """
    from app.agents.symptom_extractor import CANONICAL_SYMPTOMS, SYMPTOM_SEVERITY_MAP

    result = []
    categories = {
        "skin": ["itching", "skin_rash", "nodal_skin_eruptions", "pus_filled_pimples",
                 "blackheads", "scurring", "skin_peeling", "silver_like_dusting",
                 "blister", "red_sore_around_nose", "yellow_crust_ooze", "dischromic_patches"],
        "respiratory": ["continuous_sneezing", "cough", "breathlessness", "phlegm",
                        "throat_irritation", "runny_nose", "congestion", "sinus_pressure",
                        "mucoid_sputum", "rusty_sputum", "blood_in_sputum"],
        "gastrointestinal": ["stomach_pain", "acidity", "vomiting", "indigestion",
                             "nausea", "loss_of_appetite", "constipation", "diarrhoea",
                             "abdominal_pain", "belly_pain", "passage_of_gases",
                             "stomach_bleeding"],
        "neurological": ["headache", "dizziness", "spinning_movements", "loss_of_balance",
                         "unsteadiness", "weakness_of_one_body_side", "altered_sensorium",
                         "slurred_speech", "lack_of_concentration", "visual_disturbances",
                         "blurred_and_distorted_vision"],
        "cardiovascular": ["chest_pain", "fast_heart_rate", "palpitations",
                           "swollen_blood_vessels", "prominent_veins_on_calf"],
        "general": ["fatigue", "high_fever", "mild_fever", "malaise", "lethargy",
                     "sweating", "dehydration", "weight_loss", "weight_gain",
                     "restlessness", "chills", "shivering"],
    }

    # Build a reverse lookup: symptom → category
    symptom_cat: dict[str, str] = {}
    for cat, symptoms_list in categories.items():
        for s in symptoms_list:
            symptom_cat[s] = cat

    for symptom in CANONICAL_SYMPTOMS:
        severity = SYMPTOM_SEVERITY_MAP.get(symptom, 3)
        result.append({
            "name": symptom,
            "display_name": symptom.replace("_", " ").title(),
            "category": symptom_cat.get(symptom, "other"),
            "severity_weight": severity,
            "severity_range": [max(1, severity - 2), min(10, severity + 3)],
        })

    return result


class FullPipelineRequest(BaseModel):
    """Request schema for the full diagnostic pipeline."""

    text: str = Field(..., min_length=5, max_length=5000)
    language: str = Field(default="en")
    patient_context: Optional[dict[str, Any]] = None


@router.post(
    "/diagnose",
    summary="Run full diagnostic pipeline",
    response_model=dict[str, Any],
)
async def run_full_diagnosis(
    body: FullPipelineRequest,
    user: CurrentUser,
) -> dict[str, Any]:
    """Run the complete AI diagnostic pipeline on a symptom description.

    Executes all agents in sequence:
    1. SymptomExtractor → parse symptoms
    2. DiagnosisAgent → predict conditions
    3. TriageAgent → classify urgency
    4. PrecautionAgent → recommend actions
    5. MedicalSummaryAgent → generate summary

    Returns the full pipeline output including all agent results.
    """
    logger.info("Full pipeline requested for text of length %d", len(body.text))

    try:
        from app.agents.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        result = await orchestrator.run_full_pipeline(
            user_text=body.text,
            patient_context=body.patient_context,
            language=body.language,
        )
        return result
    except Exception as exc:
        logger.error("Full pipeline failed: %s", exc, exc_info=True)
        return {
            "error": str(exc),
            "response": (
                "I apologise, but I encountered an error processing your symptoms. "
                "Please try again or consult a healthcare professional."
            ),
            "disclaimer": (
                "⚠️ This is an AI system — always consult a qualified "
                "healthcare provider for medical advice."
            ),
        }
