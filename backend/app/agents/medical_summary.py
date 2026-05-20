"""
MedicalSummaryAgent — generates structured visit/consultation summaries.

Produces a comprehensive summary document from the full pipeline output
that can be saved, shared with a doctor, or exported as a health record.
"""

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class MedicalSummaryAgent(BaseAgent):
    """Generate structured medical consultation summaries.

    Takes the output of all other agents (symptoms, diagnosis, triage,
    precautions) and produces a unified summary suitable for:
    - Sharing with a healthcare provider
    - Patient health records
    - Follow-up visit preparation
    """

    name = "MedicalSummaryAgent"
    temperature = 0.3
    json_mode = True

    @property
    def system_prompt(self) -> str:
        """System prompt for summary generation."""
        return """You are a medical documentation specialist. Your task is to create
a clear, structured consultation summary from AI-generated diagnostic data.

The summary should be:
- Written in professional but accessible language.
- Organised into clear sections.
- Suitable for sharing with a healthcare provider.
- Include all relevant findings, recommendations, and caveats.

RESPOND WITH THIS EXACT JSON STRUCTURE:
{
    "summary_title": "AI Health Consultation Summary",
    "date": "ISO date string",
    "chief_complaint": "Patient's primary concern in one sentence",
    "symptoms_presented": [
        {"symptom": "name", "severity": 7, "duration": "3 days"}
    ],
    "assessment": {
        "primary_condition": "Condition Name",
        "confidence": 0.85,
        "differential": ["Alt 1", "Alt 2"],
        "icd10_codes": ["X99.9"]
    },
    "triage_classification": {
        "level": "URGENT",
        "reasoning": "Brief explanation"
    },
    "recommendations": {
        "immediate": ["Rest", "Stay hydrated"],
        "follow_up": "See a doctor within 24 hours",
        "tests_suggested": ["Blood work", "Chest X-ray"],
        "medications": ["OTC pain relief if needed"]
    },
    "patient_instructions": "Clear, actionable paragraph for the patient",
    "provider_notes": "Technical notes for the healthcare provider",
    "ai_confidence_note": "Explanation of AI confidence and limitations",
    "disclaimer": "Standard medical disclaimer"
}"""

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Generate a comprehensive consultation summary.

        Args:
            input_data: Must contain:
                - ``symptoms_result`` (dict): SymptomExtractorAgent output.
                - ``diagnosis_result`` (dict): DiagnosisAgent output.
                - ``triage_result`` (dict): TriageAgent output.
                - ``precaution_result`` (dict): PrecautionAgent output.
                - ``patient_context`` (dict, optional): Demographics.
                - ``conversation_history`` (list[str], optional): Chat log.

        Returns:
            Structured medical summary dictionary.
        """
        symptoms_result = input_data.get("symptoms_result", {})
        diagnosis_result = input_data.get("diagnosis_result", {})
        triage_result = input_data.get("triage_result", {})
        precaution_result = input_data.get("precaution_result", {})
        patient_context = input_data.get("patient_context", {})
        conversation = input_data.get("conversation_history", [])

        # Build a comprehensive prompt from all agent outputs
        prompt = f"""Generate a structured medical consultation summary from these findings:

SYMPTOMS EXTRACTED:
{_format_section(symptoms_result)}

DIAGNOSIS:
{_format_section(diagnosis_result)}

TRIAGE ASSESSMENT:
{_format_section(triage_result)}

PRECAUTIONS & RECOMMENDATIONS:
{_format_section(precaution_result)}

PATIENT CONTEXT:
{_format_section(patient_context)}

CONVERSATION LENGTH: {len(conversation)} messages

Create a professional, comprehensive summary suitable for both the patient and
their healthcare provider. Ensure all findings are represented accurately."""

        result = await self._call_llm(prompt)
        return self._add_disclaimer(result)


def _format_section(data: dict[str, Any] | Any) -> str:
    """Format a data section for the prompt.

    Args:
        data: Dictionary or value to format.

    Returns:
        Formatted string representation.
    """
    if not data:
        return "Not available"
    if isinstance(data, dict):
        # Remove disclaimer from nested data to avoid duplication
        cleaned = {k: v for k, v in data.items() if k != "disclaimer"}
        import json
        try:
            return json.dumps(cleaned, indent=2, default=str)[:2000]
        except (TypeError, ValueError):
            return str(cleaned)[:2000]
    return str(data)[:2000]
