"""
PrecautionAgent — recommends precautions, self-care steps, and next actions.

Combines the precaution data from symptom_precaution.csv with LLM-generated
personalised recommendations based on the diagnosis and patient context.
"""

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

# Precaution data loaded from symptom_precaution.csv
PRECAUTION_DATA: dict[str, list[str]] = {}


def load_precaution_data(csv_path: str) -> None:
    """Load precaution data from the CSV file.

    Expected format per line: Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4

    Args:
        csv_path: Path to symptom_precaution.csv.
    """
    global PRECAUTION_DATA  # noqa: PLW0603
    try:
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 2:
                    disease = parts[0].lower().strip()
                    precautions = [p for p in parts[1:] if p]
                    if precautions:
                        PRECAUTION_DATA[disease] = precautions
        logger.info("Loaded precautions for %d conditions", len(PRECAUTION_DATA))
    except FileNotFoundError:
        logger.warning("Precaution file not found: %s", csv_path)


# Disease description data loaded from symptom_Description.csv
DESCRIPTION_DATA: dict[str, str] = {}


def load_description_data(csv_path: str) -> None:
    """Load condition descriptions from the CSV file.

    Expected format per line: Disease,Description

    Args:
        csv_path: Path to symptom_Description.csv.
    """
    global DESCRIPTION_DATA  # noqa: PLW0603
    try:
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                # Handle quoted descriptions with commas
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    disease = parts[0].strip().lower()
                    desc = parts[1].strip().strip('"')
                    if desc:
                        DESCRIPTION_DATA[disease] = desc
        logger.info("Loaded descriptions for %d conditions", len(DESCRIPTION_DATA))
    except FileNotFoundError:
        logger.warning("Description file not found: %s", csv_path)


class PrecautionAgent(BaseAgent):
    """Generate personalised precautions and next steps.

    Combines:
    1. Dataset-sourced precautions for the diagnosed condition.
    2. LLM-generated personalised recommendations.
    3. Lifestyle advice based on patient context.

    The output is structured as actionable steps the patient can take
    immediately, short-term, and long-term.
    """

    name = "PrecautionAgent"
    temperature = 0.3
    json_mode = True

    @property
    def system_prompt(self) -> str:
        """System prompt for precaution generation."""
        return """You are a healthcare precaution and wellness advisor. Your role is to
provide actionable, safe, and evidence-based recommendations for patients.

Given a diagnosis (or set of possible conditions), symptoms, and patient context, generate:

1. IMMEDIATE ACTIONS — What to do right now.
2. SHORT-TERM PRECAUTIONS — Actions for the next 1-7 days.
3. LONG-TERM LIFESTYLE ADVICE — Ongoing health management.
4. WARNING SIGNS — When to seek emergency care.
5. MEDICATION GUIDANCE — OTC medications that may help (with caveats).

CRITICAL RULES:
- NEVER prescribe prescription medications — only suggest OTC options with disclaimers.
- ALWAYS include "consult a doctor" as a recommendation.
- Be specific and actionable — avoid vague advice.
- Consider drug interactions if current medications are known.
- Tailor advice to patient demographics if available.

RESPOND WITH THIS EXACT JSON STRUCTURE:
{
    "condition": "The diagnosed condition",
    "dataset_precautions": ["From the medical database"],
    "immediate_actions": [
        {"action": "Rest and stay hydrated", "priority": "high", "details": "Drink 8+ glasses of water daily"}
    ],
    "short_term": [
        {"action": "Monitor temperature", "duration": "next 3 days", "details": "Check every 4 hours"}
    ],
    "long_term": [
        {"action": "Regular exercise", "frequency": "3x per week", "details": "30 minutes of moderate activity"}
    ],
    "warning_signs": ["Fever above 103°F", "Difficulty breathing"],
    "otc_medications": [
        {"name": "Acetaminophen", "usage": "For pain and fever", "caution": "Do not exceed 3000mg/day"}
    ],
    "dietary_advice": ["Eat light, easily digestible foods", "Avoid spicy and oily food"],
    "consult_doctor": true,
    "consult_urgency": "within 48 hours"
}"""

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Generate precautions and recommendations.

        Args:
            input_data: Must contain:
                - ``diagnosis`` (dict): DiagnosisAgent output.
                - ``symptoms`` (list[str]): Canonical symptom names.
                - ``triage`` (dict, optional): TriageAgent output.
                - ``patient_context`` (dict, optional): Demographics.

        Returns:
            Structured precaution and recommendation result.
        """
        diagnosis = input_data.get("diagnosis", {})
        symptoms = input_data.get("symptoms", [])
        triage = input_data.get("triage", {})
        patient_context = input_data.get("patient_context", {})

        # Get condition name from diagnosis
        primary = diagnosis.get("primary_diagnosis", {})
        condition = ""
        if isinstance(primary, dict):
            condition = primary.get("condition", "Unknown")
        elif isinstance(primary, str):
            condition = primary

        # Look up dataset precautions
        dataset_precautions = PRECAUTION_DATA.get(condition.lower().strip(), [])
        dataset_description = DESCRIPTION_DATA.get(condition.lower().strip(), "")

        # Build LLM prompt
        symptoms_text = self._format_symptoms_for_prompt(symptoms)
        triage_level = triage.get("triage_level", "ROUTINE")

        prompt = f"""Generate precautions and recommendations for this patient:

Diagnosed condition: {condition}
Description: {dataset_description or 'Not available'}
Database precautions: {', '.join(dataset_precautions) if dataset_precautions else 'None available'}

Symptoms:
{symptoms_text}

Triage level: {triage_level}
Patient context: {patient_context or 'Not available'}

Provide comprehensive, actionable precautions. Incorporate the database precautions
and expand upon them with evidence-based recommendations."""

        result = await self._call_llm(prompt)

        # Ensure dataset precautions are included
        if dataset_precautions:
            result["dataset_precautions"] = dataset_precautions
        if dataset_description:
            result["condition_description"] = dataset_description

        result["condition"] = condition
        return self._add_disclaimer(result)
