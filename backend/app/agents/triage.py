"""
TriageAgent — classifies urgency level for a set of symptoms.

Determines whether a patient's presentation requires:
- EMERGENCY: Call 911 / go to ER immediately
- URGENT: See a doctor within 24 hours
- ROUTINE: Schedule an appointment within a week
- SELF_CARE: Manageable at home with OTC treatment

Uses a combination of rule-based severity thresholds and LLM reasoning
for nuanced triage decisions.
"""

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Rule-based red-flag symptoms ────────────────────────────────────
# Symptoms that ALWAYS trigger EMERGENCY or URGENT triage regardless
# of LLM assessment
EMERGENCY_SYMPTOMS: set[str] = {
    "chest_pain",
    "coma",
    "acute_liver_failure",
    "stomach_bleeding",
    "heart_attack",
    "paralysis_(brain_hemorrhage)",
    "altered_sensorium",
    "slurred_speech",
}

URGENT_SYMPTOMS: set[str] = {
    "high_fever",
    "breathlessness",
    "fast_heart_rate",
    "bloody_stool",
    "blood_in_sputum",
    "weakness_of_one_body_side",
    "loss_of_balance",
    "yellowish_skin",
    "yellowing_of_eyes",
    "fluid_overload",
    "toxic_look_(typhos)",
}


class TriageAgent(BaseAgent):
    """Classify the urgency of a patient's symptoms.

    The triage process works in two phases:
    1. **Rule-based check**: Instantly escalate if any red-flag symptoms
       are present (no LLM call needed).
    2. **LLM reasoning**: For non-obvious cases, Claude evaluates the
       full symptom profile with severity and context.

    The rule-based check ensures that critical symptoms are NEVER
    downgraded by a potentially uncertain LLM response.
    """

    name = "TriageAgent"
    temperature = 0.1  # Very low temp for safety-critical decisions
    json_mode = True

    @property
    def system_prompt(self) -> str:
        """System prompt for triage classification."""
        return """You are a medical triage specialist AI. Your role is to assess the
urgency of a patient's symptoms and classify them into one of four levels.

TRIAGE LEVELS (choose exactly one):
- EMERGENCY: Life-threatening — requires immediate emergency care (call 911 / go to ER).
  Examples: chest pain + breathlessness, loss of consciousness, severe bleeding,
  stroke symptoms (facial droop, arm weakness, speech difficulty).

- URGENT: Serious but not immediately life-threatening — see a doctor within 24 hours.
  Examples: high fever (>103°F / 39.4°C) lasting >2 days, severe abdominal pain,
  significant breathing difficulty, signs of dehydration.

- ROUTINE: Non-urgent — schedule an appointment within the week.
  Examples: persistent cough without fever, mild joint pain, skin rash without
  fever, recurring headache.

- SELF_CARE: Minor — manageable at home with OTC medication and rest.
  Examples: common cold symptoms, mild headache, minor muscle ache,
  occasional indigestion.

DECISION CRITERIA:
1. Consider the COMBINATION of symptoms — individual mild symptoms can
   become urgent when combined.
2. Account for DURATION — symptoms persisting for a long time may need
   escalation.
3. Consider SEVERITY scores if provided.
4. When in doubt, ALWAYS err on the side of higher urgency.
5. Consider patient age and pre-existing conditions if available.

RESPOND WITH THIS EXACT JSON STRUCTURE:
{
    "triage_level": "URGENT",
    "confidence": 0.88,
    "reasoning": "The combination of high fever (severity 8) and breathlessness suggests...",
    "red_flags": ["high_fever", "breathlessness"],
    "immediate_actions": [
        "Monitor temperature every 2 hours",
        "Seek medical attention if breathing worsens"
    ],
    "when_to_escalate": "Go to the emergency room if you experience chest pain, confusion, or oxygen saturation drops below 92%",
    "safe_to_wait": false,
    "recommended_timeframe": "See a doctor within 24 hours"
}"""

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Classify the urgency of the patient's symptoms.

        Args:
            input_data: Must contain:
                - ``symptoms`` (list[str]): Canonical symptom names.
                - ``severity_scores`` (dict[str, int], optional): Per-symptom severity.
                - ``overall_severity`` (int, optional): Aggregate severity (1–10).
                - ``patient_context`` (dict, optional): Age, sex, history.
                - ``diagnosis`` (dict, optional): DiagnosisAgent output.

        Returns:
            Triage result with level, reasoning, and action items.
        """
        symptoms = input_data.get("symptoms", [])
        severity_scores = input_data.get("severity_scores", {})
        overall_severity = input_data.get("overall_severity", 0)
        patient_context = input_data.get("patient_context", {})
        diagnosis = input_data.get("diagnosis", {})

        if not symptoms:
            return self._add_disclaimer({
                "triage_level": "ROUTINE",
                "confidence": 0.0,
                "reasoning": "No symptoms provided for triage.",
                "error": "No symptoms provided",
            })

        # ── Phase 1: Rule-based check ───────────────────────────────
        rule_result = self._rule_based_check(symptoms, overall_severity)
        if rule_result:
            logger.info(
                "[%s] Rule-based triage: %s (symptoms: %s)",
                self.name, rule_result["triage_level"], rule_result.get("red_flags"),
            )
            return self._add_disclaimer(rule_result)

        # ── Phase 2: LLM-based triage ──────────────────────────────
        symptoms_text = self._format_symptoms_for_prompt(symptoms)

        severity_text = ""
        if severity_scores:
            lines = [f"  - {k.replace('_', ' ')}: {v}/10" for k, v in severity_scores.items()]
            severity_text = f"\nSeverity scores:\n" + "\n".join(lines)

        diagnosis_text = ""
        if diagnosis:
            primary = diagnosis.get("primary_diagnosis", {})
            if isinstance(primary, dict):
                diagnosis_text = (
                    f"\nPreliminary diagnosis: {primary.get('condition', 'unknown')} "
                    f"(confidence: {primary.get('confidence', 0):.0%})"
                )

        context_text = ""
        if patient_context:
            context_text = f"\nPatient context: {patient_context}"

        prompt = f"""Assess the urgency of this patient's presentation:

Symptoms:
{symptoms_text}
{severity_text}
Overall severity: {overall_severity}/10
{diagnosis_text}
{context_text}

Classify the triage level and provide your reasoning."""

        result = await self._call_llm(prompt)

        # Validate the triage level
        valid_levels = {"EMERGENCY", "URGENT", "ROUTINE", "SELF_CARE"}
        triage = result.get("triage_level", "ROUTINE").upper()
        if triage not in valid_levels:
            triage = "ROUTINE"
        result["triage_level"] = triage

        # Safety: never allow LLM to downgrade below rule-based check
        result = self._enforce_minimum_triage(symptoms, result)

        return self._add_disclaimer(result)

    def _rule_based_check(
        self,
        symptoms: list[str],
        overall_severity: int,
    ) -> dict[str, Any] | None:
        """Apply rule-based checks for red-flag symptoms.

        Returns a triage result dict if red-flags are found, or None
        to proceed to LLM-based assessment.
        """
        symptom_set = set(s.lower() for s in symptoms)

        # Check for EMERGENCY symptoms
        emergency_hits = symptom_set & EMERGENCY_SYMPTOMS
        if emergency_hits:
            return {
                "triage_level": "EMERGENCY",
                "confidence": 0.99,
                "reasoning": (
                    f"CRITICAL: The following emergency-level symptoms were detected: "
                    f"{', '.join(s.replace('_', ' ') for s in emergency_hits)}. "
                    "Immediate medical attention is required."
                ),
                "red_flags": list(emergency_hits),
                "immediate_actions": [
                    "Call emergency services (911) immediately",
                    "Do not drive yourself — have someone take you or call an ambulance",
                    "If experiencing chest pain, chew an aspirin if not allergic",
                ],
                "when_to_escalate": "This IS an emergency — seek help NOW.",
                "safe_to_wait": False,
                "recommended_timeframe": "Immediately — call emergency services",
                "rule_triggered": True,
            }

        # Check for URGENT symptoms
        urgent_hits = symptom_set & URGENT_SYMPTOMS
        if urgent_hits or overall_severity >= 8:
            return {
                "triage_level": "URGENT",
                "confidence": 0.95,
                "reasoning": (
                    f"Urgent symptoms detected: "
                    f"{', '.join(s.replace('_', ' ') for s in urgent_hits)}. "
                    f"Overall severity: {overall_severity}/10. "
                    "Medical attention within 24 hours is recommended."
                ),
                "red_flags": list(urgent_hits),
                "immediate_actions": [
                    "Contact your doctor or visit an urgent care clinic today",
                    "Monitor symptoms for worsening",
                    "Keep a log of symptom changes to share with your doctor",
                ],
                "when_to_escalate": (
                    "Go to the emergency room if symptoms suddenly worsen, "
                    "you develop chest pain, difficulty breathing, or lose consciousness."
                ),
                "safe_to_wait": False,
                "recommended_timeframe": "Within 24 hours",
                "rule_triggered": True,
            }

        return None  # No rule triggered — proceed to LLM

    def _enforce_minimum_triage(
        self,
        symptoms: list[str],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure the LLM hasn't dangerously downgraded a triage level.

        If any urgent symptoms are present, the level cannot go below URGENT.
        """
        symptom_set = set(s.lower() for s in symptoms)
        level_order = {"SELF_CARE": 0, "ROUTINE": 1, "URGENT": 2, "EMERGENCY": 3}

        current_level = result.get("triage_level", "ROUTINE")
        current_rank = level_order.get(current_level, 1)

        # If urgent symptoms exist but LLM says ROUTINE or SELF_CARE
        if symptom_set & URGENT_SYMPTOMS and current_rank < 2:
            logger.warning(
                "[%s] Safety override: upgrading from %s to URGENT",
                self.name, current_level,
            )
            result["triage_level"] = "URGENT"
            result["safety_override"] = True
            result["safety_note"] = (
                "Triage level upgraded due to presence of clinically significant symptoms."
            )

        return result
