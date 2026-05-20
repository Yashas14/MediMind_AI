"""
Agent orchestrator — coordinates the full diagnostic pipeline.

Manages the sequential execution of all agents:
1. SymptomExtractorAgent → parse user input
2. DiagnosisAgent → predict conditions
3. TriageAgent → classify urgency
4. PrecautionAgent → recommend actions
5. MedicalSummaryAgent → generate summary

Also provides the conversational interface used by the chat endpoints.
"""

from typing import Any

from app.agents.diagnosis import DiagnosisAgent
from app.agents.medical_summary import MedicalSummaryAgent
from app.agents.precaution import PrecautionAgent
from app.agents.symptom_extractor import SymptomExtractorAgent
from app.agents.triage import TriageAgent
from app.core.claude_client import call_claude
from app.core.logging import get_logger
from app.rag.retrieval import RAGService
from app.rag.vector_store import VectorStoreService

logger = get_logger(__name__)

# Medical disclaimer for every response
DISCLAIMER = (
    "⚠️ DISCLAIMER: This is an AI-generated response for informational purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider. If you think you may have a medical "
    "emergency, call your doctor or emergency services immediately."
)


class AgentOrchestrator:
    """Coordinates all AI agents for the healthcare platform.

    Manages agent lifecycle, data flow between agents, and provides
    both full-pipeline and individual-agent access patterns.

    Attributes:
        symptom_agent: Extracts symptoms from free text.
        diagnosis_agent: Predicts conditions with confidence.
        triage_agent: Classifies urgency level.
        precaution_agent: Recommends next steps.
        summary_agent: Generates visit summaries.
        rag_service: RAG retrieval pipeline.
    """

    def __init__(self) -> None:
        self.symptom_agent = SymptomExtractorAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.triage_agent = TriageAgent()
        self.precaution_agent = PrecautionAgent()
        self.summary_agent = MedicalSummaryAgent()
        self.rag_service = RAGService(VectorStoreService())

        # Connect RAG to diagnosis agent
        self.diagnosis_agent.set_rag_service(self.rag_service)

    def load_ml_model(self, model_path: str) -> None:
        """Load the ML ensemble model into the DiagnosisAgent.

        Args:
            model_path: Path to the joblib model file.
        """
        self.diagnosis_agent.load_model(model_path)

    async def run_full_pipeline(
        self,
        user_text: str,
        patient_context: dict[str, Any] | None = None,
        language: str = "en",
    ) -> dict[str, Any]:
        """Execute the complete diagnostic pipeline.

        Runs all agents in sequence, passing data between them:
        text → symptoms → diagnosis → triage → precautions → summary

        Args:
            user_text: The patient's free-text symptom description.
            patient_context: Optional demographics, history, medications.
            language: ISO 639-1 language code.

        Returns:
            Complete pipeline result with all agent outputs.
        """
        patient_context = patient_context or {}

        logger.info("Starting full diagnostic pipeline for input: %s…", user_text[:100])

        # ── Step 1: Extract Symptoms ────────────────────────────────
        symptoms_result = await self.symptom_agent.process({
            "text": user_text,
            "language": language,
        })

        canonical_symptoms = self.symptom_agent.get_canonical_names(symptoms_result)
        feature_vector = self.symptom_agent.build_feature_vector(canonical_symptoms)

        logger.info("Extracted %d symptoms: %s", len(canonical_symptoms), canonical_symptoms)

        if not canonical_symptoms:
            return {
                "stage": "symptom_extraction",
                "symptoms_result": symptoms_result,
                "response": (
                    "I wasn't able to identify specific symptoms from your description. "
                    "Could you please describe your symptoms in more detail? "
                    "For example: 'I have a headache and feel nauseous.'"
                ),
                "follow_up_questions": symptoms_result.get("follow_up_questions", []),
                "disclaimer": DISCLAIMER,
            }

        # ── Step 2: Diagnose ────────────────────────────────────────
        severity_scores = {
            s.get("canonical_name", ""): s.get("severity", 5)
            for s in symptoms_result.get("matched_symptoms", [])
        }

        diagnosis_result = await self.diagnosis_agent.process({
            "symptoms": canonical_symptoms,
            "feature_vector": feature_vector,
            "patient_context": patient_context,
            "severity_scores": severity_scores,
        })

        # ── Step 3: Triage ──────────────────────────────────────────
        triage_result = await self.triage_agent.process({
            "symptoms": canonical_symptoms,
            "severity_scores": severity_scores,
            "overall_severity": symptoms_result.get("overall_severity", 5),
            "patient_context": patient_context,
            "diagnosis": diagnosis_result,
        })

        # ── Step 4: Precautions ─────────────────────────────────────
        precaution_result = await self.precaution_agent.process({
            "diagnosis": diagnosis_result,
            "symptoms": canonical_symptoms,
            "triage": triage_result,
            "patient_context": patient_context,
        })

        # ── Step 5: Summary ─────────────────────────────────────────
        summary_result = await self.summary_agent.process({
            "symptoms_result": symptoms_result,
            "diagnosis_result": diagnosis_result,
            "triage_result": triage_result,
            "precaution_result": precaution_result,
            "patient_context": patient_context,
        })

        # ── Assemble Final Response ─────────────────────────────────
        response = self._build_response_text(
            symptoms_result, diagnosis_result, triage_result, precaution_result,
        )

        return {
            "response": response,
            "symptoms_result": symptoms_result,
            "diagnosis_result": diagnosis_result,
            "triage_result": triage_result,
            "precaution_result": precaution_result,
            "summary_result": summary_result,
            "extracted_symptoms": canonical_symptoms,
            "confidence_score": self._get_primary_confidence(diagnosis_result),
            "triage_level": triage_result.get("triage_level", "ROUTINE"),
            "disclaimer": DISCLAIMER,
        }

    async def chat_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a conversational response for the chat interface.

        For messages that describe symptoms, runs the diagnostic pipeline.
        For other messages (greetings, questions), uses Claude directly.

        Args:
            user_message: The user's chat message.
            conversation_history: Previous messages for context.
            patient_context: Patient demographics and history.

        Returns:
            Chat response dict with the assistant's reply.
        """
        conversation_history = conversation_history or []

        # Check if the message contains symptom-related content
        if self._is_symptom_message(user_message):
            pipeline_result = await self.run_full_pipeline(
                user_message,
                patient_context=patient_context,
            )
            return {
                "type": "diagnosis",
                "content": pipeline_result["response"],
                "confidence_score": pipeline_result.get("confidence_score", 0),
                "extracted_symptoms": pipeline_result.get("extracted_symptoms", []),
                "triage_level": pipeline_result.get("triage_level", "ROUTINE"),
                "full_result": pipeline_result,
                "disclaimer": DISCLAIMER,
            }

        # General conversational response
        system_prompt = """You are a friendly, professional healthcare AI assistant.
You help patients understand their symptoms and health concerns.

RULES:
1. Be empathetic but professional.
2. Never diagnose without running through the symptom analysis pipeline.
3. Encourage patients to describe their symptoms if they haven't already.
4. Always remind patients to consult a real healthcare provider.
5. If asked about medication dosages or prescriptions, decline and recommend a doctor.
6. You can answer general health questions with evidence-based information.
7. End each response with a brief reminder that you are an AI assistant."""

        # Build conversation context
        history_text = ""
        if conversation_history:
            recent = conversation_history[-10:]  # Last 10 messages
            history_text = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in recent
            )
            history_text = f"\nConversation history:\n{history_text}\n"

        prompt = f"{history_text}\nUSER: {user_message}"

        response_text = await call_claude(
            system_prompt,
            prompt,
            temperature=0.4,
        )

        return {
            "type": "conversation",
            "content": response_text,
            "confidence_score": None,
            "extracted_symptoms": [],
            "triage_level": None,
            "disclaimer": DISCLAIMER,
        }

    def _is_symptom_message(self, message: str) -> bool:
        """Heuristic to detect if a message contains symptom descriptions.

        Args:
            message: The user's message.

        Returns:
            True if the message likely contains symptoms.
        """
        symptom_keywords = {
            "symptom", "pain", "ache", "fever", "cough", "headache",
            "nausea", "vomit", "dizzy", "rash", "itch", "sore",
            "throat", "tired", "fatigue", "breathing", "chest",
            "stomach", "diarr", "constipat", "bleed", "swollen",
            "weak", "burn", "cramp", "sneez", "runny", "congest",
            "feeling sick", "not feeling well", "i have", "i feel",
            "i'm experiencing", "suffering from", "hurts", "hurting",
            "diagnos", "what's wrong", "what could", "body",
        }
        lower = message.lower()
        return any(kw in lower for kw in symptom_keywords)

    def _build_response_text(
        self,
        symptoms: dict[str, Any],
        diagnosis: dict[str, Any],
        triage: dict[str, Any],
        precautions: dict[str, Any],
    ) -> str:
        """Build a human-readable response from agent outputs.

        Args:
            symptoms: SymptomExtractorAgent output.
            diagnosis: DiagnosisAgent output.
            triage: TriageAgent output.
            precautions: PrecautionAgent output.

        Returns:
            Formatted response text with markdown.
        """
        sections: list[str] = []

        # Triage banner (if urgent/emergency)
        triage_level = triage.get("triage_level", "ROUTINE")
        if triage_level == "EMERGENCY":
            sections.append(
                "🚨 **EMERGENCY — SEEK IMMEDIATE MEDICAL ATTENTION**\n"
                f"{triage.get('reasoning', 'Critical symptoms detected.')}\n"
            )
        elif triage_level == "URGENT":
            sections.append(
                "⚠️ **URGENT — Please see a doctor within 24 hours**\n"
                f"{triage.get('reasoning', 'Serious symptoms detected.')}\n"
            )

        # Symptoms identified
        matched = symptoms.get("matched_symptoms", [])
        if matched:
            symptom_lines = []
            for s in matched:
                name = s.get("canonical_name", "").replace("_", " ").title()
                severity = s.get("severity", "?")
                symptom_lines.append(f"  • {name} (severity: {severity}/10)")
            sections.append(
                f"**Symptoms Identified ({len(matched)}):**\n" + "\n".join(symptom_lines)
            )

        # Diagnosis
        primary = diagnosis.get("primary_diagnosis", {})
        if isinstance(primary, dict) and primary.get("condition"):
            confidence = primary.get("confidence", 0)
            sections.append(
                f"**Preliminary Assessment:**\n"
                f"  • {primary['condition']} (confidence: {confidence:.0%})\n"
                f"  {primary.get('explanation', '')}"
            )

            # Differential
            diffs = diagnosis.get("differential_diagnoses", [])
            if diffs:
                diff_lines = [
                    f"  • {d.get('condition', '?')} ({d.get('confidence', 0):.0%})"
                    for d in diffs[:3]
                ]
                sections.append(
                    "**Other Possibilities:**\n" + "\n".join(diff_lines)
                )

        # Key recommendations
        immediate = precautions.get("immediate_actions", [])
        if immediate:
            action_lines = []
            for a in immediate[:4]:
                if isinstance(a, dict):
                    action_lines.append(f"  • {a.get('action', '')}")
                else:
                    action_lines.append(f"  • {a}")
            sections.append(
                "**Recommended Actions:**\n" + "\n".join(action_lines)
            )

        # Warning signs
        warnings = precautions.get("warning_signs", [])
        if warnings:
            sections.append(
                "**Warning Signs — Seek Emergency Care If:**\n"
                + "\n".join(f"  🔴 {w}" for w in warnings[:4])
            )

        # Always end with disclaimer
        sections.append(f"\n---\n{DISCLAIMER}")

        return "\n\n".join(sections)

    def _get_primary_confidence(self, diagnosis: dict[str, Any]) -> float:
        """Extract the primary diagnosis confidence score.

        Args:
            diagnosis: DiagnosisAgent output.

        Returns:
            Confidence score (0–1) or 0.0 if unavailable.
        """
        primary = diagnosis.get("primary_diagnosis", {})
        if isinstance(primary, dict):
            return float(primary.get("confidence", 0.0))
        return 0.0


# Module-level singleton for convenience
_orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the singleton agent orchestrator.

    Creates on first access and reuses thereafter.

    Returns:
        Configured :class:`AgentOrchestrator` instance.
    """
    global _orchestrator  # noqa: PLW0603
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
