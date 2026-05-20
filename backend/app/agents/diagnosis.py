"""
DiagnosisAgent — predicts conditions using ML ensemble + RAG + LLM reasoning.

This is the core diagnostic engine that combines three approaches:
1. **ML Ensemble** — XGBoost + RandomForest voting classifier on structured
   symptom features (trained on Training.csv).
2. **RAG Retrieval** — Medical knowledge retrieval from ChromaDB for
   evidence-based reasoning.
3. **LLM Reasoning** — Claude synthesises all signals into a final
   differential diagnosis with confidence scores and ICD-10 codes.
"""

from typing import Any

import numpy as np

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── ICD-10 Code Mapping (common conditions from the dataset) ────────
ICD10_MAP: dict[str, str] = {
    "fungal infection": "B49",
    "allergy": "T78.4",
    "gerd": "K21.0",
    "chronic cholestasis": "K71.0",
    "drug reaction": "T88.7",
    "peptic ulcer disease": "K27.9",
    "aids": "B24",
    "diabetes": "E14",
    "gastroenteritis": "A09",
    "bronchial asthma": "J45.9",
    "hypertension": "I10",
    "migraine": "G43.9",
    "cervical spondylosis": "M47.8",
    "paralysis (brain hemorrhage)": "I61.9",
    "jaundice": "R17",
    "malaria": "B54",
    "chicken pox": "B01.9",
    "dengue": "A90",
    "typhoid": "A01.0",
    "hepatitis a": "B15.9",
    "hepatitis b": "B16.9",
    "hepatitis c": "B17.1",
    "hepatitis d": "B17.0",
    "hepatitis e": "B17.2",
    "alcoholic hepatitis": "K70.1",
    "tuberculosis": "A16.9",
    "common cold": "J00",
    "pneumonia": "J18.9",
    "dimorphic hemmorhoids(piles)": "K64.9",
    "heart attack": "I21.9",
    "varicose veins": "I83.9",
    "hypothyroidism": "E03.9",
    "hyperthyroidism": "E05.9",
    "hypoglycemia": "E16.2",
    "osteoarthritis": "M19.9",
    "arthritis": "M13.9",
    "(vertigo) paroxymal positional vertigo": "H81.1",
    "acne": "L70.0",
    "urinary tract infection": "N39.0",
    "psoriasis": "L40.9",
    "impetigo": "L01.0",
}


class DiagnosisAgent(BaseAgent):
    """Hybrid diagnosis engine combining ML models, RAG, and LLM reasoning.

    The agent operates in three stages:
    1. **ML Prediction** — Run the binary symptom vector through the
       pre-trained ensemble model to get top-k conditions with probabilities.
    2. **RAG Enrichment** — Query ChromaDB for relevant medical context
       about the top candidate conditions.
    3. **LLM Synthesis** — Feed all signals to Claude for final differential
       diagnosis, ICD-10 mapping, and narrative explanation.

    Attributes:
        model: The loaded ML ensemble model (scikit-learn compatible).
        rag_service: The RAG retrieval service (injected at init or runtime).
    """

    name = "DiagnosisAgent"
    temperature = 0.2
    json_mode = True

    def __init__(self) -> None:
        self._model = None
        self._label_encoder = None
        self._rag_service = None

    @property
    def system_prompt(self) -> str:
        """System prompt for the diagnostic reasoning phase."""
        return """You are a senior diagnostic AI assistant with expertise in internal
medicine. You synthesise evidence from multiple sources to produce a differential
diagnosis.

INPUT YOU WILL RECEIVE:
- Patient's symptoms (canonical names from a medical dataset)
- ML model predictions with confidence scores (from an ensemble classifier)
- Retrieved medical knowledge (from a RAG knowledge base)
- Patient context (age, sex, medical history — if available)

YOUR TASK:
1. Evaluate the ML model's top predictions critically.
2. Cross-reference with the retrieved medical knowledge.
3. Apply clinical reasoning to produce a FINAL differential diagnosis.
4. For each condition, provide:
   - Confidence score (0.0–1.0)
   - ICD-10 code if known
   - Brief explanation of why this condition fits the symptoms
5. Assign an overall severity assessment.
6. Note any RED FLAG symptoms requiring immediate attention.

RESPOND WITH THIS EXACT JSON STRUCTURE:
{
    "primary_diagnosis": {
        "condition": "Condition Name",
        "confidence": 0.85,
        "icd10_code": "X99.9",
        "explanation": "Brief explanation of why this is the top diagnosis"
    },
    "differential_diagnoses": [
        {
            "condition": "Alternative 1",
            "confidence": 0.65,
            "icd10_code": "Y88.8",
            "explanation": "Why this is a possibility"
        }
    ],
    "red_flags": ["List of any alarming symptoms"],
    "overall_severity": 7,
    "reasoning_summary": "Overall clinical reasoning narrative",
    "recommended_tests": ["Blood test", "X-ray"],
    "confidence_explanation": "Why the model is/isn't confident about this diagnosis"
}"""

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained ML ensemble model from disk.

        Args:
            model_path: Path to the joblib-serialised model.
        """
        try:
            import joblib
            artifacts = joblib.load(model_path)
            self._model = artifacts.get("model")
            self._label_encoder = artifacts.get("label_encoder")
            logger.info("Loaded ML model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load ML model: %s", exc)

    def set_rag_service(self, rag_service: Any) -> None:
        """Inject the RAG retrieval service.

        Args:
            rag_service: An instance with a ``query()`` async method.
        """
        self._rag_service = rag_service

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the full diagnostic pipeline.

        Args:
            input_data: Must contain:
                - ``symptoms`` (list[str]): Canonical symptom names.
                - ``feature_vector`` (list[int], optional): Binary feature vector.
                - ``patient_context`` (dict, optional): Age, sex, history.
                - ``severity_scores`` (dict, optional): Per-symptom severity.

        Returns:
            Complete diagnosis result with differential, ICD-10, and disclaimer.
        """
        symptoms = input_data.get("symptoms", [])
        feature_vector = input_data.get("feature_vector")
        patient_context = input_data.get("patient_context", {})
        severity_scores = input_data.get("severity_scores", {})

        if not symptoms:
            return self._add_disclaimer({
                "primary_diagnosis": None,
                "differential_diagnoses": [],
                "error": "No symptoms provided for diagnosis.",
            })

        # ── Stage 1: ML Ensemble Prediction ─────────────────────────
        ml_predictions = await self._run_ml_prediction(symptoms, feature_vector)

        # ── Stage 2: RAG Retrieval ──────────────────────────────────
        rag_context = await self._retrieve_context(symptoms, ml_predictions)

        # ── Stage 3: LLM Synthesis ──────────────────────────────────
        result = await self._synthesise_diagnosis(
            symptoms, ml_predictions, rag_context, patient_context, severity_scores,
        )

        # Enrich with ICD-10 codes from our mapping if LLM missed them
        result = self._enrich_icd10(result)

        return self._add_disclaimer(result)

    async def _run_ml_prediction(
        self,
        symptoms: list[str],
        feature_vector: list[int] | None,
    ) -> list[dict[str, Any]]:
        """Run the ML ensemble model if loaded.

        Args:
            symptoms: Canonical symptom names.
            feature_vector: Binary feature vector (132 features).

        Returns:
            List of {condition, confidence} dicts from the model.
        """
        if self._model is None or feature_vector is None:
            logger.info("ML model not available; skipping ensemble prediction")
            return []

        try:
            X = np.array(feature_vector).reshape(1, -1)

            # Get probability predictions
            if hasattr(self._model, "predict_proba"):
                probas = self._model.predict_proba(X)[0]
                top_indices = np.argsort(probas)[::-1][:5]

                predictions = []
                for idx in top_indices:
                    if probas[idx] > 0.01:  # Only include non-trivial predictions
                        condition = (
                            self._label_encoder.inverse_transform([idx])[0]
                            if self._label_encoder
                            else f"condition_{idx}"
                        )
                        predictions.append({
                            "condition": condition,
                            "confidence": round(float(probas[idx]), 4),
                            "source": "ml_ensemble",
                        })
                return predictions
            else:
                # Fallback: single prediction without probabilities
                pred = self._model.predict(X)[0]
                condition = (
                    self._label_encoder.inverse_transform([pred])[0]
                    if self._label_encoder
                    else str(pred)
                )
                return [{"condition": condition, "confidence": 0.7, "source": "ml_ensemble"}]

        except Exception as exc:
            logger.error("ML prediction failed: %s", exc)
            return []

    async def _retrieve_context(
        self,
        symptoms: list[str],
        ml_predictions: list[dict[str, Any]],
    ) -> str:
        """Retrieve relevant medical context from the RAG pipeline.

        Args:
            symptoms: Canonical symptom names.
            ml_predictions: ML model predictions for targeted retrieval.

        Returns:
            Concatenated context string from retrieved documents.
        """
        if self._rag_service is None:
            return "No RAG knowledge base available."

        try:
            # Build query from symptoms and top predictions
            query_parts = [s.replace("_", " ") for s in symptoms[:10]]
            for pred in ml_predictions[:3]:
                query_parts.append(pred["condition"])

            query = " ".join(query_parts)
            results = await self._rag_service.query(query, top_k=5)

            if not results:
                return "No relevant medical context found in knowledge base."

            context_chunks = []
            for i, doc in enumerate(results, 1):
                content = doc.get("content", doc.get("text", ""))
                source = doc.get("source", "unknown")
                context_chunks.append(f"[Source {i}: {source}]\n{content}")

            return "\n\n".join(context_chunks)

        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc)
            return "RAG retrieval error — proceeding without knowledge base context."

    async def _synthesise_diagnosis(
        self,
        symptoms: list[str],
        ml_predictions: list[dict[str, Any]],
        rag_context: str,
        patient_context: dict[str, Any],
        severity_scores: dict[str, Any],
    ) -> dict[str, Any]:
        """Use Claude to synthesise all data into a final diagnosis.

        Args:
            symptoms: Canonical symptom names.
            ml_predictions: ML model output.
            rag_context: Retrieved knowledge base context.
            patient_context: Patient demographics and history.
            severity_scores: Per-symptom severity assessments.

        Returns:
            Structured diagnosis dictionary.
        """
        # Build the comprehensive prompt
        symptoms_text = self._format_symptoms_for_prompt(symptoms)

        ml_text = "No ML model predictions available."
        if ml_predictions:
            ml_lines = [
                f"  {i}. {p['condition']} (confidence: {p['confidence']:.2%})"
                for i, p in enumerate(ml_predictions, 1)
            ]
            ml_text = "ML Ensemble Predictions:\n" + "\n".join(ml_lines)

        context_text = f"Patient context: {patient_context}" if patient_context else "No patient context available."
        severity_text = f"Severity scores: {severity_scores}" if severity_scores else ""

        prompt = f"""PATIENT SYMPTOMS:
{symptoms_text}

{ml_text}

RETRIEVED MEDICAL KNOWLEDGE:
{rag_context}

{context_text}
{severity_text}

Based on ALL the above information, provide your differential diagnosis.
Consider the ML model's predictions but apply your own clinical reasoning.
If the ML predictions seem incorrect given the symptoms, explain why."""

        return await self._call_llm(prompt)

    def _enrich_icd10(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add ICD-10 codes from our mapping if the LLM didn't provide them.

        Args:
            result: Diagnosis result dict.

        Returns:
            Enriched result with ICD-10 codes.
        """
        # Primary diagnosis
        primary = result.get("primary_diagnosis")
        if isinstance(primary, dict) and not primary.get("icd10_code"):
            condition = primary.get("condition", "").lower().strip()
            if condition in ICD10_MAP:
                primary["icd10_code"] = ICD10_MAP[condition]

        # Differential diagnoses
        for diff in result.get("differential_diagnoses", []):
            if isinstance(diff, dict) and not diff.get("icd10_code"):
                condition = diff.get("condition", "").lower().strip()
                if condition in ICD10_MAP:
                    diff["icd10_code"] = ICD10_MAP[condition]

        return result
