"""
Test suite for AI agents.

Tests the agent system with mocked Claude API responses to verify:
- SymptomExtractorAgent extraction logic
- TriageAgent rule-based checks
- DiagnosisAgent pipeline
- Orchestrator flow
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.symptom_extractor import (
    CANONICAL_SYMPTOMS,
    SymptomExtractorAgent,
)
from app.agents.triage import (
    EMERGENCY_SYMPTOMS,
    URGENT_SYMPTOMS,
    TriageAgent,
)
from app.agents.diagnosis import DiagnosisAgent, ICD10_MAP
from app.agents.precaution import PrecautionAgent
from app.agents.medical_summary import MedicalSummaryAgent


# ── SymptomExtractorAgent Tests ─────────────────────────────────────

class TestSymptomExtractor:
    """Tests for the SymptomExtractorAgent."""

    def setup_method(self) -> None:
        """Initialize agent for each test."""
        self.agent = SymptomExtractorAgent()

    def test_canonical_symptoms_count(self) -> None:
        """Should have 132 canonical symptoms matching the dataset."""
        assert len(CANONICAL_SYMPTOMS) == 132

    def test_build_feature_vector_all_zeros(self) -> None:
        """Empty symptoms should produce a zero vector."""
        vector = self.agent.build_feature_vector([])
        assert len(vector) == 132
        assert sum(vector) == 0

    def test_build_feature_vector_with_symptoms(self) -> None:
        """Should set 1 for matching symptoms, 0 for others."""
        symptoms = ["headache", "nausea", "high_fever"]
        vector = self.agent.build_feature_vector(symptoms)
        assert len(vector) == 132
        assert sum(vector) == 3

        # Verify specific positions
        headache_idx = CANONICAL_SYMPTOMS.index("headache")
        nausea_idx = CANONICAL_SYMPTOMS.index("nausea")
        assert vector[headache_idx] == 1
        assert vector[nausea_idx] == 1

    def test_get_canonical_names(self) -> None:
        """Should extract canonical names from extraction result."""
        result = {
            "matched_symptoms": [
                {"canonical_name": "headache", "severity": 7},
                {"canonical_name": "nausea", "severity": 5},
            ],
        }
        names = self.agent.get_canonical_names(result)
        assert names == ["headache", "nausea"]

    def test_get_canonical_names_empty(self) -> None:
        """Should return empty list for empty result."""
        names = self.agent.get_canonical_names({})
        assert names == []

    @pytest.mark.asyncio
    async def test_process_empty_input(self) -> None:
        """Should return error for empty text input."""
        result = await self.agent.process({"text": ""})
        assert result.get("error") == "No text provided"
        assert "disclaimer" in result

    @pytest.mark.asyncio
    async def test_process_with_mocked_llm(self) -> None:
        """Should process text and return structured symptoms."""
        mock_response = {
            "matched_symptoms": [
                {
                    "canonical_name": "headache",
                    "original_text": "bad headache",
                    "severity": 7,
                    "body_region": "head",
                    "duration": "2 days",
                    "confidence": 0.95,
                },
            ],
            "unmatched_symptoms": [],
            "overall_severity": 7,
            "symptom_count": 1,
            "follow_up_questions": ["How would you rate the pain?"],
            "language_detected": "en",
        }

        with patch(
            "app.agents.base.call_claude_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await self.agent.process({
                "text": "I have a bad headache for two days",
            })

        assert len(result["matched_symptoms"]) == 1
        assert result["matched_symptoms"][0]["canonical_name"] == "headache"
        assert "disclaimer" in result


# ── TriageAgent Tests ───────────────────────────────────────────────

class TestTriageAgent:
    """Tests for the TriageAgent."""

    def setup_method(self) -> None:
        """Initialize agent for each test."""
        self.agent = TriageAgent()

    @pytest.mark.asyncio
    async def test_emergency_rule_based(self) -> None:
        """Chest pain should trigger EMERGENCY without LLM call."""
        result = await self.agent.process({
            "symptoms": ["chest_pain", "breathlessness"],
        })
        assert result["triage_level"] == "EMERGENCY"
        assert result.get("rule_triggered") is True
        assert result["confidence"] >= 0.95

    @pytest.mark.asyncio
    async def test_urgent_rule_based(self) -> None:
        """High fever should trigger URGENT without LLM call."""
        result = await self.agent.process({
            "symptoms": ["high_fever", "headache"],
        })
        assert result["triage_level"] == "URGENT"

    @pytest.mark.asyncio
    async def test_empty_symptoms(self) -> None:
        """Empty symptoms should return ROUTINE with error."""
        result = await self.agent.process({"symptoms": []})
        assert result["triage_level"] == "ROUTINE"
        assert result.get("error") == "No symptoms provided"

    @pytest.mark.asyncio
    async def test_safety_override(self) -> None:
        """LLM downgrading urgent symptoms should be overridden."""
        mock_result = {
            "triage_level": "SELF_CARE",
            "confidence": 0.8,
            "reasoning": "Seems mild",
        }

        with patch(
            "app.agents.base.call_claude_json",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await self.agent.process({
                "symptoms": ["high_fever", "breathlessness", "fatigue"],
                "overall_severity": 7,
            })

        # Should be upgraded to at least URGENT due to safety override
        assert result["triage_level"] in ("URGENT", "EMERGENCY")

    def test_emergency_symptoms_set(self) -> None:
        """Emergency symptoms should all be recognized canonical names."""
        for symptom in EMERGENCY_SYMPTOMS:
            assert symptom in CANONICAL_SYMPTOMS or "_" in symptom

    def test_urgent_symptoms_set(self) -> None:
        """Urgent symptoms should all be recognized canonical names."""
        for symptom in URGENT_SYMPTOMS:
            assert symptom in CANONICAL_SYMPTOMS


# ── DiagnosisAgent Tests ────────────────────────────────────────────

class TestDiagnosisAgent:
    """Tests for the DiagnosisAgent."""

    def setup_method(self) -> None:
        """Initialize agent for each test."""
        self.agent = DiagnosisAgent()

    def test_icd10_mapping(self) -> None:
        """Should have ICD-10 codes for common conditions."""
        assert ICD10_MAP["malaria"] == "B54"
        assert ICD10_MAP["diabetes"] == "E14"
        assert ICD10_MAP["common cold"] == "J00"
        assert len(ICD10_MAP) > 30

    @pytest.mark.asyncio
    async def test_process_empty_symptoms(self) -> None:
        """Should return error for empty symptom list."""
        result = await self.agent.process({"symptoms": []})
        assert result.get("error") == "No symptoms provided for diagnosis."
        assert "disclaimer" in result

    @pytest.mark.asyncio
    async def test_process_with_mocked_llm(self) -> None:
        """Should produce diagnosis with mocked LLM and no ML model."""
        mock_response = {
            "primary_diagnosis": {
                "condition": "Common Cold",
                "confidence": 0.78,
                "icd10_code": "J00",
                "explanation": "Symptoms consistent with viral upper respiratory infection",
            },
            "differential_diagnoses": [
                {"condition": "Allergic Rhinitis", "confidence": 0.45, "icd10_code": "J30.9"},
            ],
            "red_flags": [],
            "overall_severity": 3,
            "reasoning_summary": "Mild viral infection likely",
        }

        with patch(
            "app.agents.base.call_claude_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await self.agent.process({
                "symptoms": ["cough", "runny_nose", "mild_fever"],
            })

        assert result["primary_diagnosis"]["condition"] == "Common Cold"
        assert "disclaimer" in result

    def test_enrich_icd10(self) -> None:
        """Should add ICD-10 codes from the mapping."""
        result = {
            "primary_diagnosis": {
                "condition": "Malaria",
                "confidence": 0.9,
            },
            "differential_diagnoses": [
                {"condition": "Dengue", "confidence": 0.4},
            ],
        }
        enriched = self.agent._enrich_icd10(result)
        assert enriched["primary_diagnosis"]["icd10_code"] == "B54"
        assert enriched["differential_diagnoses"][0]["icd10_code"] == "A90"


# ── PrecautionAgent Tests ──────────────────────────────────────────

class TestPrecautionAgent:
    """Tests for the PrecautionAgent."""

    @pytest.mark.asyncio
    async def test_process_with_mocked_llm(self) -> None:
        """Should generate precautions from diagnosis."""
        agent = PrecautionAgent()

        mock_response = {
            "condition": "Common Cold",
            "immediate_actions": [
                {"action": "Rest", "priority": "high", "details": "Get plenty of sleep"},
            ],
            "warning_signs": ["High fever", "Difficulty breathing"],
            "consult_doctor": True,
        }

        with patch(
            "app.agents.base.call_claude_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.process({
                "diagnosis": {"primary_diagnosis": {"condition": "Common Cold"}},
                "symptoms": ["cough", "runny_nose"],
            })

        assert result["condition"] == "Common Cold"
        assert "disclaimer" in result


# ── MedicalSummaryAgent Tests ──────────────────────────────────────

class TestMedicalSummaryAgent:
    """Tests for the MedicalSummaryAgent."""

    @pytest.mark.asyncio
    async def test_process_with_mocked_llm(self) -> None:
        """Should generate a summary from all agent outputs."""
        agent = MedicalSummaryAgent()

        mock_response = {
            "summary_title": "AI Health Consultation Summary",
            "chief_complaint": "Headache and nausea",
            "assessment": {
                "primary_condition": "Migraine",
                "confidence": 0.82,
            },
            "recommendations": {
                "immediate": ["Rest in a dark room"],
                "follow_up": "See a neurologist if recurring",
            },
        }

        with patch(
            "app.agents.base.call_claude_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await agent.process({
                "symptoms_result": {"matched_symptoms": []},
                "diagnosis_result": {"primary_diagnosis": {"condition": "Migraine"}},
                "triage_result": {"triage_level": "ROUTINE"},
                "precaution_result": {},
            })

        assert result["summary_title"] == "AI Health Consultation Summary"
        assert "disclaimer" in result
