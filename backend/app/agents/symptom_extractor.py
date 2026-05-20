"""
SymptomExtractorAgent — extracts structured symptoms from free-text input.

Uses Claude to parse natural language descriptions into structured
symptom data with severity scores, body regions, duration, and
confidence levels. Maps extracted symptoms to the canonical symptom
list from the training dataset.
"""

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

# Canonical symptoms from the training dataset (132 features)
CANONICAL_SYMPTOMS: list[str] = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability",
    "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
    "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking",
    "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze",
]

# Severity data from Symptom_severity.csv (loaded at import time)
SYMPTOM_SEVERITY_MAP: dict[str, int] = {}


def load_severity_data(csv_path: str) -> None:
    """Load symptom severity weights from the CSV file.

    Args:
        csv_path: Path to Symptom_severity.csv
    """
    global SYMPTOM_SEVERITY_MAP  # noqa: PLW0603
    try:
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    symptom = parts[0].strip().lower()
                    try:
                        weight = int(parts[1].strip())
                        SYMPTOM_SEVERITY_MAP[symptom] = weight
                    except ValueError:
                        continue
        logger.info("Loaded severity data for %d symptoms", len(SYMPTOM_SEVERITY_MAP))
    except FileNotFoundError:
        logger.warning("Symptom severity file not found: %s", csv_path)


class SymptomExtractorAgent(BaseAgent):
    """Extract structured symptoms from natural-language descriptions.

    Given free-text input like *"I've had a pounding headache for three
    days and I feel nauseous"*, this agent produces:
    - Matched canonical symptoms with confidence scores
    - Severity estimates (1–10)
    - Body region mapping
    - Duration parsing
    - Any symptoms mentioned that don't match the canonical list

    The output feeds downstream agents (Diagnosis, Triage).
    """

    name = "SymptomExtractorAgent"
    temperature = 0.1  # Low temp for deterministic extraction
    json_mode = True

    @property
    def system_prompt(self) -> str:
        """System prompt defining the extraction task."""
        symptom_list = ", ".join(CANONICAL_SYMPTOMS[:60])
        symptom_list_2 = ", ".join(CANONICAL_SYMPTOMS[60:])

        return f"""You are a medical symptom extraction specialist. Your task is to
extract structured symptom information from a patient's free-text description.

CANONICAL SYMPTOM LIST (use ONLY these exact names when matching):
{symptom_list},
{symptom_list_2}

INSTRUCTIONS:
1. Read the patient's description carefully.
2. Identify ALL symptoms mentioned — explicit or implied.
3. Map each symptom to the closest match in the CANONICAL SYMPTOM LIST above.
4. For each symptom, estimate:
   - severity: integer 1-10 (1=mild, 5=moderate, 10=severe/life-threatening)
   - body_region: the anatomical area (head, chest, abdomen, limbs, skin, etc.)
   - duration: how long they've had it (parse from text, or "unknown")
   - confidence: float 0.0-1.0 indicating how confident you are in the match
5. Also note any symptoms that DON'T match the canonical list as "unmatched".
6. Provide follow-up questions the assistant should ask to clarify ambiguity.

RESPOND WITH THIS EXACT JSON STRUCTURE:
{{
    "matched_symptoms": [
        {{
            "canonical_name": "headache",
            "original_text": "pounding headache",
            "severity": 7,
            "body_region": "head",
            "duration": "3 days",
            "confidence": 0.95
        }}
    ],
    "unmatched_symptoms": [
        {{
            "original_text": "tingling sensation in fingertips",
            "suggested_canonical": "cold_hands_and_feets",
            "confidence": 0.4
        }}
    ],
    "overall_severity": 6,
    "symptom_count": 2,
    "follow_up_questions": [
        "How would you rate your headache on a scale of 1-10?",
        "Have you experienced any fever?"
    ],
    "language_detected": "en"
}}"""

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Extract symptoms from the user's text.

        Args:
            input_data: Must contain:
                - ``text`` (str): The patient's free-text description.
                - ``language`` (str, optional): ISO code, default "en".

        Returns:
            Structured symptom extraction result with medical disclaimer.
        """
        text = input_data.get("text", "")
        language = input_data.get("language", "en")

        if not text.strip():
            return self._add_disclaimer({
                "matched_symptoms": [],
                "unmatched_symptoms": [],
                "overall_severity": 0,
                "symptom_count": 0,
                "follow_up_questions": ["Could you describe what symptoms you're experiencing?"],
                "error": "No text provided",
            })

        prompt = f"""Patient description (language: {language}):
\"{text}\"

Extract all symptoms from the above description. Be thorough — look for both
explicitly stated and implied symptoms. Use the canonical symptom names provided."""

        result = await self._call_llm(prompt)

        # Enrich with severity weights from the dataset
        if "matched_symptoms" in result:
            for symptom in result["matched_symptoms"]:
                canonical = symptom.get("canonical_name", "")
                if canonical in SYMPTOM_SEVERITY_MAP:
                    symptom["dataset_severity_weight"] = SYMPTOM_SEVERITY_MAP[canonical]

        # Ensure required keys exist
        result.setdefault("matched_symptoms", [])
        result.setdefault("unmatched_symptoms", [])
        result.setdefault("overall_severity", 0)
        result.setdefault("symptom_count", len(result.get("matched_symptoms", [])))
        result.setdefault("follow_up_questions", [])

        return self._add_disclaimer(result)

    def get_canonical_names(self, extraction_result: dict[str, Any]) -> list[str]:
        """Extract just the canonical symptom names from a result.

        Args:
            extraction_result: Output from ``process()``.

        Returns:
            List of canonical symptom name strings.
        """
        return [
            s["canonical_name"]
            for s in extraction_result.get("matched_symptoms", [])
            if "canonical_name" in s
        ]

    def build_feature_vector(self, canonical_names: list[str]) -> list[int]:
        """Convert canonical symptom names into a binary feature vector.

        The vector matches the column order of the Training.csv dataset,
        enabling direct input to the ML ensemble model.

        Args:
            canonical_names: List of canonical symptom names from extraction.

        Returns:
            Binary feature vector (list of 0/1 integers).
        """
        name_set = set(canonical_names)
        return [1 if symptom in name_set else 0 for symptom in CANONICAL_SYMPTOMS]
