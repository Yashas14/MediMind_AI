"""
Data loading service — initialise agents and RAG on startup.

Loads the CSV datasets into agent data structures and ingests
documents into ChromaDB. Called during application startup.
"""

import os
from pathlib import Path
from typing import Any

from app.agents.orchestrator import get_orchestrator
from app.agents.precaution import load_description_data, load_precaution_data
from app.agents.symptom_extractor import load_severity_data
from app.core.logging import get_logger
from app.rag.ingestion import load_all_medical_data
from app.rag.vector_store import VectorStoreService

logger = get_logger(__name__)

# Default data paths (can be overridden via env vars)
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data/original")
DEFAULT_MASTER_DIR = os.getenv("MASTER_DATA_DIR", "./data/master")
DEFAULT_MODEL_PATH = os.getenv(
    "ML_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent.parent.parent / "ml-training" / "model_artifacts" / "ensemble_model.joblib"),
)

# Fallback to the original project paths for local dev
# (one level up from backend/ to project root, then chatbot/)
FALLBACK_DATA_DIR = "../chatbot/Data"
FALLBACK_MASTER_DIR = "../chatbot/MasterData"


def _resolve_dir(primary: str, fallback: str) -> str:
    """Find the first directory that exists."""
    if Path(primary).exists():
        return primary
    if Path(fallback).exists():
        return fallback
    return primary  # Return primary even if missing; errors handled downstream


def load_agent_data() -> None:
    """Load CSV data into agent in-memory structures.

    Loads:
    - Symptom severity weights into SymptomExtractorAgent
    - Disease descriptions into PrecautionAgent
    - Disease precautions into PrecautionAgent
    """
    master_dir = _resolve_dir(DEFAULT_MASTER_DIR, FALLBACK_MASTER_DIR)

    severity_path = os.path.join(master_dir, "Symptom_severity.csv")
    description_path = os.path.join(master_dir, "symptom_Description.csv")
    precaution_path = os.path.join(master_dir, "symptom_precaution.csv")

    load_severity_data(severity_path)
    load_description_data(description_path)
    load_precaution_data(precaution_path)

    logger.info("Agent data loaded from %s", master_dir)


def ingest_to_vector_store() -> dict[str, Any]:
    """Ingest all medical CSV data into ChromaDB.

    Returns:
        Stats dict with document count and status.
    """
    data_dir = _resolve_dir(DEFAULT_DATA_DIR, FALLBACK_DATA_DIR)
    master_dir = _resolve_dir(DEFAULT_MASTER_DIR, FALLBACK_MASTER_DIR)

    logger.info("Ingesting medical data from data=%s master=%s", data_dir, master_dir)

    try:
        documents = load_all_medical_data(data_dir, master_dir)

        if not documents:
            logger.warning("No documents found to ingest")
            return {"status": "warning", "documents_ingested": 0}

        vs = VectorStoreService()
        count = vs.ingest_documents(documents)

        stats = vs.get_stats()
        logger.info("Ingestion complete: %d documents in store", stats["document_count"])

        return {
            "status": "success",
            "documents_ingested": count,
            "total_in_store": stats["document_count"],
        }

    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


def load_ml_model() -> None:
    """Load the trained ML ensemble model into the DiagnosisAgent.

    Searches for the model file at DEFAULT_MODEL_PATH (configurable
    via ML_MODEL_PATH env var). If the model file doesn't exist,
    diagnosis will fall back to LLM-only reasoning.
    """
    model_path = DEFAULT_MODEL_PATH

    # Also check a few common locations
    candidates = [
        model_path,
        "./ml-training/model_artifacts/ensemble_model.joblib",
        "../ml-training/model_artifacts/ensemble_model.joblib",
    ]

    resolved = None
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            resolved = str(p.resolve())
            break

    if resolved is None:
        logger.warning(
            "ML ensemble model not found at any of: %s — "
            "diagnosis will use LLM-only mode. Run ml-training/train_ensemble.py first.",
            candidates,
        )
        return

    try:
        orchestrator = get_orchestrator()
        orchestrator.load_ml_model(resolved)
        logger.info("ML ensemble model loaded from %s", resolved)
    except Exception as exc:
        logger.error("Failed to load ML model: %s", exc)


def initialise_all() -> None:
    """Full startup initialisation: load agent data + ML model + vector store.

    Called during application lifespan startup.
    """
    load_agent_data()

    # Load ML ensemble model into DiagnosisAgent
    load_ml_model()

    try:
        result = ingest_to_vector_store()
        logger.info("Vector store initialisation: %s", result.get("status"))
    except Exception as exc:
        logger.warning(
            "Vector store initialisation skipped (ChromaDB may not be available): %s",
            exc,
        )
