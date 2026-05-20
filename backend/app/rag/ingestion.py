"""
Medical knowledge ingestion pipeline.

Reads the original healthcare-chatbot CSV datasets and transforms them
into structured documents suitable for vector embedding and storage
in ChromaDB. Also supports ingestion of PubMed abstracts and custom
medical knowledge sources.

Data sources:
- Training.csv / Testing.csv — symptom–disease mappings
- symptom_Description.csv — disease descriptions
- symptom_precaution.csv — recommended precautions
- Symptom_severity.csv — symptom severity weights
"""

import csv
import hashlib
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def generate_doc_id(text: str) -> str:
    """Generate a deterministic document ID from content.

    Args:
        text: Document text.

    Returns:
        MD5 hex digest truncated to 16 chars.
    """
    return hashlib.md5(text.encode()).hexdigest()[:16]


def load_disease_descriptions(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse symptom_Description.csv into structured documents.

    Each row becomes a document with the disease name as metadata.

    Args:
        csv_path: Path to symptom_Description.csv.

    Returns:
        List of document dicts with ``id``, ``text``, and ``metadata``.
    """
    docs: list[dict[str, Any]] = []
    path = Path(csv_path)

    if not path.exists():
        logger.warning("Description file not found: %s", csv_path)
        return docs

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) < 2 or not parts[1].strip():
                continue
            disease = parts[0].strip()
            description = parts[1].strip().strip('"')

            text = f"Disease: {disease}\n\nDescription: {description}"
            docs.append({
                "id": generate_doc_id(text),
                "text": text,
                "metadata": {
                    "source": "symptom_description_csv",
                    "disease": disease,
                    "type": "disease_description",
                },
            })

    logger.info("Loaded %d disease descriptions from %s", len(docs), csv_path)
    return docs


def load_disease_precautions(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse symptom_precaution.csv into structured documents.

    Args:
        csv_path: Path to symptom_precaution.csv.

    Returns:
        List of document dicts.
    """
    docs: list[dict[str, Any]] = []
    path = Path(csv_path)

    if not path.exists():
        logger.warning("Precaution file not found: %s", csv_path)
        return docs

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            disease = parts[0]
            precautions = [p for p in parts[1:] if p]

            if not precautions:
                continue

            text = (
                f"Disease: {disease}\n\n"
                f"Recommended Precautions:\n"
                + "\n".join(f"- {p}" for p in precautions)
            )
            docs.append({
                "id": generate_doc_id(text),
                "text": text,
                "metadata": {
                    "source": "symptom_precaution_csv",
                    "disease": disease,
                    "type": "precautions",
                },
            })

    logger.info("Loaded %d precaution records from %s", len(docs), csv_path)
    return docs


def load_symptom_severity(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse Symptom_severity.csv into structured documents.

    Args:
        csv_path: Path to Symptom_severity.csv.

    Returns:
        List of document dicts.
    """
    docs: list[dict[str, Any]] = []
    path = Path(csv_path)

    if not path.exists():
        logger.warning("Severity file not found: %s", csv_path)
        return docs

    severity_map: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                symptom = parts[0].strip()
                try:
                    weight = int(parts[1].strip())
                    severity_map[symptom] = weight
                except ValueError:
                    continue

    # Group by severity level for richer context
    for level_name, (low, high) in [
        ("mild", (1, 3)),
        ("moderate", (4, 6)),
        ("severe", (7, 10)),
    ]:
        symptoms = [s for s, w in severity_map.items() if low <= w <= high]
        if symptoms:
            text = (
                f"Symptom Severity Level: {level_name.upper()} (weight {low}-{high})\n\n"
                f"Symptoms in this category:\n"
                + "\n".join(f"- {s.replace('_', ' ')} (weight: {severity_map[s]})" for s in symptoms)
            )
            docs.append({
                "id": generate_doc_id(text),
                "text": text,
                "metadata": {
                    "source": "symptom_severity_csv",
                    "severity_level": level_name,
                    "type": "severity_classification",
                },
            })

    logger.info("Loaded severity data for %d symptoms from %s", len(severity_map), csv_path)
    return docs


def load_disease_symptom_mappings(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse Training.csv to create disease–symptom mapping documents.

    Each unique disease gets a document listing all its associated symptoms.

    Args:
        csv_path: Path to Training.csv.

    Returns:
        List of document dicts.
    """
    docs: list[dict[str, Any]] = []
    path = Path(csv_path)

    if not path.exists():
        logger.warning("Training file not found: %s", csv_path)
        return docs

    # Read CSV
    disease_symptoms: dict[str, set[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Last column is the prognosis/disease
            symptom_cols = header[:-1]

            for row in reader:
                if len(row) < len(header):
                    continue
                disease = row[-1].strip()
                if not disease:
                    continue

                active_symptoms = set()
                for i, val in enumerate(row[:-1]):
                    try:
                        if int(val.strip()) == 1:
                            active_symptoms.add(symptom_cols[i].strip())
                    except (ValueError, IndexError):
                        continue

                if disease not in disease_symptoms:
                    disease_symptoms[disease] = set()
                disease_symptoms[disease] |= active_symptoms

    except Exception as exc:
        logger.error("Failed to parse Training.csv: %s", exc)
        return docs

    # Create documents
    for disease, symptoms in disease_symptoms.items():
        sorted_symptoms = sorted(symptoms)
        text = (
            f"Disease: {disease}\n\n"
            f"Associated Symptoms ({len(sorted_symptoms)} total):\n"
            + "\n".join(f"- {s.replace('_', ' ')}" for s in sorted_symptoms)
        )
        docs.append({
            "id": generate_doc_id(text),
            "text": text,
            "metadata": {
                "source": "training_csv",
                "disease": disease,
                "type": "disease_symptom_mapping",
                "symptom_count": len(sorted_symptoms),
            },
        })

    logger.info(
        "Loaded symptom mappings for %d diseases from %s",
        len(disease_symptoms), csv_path,
    )
    return docs


def load_all_medical_data(data_dir: str, master_dir: str) -> list[dict[str, Any]]:
    """Load and merge all medical data sources into a unified document list.

    Args:
        data_dir: Path to the ``Data/`` directory (Training.csv, etc.).
        master_dir: Path to the ``MasterData/`` directory.

    Returns:
        Combined list of all documents ready for vector store ingestion.
    """
    all_docs: list[dict[str, Any]] = []

    all_docs.extend(load_disease_descriptions(Path(master_dir) / "symptom_Description.csv"))
    all_docs.extend(load_disease_precautions(Path(master_dir) / "symptom_precaution.csv"))
    all_docs.extend(load_symptom_severity(Path(master_dir) / "Symptom_severity.csv"))
    all_docs.extend(load_disease_symptom_mappings(Path(data_dir) / "Training.csv"))

    logger.info("Total documents prepared for ingestion: %d", len(all_docs))
    return all_docs
