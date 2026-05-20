"""
OpenFDA drug interaction and adverse-event lookup service.

Queries the US FDA's public API for:
- Drug label information (indications, warnings, interactions)
- Adverse event reports (FAERS data)
- Drug recall information

API docs: https://open.fda.gov/apis/
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

OPENFDA_DRUG_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_DRUG_EVENT_URL = "https://api.fda.gov/drug/event.json"
OPENFDA_DRUG_RECALL_URL = "https://api.fda.gov/drug/enforcement.json"

# Max results per query
_DEFAULT_LIMIT = 5
_TIMEOUT = 15.0


def _build_params(
    search: str,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, str]:
    """Build query params, including the optional API key."""
    params: dict[str, str] = {"search": search, "limit": str(limit)}
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    return params


class OpenFDAService:
    """Async client for OpenFDA drug APIs."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Drug Label Search ───────────────────────────────────────────

    async def search_drug_label(
        self,
        drug_name: str,
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Search FDA drug labels for a given drug name.

        Returns structured label data including indications, warnings,
        contraindications, and drug interactions.

        Args:
            drug_name: Generic or brand name of the drug.
            limit: Max results to return.

        Returns:
            Dict with ``results`` list and ``meta`` information.
        """
        client = await self._get_client()
        search = f'openfda.brand_name:"{drug_name}"+openfda.generic_name:"{drug_name}"'
        params = _build_params(search, limit)

        try:
            resp = await client.get(OPENFDA_DRUG_LABEL_URL, params=params)
            if resp.status_code == 404:
                return {"results": [], "meta": {"total": 0}, "query": drug_name}
            resp.raise_for_status()
            data = resp.json()

            # Extract key fields from each result
            results = []
            for item in data.get("results", []):
                results.append({
                    "brand_name": _first(item.get("openfda", {}).get("brand_name")),
                    "generic_name": _first(item.get("openfda", {}).get("generic_name")),
                    "manufacturer": _first(item.get("openfda", {}).get("manufacturer_name")),
                    "route": _first(item.get("openfda", {}).get("route")),
                    "indications_and_usage": _truncate(_first(item.get("indications_and_usage"))),
                    "warnings": _truncate(_first(item.get("warnings"))),
                    "drug_interactions": _truncate(_first(item.get("drug_interactions"))),
                    "contraindications": _truncate(_first(item.get("contraindications"))),
                    "adverse_reactions": _truncate(_first(item.get("adverse_reactions"))),
                    "dosage_and_administration": _truncate(
                        _first(item.get("dosage_and_administration"))
                    ),
                    "pregnancy": _truncate(_first(item.get("pregnancy"))),
                })

            return {
                "results": results,
                "meta": {"total": data.get("meta", {}).get("results", {}).get("total", 0)},
                "query": drug_name,
            }

        except httpx.HTTPStatusError as exc:
            logger.error("OpenFDA label search failed: %s", exc)
            return {"results": [], "meta": {"total": 0}, "error": str(exc), "query": drug_name}
        except httpx.HTTPError as exc:
            logger.error("OpenFDA HTTP error: %s", exc)
            return {"results": [], "meta": {"total": 0}, "error": str(exc), "query": drug_name}

    # ── Drug Interaction Check ──────────────────────────────────────

    async def check_drug_interactions(
        self,
        drug_names: list[str],
    ) -> dict[str, Any]:
        """Check for known interactions between a list of drugs.

        Queries the drug label API for each drug and extracts the
        ``drug_interactions`` section, then cross-references mentions.

        Args:
            drug_names: List of drug names to check pairwise.

        Returns:
            Dict with interaction warnings and per-drug details.
        """
        if len(drug_names) < 2:
            return {
                "drugs": drug_names,
                "interactions_found": False,
                "interactions": [],
                "details": {},
            }

        details: dict[str, Any] = {}
        interactions: list[dict[str, Any]] = []

        for drug in drug_names:
            label = await self.search_drug_label(drug, limit=1)
            details[drug] = label

            if label["results"]:
                interaction_text = label["results"][0].get("drug_interactions", "") or ""
                # Check if any other drug name appears in the interaction text
                for other in drug_names:
                    if other != drug and other.lower() in interaction_text.lower():
                        interactions.append({
                            "drug_a": drug,
                            "drug_b": other,
                            "warning": interaction_text[:500],
                            "source": "FDA Drug Label",
                        })

        return {
            "drugs": drug_names,
            "interactions_found": len(interactions) > 0,
            "interaction_count": len(interactions),
            "interactions": interactions,
            "details": details,
            "disclaimer": (
                "This interaction check uses FDA drug label data. "
                "It may not capture all possible interactions. "
                "Always consult a pharmacist or physician."
            ),
        }

    # ── Adverse Event Reports ───────────────────────────────────────

    async def search_adverse_events(
        self,
        drug_name: str,
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Search FDA Adverse Event Reporting System (FAERS) for a drug.

        Args:
            drug_name: Name of the drug.
            limit: Max results.

        Returns:
            Dict with adverse event reports.
        """
        client = await self._get_client()
        search = f'patient.drug.medicinalproduct:"{drug_name}"'
        params = _build_params(search, limit)

        try:
            resp = await client.get(OPENFDA_DRUG_EVENT_URL, params=params)
            if resp.status_code == 404:
                return {"results": [], "meta": {"total": 0}, "query": drug_name}
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", []):
                reactions = [
                    r.get("reactionmeddrapt", "")
                    for r in item.get("patient", {}).get("reaction", [])
                ]
                results.append({
                    "report_date": item.get("receivedate"),
                    "serious": item.get("serious", "0") == "1",
                    "seriousness_detail": {
                        "death": item.get("seriousnessother") == "1",
                        "hospitalization": item.get("seriousnesshospitalization") == "1",
                        "life_threatening": item.get("seriousnesslifethreatening") == "1",
                    },
                    "reactions": reactions,
                    "patient_age": item.get("patient", {}).get("patientonsetage"),
                    "patient_sex": item.get("patient", {}).get("patientsex"),
                })

            return {
                "results": results,
                "meta": {"total": data.get("meta", {}).get("results", {}).get("total", 0)},
                "query": drug_name,
            }

        except httpx.HTTPError as exc:
            logger.error("OpenFDA event search failed: %s", exc)
            return {"results": [], "meta": {"total": 0}, "error": str(exc), "query": drug_name}

    # ── Drug Recall Search ──────────────────────────────────────────

    async def search_recalls(
        self,
        drug_name: str,
        limit: int = _DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        """Search FDA drug recall/enforcement data.

        Args:
            drug_name: Name of the drug or product.
            limit: Max results.

        Returns:
            Dict with recall records.
        """
        client = await self._get_client()
        search = f'openfda.brand_name:"{drug_name}"+product_description:"{drug_name}"'
        params = _build_params(search, limit)

        try:
            resp = await client.get(OPENFDA_DRUG_RECALL_URL, params=params)
            if resp.status_code == 404:
                return {"results": [], "meta": {"total": 0}, "query": drug_name}
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "recall_number": item.get("recall_number"),
                    "status": item.get("status"),
                    "classification": item.get("classification"),
                    "reason": item.get("reason_for_recall"),
                    "product_description": item.get("product_description"),
                    "recall_initiation_date": item.get("recall_initiation_date"),
                    "voluntary_mandated": item.get("voluntary_mandated"),
                    "distribution_pattern": item.get("distribution_pattern"),
                })

            return {
                "results": results,
                "meta": {"total": data.get("meta", {}).get("results", {}).get("total", 0)},
                "query": drug_name,
            }

        except httpx.HTTPError as exc:
            logger.error("OpenFDA recall search failed: %s", exc)
            return {"results": [], "meta": {"total": 0}, "error": str(exc), "query": drug_name}


# ── Helpers ─────────────────────────────────────────────────────────

def _first(value: Any) -> str | None:
    """Return the first element if a list, or the value itself."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _truncate(text: str | None, max_len: int = 1000) -> str | None:
    """Truncate long text fields to keep payloads manageable."""
    if not text:
        return text
    return text[:max_len] + ("…" if len(text) > max_len else "")


# ── Singleton ───────────────────────────────────────────────────────

_service: OpenFDAService | None = None


def get_openfda_service() -> OpenFDAService:
    """Return the global OpenFDAService singleton."""
    global _service  # noqa: PLW0603
    if _service is None:
        _service = OpenFDAService()
    return _service
