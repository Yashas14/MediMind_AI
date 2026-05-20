"""
Drug information endpoints — powered by OpenFDA.

Provides drug label search, interaction checks, adverse event lookups,
and recall information from the US FDA's public API.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.core.logging import get_logger
from app.services.openfda import get_openfda_service

router = APIRouter(prefix="/drugs", tags=["Drugs"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class DrugSearchRequest(BaseModel):
    """Request for searching drug information."""

    drug_name: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=3, ge=1, le=10)


class DrugInteractionRequest(BaseModel):
    """Request for checking drug interactions."""

    drug_names: list[str] = Field(..., min_length=2, max_length=10)


class DrugEventRequest(BaseModel):
    """Request for adverse event search."""

    drug_name: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)


# ── Endpoints ───────────────────────────────────────────────────────

@router.post(
    "/search",
    summary="Search drug label information",
    response_model=dict[str, Any],
)
async def search_drug(
    body: DrugSearchRequest,
    user: CurrentUser,
) -> dict[str, Any]:
    """Search FDA drug labels for indications, warnings, and interactions.

    Returns structured data from the FDA drug label database including
    brand name, generic name, indications, warnings, and interactions.
    """
    logger.info("Drug search by user %s: %s", user.id, body.drug_name)
    service = get_openfda_service()
    return await service.search_drug_label(body.drug_name, body.limit)


@router.post(
    "/interactions",
    summary="Check drug-drug interactions",
    response_model=dict[str, Any],
)
async def check_interactions(
    body: DrugInteractionRequest,
    user: CurrentUser,
) -> dict[str, Any]:
    """Check for known interactions between multiple drugs.

    Cross-references FDA drug label data to find interaction warnings.
    Useful for patients taking multiple medications.
    """
    logger.info(
        "Drug interaction check by user %s: %s",
        user.id,
        ", ".join(body.drug_names),
    )
    service = get_openfda_service()
    return await service.check_drug_interactions(body.drug_names)


@router.post(
    "/adverse-events",
    summary="Search adverse event reports",
    response_model=dict[str, Any],
)
async def search_adverse_events(
    body: DrugEventRequest,
    user: CurrentUser,
) -> dict[str, Any]:
    """Search FDA Adverse Event Reporting System (FAERS) data.

    Returns reported adverse events for a given drug, including
    seriousness classification and patient demographics.
    """
    logger.info("Adverse event search by user %s: %s", user.id, body.drug_name)
    service = get_openfda_service()
    return await service.search_adverse_events(body.drug_name, body.limit)


@router.post(
    "/recalls",
    summary="Search drug recall/enforcement data",
    response_model=dict[str, Any],
)
async def search_recalls(
    body: DrugSearchRequest,
    user: CurrentUser,
) -> dict[str, Any]:
    """Search FDA drug recall and enforcement actions.

    Returns active and past recall information for a given drug product.
    """
    logger.info("Drug recall search by user %s: %s", user.id, body.drug_name)
    service = get_openfda_service()
    return await service.search_recalls(body.drug_name, body.limit)
