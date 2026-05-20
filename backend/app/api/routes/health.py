"""
Health-check and system status endpoints.

These endpoints are unauthenticated and used by load balancers,
Kubernetes probes, and monitoring systems.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get(
    "/health",
    summary="Basic health check",
    response_model=dict[str, Any],
)
async def health_check() -> dict[str, Any]:
    """Return a simple liveness probe response.

    Used by Kubernetes ``livenessProbe`` and Docker ``HEALTHCHECK``.
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/ready",
    summary="Readiness check (includes DB connectivity)",
    response_model=dict[str, Any],
)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Verify that all critical dependencies are reachable.

    Checks:
    - PostgreSQL connectivity
    - (future) Redis, ChromaDB
    """
    checks: dict[str, str] = {}

    # PostgreSQL
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
