"""
Healthcare AI Platform — FastAPI Application Entry Point.

This module initialises the FastAPI application, registers all routers,
configures middleware (CORS, rate limiting, request tracing, audit logging),
and sets up startup/shutdown lifecycle hooks for database and cache connections.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.core.logging import get_logger, setup_logging
from app.core.middleware import (
    AuditLogMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)

# ── Route imports ───────────────────────────────────────────────────
from app.api.routes.admin import router as admin_router
from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.diagnosis import router as diagnosis_router
from app.api.routes.drugs import router as drugs_router
from app.api.routes.health import router as health_router
from app.api.routes.hospitals import router as hospitals_router
from app.api.routes.rag import router as rag_router
from app.api.routes.symptoms import router as symptoms_router
from app.api.routes.users import router as users_router

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


# ── Lifecycle ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle manager.

    Startup:
        - Initialise database tables (dev only; use Alembic in production)
        - Pre-warm caches
        - Log startup banner

    Shutdown:
        - Close database connection pool
        - Flush logs
    """
    logger.info(
        "🚀 Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )

    # Database initialisation (dev only)
    if not settings.is_production:
        try:
            await init_db()
            logger.info("Database tables initialised (dev mode)")
        except Exception as exc:
            logger.warning("Database init skipped: %s", exc)

    # Load agent data and ingest to vector store
    try:
        from app.services.data_loader import initialise_all
        initialise_all()
        logger.info("AI agents and RAG pipeline initialised")
    except Exception as exc:
        logger.warning("Agent initialisation skipped: %s", exc)

    yield

    # Shutdown
    logger.info("Shutting down %s…", settings.app_name)

    # Close external HTTP clients
    try:
        from app.services.openfda import get_openfda_service
        await get_openfda_service().close()
    except Exception:
        pass

    try:
        from app.services.google_maps import get_google_maps_service
        await get_google_maps_service().close()
    except Exception:
        pass

    await close_db()


# ── App Factory ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered healthcare assistant platform providing symptom analysis, "
        "disease prediction with confidence scoring, triage assessment, and "
        "personalised health recommendations. Built with FastAPI, LLM agents, "
        "and a medical knowledge RAG pipeline."
    ),
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)


# ── Middleware (order matters — outermost first) ────────────────────

# CORS — must be first so preflight requests are handled before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
)

# Request tracing
app.add_middleware(RequestIDMiddleware)

# Performance timing
app.add_middleware(TimingMiddleware)

# Rate limiting (100 requests / 60 seconds per IP)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# HIPAA audit logging for sensitive endpoints
app.add_middleware(AuditLogMiddleware)


# ── Global Exception Handler ───────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a safe error response.

    In production, error details are hidden to avoid leaking internals.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    detail = "An internal server error occurred."
    if settings.debug:
        detail = f"{type(exc).__name__}: {exc}"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": detail,
            "disclaimer": (
                "If you are experiencing a medical emergency, "
                "please call your local emergency services immediately."
            ),
        },
    )


# ── Router Registration ────────────────────────────────────────────

API_V1_PREFIX = "/api/v1"

app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(chat_router, prefix=API_V1_PREFIX)
app.include_router(symptoms_router, prefix=API_V1_PREFIX)
app.include_router(diagnosis_router, prefix=API_V1_PREFIX)
app.include_router(hospitals_router, prefix=API_V1_PREFIX)
app.include_router(rag_router, prefix=API_V1_PREFIX)
app.include_router(drugs_router, prefix=API_V1_PREFIX)
app.include_router(users_router, prefix=API_V1_PREFIX)
app.include_router(admin_router, prefix=API_V1_PREFIX)


# ── Root Endpoint ──────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """Root endpoint with platform information and API documentation links."""
    return {
        "platform": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{API_V1_PREFIX}/health",
        "disclaimer": (
            "This platform is for informational purposes only and is not a "
            "substitute for professional medical advice, diagnosis, or treatment."
        ),
    }
