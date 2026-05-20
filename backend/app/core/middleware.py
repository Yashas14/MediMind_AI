"""
Custom FastAPI middleware.

Includes:
- Request ID injection for distributed tracing
- Request/response timing
- Rate limiting via Redis (token bucket)
- HIPAA audit logging for data-access endpoints
"""

import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique ``X-Request-ID`` header into every request/response.

    If the client already provides the header, it is reused for tracing.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request, injecting a request ID."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure and log request processing time."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and log timing information."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter (token-bucket per client IP).

    For production, replace with a Redis-backed implementation
    or use an API gateway (e.g., Cloudflare, AWS WAF).
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60) -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Enforce rate limits per client IP."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune expired entries
        timestamps = self._buckets.setdefault(client_ip, [])
        cutoff = now - self.window_seconds
        self._buckets[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._buckets[client_ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return Response(
                content='{"detail":"Rate limit exceeded. Please try again later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )

        self._buckets[client_ip].append(now)
        return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log access to sensitive health-data endpoints for HIPAA compliance.

    Captures the user (from JWT), action, resource, and timestamp.
    """

    AUDIT_PATHS: set[str] = {
        "/api/v1/diagnosis",
        "/api/v1/health/summary",
        "/api/v1/health/profile",
        "/api/v1/chat",
    }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log audit events for protected endpoints."""
        response = await call_next(request)

        # Only audit matching paths
        if any(request.url.path.startswith(p) for p in self.AUDIT_PATHS):
            user_id = getattr(request.state, "user_id", "anonymous")
            logger.info(
                "AUDIT │ user=%s │ %s %s │ status=%s",
                user_id,
                request.method,
                request.url.path,
                response.status_code,
            )

        return response
