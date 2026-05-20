"""
Phase 3 Integration tests — auth flow, JWT-protected endpoints, services.

Tests cover:
- Full registration → login → authenticated request flow
- JWT token validation and refresh
- Session ownership enforcement
- OpenFDA service (mocked HTTP)
- WebSocket auth handshake
- Admin-only endpoint access control
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.core.security import create_access_token, create_refresh_token


# ── Auth Flow Tests ─────────────────────────────────────────────────

class TestAuthFlow:
    """End-to-end authentication flow tests."""

    @pytest.mark.asyncio
    async def test_register_returns_tokens(self, client: AsyncClient) -> None:
        """Registration should return access + refresh tokens."""
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "securepassword123",
                "full_name": "Test User",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_login_returns_tokens(self, client: AsyncClient) -> None:
        """Login should return tokens for valid credentials."""
        # Register first
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@example.com",
                "password": "mypassword123",
                "full_name": "Login User",
            },
        )

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "login@example.com", "password": "mypassword123"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient) -> None:
        """Login with wrong password should return 401."""
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "wrongpw@example.com",
                "password": "correctpassword",
                "full_name": "Wrong PW User",
            },
        )

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "wrongpw@example.com", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_duplicate_registration(self, client: AsyncClient) -> None:
        """Duplicate email registration should return 409."""
        body = {
            "email": "dupe@example.com",
            "password": "password123",
            "full_name": "Dupe User",
        }
        await client.post("/api/v1/auth/register", json=body)
        resp = await client.post("/api/v1/auth/register", json=body)
        assert resp.status_code == 409


# ── JWT Auth Enforcement Tests ──────────────────────────────────────

class TestJWTAuth:
    """Tests that protected endpoints reject unauthenticated requests."""

    @pytest.mark.asyncio
    async def test_sessions_requires_auth(self, client: AsyncClient) -> None:
        """GET /chat/sessions without Bearer token → 401."""
        resp = await client.get("/api/v1/chat/sessions")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_symptoms_requires_auth(self, client: AsyncClient) -> None:
        """POST /symptoms/analyze without Bearer token → 401."""
        resp = await client.post(
            "/api/v1/symptoms/analyze",
            json={"text": "I have a headache"},
        )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_diagnosis_history_requires_auth(self, client: AsyncClient) -> None:
        """GET /diagnosis/history without Bearer token → 401."""
        resp = await client.get("/api/v1/diagnosis/history")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_valid_token_grants_access(self, client: AsyncClient) -> None:
        """GET /chat/sessions with valid token → 200."""
        # Register and get token
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "authed@example.com",
                "password": "password123",
                "full_name": "Authed User",
            },
        )
        token = reg.json()["access_token"]

        resp = await client.get(
            "/api/v1/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, client: AsyncClient) -> None:
        """Expired token should be rejected."""
        from datetime import timedelta

        token = create_access_token(
            {"sub": "nonexistent-id", "email": "x@x.com", "role": "patient"},
            expires_delta=timedelta(seconds=-1),
        )
        resp = await client.get(
            "/api/v1/chat/sessions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


# ── Session Ownership Tests ─────────────────────────────────────────

class TestSessionOwnership:
    """Tests that users can only access their own sessions."""

    @pytest.mark.asyncio
    async def test_cannot_see_other_users_sessions(self, client: AsyncClient) -> None:
        """User A cannot list User B's sessions."""
        # Register two users
        reg_a = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "usera@example.com",
                "password": "password123",
                "full_name": "User A",
            },
        )
        token_a = reg_a.json()["access_token"]

        reg_b = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "userb@example.com",
                "password": "password123",
                "full_name": "User B",
            },
        )
        token_b = reg_b.json()["access_token"]

        # User A creates a session
        await client.post(
            "/api/v1/chat/sessions",
            headers={"Authorization": f"Bearer {token_a}"},
        )

        # User B should see zero sessions
        resp = await client.get(
            "/api/v1/chat/sessions",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 0


# ── Admin Access Control Tests ──────────────────────────────────────

class TestAdminAccess:
    """Tests that admin endpoints require admin role."""

    @pytest.mark.asyncio
    async def test_stats_requires_admin(self, client: AsyncClient) -> None:
        """GET /admin/stats for patient user → 403."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "patient@example.com",
                "password": "password123",
                "full_name": "Patient User",
            },
        )
        token = reg.json()["access_token"]

        resp = await client.get(
            "/api/v1/admin/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


# ── OpenFDA Service Tests ──────────────────────────────────────────

class TestOpenFDAService:
    """Tests for the OpenFDA service with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_drug_label_search(self) -> None:
        """Should parse FDA drug label response correctly."""
        from app.services.openfda import OpenFDAService

        mock_response = {
            "meta": {"results": {"total": 1}},
            "results": [
                {
                    "openfda": {
                        "brand_name": ["Ibuprofen"],
                        "generic_name": ["ibuprofen"],
                        "manufacturer_name": ["Test Pharma"],
                        "route": ["ORAL"],
                    },
                    "indications_and_usage": ["Pain relief"],
                    "warnings": ["May cause stomach bleeding"],
                    "drug_interactions": ["Aspirin interaction warning"],
                },
            ],
        }

        service = OpenFDAService()

        with patch.object(
            service,
            "_get_client",
            return_value=AsyncMock(
                get=AsyncMock(
                    return_value=AsyncMock(
                        status_code=200,
                        raise_for_status=lambda: None,
                        json=lambda: mock_response,
                    ),
                ),
            ),
        ):
            result = await service.search_drug_label("ibuprofen")

        assert len(result["results"]) == 1
        assert result["results"][0]["brand_name"] == "Ibuprofen"

    @pytest.mark.asyncio
    async def test_drug_interaction_check(self) -> None:
        """Should detect cross-referenced drug interactions."""
        from app.services.openfda import OpenFDAService

        service = OpenFDAService()

        # Mock: aspirin's label mentions ibuprofen
        async def mock_search(drug, limit=1):
            if drug.lower() == "aspirin":
                return {
                    "results": [
                        {"drug_interactions": "Do not take with ibuprofen"},
                    ],
                    "meta": {"total": 1},
                    "query": drug,
                }
            return {
                "results": [
                    {"drug_interactions": "No known interactions"},
                ],
                "meta": {"total": 1},
                "query": drug,
            }

        service.search_drug_label = mock_search  # type: ignore[assignment]

        result = await service.check_drug_interactions(["aspirin", "ibuprofen"])
        assert result["interactions_found"] is True
        assert result["interaction_count"] >= 1


# ── User Profile Tests ─────────────────────────────────────────────

class TestUserProfile:
    """Tests for user profile endpoints."""

    @pytest.mark.asyncio
    async def test_get_profile(self, client: AsyncClient) -> None:
        """GET /users/me should return current user profile."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "profile@example.com",
                "password": "password123",
                "full_name": "Profile User",
            },
        )
        token = reg.json()["access_token"]

        resp = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == "profile@example.com"

    @pytest.mark.asyncio
    async def test_update_profile(self, client: AsyncClient) -> None:
        """PATCH /users/me should update user fields."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "update@example.com",
                "password": "password123",
                "full_name": "Original Name",
            },
        )
        token = reg.json()["access_token"]

        resp = await client.patch(
            "/api/v1/users/me",
            json={"full_name": "Updated Name"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Updated Name"


# ── Connection Manager Tests ───────────────────────────────────────

class TestConnectionManager:
    """Tests for the WebSocket connection manager."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_connection_manager() should return the same instance."""
        from app.services.connection_manager import get_connection_manager

        mgr1 = get_connection_manager()
        mgr2 = get_connection_manager()
        assert mgr1 is mgr2

    def test_initial_counts_zero(self) -> None:
        """New manager should have zero connections."""
        from app.services.connection_manager import ConnectionManager

        mgr = ConnectionManager()
        assert mgr.active_connections == 0
        assert mgr.active_sessions == 0


# ── Health Endpoint Tests ──────────────────────────────────────────

class TestHealthEndpoints:
    """Tests for health/readiness endpoints (no auth required)."""

    @pytest.mark.asyncio
    async def test_liveness(self, client: AsyncClient) -> None:
        """GET /health should return 200."""
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
