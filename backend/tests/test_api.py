"""
Test suite for health check endpoints.

Validates that /health and /health/ready respond correctly.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """GET /api/v1/health should return 200 with status healthy."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """GET / should return platform info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "platform" in data
    assert "disclaimer" in data


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient, sample_user_data: dict) -> None:
    """POST /api/v1/auth/register should create a user and return tokens."""
    response = await client.post("/api/v1/auth/register", json=sample_user_data)
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == sample_user_data["email"]


@pytest.mark.asyncio
async def test_register_duplicate_email(
    client: AsyncClient, sample_user_data: dict,
) -> None:
    """Registering the same email twice should return 409."""
    await client.post("/api/v1/auth/register", json=sample_user_data)
    response = await client.post("/api/v1/auth/register", json=sample_user_data)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login(client: AsyncClient, sample_user_data: dict) -> None:
    """POST /api/v1/auth/login should authenticate and return tokens."""
    # Register first
    await client.post("/api/v1/auth/register", json=sample_user_data)

    # Login
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": sample_user_data["email"], "password": sample_user_data["password"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(
    client: AsyncClient, sample_user_data: dict,
) -> None:
    """Login with wrong password should return 401."""
    await client.post("/api/v1/auth/register", json=sample_user_data)

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": sample_user_data["email"], "password": "wrong-password"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_symptom_list(client: AsyncClient) -> None:
    """GET /api/v1/symptoms/list should return symptom catalogue."""
    response = await client.get("/api/v1/symptoms/list")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.asyncio
async def test_symptom_analyze(client: AsyncClient) -> None:
    """POST /api/v1/symptoms/analyze should extract symptoms."""
    response = await client.post(
        "/api/v1/symptoms/analyze",
        json={"text": "I have a headache and fever for two days"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "extracted_symptoms" in data
    assert "disclaimer" in data
