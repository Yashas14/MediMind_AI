"""
Google Maps Places API service — hospital/clinic search.

Queries Google Maps Places API (New) to find nearby hospitals,
clinics, and urgent-care facilities based on user location and
urgency level.

Docs: https://developers.google.com/maps/documentation/places/web-service
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

_TIMEOUT = 10.0


class GoogleMapsService:
    """Async client for Google Maps Places API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def search_nearby_hospitals(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        urgency: str = "routine",
    ) -> list[dict[str, Any]]:
        """Search for hospitals and clinics near coordinates.

        Args:
            latitude: User's latitude.
            longitude: User's longitude.
            radius_km: Search radius in kilometres.
            urgency: Triage-based urgency (emergency, urgent, routine).

        Returns:
            List of hospital results with name, address, distance, etc.
        """
        if not settings.google_maps_api_key:
            logger.warning("Google Maps API key not configured — returning mock data")
            return _mock_hospitals(latitude, longitude)

        client = await self._get_client()
        radius_m = int(radius_km * 1000)

        # For emergencies, search specifically for hospitals
        keyword = "hospital emergency" if urgency == "emergency" else "hospital clinic"
        place_type = "hospital"

        params = {
            "location": f"{latitude},{longitude}",
            "radius": str(radius_m),
            "type": place_type,
            "keyword": keyword,
            "key": settings.google_maps_api_key,
        }

        try:
            resp = await client.get(PLACES_NEARBY_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "OK":
                logger.error("Google Maps API error: %s", data.get("status"))
                return _mock_hospitals(latitude, longitude)

            results = []
            for place in data.get("results", [])[:10]:
                loc = place.get("geometry", {}).get("location", {})
                plat = loc.get("lat", 0)
                plng = loc.get("lng", 0)

                distance = _haversine(latitude, longitude, plat, plng)

                results.append({
                    "name": place.get("name", "Unknown"),
                    "address": place.get("vicinity", ""),
                    "latitude": plat,
                    "longitude": plng,
                    "distance_km": round(distance, 2),
                    "rating": place.get("rating"),
                    "phone": None,  # Requires Place Details call
                    "open_now": place.get("opening_hours", {}).get("open_now"),
                    "place_id": place.get("place_id", ""),
                })

            # Sort by distance
            results.sort(key=lambda x: x["distance_km"])
            return results

        except httpx.HTTPError as exc:
            logger.error("Google Maps HTTP error: %s", exc)
            return _mock_hospitals(latitude, longitude)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance in km between two points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _mock_hospitals(lat: float, lon: float) -> list[dict[str, Any]]:
    """Return mock hospital data when Google Maps is not configured."""
    return [
        {
            "name": "City General Hospital",
            "address": "123 Medical Center Drive",
            "latitude": lat + 0.01,
            "longitude": lon + 0.01,
            "distance_km": 1.2,
            "rating": 4.5,
            "phone": "+1-555-0100",
            "open_now": True,
            "place_id": "mock_place_id_1",
        },
        {
            "name": "Community Health Clinic",
            "address": "456 Healthcare Avenue",
            "latitude": lat + 0.02,
            "longitude": lon - 0.01,
            "distance_km": 2.8,
            "rating": 4.2,
            "phone": "+1-555-0200",
            "open_now": True,
            "place_id": "mock_place_id_2",
        },
        {
            "name": "Regional Medical Center",
            "address": "789 Hospital Boulevard",
            "latitude": lat - 0.015,
            "longitude": lon + 0.02,
            "distance_km": 3.5,
            "rating": 4.7,
            "phone": "+1-555-0300",
            "open_now": True,
            "place_id": "mock_place_id_3",
        },
    ]


# ── Singleton ───────────────────────────────────────────────────────

_service: GoogleMapsService | None = None


def get_google_maps_service() -> GoogleMapsService:
    """Return the global GoogleMapsService singleton."""
    global _service  # noqa: PLW0603
    if _service is None:
        _service = GoogleMapsService()
    return _service
