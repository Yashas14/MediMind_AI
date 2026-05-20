"""
WebSocket connection manager — tracks active connections with auth.

Handles connection lifecycle, room-based broadcasting (per session),
and graceful disconnection with proper cleanup.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectedClient:
    """Metadata for a connected WebSocket client."""

    websocket: WebSocket
    user_id: UUID | None
    session_id: UUID
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConnectionManager:
    """Manages WebSocket connections with session-based rooms.

    Each chat session acts as a room. Multiple browser tabs for the
    same session are grouped together.
    """

    def __init__(self) -> None:
        # session_id → list of connected clients
        self._rooms: dict[UUID, list[ConnectedClient]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: UUID,
        user_id: UUID | None = None,
    ) -> ConnectedClient:
        """Accept a WebSocket connection and register it in a room."""
        await websocket.accept()

        client = ConnectedClient(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
        )

        async with self._lock:
            self._rooms[session_id].append(client)

        logger.info(
            "WebSocket connected: session=%s user=%s (total in room: %d)",
            session_id,
            user_id,
            len(self._rooms[session_id]),
        )

        return client

    async def disconnect(self, client: ConnectedClient) -> None:
        """Remove a client from its room."""
        async with self._lock:
            room = self._rooms.get(client.session_id, [])
            self._rooms[client.session_id] = [c for c in room if c is not client]
            if not self._rooms[client.session_id]:
                del self._rooms[client.session_id]

        logger.info(
            "WebSocket disconnected: session=%s user=%s",
            client.session_id,
            client.user_id,
        )

    async def send_json(self, client: ConnectedClient, data: dict[str, Any]) -> None:
        """Send JSON data to a specific client, handling errors."""
        try:
            await client.websocket.send_json(data)
        except Exception as exc:
            logger.warning("Failed to send to client: %s", exc)
            await self.disconnect(client)

    @property
    def active_connections(self) -> int:
        """Total number of active WebSocket connections."""
        return sum(len(clients) for clients in self._rooms.values())

    @property
    def active_sessions(self) -> int:
        """Number of active session rooms."""
        return len(self._rooms)


# ── Singleton ───────────────────────────────────────────────────────

_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Return the global ConnectionManager singleton."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
