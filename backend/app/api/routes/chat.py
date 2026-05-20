"""
Chat endpoints — WebSocket real-time chat and REST session management.

Phase 3: Full JWT auth on all endpoints, WebSocket token auth via
query-string, session ownership enforcement, message persistence,
connection manager integration.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession, Pagination, get_ws_user
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.chat import ChatSession, Message, MessageRole, SessionStatus
from app.models.user import User
from app.services.connection_manager import get_connection_manager

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    """Schema for sending a message via REST."""

    content: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[UUID] = None


class ChatMessageResponse(BaseModel):
    """Schema for a chat message response."""

    id: str
    role: str
    content: str
    confidence_score: Optional[float] = None
    extracted_symptoms: Optional[list[str]] = None
    triage_level: Optional[str] = None
    created_at: str
    disclaimer: str = (
        "⚠️ This is an AI-generated response for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )


class SessionResponse(BaseModel):
    """Schema for a chat session summary."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: Optional[str] = None
    message_count: int


class SessionUpdateRequest(BaseModel):
    """Schema for updating a session (rename, close)."""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[str] = Field(None, pattern="^(active|closed|archived)$")


# ── Session Ownership Helper ───────────────────────────────────────

async def _get_user_session(
    session_id: UUID,
    user: User,
    db: AsyncSession,
) -> ChatSession:
    """Fetch a session and verify ownership."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user.id,
        ),
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or access denied.",
        )
    return session


# ── REST Endpoints ──────────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="List chat sessions for the current user",
)
async def list_sessions(
    user: CurrentUser,
    db: DBSession,
    page: Pagination,
    status_filter: Optional[str] = Query(
        None, alias="status", pattern="^(active|closed|archived)$",
    ),
) -> list[SessionResponse]:
    """Return all chat sessions for the authenticated user."""
    query = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    if status_filter:
        query = query.where(ChatSession.status == SessionStatus(status_filter))

    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        SessionResponse(
            id=str(s.id),
            title=s.title,
            status=s.status.value,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat() if s.updated_at else None,
            message_count=len(s.messages) if s.messages else 0,
        )
        for s in sessions
    ]


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_session(
    user: CurrentUser,
    db: DBSession,
) -> SessionResponse:
    """Start a new conversation session for the authenticated user."""
    session = ChatSession(
        user_id=user.id,
        title="New Consultation",
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    logger.info("New chat session created: %s for user %s", session.id, user.id)

    return SessionResponse(
        id=str(session.id),
        title=session.title,
        status=session.status.value,
        created_at=session.created_at.isoformat(),
        message_count=0,
    )


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Update a chat session (rename or close)",
)
async def update_session(
    session_id: UUID,
    body: SessionUpdateRequest,
    user: CurrentUser,
    db: DBSession,
) -> SessionResponse:
    """Rename or close a chat session."""
    session = await _get_user_session(session_id, user, db)

    if body.title is not None:
        session.title = body.title
    if body.status is not None:
        session.status = SessionStatus(body.status)

    await db.commit()
    await db.refresh(session)

    return SessionResponse(
        id=str(session.id),
        title=session.title,
        status=session.status.value,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat() if session.updated_at else None,
        message_count=len(session.messages) if session.messages else 0,
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Delete (archive) a chat session",
)
async def delete_session(
    session_id: UUID,
    user: CurrentUser,
    db: DBSession,
) -> Response:
    """Archive a chat session (soft-delete)."""
    session = await _get_user_session(session_id, user, db)
    session.status = SessionStatus.ARCHIVED
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[ChatMessageResponse],
    summary="Get all messages in a session",
)
async def get_session_messages(
    session_id: UUID,
    user: CurrentUser,
    db: DBSession,
    page: Pagination,
) -> list[ChatMessageResponse]:
    """Retrieve the full message history for a chat session."""
    await _get_user_session(session_id, user, db)

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
        .limit(page.limit)
        .offset(page.offset),
    )
    messages = result.scalars().all()
    return [
        ChatMessageResponse(
            id=str(m.id),
            role=m.role.value,
            content=m.content,
            confidence_score=m.confidence_score,
            extracted_symptoms=m.extracted_symptoms,
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Send a message and get AI response (REST)",
)
async def send_message(
    body: ChatMessageRequest,
    user: CurrentUser,
    db: DBSession,
) -> ChatMessageResponse:
    """Send a message via REST and receive an AI response.

    Creates a session if ``session_id`` is not provided.
    Persists both user and assistant messages.
    """
    # Get or create session
    if body.session_id:
        session = await _get_user_session(body.session_id, user, db)
        if session.status != SessionStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot send messages to a closed session.",
            )
    else:
        session = ChatSession(user_id=user.id, title="New Consultation")
        db.add(session)
        await db.flush()

    # Save user message
    user_msg = Message(
        session_id=session.id,
        role=MessageRole.USER,
        content=body.content,
    )
    db.add(user_msg)

    # Run through agent orchestrator
    try:
        from app.agents.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        agent_result = await orchestrator.chat_response(body.content)
        ai_content = agent_result.get("content", "")
        confidence = agent_result.get("confidence_score", 0.0)
        extracted = agent_result.get("extracted_symptoms", [])
        triage = agent_result.get("triage_level")
    except Exception as exc:
        logger.error("Agent pipeline error (REST): %s", exc)
        ai_content = (
            "I apologise, but I encountered an error processing your request. "
            "Please try again or consult a healthcare professional."
        )
        confidence = 0.0
        extracted = []
        triage = None

    # Save assistant message
    ai_msg = Message(
        session_id=session.id,
        role=MessageRole.ASSISTANT,
        content=ai_content,
        confidence_score=confidence,
        extracted_symptoms=extracted,
    )
    db.add(ai_msg)

    # Auto-title from first message
    if session.title == "New Consultation" and len(body.content) > 5:
        session.title = body.content[:80] + ("…" if len(body.content) > 80 else "")

    await db.commit()
    await db.refresh(ai_msg)

    return ChatMessageResponse(
        id=str(ai_msg.id),
        role=ai_msg.role.value,
        content=ai_msg.content,
        confidence_score=ai_msg.confidence_score,
        extracted_symptoms=ai_msg.extracted_symptoms,
        triage_level=triage,
        created_at=ai_msg.created_at.isoformat(),
    )


# ── WebSocket Endpoint ─────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: UUID,
    user: User | None = Depends(get_ws_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Real-time bidirectional chat over WebSocket.

    Authentication via query-string: ``ws://…/ws/{session_id}?token=<jwt>``

    Protocol:
    - Client sends JSON: ``{"content": "...", "type": "message"}``
    - Server responds with JSON: ``{"role": "assistant", "content": "...", ...}``
    - Server sends ``{"type": "typing", "status": true}`` during processing.
    - Server sends ``{"type": "ping"}`` keepalives every 30s.
    """
    manager = get_connection_manager()

    # Require auth in production
    if user is None:
        from app.core.config import get_settings

        if get_settings().is_production:
            await websocket.close(code=4001, reason="Authentication required")
            return

    client = await manager.connect(
        websocket,
        session_id,
        user_id=user.id if user else None,
    )

    logger.info(
        "WebSocket connected: session=%s user=%s",
        session_id,
        user.id if user else "anonymous",
    )

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "ping":
                await manager.send_json(client, {"type": "pong"})
                continue

            content = data.get("content", "")
            if not content.strip():
                await manager.send_json(client, {"type": "error", "detail": "Empty message"})
                continue

            # Typing indicator
            await manager.send_json(client, {"type": "typing", "status": True})

            # Persist user message
            if user:
                user_msg = Message(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=content,
                )
                db.add(user_msg)
                await db.flush()

            # Run through agent orchestrator
            try:
                from app.agents.orchestrator import get_orchestrator

                orchestrator = get_orchestrator()
                agent_result = await orchestrator.chat_response(content)

                response = {
                    "type": "message",
                    "role": "assistant",
                    "content": agent_result.get("content", ""),
                    "confidence_score": agent_result.get("confidence_score"),
                    "extracted_symptoms": agent_result.get("extracted_symptoms", []),
                    "triage_level": agent_result.get("triage_level"),
                    "disclaimer": agent_result.get("disclaimer", ""),
                }

                # Persist assistant message
                if user:
                    ai_msg = Message(
                        session_id=session_id,
                        role=MessageRole.ASSISTANT,
                        content=response["content"],
                        confidence_score=response.get("confidence_score"),
                        extracted_symptoms=response.get("extracted_symptoms"),
                    )
                    db.add(ai_msg)
                    await db.commit()

            except Exception as exc:
                logger.error("Agent pipeline error: %s", exc)
                response = {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        "I apologise, but I encountered an error processing your request. "
                        "Please try again. If you are experiencing a medical emergency, "
                        "call your local emergency services immediately."
                    ),
                    "confidence_score": 0.0,
                    "disclaimer": (
                        "⚠️ AI-generated response — not a substitute for "
                        "professional medical advice."
                    ),
                }

            # Stop typing + send response
            await manager.send_json(client, {"type": "typing", "status": False})
            await manager.send_json(client, response)

    except WebSocketDisconnect:
        await manager.disconnect(client)
    except Exception as exc:
        logger.error("WebSocket error: session=%s error=%s", session_id, exc)
        await manager.disconnect(client)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
