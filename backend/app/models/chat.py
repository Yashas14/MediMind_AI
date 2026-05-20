"""
Chat models — sessions and individual messages.

Each ``ChatSession`` groups a conversation between a user and the AI.
Messages track sender role, content, metadata (symptom extractions,
diagnosis references), and token usage for cost monitoring.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class SessionStatus(str, enum.Enum):
    """Chat session lifecycle states."""

    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"


class MessageRole(str, enum.Enum):
    """Who sent the message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single conversation thread between a user and the AI assistant.

    Attributes:
        user_id: FK to the owning user.
        title: Auto-generated or user-defined session title.
        status: Current lifecycle state.
        summary: AI-generated summary of the conversation (populated on close).
        metadata_: Flexible JSON for session-level metadata.
    """

    __tablename__ = "chat_sessions"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(500),
        default="New Consultation",
        nullable=False,
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status_enum"),
        default=SessionStatus.ACTIVE,
        nullable=False,
    )
    summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True,
    )

    # ── Relationships ───────────────────────────────────────────────
    user: Mapped[User] = relationship(
        "User", back_populates="chat_sessions",
    )
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user={self.user_id} status={self.status.value}>"


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """An individual message within a chat session.

    Attributes:
        session_id: FK to the parent session.
        role: Sender role (user / assistant / system).
        content: The raw text content.
        content_encrypted: Encrypted version of sensitive content.
        extracted_symptoms: Symptoms identified by the SymptomExtractorAgent.
        diagnosis_ref_id: Optional link to a generated DiagnosisRecord.
        confidence_score: Agent confidence for assistant-generated messages.
        token_count: Token usage for LLM-generated messages.
        is_flagged: Whether the message was flagged for review.
        metadata_: Flexible JSON blob (e.g., voice input metadata).
    """

    __tablename__ = "messages"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role_enum"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    content_encrypted: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    extracted_symptoms: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True,
    )
    diagnosis_ref_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("diagnosis_records.id", ondelete="SET NULL"),
        nullable=True,
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    is_flagged: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
    )
    metadata_: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True,
    )

    # ── Relationships ───────────────────────────────────────────────
    session: Mapped[ChatSession] = relationship(
        "ChatSession", back_populates="messages",
    )

    def __repr__(self) -> str:
        preview = self.content[:50] + "…" if len(self.content) > 50 else self.content
        return f"<Message id={self.id} role={self.role.value} content={preview!r}>"
