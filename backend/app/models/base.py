"""
SQLAlchemy declarative base and common mixins.

Every model inherits from ``Base`` and optionally from ``TimestampMixin``
to get automatic ``created_at`` / ``updated_at`` columns.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class TimestampMixin:
    """Mixin that adds ``created_at`` and ``updated_at`` columns.

    ``created_at`` is set automatically on INSERT.
    ``updated_at`` is refreshed automatically on every UPDATE.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """Mixin that provides a UUID v4 primary key column named ``id``.

    Uses String(36) storage for cross-database compatibility (SQLite + PostgreSQL).
    """

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
