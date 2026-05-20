"""
Async database engine and session management.

Uses SQLAlchemy 2.0 async API. Supports both PostgreSQL (asyncpg) and
SQLite (aiosqlite) backends depending on DATABASE_URL.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

settings = get_settings()

# SQLite doesn't support pool_size / max_overflow / pool_pre_ping
_is_sqlite = settings.database_url.startswith("sqlite")

_engine_kwargs: dict = {
    "echo": settings.debug,
}

if _is_sqlite:
    # aiosqlite requires check_same_thread=False for async usage
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs.update(
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )

engine = create_async_engine(settings.database_url, **_engine_kwargs)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    The session is automatically closed after the request completes.
    Commits must be called explicitly — the session does **not** auto-commit.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables. Used for development bootstrapping only.

    In production, run migrations via Alembic instead.
    """
    from app.models.base import Base  # noqa: F811

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed a default demo user for local development
    await _seed_demo_user()


async def _seed_demo_user() -> None:
    """Create a demo user if it doesn't already exist."""
    from sqlalchemy import select

    from app.core.security import hash_password
    from app.models.user import User

    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.email == "demo@123")
        )
        if result.scalar_one_or_none() is None:
            demo_user = User(
                email="demo@123",
                hashed_password=hash_password("demo@123"),
                full_name="Demo User",
                is_active=True,
            )
            session.add(demo_user)
            await session.commit()


async def close_db() -> None:
    """Dispose the engine connection pool on shutdown."""
    await engine.dispose()
