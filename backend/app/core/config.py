"""
Application configuration using Pydantic Settings.

All configuration is loaded from environment variables or .env files.
Secrets are never hardcoded — use .env or a secrets manager in production.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "Healthcare AI Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # ── Server ───────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:3000"

    # ── Database ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://healthcare_user:password@localhost:5432/healthcare_db",
        description="Async PostgreSQL connection string.",
    )
    postgres_user: str = "healthcare_user"
    postgres_password: str = "password"
    postgres_db: str = "healthcare_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # ── Redis ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── JWT / Auth ───────────────────────────────────────────────────
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ── Google OAuth2 ────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"

    # ── Anthropic Claude ─────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4096
    claude_timeout: int = 30

    # ── ChromaDB ─────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "medical_knowledge"

    # ── External APIs ────────────────────────────────────────────────
    openfda_api_key: str = ""
    google_maps_api_key: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@healthcareai.com"

    # ── Monitoring ───────────────────────────────────────────────────
    sentry_dsn: str = ""

    # ── Encryption ───────────────────────────────────────────────────
    encryption_key: str = "CHANGE-ME-32-BYTE-KEY-1234567890"

    # ── Computed helpers ─────────────────────────────────────────────
    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if the application is running in production."""
        return self.environment == "production"

    @field_validator("jwt_secret_key")
    @classmethod
    def _warn_default_secret(cls, v: str) -> str:
        if v == "CHANGE-ME-IN-PRODUCTION":
            import warnings
            warnings.warn(
                "JWT_SECRET_KEY is set to the default value. "
                "Change it before deploying to production!",
                UserWarning,
                stacklevel=2,
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
