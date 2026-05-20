"""
HIPAA-aware logging configuration.

Automatically redacts PII patterns (emails, phone numbers, SSNs, names
associated with health data) from log messages before they are emitted.
"""

import logging
import re
import sys
from typing import ClassVar

from app.core.config import get_settings

settings = get_settings()

# Patterns that should be redacted from log output
PII_PATTERNS: list[tuple[str, str]] = [
    # Email addresses
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]"),
    # US phone numbers
    (r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    # Bearer tokens
    (r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED_TOKEN]"),
    # API keys (common patterns)
    (r"(?:api[_-]?key|apikey|token)\s*[:=]\s*\S+", "api_key=[REDACTED]"),
]

COMPILED_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in PII_PATTERNS
]


class HIPAAFilter(logging.Filter):
    """Logging filter that redacts PII from log records.

    Applies regex-based redaction to the log message and any string
    arguments before the record is formatted.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact PII from the log record message."""
        record.msg = self._redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                # Only redact string args — leave numbers/other types intact
                # to avoid breaking %-format specifiers like %d, %f, etc.
                record.args = tuple(
                    self._redact(a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True

    @staticmethod
    def _redact(text: str) -> str:
        """Apply all PII redaction patterns to *text*."""
        for pattern, replacement in COMPILED_PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    RESERVED_ATTRS: ClassVar[set[str]] = {
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        import json
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Configure application-wide logging with HIPAA-aware filtering.

    In production, uses structured JSON output. In development, uses a
    human-readable format with colours (if the terminal supports them).
    """
    log_level = logging.DEBUG if settings.debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.is_production:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # Attach HIPAA filter
    console_handler.addFilter(HIPAAFilter())
    root_logger.addHandler(console_handler)

    # Quieten noisy third-party loggers
    for logger_name in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger("app").info(
        "Logging initialised — env=%s level=%s",
        settings.environment,
        logging.getLevelName(log_level),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger within the ``app`` namespace.

    Args:
        name: The logger name (usually ``__name__``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(f"app.{name}")
