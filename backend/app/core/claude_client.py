"""
Anthropic Claude API client with retry logic, timeout handling, and fallbacks.

Provides a singleton async client with exponential backoff retries,
structured JSON output parsing, and graceful degradation when the
API is unavailable.
"""

import asyncio
import json
from typing import Any

from anthropic import AsyncAnthropic, APIError, APITimeoutError, RateLimitError

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Singleton client (lazy init)
_client: AsyncAnthropic | None = None


def get_claude_client() -> AsyncAnthropic:
    """Return a reusable async Anthropic client.

    Creates the client on first call and reuses it thereafter.
    Requires ``ANTHROPIC_API_KEY`` to be set in the environment.

    Returns:
        An :class:`AsyncAnthropic` instance.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.claude_timeout,
            max_retries=0,  # We handle retries ourselves
        )
    return _client


async def call_claude(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    json_mode: bool = False,
) -> str:
    """Send a message to Claude and return the response text.

    Implements exponential backoff retries for transient errors
    (rate limits, timeouts, 5xx server errors). Returns a fallback
    message instead of raising if all retries fail.

    Args:
        system_prompt: The system instructions for Claude.
        user_message: The user's input message.
        model: Claude model identifier. Defaults to config value.
        max_tokens: Maximum response tokens. Defaults to config value.
        temperature: Sampling temperature (0.0–1.0).
        max_retries: Number of retry attempts for transient failures.
        retry_delay: Initial delay in seconds (doubled each retry).
        json_mode: If True, instruct Claude to respond with valid JSON only.

    Returns:
        The assistant's response text, or a fallback error message.
    """
    client = get_claude_client()
    model = model or settings.claude_model
    max_tokens = max_tokens or settings.claude_max_tokens

    if json_mode:
        system_prompt += (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "Do not include markdown code fences, explanatory text, or any "
            "content outside the JSON object."
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text.strip()
            logger.debug(
                "Claude response (attempt %d): tokens_in=%d tokens_out=%d",
                attempt,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return text

        except RateLimitError:
            logger.warning(
                "Claude rate limited (attempt %d/%d). Retrying in %.1fs…",
                attempt, max_retries, retry_delay,
            )
        except APITimeoutError:
            logger.warning(
                "Claude timeout (attempt %d/%d). Retrying in %.1fs…",
                attempt, max_retries, retry_delay,
            )
        except APIError as exc:
            if exc.status_code and exc.status_code >= 500:
                logger.warning(
                    "Claude server error %d (attempt %d/%d). Retrying…",
                    exc.status_code, attempt, max_retries,
                )
            else:
                logger.error("Claude API error: %s", exc)
                break  # Non-retryable client error
        except Exception as exc:
            logger.error("Unexpected Claude error: %s", exc, exc_info=True)
            break

        if attempt < max_retries:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    # All retries exhausted — return fallback
    logger.error("Claude API unavailable after %d attempts", max_retries)
    return _fallback_response(json_mode)


async def call_claude_json(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call Claude and parse the response as JSON.

    A convenience wrapper around :func:`call_claude` that enables
    ``json_mode`` and parses the response into a Python dict.

    Args:
        system_prompt: System instructions.
        user_message: User input.
        temperature: Sampling temperature.
        max_retries: Retry attempts.

    Returns:
        Parsed JSON dictionary. Returns a fallback dict on failure.
    """
    raw = await call_claude(
        system_prompt,
        user_message,
        temperature=temperature,
        max_retries=max_retries,
        json_mode=True,
    )

    try:
        # Handle potential markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude JSON response: %s\nRaw: %s", exc, raw[:500])
        return {"error": "Failed to parse AI response", "raw": raw[:500]}


async def call_claude_streaming(
    system_prompt: str,
    user_message: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
):
    """Stream Claude's response token by token.

    Yields text chunks as they arrive from the API. Useful for
    real-time chat over WebSocket.

    Args:
        system_prompt: System instructions.
        user_message: User input.
        model: Model identifier.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Yields:
        Text chunks (strings) as they stream from Claude.
    """
    client = get_claude_client()
    model = model or settings.claude_model
    max_tokens = max_tokens or settings.claude_max_tokens

    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except Exception as exc:
        logger.error("Claude streaming error: %s", exc)
        yield _fallback_response(json_mode=False)


def _fallback_response(json_mode: bool) -> str:
    """Return a safe fallback when Claude is unavailable."""
    if json_mode:
        return json.dumps({
            "error": "AI service temporarily unavailable",
            "fallback": True,
            "message": (
                "I'm unable to process your request right now. "
                "Please try again in a few moments or consult a healthcare professional."
            ),
        })
    return (
        "I apologise, but I'm experiencing temporary difficulties processing your request. "
        "Please try again in a few moments. If you are experiencing a medical emergency, "
        "please call your local emergency services immediately."
    )
