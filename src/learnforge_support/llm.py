"""LLM interface module for LearnForge Support Assistant.

Wraps the Groq provider client, requests structured JSON output, enforces
Pydantic schema validation with a single bounded retry, and measures latency.
Falls back safely to an escalation response on provider failures.
"""

import json
import re
import time
from threading import Lock
from typing import Protocol

from groq import Groq, RateLimitError

from learnforge_support.config import Settings
from learnforge_support.logging_utils import Stopwatch, setup_logger
from learnforge_support.reliability import sanitize_decision
from learnforge_support.schemas import SupportDecision

logger = setup_logger()


class LLMClientProtocol(Protocol):
    """Protocol for LLM provider client to allow straightforward testing/mocking."""

    def complete_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """Returns raw JSON string response from LLM."""
        ...


class GroqLLMClient:
    """Groq client with process-local request pacing and bounded 429 retries."""

    def __init__(
        self,
        api_key: str,
        min_request_interval_seconds: float = 2.0,
        rate_limit_max_attempts: int = 3,
    ) -> None:
        self.client = Groq(api_key=api_key)
        self._pacer = RequestPacer(min_request_interval_seconds)
        self._rate_limit_max_attempts = rate_limit_max_attempts

    def complete_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        for attempt in range(self._rate_limit_max_attempts):
            try:
                self._pacer.wait()
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=600,
                )
                content = completion.choices[0].message.content
                if not content:
                    raise ValueError("Groq returned empty response content")
                return content
            except RateLimitError as rle:
                if attempt < self._rate_limit_max_attempts - 1:
                    wait_s = rate_limit_retry_delay(rle, attempt)
                    if wait_s is None:
                        logger.warning(
                            "Groq rate limit will not reset within the request retry window."
                        )
                        raise rle
                    logger.warning(
                        "Groq rate limit encountered (attempt %d/%d). Backing off for %.1fs...",
                        attempt + 1,
                        self._rate_limit_max_attempts,
                        wait_s,
                    )
                    time.sleep(wait_s)
                else:
                    raise rle


class RequestPacer:
    """Coordinates Groq calls made by one application process.

    Groq also enforces account-level limits. This small local guard reduces avoidable
    bursts from concurrent browser requests; it is not presented as a replacement for
    provider limits or a distributed rate limiter.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._lock = Lock()
        self._next_request_at = 0.0

    def wait(self) -> None:
        if self._min_interval_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self._min_interval_seconds

        if wait_seconds:
            time.sleep(wait_seconds)


def rate_limit_retry_delay(error: RateLimitError, attempt: int) -> float | None:
    """Returns a short retry delay, or None when the provider reset is too far away."""
    headers = getattr(getattr(error, "response", None), "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    delay_candidates: list[float] = []
    if retry_after:
        try:
            delay_candidates.append(float(retry_after))
        except ValueError:
            pass

    message_delay = _provider_reset_delay_seconds(str(error))
    if message_delay is not None:
        delay_candidates.append(message_delay)

    if delay_candidates:
        provider_delay = max(delay_candidates)
        if provider_delay > 30.0:
            return None
        return provider_delay
    return min(2.0**attempt, 8.0)


def _provider_reset_delay_seconds(message: str) -> float | None:
    """Parses Groq's human-readable `try again in 15m24s` quota message."""
    match = re.search(
        r"try again in\s+(?:(?P<minutes>\d+)m)?(?P<seconds>\d+(?:\.\d+)?)s",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))
    return minutes * 60 + seconds


def parse_and_validate_decision(raw_json: str, available_record_ids: set[str]) -> SupportDecision:
    """Parses raw JSON string, validates schema with Pydantic, and sanitizes decisions."""
    data = json.loads(raw_json)
    decision = SupportDecision.model_validate(data)
    return sanitize_decision(decision, available_record_ids)


def generate_decision(
    messages: list[dict[str, str]],
    available_record_ids: set[str],
    settings: Settings,
    client: LLMClientProtocol | None = None,
) -> tuple[SupportDecision, float]:
    """Invokes LLM with messages, parses structured output, and executes at most one bounded retry.

    Returns (SupportDecision, llm_ms). Never throws unhandled exceptions or fabricates facts.
    """
    stopwatch = Stopwatch()

    # Determine client instance
    llm_client: LLMClientProtocol
    if client is not None:
        llm_client = client
    else:
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY is not set. Returning safe escalation fallback.")
            return (
                SupportDecision(
                    decision="escalate",
                    message="Automated support service is currently offline. Please contact LearnForge human support.",
                    reason_code="insufficient_evidence",
                    citations=[],
                    handoff_summary="GROQ_API_KEY unconfigured; cannot invoke LLM.",
                ),
                stopwatch.elapsed_ms(),
            )
        llm_client = GroqLLMClient(
            api_key=settings.groq_api_key,
            min_request_interval_seconds=settings.llm_min_request_interval_seconds,
            rate_limit_max_attempts=settings.llm_rate_limit_max_attempts,
        )

    # First generation attempt
    attempt_messages = list(messages)
    try:
        raw_output = llm_client.complete_chat(
            model=settings.llm_model,
            messages=attempt_messages,
            temperature=0.0,
        )
        decision = parse_and_validate_decision(raw_output, available_record_ids)
        return decision, stopwatch.elapsed_ms()
    except RateLimitError:
        logger.warning(
            "Groq rate limit persisted after %d attempts.",
            settings.llm_rate_limit_max_attempts,
        )
        return (
            SupportDecision(
                decision="escalate",
                message=(
                    "LearnForge Support is temporarily busy. Please try your request again shortly, "
                    "or contact LearnForge Support directly if it is urgent."
                ),
                reason_code="insufficient_evidence",
                citations=[],
                handoff_summary="Groq rate limit persisted after bounded retries.",
            ),
            stopwatch.elapsed_ms(),
        )
    except Exception as first_exc:
        logger.warning(
            "First LLM structured output attempt failed (%s: %s). Initiating single retry.",
            type(first_exc).__name__,
            first_exc,
        )

        # Bounded Retry (Attempt 2)
        try:
            retry_messages = list(attempt_messages)
            retry_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response caused validation error: {first_exc}. "
                        "Respond with ONLY a valid JSON object matching the requested schema. "
                        "Ensure 'decision', 'message', 'reason_code', and 'citations' are strictly valid."
                    ),
                }
            )
            retry_raw = llm_client.complete_chat(
                model=settings.llm_model,
                messages=retry_messages,
                temperature=0.0,
            )
            decision = parse_and_validate_decision(retry_raw, available_record_ids)
            return decision, stopwatch.elapsed_ms()
        except Exception as retry_exc:
            logger.error(
                "LLM generation failed after retry (%s: %s). Falling back safely to ESCALATE.",
                type(retry_exc).__name__,
                retry_exc,
            )
            fallback = SupportDecision(
                decision="escalate",
                message=(
                    "I am currently unable to process this request reliably. "
                    "This inquiry requires human support review. "
                    "Please contact LearnForge Support directly."
                ),
                reason_code="insufficient_evidence",
                citations=[],
                handoff_summary=f"Automated processing error: {type(retry_exc).__name__} - {retry_exc}",
            )
            return fallback, stopwatch.elapsed_ms()
