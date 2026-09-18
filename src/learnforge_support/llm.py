"""LLM interface module for LearnForge Support Assistant.

Wraps the Groq provider client, requests structured JSON output, enforces
Pydantic schema validation with a single bounded retry, and measures latency.
Falls back safely to an escalation response on provider failures.
"""

import json
from typing import Protocol

from groq import Groq

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
    """Production LLM client utilizing the official Groq SDK."""

    def __init__(self, api_key: str) -> None:
        self.client = Groq(api_key=api_key)

    def complete_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Groq returned empty response content")
        return content


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
        llm_client = GroqLLMClient(api_key=settings.groq_api_key)

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
                    "I have escalated your question to LearnForge human support."
                ),
                reason_code="insufficient_evidence",
                citations=[],
                handoff_summary=f"Automated processing error: {type(retry_exc).__name__} - {retry_exc}",
            )
            return fallback, stopwatch.elapsed_ms()
