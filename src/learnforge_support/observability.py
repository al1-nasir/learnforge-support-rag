"""Observability module for LearnForge Support Assistant.

Provides optional, fail-open tracing integration using Langfuse and OpenInference
instrumentation for the Groq SDK. Observability is strictly non-participating: if
Langfuse is disabled, unconfigured, or encounters any network/auth failures, the
RAG assistant continues operating normally with zero disruption.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from openinference.instrumentation.groq import GroqInstrumentor

from learnforge_support.config import Settings
from learnforge_support.logging_utils import setup_logger

logger = setup_logger()

# Module-level singletons
_client: Langfuse | None = None
_enabled: bool = False
_groq_instrumented: bool = False
_tracing_environment: str = "demo"


class NullObservation:
    """Safe no-op observation object returned when observability is disabled or unavailable."""

    def update(self, **kwargs: Any) -> None:
        pass


def is_observability_enabled() -> bool:
    """Returns True if Langfuse client is initialized, authenticated, and enabled."""
    return _enabled and _client is not None


def get_langfuse_client() -> Langfuse | None:
    """Returns the active Langfuse client if enabled, else None."""
    return _client if _enabled else None


def initialize_observability(settings: Settings) -> None:
    """Initializes Langfuse client and Groq OpenInference instrumentation once.

    Fails open: catches any configuration, network, or authentication exceptions,
    logs a warning, and safely disables observability without interrupting startup.
    """
    global _client, _enabled, _groq_instrumented, _tracing_environment
    _enabled = False
    _client = None

    if not settings.langfuse_enabled:
        logger.info("Langfuse observability is disabled by configuration.")
        return

    pub_key = settings.langfuse_public_key.strip()
    sec_key = settings.langfuse_secret_key.strip()

    if not pub_key or not sec_key:
        logger.warning(
            "Langfuse is enabled, but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing. "
            "Disabling observability (fail-open)."
        )
        return

    try:
        # Propagate environment settings so Otel and Langfuse internals share the configuration
        os.environ["LANGFUSE_PUBLIC_KEY"] = pub_key
        os.environ["LANGFUSE_SECRET_KEY"] = sec_key
        base_url = settings.langfuse_base_url.strip() or "https://cloud.langfuse.com"
        os.environ["LANGFUSE_BASE_URL"] = base_url
        os.environ["LANGFUSE_HOST"] = base_url
        _tracing_environment = settings.langfuse_tracing_environment.strip() or "demo"
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = _tracing_environment

        client = Langfuse(
            public_key=pub_key,
            secret_key=sec_key,
            base_url=base_url,
            environment=_tracing_environment,
        )
        if not client.auth_check():
            logger.warning(
                "Langfuse authentication check failed. Disabling observability (fail-open)."
            )
            return

        # Instrument Groq OpenInference exactly once
        if not _groq_instrumented:
            instrumentor = GroqInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            _groq_instrumented = True

        _client = client
        _enabled = True
        logger.info(
            "Langfuse observability initialized successfully (environment: %s).",
            _tracing_environment,
        )
    except Exception as exc:
        logger.warning(
            "Langfuse initialization failed (%s: %s). Disabling observability (fail-open).",
            type(exc).__name__,
            exc,
        )
        _enabled = False
        _client = None


def flush_observability() -> None:
    """Flushes queued observations to the Langfuse backend."""
    if _enabled and _client is not None:
        try:
            _client.flush()
        except Exception as exc:
            logger.warning("Error during Langfuse flush: %s", exc)


def shutdown_observability() -> None:
    """Flushes and shuts down the Langfuse client."""
    global _enabled, _client
    if _enabled and _client is not None:
        try:
            _client.shutdown()
        except Exception as exc:
            logger.warning("Error during Langfuse shutdown: %s", exc)
        finally:
            _enabled = False
            _client = None


def create_trace_id(seed: str | None = None) -> str:
    """Creates a deterministic 32-character hexadecimal trace ID seeded by request_id."""
    try:
        return Langfuse.create_trace_id(seed=seed)
    except Exception:
        import hashlib
        import uuid

        if seed:
            return hashlib.md5(seed.encode()).hexdigest()
        return uuid.uuid4().hex


@contextmanager
def start_support_trace(
    request_id: str,
    session_id: str,
    message: str,
    environment: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Context manager wrapping SupportService.process_chat() in a root 'support-request' span."""
    if not is_observability_enabled() or _client is None:
        yield NullObservation()
        return

    try:
        from langfuse import propagate_attributes

        trace_id = create_trace_id(seed=request_id)
        with _client.start_as_current_observation(
            name="support-request",
            as_type="span",
            trace_context={"trace_id": trace_id},
            input={"message": message},
        ) as root_obs:
            env = environment or _tracing_environment
            combined_tags = list(tags) if tags else ["learnforge", "support-rag"]
            trace_meta = dict(metadata) if metadata else {}
            trace_meta["request_id"] = request_id

            with propagate_attributes(
                session_id=session_id,
                environment=env,
                tags=combined_tags,
                metadata=trace_meta,
            ):
                yield root_obs
    except Exception as exc:
        logger.warning("Error in support-request root trace: %s", exc)
        yield NullObservation()


@contextmanager
def start_child_span(
    name: str,
    input_data: Any = None,
    metadata: Any = None,
) -> Iterator[Any]:
    """Context manager creating a child span under the active support-request observation."""
    if not is_observability_enabled() or _client is None:
        yield NullObservation()
        return

    try:
        with _client.start_as_current_observation(
            name=name,
            as_type="span",
            input=input_data,
            metadata=metadata,
        ) as child_obs:
            yield child_obs
    except Exception as exc:
        logger.warning("Error in child span '%s': %s", name, exc)
        yield NullObservation()


def record_eval_score(
    trace_id: str,
    name: str,
    value: int | float,
    data_type: str = "BOOLEAN",
    comment: str | None = None,
) -> None:
    """Records an evaluation score for a trace if observability is enabled."""
    if not is_observability_enabled() or _client is None:
        return

    try:
        _client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,  # type: ignore[arg-type]
            comment=comment,
        )
    except Exception as exc:
        logger.warning(
            "Failed to record evaluation score '%s' on trace '%s': %s",
            name,
            trace_id,
            exc,
        )
