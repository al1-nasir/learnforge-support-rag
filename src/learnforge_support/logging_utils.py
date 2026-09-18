"""Logging utilities for LearnForge AI Support Assistant.

Provides structured, readable operational logging. Tracks request IDs,
latencies, retrieval records, and decision codes without exposing secrets
or full conversation transcripts.
"""

import json
import logging
import sys
import time
from typing import Any

# Default application logger name
LOGGER_NAME = "learnforge_support"


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configures and returns the application logger.

    Avoids duplicating handlers if called multiple times.
    """
    logger = logging.getLogger(LOGGER_NAME)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_request_event(
    logger: logging.Logger,
    request_id: str,
    session_id: str,
    decision: str,
    reason_code: str,
    retrieved_record_ids: list[str],
    retrieval_ms: float,
    rerank_ms: float,
    llm_ms: float,
    total_ms: float,
    error_category: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emits a structured JSON log entry capturing engineering signals."""
    payload: dict[str, Any] = {
        "event": "chat_request",
        "request_id": request_id,
        "session_id": session_id,
        "decision": decision,
        "reason_code": reason_code,
        "retrieved_record_ids": retrieved_record_ids,
        "timings_ms": {
            "retrieval": round(retrieval_ms, 2),
            "rerank": round(rerank_ms, 2),
            "llm": round(llm_ms, 2),
            "total": round(total_ms, 2),
        },
    }
    if error_category:
        payload["error_category"] = error_category
    if extra:
        payload.update(extra)

    logger.info(json.dumps(payload))


class Stopwatch:
    """Simple high-resolution execution timer."""

    def __init__(self) -> None:
        self._start_time: float = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Returns elapsed time in milliseconds since timer creation."""
        return (time.perf_counter() - self._start_time) * 1000.0
