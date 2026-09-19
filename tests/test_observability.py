"""Unit tests for Langfuse observability integration.

Verifies fail-open behavior, missing credentials handling, error resilience,
instrumentation idempotency, and that SupportService.process_chat operates
normally whether observability is enabled, disabled, or encountering errors.
Runs completely offline with mocked provider boundaries.
"""

from unittest.mock import MagicMock, patch

import pytest

from learnforge_support.config import Settings
from learnforge_support.observability import (
    flush_observability,
    get_langfuse_client,
    initialize_observability,
    is_observability_enabled,
    record_eval_score,
    shutdown_observability,
    start_child_span,
    start_support_trace,
)
from learnforge_support.schemas import ChatRequest, EvidenceItem, KnowledgeRecord
from learnforge_support.service import SupportService


@pytest.fixture(autouse=True)
def cleanup_observability() -> None:
    """Ensures observability state and environment are cleanly reset before and after each test."""
    import os

    import learnforge_support.observability as obs

    shutdown_observability()
    obs._groq_instrumented = False
    for key in [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
        "LANGFUSE_TRACING_ENVIRONMENT",
    ]:
        os.environ.pop(key, None)
    yield
    shutdown_observability()
    obs._groq_instrumented = False
    for key in [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
        "LANGFUSE_TRACING_ENVIRONMENT",
    ]:
        os.environ.pop(key, None)


def test_observability_disabled_by_default() -> None:
    """Verifies that observability defaults to disabled and functions safely as no-op."""
    settings = Settings(langfuse_enabled=False)
    initialize_observability(settings)

    assert not is_observability_enabled()
    assert get_langfuse_client() is None

    # Test context managers and scoring fail open safely as no-ops
    with start_support_trace("req-1", "sess-1", "hello") as root:
        root.update(output={"status": "ok"})
        with start_child_span("child-span") as child:
            child.update(output={"sub": 1})

    record_eval_score("trace-1", "decision_correct", 1)
    flush_observability()
    shutdown_observability()


def test_missing_credentials_fails_open() -> None:
    """Verifies that enabling Langfuse without keys safely disables tracing without error."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="",
        langfuse_secret_key="",
    )
    initialize_observability(settings)

    assert not is_observability_enabled()
    assert get_langfuse_client() is None


def test_auth_check_failure_fails_open() -> None:
    """Verifies that a failed auth_check safely disables observability."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-invalid",
        langfuse_secret_key="sk-invalid",
    )
    mock_client = MagicMock()
    mock_client.auth_check.return_value = False

    with patch("learnforge_support.observability.Langfuse", return_value=mock_client):
        initialize_observability(settings)

    assert not is_observability_enabled()
    assert get_langfuse_client() is None


def test_initialization_exception_fails_open() -> None:
    """Verifies that an unexpected exception during initialization does not propagate."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )

    with patch(
        "learnforge_support.observability.Langfuse",
        side_effect=RuntimeError("Connection refused"),
    ):
        # Must not raise
        initialize_observability(settings)

    assert not is_observability_enabled()
    assert get_langfuse_client() is None


def test_groq_instrumentation_idempotent() -> None:
    """Verifies that Groq OpenInference instrumentation is called safely and idempotently."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    mock_client = MagicMock()
    mock_client.auth_check.return_value = True

    with (
        patch("learnforge_support.observability.Langfuse", return_value=mock_client),
        patch("learnforge_support.observability.GroqInstrumentor") as mock_groq_inst,
    ):
        mock_inst_instance = MagicMock()
        mock_inst_instance.is_instrumented_by_opentelemetry = False
        mock_groq_inst.return_value = mock_inst_instance

        # First initialization
        initialize_observability(settings)
        assert is_observability_enabled()
        assert mock_inst_instance.instrument.call_count == 1

        # Second initialization (should not re-instrument Groq)
        initialize_observability(settings)
        assert mock_inst_instance.instrument.call_count == 1


def test_process_chat_operates_normally_when_observability_disabled() -> None:
    """Proves that SupportService.process_chat operates completely normally with Langfuse disabled."""
    settings = Settings(langfuse_enabled=False)
    initialize_observability(settings)

    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = (
        '{"decision": "answer", "message": "Refunds are 14 days.", '
        '"reason_code": "grounded_answer", "citations": ["POLICY-02"], "handoff_summary": null}'
    )
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False
    mock_dense = MagicMock()
    mock_sparse = MagicMock()
    mock_reranker = MagicMock()

    service = SupportService(
        settings=settings,
        qdrant_client=mock_qdrant,
        dense_model=mock_dense,
        sparse_model=mock_sparse,
        reranker=mock_reranker,
        llm_client=mock_llm,
    )
    evidence_item = EvidenceItem(
        record=KnowledgeRecord(
            record_id="POLICY-02",
            source_type="policy",
            title="Refund Policy",
            text="Refunds are 14 days.",
            temporal_status="current",
            authority_tier="policy",
            content_hash="hash-123",
        ),
        rrf_score=1.0,
        rerank_score=0.95,
    )

    service.is_index_ready = MagicMock(return_value=True)

    with (
        patch("learnforge_support.service.hybrid_retrieve", return_value=[evidence_item]),
        patch("learnforge_support.service.rerank_candidates", return_value=([evidence_item], 5.0)),
    ):
        req = ChatRequest(session_id="test-sess", message="What is the refund policy?")
        res, timings = service.process_chat(req)

        assert res.decision == "answer"
        assert "POLICY-02" in [c.record_id for c in res.citations]
        assert "total_ms" in timings


def test_eval_scoring_fails_open_when_disabled_or_errored() -> None:
    """Verifies that record_eval_score never raises exceptions even if the client errors."""
    # When disabled
    record_eval_score("trace-dummy", "decision_correct", 1)

    # When enabled but create_score raises an error
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    mock_client = MagicMock()
    mock_client.auth_check.return_value = True
    mock_client.create_score.side_effect = RuntimeError("Score export failed")

    with patch("learnforge_support.observability.Langfuse", return_value=mock_client):
        initialize_observability(settings)
        assert is_observability_enabled()
        # Must not raise
        record_eval_score("trace-dummy", "decision_correct", 1)


def test_observability_spans_mocked_execution() -> None:
    """Verifies that start_support_trace and start_child_span update outputs on active observation."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    mock_client = MagicMock()
    mock_client.auth_check.return_value = True
    mock_root_span = MagicMock()
    mock_child_span = MagicMock()

    # Configure context manager returns
    mock_client.start_as_current_observation.side_effect = [
        MagicMock(__enter__=MagicMock(return_value=mock_root_span), __exit__=MagicMock()),
        MagicMock(__enter__=MagicMock(return_value=mock_child_span), __exit__=MagicMock()),
    ]

    with (
        patch("learnforge_support.observability.Langfuse", return_value=mock_client),
        patch("langfuse.propagate_attributes") as mock_propagate,
    ):
        mock_propagate.return_value.__enter__ = MagicMock()
        mock_propagate.return_value.__exit__ = MagicMock()

        initialize_observability(settings)
        assert is_observability_enabled()

        with start_support_trace("req-100", "sess-100", "hello") as root:
            root.update(output={"decision": "answer"})
            with start_child_span("hybrid-retrieval", input_data={"query": "hello"}) as child:
                child.update(output={"candidate_record_ids": ["POLICY-02"]})

        assert mock_root_span.update.called
        assert mock_child_span.update.called


def test_process_chat_emits_expected_spans_when_enabled() -> None:
    """Verifies that SupportService.process_chat creates expected spans when Langfuse is enabled."""
    settings = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    mock_client = MagicMock()
    mock_client.auth_check.return_value = True

    created_observations = []

    def fake_start_observation(**kwargs):
        obs_mock = MagicMock()
        obs_mock.name = kwargs.get("name")
        created_observations.append((kwargs.get("name"), obs_mock))
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=obs_mock)
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    mock_client.start_as_current_observation.side_effect = fake_start_observation

    with (
        patch("learnforge_support.observability.Langfuse", return_value=mock_client),
        patch("langfuse.propagate_attributes") as mock_propagate,
    ):
        mock_propagate.return_value.__enter__ = MagicMock()
        mock_propagate.return_value.__exit__ = MagicMock()

        initialize_observability(settings)
        assert is_observability_enabled()

        mock_llm = MagicMock()
        mock_llm.complete_chat.return_value = (
            '{"decision": "answer", "message": "Refunds are 14 days.", '
            '"reason_code": "grounded_answer", "citations": ["POLICY-02"], "handoff_summary": null}'
        )
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_dense = MagicMock()
        mock_sparse = MagicMock()
        mock_reranker = MagicMock()

        service = SupportService(
            settings=settings,
            qdrant_client=mock_qdrant,
            dense_model=mock_dense,
            sparse_model=mock_sparse,
            reranker=mock_reranker,
            llm_client=mock_llm,
        )
        service.is_index_ready = MagicMock(return_value=True)

        evidence_item = EvidenceItem(
            record=KnowledgeRecord(
                record_id="POLICY-02",
                source_type="policy",
                title="Refund Policy",
                text="Refunds are 14 days.",
                temporal_status="current",
                authority_tier="policy",
                content_hash="hash-123",
            ),
            rrf_score=1.0,
            rerank_score=0.95,
        )

        with (
            patch("learnforge_support.service.hybrid_retrieve", return_value=[evidence_item]),
            patch("learnforge_support.service.rerank_candidates", return_value=([evidence_item], 5.0)),
        ):
            req = ChatRequest(session_id="test-sess", message="What is the refund policy?")
            res, timings = service.process_chat(req, tags=["test-tag"], metadata={"meta_key": "val"})

            assert res.decision == "answer"
            assert "trace_id" in timings
            assert "request_id" in timings

            span_names = [name for name, _ in created_observations]
            assert "support-request" in span_names
            assert "hybrid-retrieval" in span_names
            assert "cross-encoder-reranking" in span_names
            assert "reliability-check" in span_names
            assert "response-validation" in span_names
