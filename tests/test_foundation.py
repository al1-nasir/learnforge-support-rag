"""Foundational tests for configuration, schemas, and logging utilities.

Verifies that the core package imports and foundational types behave as
specified in Phase 1 of PLAN.md.
"""

from datetime import date

from learnforge_support.config import Settings
from learnforge_support.logging_utils import Stopwatch, setup_logger
from learnforge_support.schemas import (
    Citation,
    KnowledgeRecord,
    SupportDecision,
)


def test_settings_defaults():
    """Verify application settings initialize with expected defaults."""
    settings = Settings(groq_api_key="test-key")
    assert settings.groq_api_key == "test-key"
    assert settings.dense_top_k == 8
    assert settings.sparse_top_k == 8
    assert settings.fused_top_k == 10
    assert settings.final_top_k == 5
    assert settings.rrf_k == 60
    assert settings.qdrant_collection == "learnforge_support"


def test_knowledge_record_validation():
    """Verify KnowledgeRecord validates types, source categories, and metadata."""
    record = KnowledgeRecord(
        record_id="POLICY-02",
        source_type="policy",
        title="Cancellation and Refund Policy",
        text="Eligible refund period is 14 days.",
        source_date=date(2026, 1, 1),
        temporal_status="current",
        authority_tier="policy",
        contains_deprecated_reference=True,
        ticket_status=None,
        content_hash="abc123hash",
    )
    assert record.record_id == "POLICY-02"
    assert record.source_type == "policy"
    assert record.authority_tier == "policy"
    assert record.contains_deprecated_reference is True
    assert record.source_date == date(2026, 1, 1)


def test_support_decision_schema():
    """Verify SupportDecision schema structure and validation."""
    decision = SupportDecision(
        decision="answer",
        message="Course refunds are available within 14 days.",
        reason_code="grounded_answer",
        citations=["POLICY-02"],
        handoff_summary=None,
    )
    assert decision.decision == "answer"
    assert decision.reason_code == "grounded_answer"
    assert decision.citations == ["POLICY-02"]


def test_citation_schema():
    """Verify Citation schema maps ID and title correctly."""
    cite = Citation(record_id="FAQ-01", title="How do I access a course?")
    assert cite.record_id == "FAQ-01"
    assert cite.title == "How do I access a course?"


def test_stopwatch_measures_elapsed_time():
    """Verify Stopwatch returns positive elapsed milliseconds."""
    sw = Stopwatch()
    elapsed = sw.elapsed_ms()
    assert elapsed >= 0.0


def test_logger_setup():
    """Verify logger setup produces an active logger."""
    logger = setup_logger("DEBUG")
    assert logger.name == "learnforge_support"
