"""Unit tests for the reliability layer, citation validation, and LLM retry boundary.

Verifies source precedence sorting, capability and security boundary guards,
citation rejection, and bounded retry error handling.
"""

import pytest

from learnforge_support.config import Settings
from learnforge_support.llm import generate_decision
from learnforge_support.reliability import (
    CitationValidationError,
    sanitize_decision,
    sort_evidence_by_authority,
    validate_citations,
    violates_capability_boundary,
    violates_security_boundary,
)
from learnforge_support.schemas import EvidenceItem, KnowledgeRecord, SupportDecision


def _create_evidence_item(
    rec_id: str,
    source_type: str,
    tier: str,
    deprecated: bool = False,
    score: float = 0.0,
) -> EvidenceItem:
    rec = KnowledgeRecord(
        record_id=rec_id,
        source_type=source_type,  # type: ignore[arg-type]
        title=f"Title {rec_id}",
        text=f"Text content for {rec_id}",
        temporal_status="current" if tier == "policy" else "historical",
        authority_tier=tier,  # type: ignore[arg-type]
        contains_deprecated_reference=deprecated,
        content_hash=f"hash-{rec_id}",
    )
    return EvidenceItem(record=rec, reranker_score=score)


def test_sort_evidence_precedence():
    """Verify policies outrank FAQs, which outrank tickets."""
    ticket = _create_evidence_item("TICKET-01", "ticket", "historical_example", score=5.0)
    faq = _create_evidence_item("FAQ-01", "faq", "faq", score=4.0)
    policy = _create_evidence_item("POLICY-01", "policy", "policy", score=3.0)

    # Even though ticket had highest relevance score, policy must rank first by authority
    sorted_items = sort_evidence_by_authority([ticket, faq, policy])
    assert sorted_items[0].record.record_id == "POLICY-01"
    assert sorted_items[1].record.record_id == "FAQ-01"
    assert sorted_items[2].record.record_id == "TICKET-01"


def test_validate_citations_valid_and_invalid():
    """Verify known citations pass while unknown/hallucinated citations raise error."""
    available = {"POLICY-02", "FAQ-01"}

    # Valid
    cites = validate_citations(["POLICY-02"], available)
    assert cites == ["POLICY-02"]

    # Unknown citation (e.g. POLICY-99)
    with pytest.raises(CitationValidationError):
        validate_citations(["POLICY-99"], available)


def test_capability_boundary_detection():
    """Verify assistant cannot claim that it performed account actions."""
    assert violates_capability_boundary("I have cancelled your subscription.") is True
    assert violates_capability_boundary("I issued your refund for the course.") is True
    assert violates_capability_boundary("I restored your course progress.") is True
    assert (
        violates_capability_boundary("You can cancel your subscription in Account Settings.")
        is False
    )


def test_security_boundary_detection():
    """Verify assistant cannot request sensitive credentials or card details."""
    assert violates_security_boundary("Please provide your password to proceed.") is True
    assert violates_security_boundary("Send your full credit card number and CVV.") is True
    assert violates_security_boundary("Please provide your order number or receipt.") is False


def test_sanitize_decision_converts_capability_violation():
    """Verify an answer claiming action execution is converted to escalation."""
    bad_decision = SupportDecision(
        decision="answer",
        message="I cancelled your subscription and refunded your card.",
        reason_code="grounded_answer",
        citations=["POLICY-02"],
        handoff_summary=None,
    )
    sanitized = sanitize_decision(bad_decision, {"POLICY-02"})
    assert sanitized.decision == "escalate"
    assert sanitized.reason_code == "account_specific"
    assert "cannot directly perform" in sanitized.message


def test_sanitize_decision_rejects_hallucinated_citation():
    """Verify an answer with a hallucinated citation is converted to escalation."""
    bad_decision = SupportDecision(
        decision="answer",
        message="According to policy, refunds are allowed.",
        reason_code="grounded_answer",
        citations=["POLICY-99"],
    )
    sanitized = sanitize_decision(bad_decision, {"POLICY-02"})
    assert sanitized.decision == "escalate"
    assert sanitized.reason_code == "insufficient_evidence"


class MockLLMClient:
    """Mock client returning controlled responses for testing."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def complete_chat(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        self.call_count += 1
        return self.responses.pop(0)


def test_llm_bounded_retry_succeeds():
    """Verify that a malformed first response triggers a single retry and succeeds."""
    mock_client = MockLLMClient(
        [
            "Not valid json at all",  # Attempt 1 fails
            '{"decision": "answer", "message": "Refunds take 14 days.", "reason_code": "grounded_answer", "citations": ["POLICY-02"], "handoff_summary": null}',  # Attempt 2 succeeds
        ]
    )
    settings = Settings(groq_api_key="mock-key")
    decision, elapsed_ms = generate_decision(
        messages=[{"role": "user", "content": "test"}],
        available_record_ids={"POLICY-02"},
        settings=settings,
        client=mock_client,
    )

    assert mock_client.call_count == 2
    assert decision.decision == "answer"
    assert decision.citations == ["POLICY-02"]
    assert elapsed_ms >= 0.0


def test_llm_both_attempts_fail_safe_fallback():
    """Verify that when both attempts fail, system safely returns ESCALATE without throwing."""
    mock_client = MockLLMClient(
        [
            "invalid 1",
            "invalid 2",
        ]
    )
    settings = Settings(groq_api_key="mock-key")
    decision, _ = generate_decision(
        messages=[{"role": "user", "content": "test"}],
        available_record_ids={"POLICY-02"},
        settings=settings,
        client=mock_client,
    )

    assert mock_client.call_count == 2
    assert decision.decision == "escalate"
    assert decision.reason_code == "insufficient_evidence"
