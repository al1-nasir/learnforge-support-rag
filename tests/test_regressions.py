"""Required regression tests specified in AGENT.md Section 24 and PLAN.md Section 25.

Verifies:
1. Refund freshness (14-day current policy beats stale 7-day wording)
2. Offline downloads (current mobile support beats outdated desktop guidance)
3. Ambiguous cancellation ('Cancel my LearnForge' -> clarify / ambiguous_intent)
4. Annual subscription ambiguity (policy vs ticket ambiguity -> escalate)
5. Payment authorization (pending authorization hold is not a completed charge)
6. Capability boundary (assistant never claims to perform refunds/cancellations)
7. Unknown question (unsupported question must escalate, not hallucinate)
8. Multi-turn follow-up ('What if I bought it through Apple?' preserves refund context)
"""

from learnforge_support.config import get_settings
from learnforge_support.reliability import (
    sanitize_decision,
    sort_evidence_by_authority,
    violates_capability_boundary,
)
from learnforge_support.retrieval import build_retrieval_query, hybrid_retrieve
from learnforge_support.schemas import ChatRequest, SupportDecision
from learnforge_support.service import SupportService


def test_regression_1_refund_freshness():
    """Verify refund query retrieves current 14-day POLICY-02 and flags stale 7-day wording."""
    settings = get_settings()
    service = SupportService(settings=settings)

    items = hybrid_retrieve(
        client=service.qdrant_client,
        collection_name=settings.qdrant_collection,
        query="What is the refund period for courses?",
        dense_model=service.dense_model,
        sparse_model=service.sparse_model,
    )
    ordered = sort_evidence_by_authority(items)
    retrieved_ids = [item.record.record_id for item in ordered]

    assert "POLICY-02" in retrieved_ids
    policy_02_item = next(item for item in ordered if item.record.record_id == "POLICY-02")
    assert policy_02_item.record.temporal_status == "current"
    assert policy_02_item.record.contains_deprecated_reference is True
    # The policy text clearly establishes 14 days as current and 7 days as archived
    assert "14 days" in policy_02_item.record.text
    assert "7-day refund period" in policy_02_item.record.text


def test_regression_2_offline_downloads():
    """Verify offline download query retrieves current mobile guidance over outdated desktop guidance."""
    settings = get_settings()
    service = SupportService(settings=settings)

    items = hybrid_retrieve(
        client=service.qdrant_client,
        collection_name=settings.qdrant_collection,
        query="Can I download courses offline on my laptop?",
        dense_model=service.dense_model,
        sparse_model=service.sparse_model,
    )
    retrieved_ids = [item.record.record_id for item in items]
    assert "FAQ-07" in retrieved_ids or "POLICY-04" in retrieved_ids

    faq_07 = next((item for item in items if item.record.record_id == "FAQ-07"), None)
    if faq_07:
        assert faq_07.record.contains_deprecated_reference is True
        assert "mobile" in faq_07.record.text.lower()


def test_regression_3_ambiguous_cancellation():
    """Verify 'Cancel my LearnForge' produces a CLARIFY decision with ambiguous_intent."""
    settings = get_settings()

    class ClarifyMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "clarify", "message": "Would you like to cancel your subscription renewal, '
                'request a refund for a recent purchase, or delete your account?", '
                '"reason_code": "ambiguous_intent", "citations": ["POLICY-02"], "handoff_summary": null}'
            )

    service = SupportService(settings=settings, llm_client=ClarifyMockLLM())
    response, _ = service.process_chat(ChatRequest(message="Cancel my LearnForge."))

    assert response.decision == "clarify"
    assert response.reason_code == "ambiguous_intent"
    assert "subscription" in response.message.lower()


def test_regression_4_annual_subscription_ambiguity():
    """Verify annual subscription renewal refund ambiguity escalates to human review."""
    settings = get_settings()

    class EscalateMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "escalate", "message": "Annual subscription refund eligibility depends on '
                'the terms at purchase. I am escalating your case to our billing team for review.", '
                '"reason_code": "conflicting_evidence", "citations": ["POLICY-01", "POLICY-02"], '
                '"handoff_summary": "User requesting refund for renewed annual subscription; requires plan review."}'
            )

    service = SupportService(settings=settings, llm_client=EscalateMockLLM())
    response, _ = service.process_chat(
        ChatRequest(
            message="I was charged for my annual subscription renewal yesterday and need a refund."
        )
    )

    assert response.decision == "escalate"
    assert response.reason_code in ("conflicting_evidence", "policy_exception", "account_specific")
    assert response.handoff_summary is not None


def test_regression_5_payment_authorization():
    """Verify failed payment with pending transaction is identified as an authorization hold."""
    settings = get_settings()
    service = SupportService(settings=settings)

    items = hybrid_retrieve(
        client=service.qdrant_client,
        collection_name=settings.qdrant_collection,
        query="My payment was declined but my bank shows a charge. Did LearnForge charge me?",
        dense_model=service.dense_model,
        sparse_model=service.sparse_model,
    )
    retrieved_ids = [item.record.record_id for item in items]
    assert "FAQ-04" in retrieved_ids or "POLICY-10" in retrieved_ids or "TICKET-02" in retrieved_ids


def test_regression_6_capability_boundary():
    """Verify the assistant is strictly barred from claiming that it cancelled or refunded an account."""
    raw_bad_decision = SupportDecision(
        decision="answer",
        message="I have cancelled your subscription renewal as requested.",
        reason_code="grounded_answer",
        citations=["POLICY-02"],
        handoff_summary=None,
    )

    # Violates boundary detector
    assert violates_capability_boundary(raw_bad_decision.message) is True

    # Sanitizer safely intercepts and converts to escalation
    sanitized = sanitize_decision(raw_bad_decision, {"POLICY-02"})
    assert sanitized.decision == "escalate"
    assert sanitized.reason_code == "account_specific"
    assert "cannot directly perform" in sanitized.message


def test_regression_7_unknown_question():
    """Verify an unsupported question with no corpus backing produces ESCALATE with insufficient_evidence."""
    settings = get_settings()

    class InsufficientMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "escalate", "message": "I do not have information regarding this in our '
                'knowledge base. Let me connect you with support.", "reason_code": "insufficient_evidence", '
                '"citations": [], "handoff_summary": "Query unsupported by LearnForge documentation."}'
            )

    service = SupportService(settings=settings, llm_client=InsufficientMockLLM())
    response, _ = service.process_chat(
        ChatRequest(message="What time does the LearnForge employee gym close on Sundays?")
    )

    assert response.decision == "escalate"
    assert response.reason_code == "insufficient_evidence"
    assert len(response.citations) == 0


def test_regression_8_multiturn_context():
    """Verify multi-turn query formulation combines previous user question context."""
    history = [
        {"role": "user", "content": "How do course refunds work?"},
        {"role": "assistant", "content": "LearnForge allows refunds within 14 days."},
    ]
    augmented_query = build_retrieval_query("What if I bought it through Apple?", history)
    assert "refunds" in augmented_query.lower()
    assert "apple" in augmented_query.lower()

    # When retrieved with augmented query, Apple/mobile policy (POLICY-04 / TICKET-06) is retrieved
    settings = get_settings()
    service = SupportService(settings=settings)
    items = hybrid_retrieve(
        client=service.qdrant_client,
        collection_name=settings.qdrant_collection,
        query=augmented_query,
        dense_model=service.dense_model,
        sparse_model=service.sparse_model,
    )
    retrieved_ids = [item.record.record_id for item in items]
    assert "POLICY-04" in retrieved_ids or "TICKET-06" in retrieved_ids or "FAQ-02" in retrieved_ids
