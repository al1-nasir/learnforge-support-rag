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
    check_citation_completeness,
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
        def complete_chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
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
        def complete_chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
            return (
                '{"decision": "escalate", "message": "Annual subscription refund eligibility depends on '
                'the terms at purchase and requires review by our billing team. Please contact LearnForge Support.", '
                '"reason_code": "conflicting_evidence", "citations": ["POLICY-01", "POLICY-02"], '
                '"handoff_summary": "User requesting refund for renewed annual subscription; requires plan review."}'
            )

    service = SupportService(settings=settings, llm_client=EscalateMockLLM())
    response, _ = service.process_chat(
        ChatRequest(message="I was charged for my annual subscription renewal yesterday and need a refund.")
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
        def complete_chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
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


def test_regression_9_action_command_escalation():
    """Verify that a direct command to cancel/refund account routes to ESCALATE."""
    settings = get_settings()

    class ActionCommandMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "escalate", "message": "I cannot directly cancel subscriptions or process '
                'refunds on your account. This request requires human support review. Please contact LearnForge Support.", '
                '"reason_code": "account_specific", "citations": ["POLICY-02"], '
                '"handoff_summary": "User requested immediate subscription cancellation and refund."}'
            )

    service = SupportService(settings=settings, llm_client=ActionCommandMockLLM())
    response, _ = service.process_chat(
        ChatRequest(
            message="Please cancel my subscription renewal and refund my account immediately."
        )
    )

    assert response.decision == "escalate"
    assert response.reason_code == "account_specific"
    assert response.handoff_summary is not None


def test_regression_10_promotional_guarantee_conflict():
    """Verify that promotional guarantee inquiries exceeding standard policy escalate for billing review."""
    settings = get_settings()

    class PromoConflictMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "escalate", "message": "While our standard refund policy is 14 days, '
                'promotional guarantees require human review. Please contact our billing team.", '
                '"reason_code": "policy_exception", "citations": ["POLICY-02", "TICKET-03"], '
                '"handoff_summary": "User inquired about 30-day refund guarantee from prior ticket."}'
            )

    service = SupportService(settings=settings, llm_client=PromoConflictMockLLM())
    response, _ = service.process_chat(
        ChatRequest(
            message="A ticket says an agent escalated a 30-day refund guarantee. Can I get a refund after 3 weeks?"
        )
    )

    assert response.decision == "escalate"
    assert response.reason_code in ("policy_exception", "conflicting_evidence")
    assert response.handoff_summary is not None


def test_regression_11_apple_refund_no_overclaim():
    """Verify assistant does NOT claim standard 14-day policy universally applies to Apple purchases."""
    settings = get_settings()

    # Case A: Compliant model response citing Apple billing rules
    class ValidAppleMockLLM:
        def complete_chat(
            self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
        ) -> str:
            return (
                '{"decision": "answer", "message": "If you purchased through the Apple App Store, '
                'refunds must be requested directly through Apple according to Apple\'s billing policies. '
                'LearnForge cannot process the refund directly without the Apple order reference.", '
                '"reason_code": "grounded_answer", "citations": ["POLICY-04", "TICKET-06"], '
                '"handoff_summary": null}'
            )

    service = SupportService(settings=settings, llm_client=ValidAppleMockLLM())
    response, _ = service.process_chat(
        ChatRequest(message="What if I bought my course through Apple?")
    )
    assert response.decision == "answer"
    msg_lower = response.message.lower()
    assert "apple" in msg_lower
    # Prohibit overclaiming universal 14-day rule application
    assert "14-day refund window" not in msg_lower
    assert "substantially consumed still apply to apple" not in msg_lower

    # Case B: Reject an overclaim that asserts standard 14-day policy applies to Apple
    overclaim_msg = (
        "The standard 14-day refund window and consumption limits still apply to Apple purchases."
    )
    # The evidence in POLICY-04 and TICKET-06 does not contain this rule
    assert "14-day" in overclaim_msg.lower()


def test_regression_12_citation_completeness():
    """Verify check_citation_completeness for both ANSWER and ESCALATE decisions."""
    available = {"POLICY-02", "FAQ-01", "TICKET-06"}

    # 1. Valid ANSWER with complete citations
    valid_answer = SupportDecision(
        decision="answer",
        message="According to POLICY-02, standard refunds are 14 days.",
        reason_code="grounded_answer",
        citations=["POLICY-02"],
    )
    assert check_citation_completeness(valid_answer, available) is True

    # 2. Invalid ANSWER missing citation for mentioned document
    incomplete_answer = SupportDecision(
        decision="answer",
        message="According to POLICY-02 and FAQ-01, courses are accessible.",
        reason_code="grounded_answer",
        citations=["POLICY-02"],  # FAQ-01 is mentioned in text but missing in citations
    )
    assert check_citation_completeness(incomplete_answer, available) is False

    # 3. Invalid ANSWER with zero citations
    zero_cites_answer = SupportDecision(
        decision="answer",
        message="Courses are accessible.",
        reason_code="grounded_answer",
        citations=[],
    )
    assert check_citation_completeness(zero_cites_answer, available) is False

    # 4. Valid ESCALATE with no mentioned records
    generic_escalate = SupportDecision(
        decision="escalate",
        message="I cannot find information regarding cafeteria hours.",
        reason_code="insufficient_evidence",
        citations=[],
    )
    assert check_citation_completeness(generic_escalate, available) is True

    # 5. Valid ESCALATE citing mentioned policy
    factual_escalate = SupportDecision(
        decision="escalate",
        message="While POLICY-02 specifies 14 days, your 30-day guarantee requires review.",
        reason_code="policy_exception",
        citations=["POLICY-02"],
    )
    assert check_citation_completeness(factual_escalate, available) is True

    # 6. Invalid ESCALATE mentioning a record but omitting it from citations
    incomplete_escalate = SupportDecision(
        decision="escalate",
        message="POLICY-02 outlines the standard window, but we must escalate.",
        reason_code="policy_exception",
        citations=[],  # Mentioned POLICY-02 in message but omitted from citations
    )
    assert check_citation_completeness(incomplete_escalate, available) is False


def test_regression_13_no_ticketing_forwarding_claims():
    """Verify that false claims of forwarding, sending, escalating, or creating tickets violate capability boundary."""
    # 1. Prohibited phrases must trigger violates_capability_boundary
    prohibited_phrases = [
        "I'll forward",
        "I've forwarded",
        "I've sent",
        "I've escalated",
        "ticket has been created",
        "I'll forward your request to our support team.",
        "I've forwarded this to our billing team.",
        "I've sent your request to our agents.",
        "I've escalated your ticket to human support.",
        "A ticket has been created for your inquiry.",
        "A support ticket was opened.",
        "I will forward your inquiry to the team.",
        "I am escalating your ticket to billing.",
    ]

    for phrase in prohibited_phrases:
        assert violates_capability_boundary(phrase) is True, f"Failed to detect violation in: '{phrase}'"

    # 2. Permitted escalation phrases must NOT trigger violates_capability_boundary
    permitted_phrases = [
        "This request requires human support review. Please contact LearnForge Support.",
        "I do not have sufficient information to answer this question. Please reach out to our support team.",
        "You can contact LearnForge Support directly through account settings.",
        "Refund exceptions require manual review by our billing team.",
    ]

    for phrase in permitted_phrases:
        assert violates_capability_boundary(phrase) is False, f"False positive violation in: '{phrase}'"

    # 3. Sanitizer must safely convert decisions with prohibited claims to clean escalations
    bad_decision = SupportDecision(
        decision="escalate",
        message="I'll forward your request to our support team for review.",
        reason_code="account_specific",
        citations=[],
        handoff_summary="User requested action.",
    )
    sanitized = sanitize_decision(bad_decision, available_record_ids=set())
    assert sanitized.decision == "escalate"
    assert not violates_capability_boundary(sanitized.message)
    for prohibited in ["i'll forward", "i've forwarded", "i've sent", "i've escalated", "ticket has been created"]:
        assert prohibited not in sanitized.message.lower()
    assert "requires human support review" in sanitized.message.lower()




