"""Reliability module for LearnForge Support Assistant.

Applies source precedence rules, evidence context analysis, citation validation,
and capability/security boundary checks. Contains no LLM network calls.
"""

import re
from collections.abc import Sequence

from learnforge_support.schemas import (
    EvidenceContext,
    EvidenceItem,
    SupportDecision,
)

# Numeric authority weights: Lower value = higher authority
AUTHORITY_WEIGHTS: dict[str, int] = {
    "policy": 1,
    "faq": 2,
    "historical_example": 3,
}

# Regex patterns detecting prohibited claims of direct account execution or ticketing actions
_CAPABILITY_VIOLATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:cancelled|canceled)\s+your\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:issued|processed)\s+(?:your|a)\s+refund\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:restored|reset)\s+your\s+(?:account|course|progress)\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:deleted|closed)\s+your\s+account\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:checked|verified)\s+your\s+bank\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ll| will|'ve| have)?|we(?:'ll| will|'ve| have)?)\s+forward(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)|we(?:'ve| have))\s+sent\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ll| will)?|we(?:'ll| will)?)\s+send\s+(?:your|this|the|a)\s+(?:ticket|case|request|inquiry|details|message)\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+escalated\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ll| will| am|'m)?|we(?:'ll| will| are|'re)?)\s+escalat(?:e|ing)\s+(?:your|this|the|a)\s+(?:ticket|case|request|inquiry|issue)\b", re.IGNORECASE),
    re.compile(r"\b(?:ticket|case|support request)\s+(?:has been|was|is)\s+(?:created|opened|submitted|forwarded|routed|escalated)\b", re.IGNORECASE),
    re.compile(r"\b(?:created|opened|submitted|routed)\s+(?:a|the)\s+(?:ticket|case|support request)\b", re.IGNORECASE),
]

# Regex patterns detecting illicit requests for user credentials
_SECURITY_VIOLATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:send|provide|enter)\s+(?:your\s+)?(?:password|pin|cvv|security code)\b", re.IGNORECASE),
    re.compile(r"\b(?:send|provide)\s+(?:your\s+)?full\s+(?:card|credit card|debit card)\s+number\b", re.IGNORECASE),
]


class CitationValidationError(ValueError):
    """Raised when an LLM cites an unretrieved or hallucinated record ID."""


def sort_evidence_by_authority(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """Sorts evidence items prioritizing source authority and freshness while preserving relevance.

    Ordering priority:
    1. Authority tier (policy > faq > historical_example)
    2. Deprecation status (non-deprecated precedes deprecated)
    3. Cross-encoder reranker score (higher relevance first)
    """
    sorted_items = list(items)

    def sort_key(item: EvidenceItem) -> tuple[int, int, float]:
        tier_weight = AUTHORITY_WEIGHTS.get(item.record.authority_tier, 99)
        deprecated_penalty = 1 if item.record.contains_deprecated_reference else 0
        relevance = -item.reranker_score
        return (tier_weight, deprecated_penalty, relevance)

    sorted_items.sort(key=sort_key)
    return sorted_items


def build_evidence_context(items: Sequence[EvidenceItem]) -> EvidenceContext:
    """Aggregates signals from retrieved evidence records."""
    if not items:
        return EvidenceContext(
            items=[],
            has_current_policy=False,
            stale_only=True,
            contains_deprecated_reference=False,
            source_types=set(),
        )

    has_current_policy = any(
        item.record.authority_tier == "policy" and item.record.temporal_status == "current"
        for item in items
    )
    stale_only = all(
        item.record.contains_deprecated_reference or item.record.temporal_status == "historical"
        for item in items
    )
    contains_deprecated = any(item.record.contains_deprecated_reference for item in items)
    source_types = {item.record.source_type for item in items}

    return EvidenceContext(
        items=list(items),
        has_current_policy=has_current_policy,
        stale_only=stale_only,
        contains_deprecated_reference=contains_deprecated,
        source_types=source_types,
    )


def validate_citations(
    citations: Sequence[str],
    available_record_ids: set[str],
) -> list[str]:
    """Validates that cited record IDs exist in the retrieved evidence set.

    Raises CitationValidationError if any cited ID is not in available_record_ids.
    """
    valid_citations: list[str] = []
    seen: set[str] = set()

    for cid in citations:
        clean_id = cid.strip()
        if not clean_id:
            continue
        if clean_id not in available_record_ids:
            raise CitationValidationError(
                f"Model cited unretrieved or hallucinated record ID: '{clean_id}'. "
                f"Available records: {sorted(available_record_ids)}"
            )
        if clean_id not in seen:
            seen.add(clean_id)
            valid_citations.append(clean_id)

    return valid_citations


def violates_capability_boundary(text: str) -> bool:
    """Checks if the text erroneously claims execution of real account actions."""
    return any(pattern.search(text) for pattern in _CAPABILITY_VIOLATION_PATTERNS)


def violates_security_boundary(text: str) -> bool:
    """Checks if the text erroneously requests prohibited authentication secrets."""
    return any(pattern.search(text) for pattern in _SECURITY_VIOLATION_PATTERNS)


def sanitize_decision(
    decision: SupportDecision,
    available_record_ids: set[str],
) -> SupportDecision:
    """Guards against hallucinated citations, capability overreaches, and policy leaks.

    Ensures that factual ANSWERs contain valid citations, that unknown citations
    force an ESCALATE fallback, and that claims of real action execution are safely converted.
    """
    # 1. Capability boundary guard
    if violates_capability_boundary(decision.message):
        return SupportDecision(
            decision="escalate",
            message=(
                "I cannot directly perform account modifications, cancellations, or refunds. "
                "This request requires human support review. "
                "Please contact LearnForge Support through your account settings or official support channels."
            ),
            reason_code="account_specific",
            citations=[],
            handoff_summary="User requested an account modification that requires manual support staff intervention.",
        )

    # 2. Security boundary guard
    if violates_security_boundary(decision.message):
        return SupportDecision(
            decision="escalate",
            message=(
                "For your security, LearnForge will never ask for your password, PIN, CVV, or full card details. "
                "Please contact LearnForge Support directly through official account settings."
            ),
            reason_code="account_specific",
            citations=[],
            handoff_summary="Conversation flagged for prohibited sensitive payment or credential inquiries.",
        )

    # 3. Citation validation
    try:
        clean_cites = validate_citations(decision.citations, available_record_ids)
    except CitationValidationError:
        return SupportDecision(
            decision="escalate",
            message=(
                "I do not have sufficient verified documentation to answer this inquiry accurately. "
                "This request requires review by our support team. "
                "Please contact LearnForge Support directly."
            ),
            reason_code="insufficient_evidence",
            citations=[],
            handoff_summary="Model cited unverified documentation or records outside the retrieved evidence shortlist.",
        )

    # 4. Factual ANSWER grounding check
    if decision.decision == "answer" and not clean_cites and available_record_ids:
        # Factual answers must cite at least one supporting document
        return SupportDecision(
            decision="escalate",
            message=(
                "I do not have sufficient verified policy documentation to confirm this answer. "
                "This request requires review by our support team. "
                "Please contact LearnForge Support directly."
            ),
            reason_code="insufficient_evidence",
            citations=[],
            handoff_summary="Unverified response lacking explicit authoritative citations.",
        )

    return SupportDecision(
        decision=decision.decision,
        message=decision.message,
        reason_code=decision.reason_code,
        citations=clean_cites,
        handoff_summary=decision.handoff_summary,
    )


_RECORD_ID_REGEX = re.compile(r"\b(?:POLICY|FAQ|TICKET)-\d+\b", re.IGNORECASE)


def check_citation_completeness(
    decision: SupportDecision,
    available_record_ids: set[str],
) -> bool:
    """Checks whether records referenced in the message are completely and validly cited.

    Specifically verifies:
    1. For ANSWER decisions: At least one citation is required, and any record ID
       mentioned in the text that exists in available_record_ids must be cited.
    2. For CLARIFY / ESCALATE decisions: If record IDs are mentioned in the message text,
       any mentioned record existing in available_record_ids must be explicitly cited.
    """
    mentioned = {
        m.upper()
        for m in _RECORD_ID_REGEX.findall(decision.message)
        if m.upper() in available_record_ids
    }
    cited = {c.upper() for c in decision.citations}

    if decision.decision == "answer":
        if not cited:
            return False
        return mentioned.issubset(cited)

    # For clarify or escalate:
    if mentioned and not mentioned.issubset(cited):
        return False

    return True

