"""Domain and API schemas for LearnForge AI Support Assistant.

Strict Pydantic v2 schemas used across ingestion, indexing, retrieval,
reliability checks, LLM structured outputs, and HTTP endpoints.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# Enumerated types for strict domain boundaries
SourceType = Literal["faq", "policy", "ticket"]
TemporalStatus = Literal["current", "current_unversioned", "historical"]
AuthorityTier = Literal["policy", "faq", "historical_example"]
SupportDecisionType = Literal["answer", "clarify", "escalate"]
ReasonCode = Literal[
    "grounded_answer",
    "ambiguous_intent",
    "missing_context",
    "insufficient_evidence",
    "conflicting_evidence",
    "account_specific",
    "policy_exception",
]


class KnowledgeRecord(BaseModel):
    """A single canonical knowledge record extracted from the source corpus."""

    record_id: str = Field(
        description="Unique identifier preserving original document label (e.g. FAQ-01, POLICY-02, TICKET-08).",
    )
    source_type: SourceType = Field(
        description="Origin document category: faq, policy, or ticket.",
    )
    title: str = Field(
        description="Title, heading, or primary subject of the entry.",
    )
    text: str = Field(
        description="Complete unmodified text content of the record.",
    )
    source_date: date | None = Field(
        default=None,
        description="Explicit effective, updated, or reviewed date if stated in the source.",
    )
    temporal_status: TemporalStatus = Field(
        description="Temporal validity status: current, current_unversioned, or historical.",
    )
    authority_tier: AuthorityTier = Field(
        description="Source precedence tier: policy > faq > historical_example.",
    )
    contains_deprecated_reference: bool = Field(
        default=False,
        description="Flag indicating if the record mentions obsolete, retired, or superseded information.",
    )
    ticket_status: str | None = Field(
        default=None,
        description="Resolution status string for support ticket transcripts.",
    )
    content_hash: str = Field(
        description="SHA-256 hash of canonical content and metadata for change detection and idempotency.",
    )


class EvidenceItem(BaseModel):
    """A retrieved knowledge record paired with multi-stage ranking metrics."""

    record: KnowledgeRecord
    dense_rank: int | None = Field(
        default=None,
        description="Rank position from dense vector search (1-indexed).",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank position from BM25 sparse search (1-indexed).",
    )
    rrf_score: float = Field(
        default=0.0,
        description="Fused score computed via Reciprocal Rank Fusion.",
    )
    reranker_score: float = Field(
        default=0.0,
        description="Relevance logit score produced by the cross-encoder reranker.",
    )


class EvidenceContext(BaseModel):
    """Aggregated evidence analysis produced by the reliability layer."""

    items: list[EvidenceItem]
    has_current_policy: bool
    stale_only: bool
    contains_deprecated_reference: bool
    source_types: set[str]


class SupportDecision(BaseModel):
    """Structured decision output expected from the LLM generation step."""

    decision: SupportDecisionType = Field(
        description="Primary routing decision: answer, clarify, or escalate.",
    )
    message: str = Field(
        description="User-facing response text, clarification prompt, or escalation notice.",
    )
    reason_code: ReasonCode = Field(
        description="Formal justification code explaining why this decision was chosen.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of record IDs directly supporting factual claims in the message.",
    )
    handoff_summary: str | None = Field(
        default=None,
        description="Structured briefing for human support agents if escalated.",
    )


class Citation(BaseModel):
    """User-facing citation containing record identifier and document title."""

    record_id: str
    title: str


class ChatRequest(BaseModel):
    """API request payload for conversational chat."""

    session_id: str | None = Field(
        default=None,
        description="Optional session identifier. A new UUID is generated if omitted.",
    )
    message: str = Field(
        min_length=1,
        description="The customer's incoming message or question.",
    )


class ChatResponse(BaseModel):
    """API response payload returned to the customer or frontend."""

    session_id: str
    decision: SupportDecisionType
    message: str
    reason_code: ReasonCode
    citations: list[Citation] = Field(default_factory=list)
    handoff_summary: str | None = None


class HealthResponse(BaseModel):
    """API health status and diagnostic details."""

    status: str
    index_ready: bool
    record_count: int
    llm_model: str
    dense_model: str
