"""Unit tests for the cross-encoder reranking stage.

Verifies candidate scoring, ordering, top-k pruning, and latency measurement.
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from learnforge_support.config import get_settings
from learnforge_support.reranking import get_cross_encoder, rerank_candidates
from learnforge_support.schemas import EvidenceItem, KnowledgeRecord


def _create_item(rec_id: str, title: str, text: str) -> EvidenceItem:
    rec = KnowledgeRecord(
        record_id=rec_id,
        source_type="faq" if "FAQ" in rec_id else "policy",
        title=title,
        text=text,
        temporal_status="current_unversioned",
        authority_tier="faq",
        content_hash=f"hash-{rec_id}",
    )
    return EvidenceItem(record=rec, rrf_score=0.01)


def test_rerank_empty_candidates():
    """Verify reranking empty list returns empty list and zero elapsed time."""
    reranker = get_cross_encoder()
    results, elapsed = rerank_candidates("query", [], reranker, final_top_k=5)
    assert results == []
    assert elapsed == 0.0


def test_rerank_reorders_by_semantic_relevance():
    """Verify cross-encoder prioritizes highly relevant documents over distractor text."""
    settings = get_settings()
    reranker = TextCrossEncoder(settings.rerank_model)

    item_distractor = _create_item(
        "FAQ-05",
        "How do I reset my password?",
        "To reset password, click forgot password and check your email for a link.",
    )
    item_relevant = _create_item(
        "POLICY-02",
        "Cancellation and Refund Policy",
        "LearnForge allows eligible course purchases to be refunded within 14 days of purchase.",
    )

    query = "How long do I have to request a refund for a course?"
    # Pass distractor first
    shortlist = [item_distractor, item_relevant]

    reranked, elapsed_ms = rerank_candidates(
        query=query,
        items=shortlist,
        reranker=reranker,
        final_top_k=2,
    )

    assert len(reranked) == 2
    assert elapsed_ms > 0.0
    # Relevant item must be reranked to position 1
    assert reranked[0].record.record_id == "POLICY-02"
    assert reranked[0].reranker_score > reranked[1].reranker_score
