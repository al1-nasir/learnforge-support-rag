"""Unit and integration tests for retrieval and Reciprocal Rank Fusion (RRF).

Verifies query construction, pure mathematical properties of RRF, and
hybrid search over the Qdrant index.
"""

from fastembed import SparseTextEmbedding, TextEmbedding

from learnforge_support.config import get_settings
from learnforge_support.indexing import get_qdrant_client
from learnforge_support.retrieval import (
    build_retrieval_query,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)
from learnforge_support.schemas import KnowledgeRecord


def _make_record(record_id: str, title: str = "Title") -> KnowledgeRecord:
    return KnowledgeRecord(
        record_id=record_id,
        source_type="faq" if "FAQ" in record_id else "policy",
        title=title,
        text="Sample text content for retrieval testing.",
        temporal_status="current_unversioned",
        authority_tier="faq",
        content_hash=f"hash-{record_id}",
    )


def test_build_retrieval_query_no_history():
    """Verify query formulation without prior turns returns current message."""
    query = build_retrieval_query("How do refunds work?")
    assert query == "How do refunds work?"


def test_build_retrieval_query_with_history():
    """Verify query formulation incorporates prior user intent for follow-ups."""
    history = [
        {"role": "user", "content": "How do refunds work?"},
        {"role": "assistant", "content": "LearnForge allows refunds within 14 days."},
    ]
    query = build_retrieval_query("What if I bought it through Apple?", history)
    assert query == "How do refunds work? What if I bought it through Apple?"


def test_rrf_mathematical_properties():
    """Verify Reciprocal Rank Fusion computes expected rank-inverted scores."""
    rec_a = _make_record("REC-A")
    rec_b = _make_record("REC-B")
    rec_c = _make_record("REC-C")

    # REC-A is rank 1 in both dense and sparse
    # REC-B is rank 1 in dense only
    # REC-C is rank 1 in sparse only
    dense_candidates = [rec_a, rec_b]
    sparse_candidates = [rec_a, rec_c]

    fused = reciprocal_rank_fusion(
        dense_candidates=dense_candidates,
        sparse_candidates=sparse_candidates,
        rrf_k=60,
        fused_top_k=5,
    )

    assert len(fused) == 3
    # Top candidate must be REC-A
    assert fused[0].record.record_id == "REC-A"
    # Score for REC-A: 1/(60+1) + 1/(60+1) = 2/61 ~= 0.0327868
    expected_score_a = (1.0 / 61.0) + (1.0 / 61.0)
    assert abs(fused[0].rrf_score - expected_score_a) < 1e-6
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1

    # REC-B has dense rank 2 -> 1/(60+2) = 1/62
    expected_score_b = 1.0 / 62.0
    rec_b_item = next(item for item in fused if item.record.record_id == "REC-B")
    assert abs(rec_b_item.rrf_score - expected_score_b) < 1e-6
    assert rec_b_item.dense_rank == 2
    assert rec_b_item.sparse_rank is None


def test_hybrid_retrieval_finds_relevant_documents():
    """Verify hybrid retrieval over the built index retrieves expected policy and FAQ entries."""
    settings = get_settings()
    client = get_qdrant_client(settings.qdrant_path)
    dense_model = TextEmbedding(settings.dense_model)
    sparse_model = SparseTextEmbedding(settings.sparse_model)

    # Test refund query
    results = hybrid_retrieve(
        client=client,
        collection_name=settings.qdrant_collection,
        query="Can I get a refund for a course?",
        dense_model=dense_model,
        sparse_model=sparse_model,
        dense_top_k=8,
        sparse_top_k=8,
        fused_top_k=5,
    )
    retrieved_ids = [item.record.record_id for item in results]
    # Expect POLICY-02 and FAQ-02 in retrieved shortlist
    assert "POLICY-02" in retrieved_ids or "FAQ-02" in retrieved_ids
