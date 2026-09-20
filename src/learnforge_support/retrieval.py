"""Retrieval module for LearnForge Support Assistant.

Implements query formulation incorporating recent conversation context,
first-stage dense semantic retrieval, first-stage BM25 sparse retrieval,
and Reciprocal Rank Fusion (RRF) candidate fusion.
"""

from typing import Sequence
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from learnforge_support.schemas import EvidenceItem, KnowledgeRecord


def build_retrieval_query(
    current_message: str,
    recent_turns: Sequence[dict[str, str]] | None = None,
) -> str:
    """Builds a contextualized retrieval query incorporating recent conversation context.

    If prior turns exist, extracts key terms from the most recent user question(s)
    to resolve conversational anaphora (e.g. 'What if I bought it through Apple?'
    following 'How do refunds work?').
    """
    clean_current = current_message.strip()
    if not recent_turns:
        return clean_current

    # Collect prior user messages (up to 2 recent turns)
    prior_user_messages: list[str] = []
    for turn in reversed(recent_turns):
        if turn.get("role") == "user":
            prior_msg = turn.get("content", "").strip()
            if prior_msg and prior_msg.lower() not in clean_current.lower():
                prior_user_messages.append(prior_msg)
            if len(prior_user_messages) >= 2:
                break

    if not prior_user_messages:
        return clean_current

    prior_user_messages.reverse()
    context_prefix = " ".join(prior_user_messages)
    return f"{context_prefix} {clean_current}".strip()


def search_dense(
    client: QdrantClient,
    collection_name: str,
    query: str,
    dense_model: TextEmbedding,
    top_k: int = 8,
) -> list[tuple[KnowledgeRecord, float]]:
    """Retrieves top semantic candidates via dense cosine vector search."""
    dense_vec = list(dense_model.embed([query]))[0]
    results = client.query_points(
        collection_name=collection_name,
        query=dense_vec.tolist(),
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    candidates: list[tuple[KnowledgeRecord, float]] = []
    for point in results.points:
        if point.payload:
            record = KnowledgeRecord.model_validate(point.payload)
            candidates.append((record, float(point.score)))
    return candidates


def search_sparse(
    client: QdrantClient,
    collection_name: str,
    query: str,
    sparse_model: SparseTextEmbedding,
    top_k: int = 8,
) -> list[tuple[KnowledgeRecord, float]]:
    """Retrieves top keyword candidates via BM25 sparse vector search."""
    sparse_vec = list(sparse_model.embed([query]))[0]
    sparse_data = SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist(),
    )
    results = client.query_points(
        collection_name=collection_name,
        query=sparse_data,
        using="bm25",
        limit=top_k,
        with_payload=True,
    )
    candidates: list[tuple[KnowledgeRecord, float]] = []
    for point in results.points:
        if point.payload:
            record = KnowledgeRecord.model_validate(point.payload)
            candidates.append((record, float(point.score)))
    return candidates


def reciprocal_rank_fusion(
    dense_candidates: Sequence[KnowledgeRecord],
    sparse_candidates: Sequence[KnowledgeRecord],
    rrf_k: int = 60,
    fused_top_k: int = 10,
) -> list[EvidenceItem]:
    """Combines dense and sparse ranked candidate lists using Reciprocal Rank Fusion.

    Formula:
        RRF(d) = sum_{m in M} 1.0 / (rrf_k + rank_m(d))
    where rank_m(d) is 1-indexed. Does not sum raw dense/BM25 scores.
    """
    dense_ranks: dict[str, int] = {
        record.record_id: rank
        for rank, record in enumerate(dense_candidates, start=1)
    }
    sparse_ranks: dict[str, int] = {
        record.record_id: rank
        for rank, record in enumerate(sparse_candidates, start=1)
    }

    # Gather all unique records by record_id
    records_by_id: dict[str, KnowledgeRecord] = {}
    for rec in dense_candidates:
        records_by_id[rec.record_id] = rec
    for rec in sparse_candidates:
        records_by_id[rec.record_id] = rec

    fused_scores: list[tuple[str, float]] = []
    for rec_id in records_by_id:
        score = 0.0
        if rec_id in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[rec_id])
        if rec_id in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[rec_id])
        fused_scores.append((rec_id, score))

    # Sort descending by fused RRF score, breaking ties deterministically by record_id
    fused_scores.sort(key=lambda item: (-item[1], item[0]))

    shortlist: list[EvidenceItem] = []
    for rec_id, score in fused_scores[:fused_top_k]:
        shortlist.append(
            EvidenceItem(
                record=records_by_id[rec_id],
                dense_rank=dense_ranks.get(rec_id),
                sparse_rank=sparse_ranks.get(rec_id),
                rrf_score=score,
                reranker_score=0.0,
            )
        )
    return shortlist


def hybrid_retrieve(
    client: QdrantClient,
    collection_name: str,
    query: str,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    dense_top_k: int = 8,
    sparse_top_k: int = 8,
    rrf_k: int = 60,
    fused_top_k: int = 10,
) -> list[EvidenceItem]:
    """Orchestrates first-stage dual retrieval and Reciprocal Rank Fusion."""
    dense_results = search_dense(
        client=client,
        collection_name=collection_name,
        query=query,
        dense_model=dense_model,
        top_k=dense_top_k,
    )
    sparse_results = search_sparse(
        client=client,
        collection_name=collection_name,
        query=query,
        sparse_model=sparse_model,
        top_k=sparse_top_k,
    )

    dense_records = [record for record, _ in dense_results]
    sparse_records = [record for record, _ in sparse_results]

    return reciprocal_rank_fusion(
        dense_candidates=dense_records,
        sparse_candidates=sparse_records,
        rrf_k=rrf_k,
        fused_top_k=fused_top_k,
    )

