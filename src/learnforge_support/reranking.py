"""Reranking module for LearnForge Support Assistant.

Re-scores the hybrid retrieval shortlist using a local cross-encoder model
(Xenova/ms-marco-MiniLM-L-6-v2) to prioritize high-precision relevance
before passing evidence to the reliability layer.
"""

from collections.abc import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder

from learnforge_support.logging_utils import Stopwatch
from learnforge_support.schemas import EvidenceItem


def get_cross_encoder(
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
) -> TextCrossEncoder:
    """Initializes and returns the FastEmbed TextCrossEncoder model."""
    return TextCrossEncoder(model_name=model_name)


def rerank_candidates(
    query: str,
    items: Sequence[EvidenceItem],
    reranker: TextCrossEncoder,
    final_top_k: int = 5,
) -> tuple[list[EvidenceItem], float]:
    """Scores candidate shortlist with the cross-encoder and returns top evidence with latency.

    Never runs on the full collection; only reranks the fused candidate shortlist.
    Returns (reranked_evidence_items, elapsed_ms).
    """
    if not items:
        return [], 0.0

    stopwatch = Stopwatch()
    doc_texts = [f"{item.record.title}\n\n{item.record.text}" for item in items]
    raw_scores = list(reranker.rerank(query=query, documents=doc_texts))

    scored_items: list[EvidenceItem] = []
    for item, score in zip(items, raw_scores, strict=True):
        scored_items.append(
            EvidenceItem(
                record=item.record,
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
                rrf_score=item.rrf_score,
                reranker_score=float(score),
            )
        )

    # Sort descending by cross-encoder relevance score
    scored_items.sort(key=lambda x: -x.reranker_score)
    top_items = scored_items[:final_top_k]
    elapsed_ms = stopwatch.elapsed_ms()

    return top_items, elapsed_ms
