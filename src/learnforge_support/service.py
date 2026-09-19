"""Service orchestration module for LearnForge Support Assistant.

This is the primary coordinator for the entire request lifecycle. Connects:
1. Session history lookup
2. Contextual query formulation
3. Hybrid dense + BM25 retrieval
4. Reciprocal Rank Fusion (RRF)
5. Cross-encoder reranking
6. Source authority & reliability sorting
7. Prompt generation & structured LLM invocation
8. Citation enrichment & response packaging
9. Structured latency logging
"""

import uuid
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient

from learnforge_support.config import Settings
from learnforge_support.conversation import ConversationStore
from learnforge_support.indexing import get_qdrant_client
from learnforge_support.llm import GroqLLMClient, LLMClientProtocol, generate_decision
from learnforge_support.logging_utils import Stopwatch, log_request_event, setup_logger
from learnforge_support.observability import (
    create_trace_id,
    start_child_span,
    start_support_trace,
)
from learnforge_support.prompting import build_chat_messages
from learnforge_support.reliability import sort_evidence_by_authority
from learnforge_support.reranking import get_cross_encoder, rerank_candidates
from learnforge_support.retrieval import build_retrieval_query, hybrid_retrieve
from learnforge_support.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    EvidenceItem,
)

logger = setup_logger()


class SupportService:
    """End-to-end customer support RAG service orchestrator."""

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient | None = None,
        dense_model: TextEmbedding | None = None,
        sparse_model: SparseTextEmbedding | None = None,
        reranker: TextCrossEncoder | None = None,
        llm_client: LLMClientProtocol | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.qdrant_client = qdrant_client or get_qdrant_client(settings.qdrant_path)
        self.dense_model = dense_model or TextEmbedding(model_name=settings.dense_model)
        self.sparse_model = sparse_model or SparseTextEmbedding(model_name=settings.sparse_model)
        self.reranker = reranker or get_cross_encoder(model_name=settings.rerank_model)
        self.llm_client = llm_client or self._build_llm_client()
        self.conversation_store = conversation_store or ConversationStore(
            max_turns=settings.max_conversation_turns
        )

        self._record_titles: dict[str, str] = {}
        self._load_record_titles_cache()

    def _build_llm_client(self) -> LLMClientProtocol | None:
        if not self.settings.groq_api_key:
            return None
        return GroqLLMClient(
            api_key=self.settings.groq_api_key,
            min_request_interval_seconds=self.settings.llm_min_request_interval_seconds,
            rate_limit_max_attempts=self.settings.llm_rate_limit_max_attempts,
        )

    def _load_record_titles_cache(self) -> None:
        """Loads record IDs and titles from Qdrant into memory for citation labeling."""
        try:
            if self.qdrant_client.collection_exists(self.settings.qdrant_collection):
                points, _ = self.qdrant_client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    if point.payload:
                        rid = point.payload.get("record_id")
                        title = point.payload.get("title")
                        if rid and title:
                            self._record_titles[rid] = title
        except Exception as exc:
            logger.warning("Could not pre-load record titles cache: %s", exc)

    def is_index_ready(self) -> bool:
        """Checks if the Qdrant collection exists and has points."""
        try:
            if not self.qdrant_client.collection_exists(self.settings.qdrant_collection):
                return False
            info = self.qdrant_client.get_collection(self.settings.qdrant_collection)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def get_indexed_record_count(self) -> int:
        """Returns the total number of points in the knowledge collection."""
        try:
            if not self.qdrant_client.collection_exists(self.settings.qdrant_collection):
                return 0
            info = self.qdrant_client.get_collection(self.settings.qdrant_collection)
            return info.points_count or 0
        except Exception:
            return 0

    def process_chat(
        self,
        request: ChatRequest,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> tuple[ChatResponse, dict[str, Any]]:
        """Processes a single customer turn through the full retrieval and reliability pipeline.

        Returns (ChatResponse, timings_dict).
        """
        req_id = request_id or str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        total_timer = Stopwatch()

        trace_tags = list(tags) if tags else ["learnforge", "support-rag"]
        trace_meta = {"request_id": req_id, **(metadata or {})}

        with start_support_trace(
            request_id=req_id,
            session_id=session_id,
            message=request.message,
            tags=trace_tags,
            metadata=trace_meta,
        ) as root_obs:
            history = self.conversation_store.get_history(session_id)
            retrieval_query = build_retrieval_query(
                current_message=request.message,
                recent_turns=history,
            )

            retrieval_timer = Stopwatch()
            fused_candidates: list[EvidenceItem] = []
            with start_child_span(
                "hybrid-retrieval",
                input_data={"query": retrieval_query},
                metadata={
                    "dense_top_k": self.settings.dense_top_k,
                    "sparse_top_k": self.settings.sparse_top_k,
                    "rrf_k": self.settings.rrf_k,
                    "fused_top_k": self.settings.fused_top_k,
                },
            ) as ret_span:
                if self.is_index_ready():
                    fused_candidates = hybrid_retrieve(
                        client=self.qdrant_client,
                        collection_name=self.settings.qdrant_collection,
                        query=retrieval_query,
                        dense_model=self.dense_model,
                        sparse_model=self.sparse_model,
                        dense_top_k=self.settings.dense_top_k,
                        sparse_top_k=self.settings.sparse_top_k,
                        rrf_k=self.settings.rrf_k,
                        fused_top_k=self.settings.fused_top_k,
                    )
                ret_span.update(
                    output={
                        "candidate_count": len(fused_candidates),
                        "candidate_record_ids": [item.record.record_id for item in fused_candidates],
                    }
                )
            retrieval_ms = retrieval_timer.elapsed_ms()

            top_evidence: list[EvidenceItem] = []
            rerank_ms = 0.0
            with start_child_span(
                "cross-encoder-reranking",
                input_data={
                    "query": retrieval_query,
                    "candidate_count": len(fused_candidates),
                    "candidate_record_ids": [item.record.record_id for item in fused_candidates],
                },
                metadata={
                    "final_top_k": self.settings.final_top_k,
                    "rerank_model": self.settings.rerank_model,
                },
            ) as rerank_span:
                if fused_candidates:
                    top_evidence, rerank_ms = rerank_candidates(
                        query=retrieval_query,
                        items=fused_candidates,
                        reranker=self.reranker,
                        final_top_k=self.settings.final_top_k,
                    )
                rerank_span.update(
                    output={
                        "top_count": len(top_evidence),
                        "top_record_ids": [item.record.record_id for item in top_evidence],
                        "scores": [round(item.reranker_score, 4) for item in top_evidence],
                        "rerank_ms": round(rerank_ms, 2),
                    }
                )

            with start_child_span(
                "reliability-check",
                input_data={
                    "top_evidence_ids": [item.record.record_id for item in top_evidence],
                },
            ) as rel_span:
                authoritative_evidence = sort_evidence_by_authority(top_evidence)
                available_record_ids = {item.record.record_id for item in authoritative_evidence}
                evidence_summary = [
                    {
                        "record_id": item.record.record_id,
                        "source_type": item.record.source_type,
                        "authority_tier": item.record.authority_tier,
                        "temporal_status": item.record.temporal_status,
                        "contains_deprecated_reference": item.record.contains_deprecated_reference,
                    }
                    for item in authoritative_evidence
                ]
                rel_span.update(
                    output={
                        "authoritative_record_ids": [
                            item.record.record_id for item in authoritative_evidence
                        ],
                        "count": len(available_record_ids),
                        "evidence": evidence_summary,
                    }
                )

            llm_messages = build_chat_messages(
                user_message=request.message,
                evidence_items=authoritative_evidence,
                conversation_history=history,
            )
            decision, llm_ms = generate_decision(
                messages=llm_messages,
                available_record_ids=available_record_ids,
                settings=self.settings,
                client=self.llm_client,
            )

            self.conversation_store.add_turn(session_id=session_id, role="user", content=request.message)
            self.conversation_store.add_turn(session_id=session_id, role="assistant", content=decision.message)

            with start_child_span(
                "response-validation",
                input_data={
                    "decision": decision.decision,
                    "reason_code": decision.reason_code,
                    "citations": decision.citations,
                },
                metadata={"available_record_count": len(available_record_ids)},
            ) as val_span:
                citations: list[Citation] = []
                for cid in decision.citations:
                    title = self._record_titles.get(cid)
                    if not title:
                        match = next((item.record.title for item in authoritative_evidence if item.record.record_id == cid), cid)
                        title = match
                    citations.append(Citation(record_id=cid, title=title))
                val_span.update(
                    output={
                        "final_decision": decision.decision,
                        "reason_code": decision.reason_code,
                        "citations_count": len(citations),
                        "citations": [c.record_id for c in citations],
                        "handoff_required": decision.handoff_summary is not None,
                    }
                )

            total_ms = total_timer.elapsed_ms()

            log_request_event(
                logger=logger,
                request_id=req_id,
                session_id=session_id,
                decision=decision.decision,
                reason_code=decision.reason_code,
                retrieved_record_ids=[item.record.record_id for item in authoritative_evidence],
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            )

            response = ChatResponse(
                session_id=session_id,
                decision=decision.decision,
                message=decision.message,
                reason_code=decision.reason_code,
                citations=citations,
                handoff_summary=decision.handoff_summary,
            )

            root_obs.update(
                output={
                    "decision": response.decision,
                    "reason_code": response.reason_code,
                    "citations": [c.record_id for c in response.citations],
                    "message": response.message,
                    "handoff_summary": response.handoff_summary,
                },
                metadata={
                    "actual_decision": response.decision,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "rerank_ms": round(rerank_ms, 2),
                    "llm_ms": round(llm_ms, 2),
                    "total_ms": round(total_ms, 2),
                },
            )

        trace_id = create_trace_id(seed=req_id)
        timings: dict[str, Any] = {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "llm_ms": round(llm_ms, 2),
            "total_ms": round(total_ms, 2),
            "request_id": req_id,
            "trace_id": trace_id,
        }
        return response, timings
