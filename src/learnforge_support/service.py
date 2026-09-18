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

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient

from learnforge_support.config import Settings
from learnforge_support.conversation import ConversationStore
from learnforge_support.indexing import get_qdrant_client
from learnforge_support.llm import LLMClientProtocol, generate_decision
from learnforge_support.logging_utils import Stopwatch, log_request_event, setup_logger
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
        self.llm_client = llm_client
        self.conversation_store = conversation_store or ConversationStore(
            max_turns=settings.max_conversation_turns
        )

        # Cache of record titles for rapid citation formatting
        self._record_titles: dict[str, str] = {}
        self._load_record_titles_cache()

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

    def process_chat(self, request: ChatRequest) -> tuple[ChatResponse, dict[str, float]]:
        """Processes a single customer turn through the full retrieval and reliability pipeline.

        Returns (ChatResponse, timings_dict).
        """
        request_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        total_timer = Stopwatch()

        # Step 1: Bounded conversation context lookup
        history = self.conversation_store.get_history(session_id)

        # Step 2: Contextual query formulation
        retrieval_query = build_retrieval_query(
            current_message=request.message,
            recent_turns=history,
        )

        # Step 3: First-stage hybrid retrieval & Reciprocal Rank Fusion
        retrieval_timer = Stopwatch()
        fused_candidates: list[EvidenceItem] = []
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
        retrieval_ms = retrieval_timer.elapsed_ms()

        # Step 4: Cross-encoder reranking
        top_evidence: list[EvidenceItem] = []
        rerank_ms = 0.0
        if fused_candidates:
            top_evidence, rerank_ms = rerank_candidates(
                query=retrieval_query,
                items=fused_candidates,
                reranker=self.reranker,
                final_top_k=self.settings.final_top_k,
            )

        # Step 5: Source authority and freshness ordering
        authoritative_evidence = sort_evidence_by_authority(top_evidence)
        available_record_ids = {item.record.record_id for item in authoritative_evidence}

        # Step 6: LLM generation & reliability decision
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

        # Step 7: Update session conversation store
        self.conversation_store.add_turn(
            session_id=session_id, role="user", content=request.message
        )
        self.conversation_store.add_turn(
            session_id=session_id, role="assistant", content=decision.message
        )

        # Step 8: Enrich citations with document titles
        citations: list[Citation] = []
        for cid in decision.citations:
            title = self._record_titles.get(cid)
            if not title:
                # Attempt lookup from retrieved records
                match = next(
                    (
                        item.record.title
                        for item in authoritative_evidence
                        if item.record.record_id == cid
                    ),
                    cid,
                )
                title = match
            citations.append(Citation(record_id=cid, title=title))

        total_ms = total_timer.elapsed_ms()

        # Step 9: Structured telemetry logging
        log_request_event(
            logger=logger,
            request_id=request_id,
            session_id=session_id,
            decision=decision.decision,
            reason_code=decision.reason_code,
            retrieved_record_ids=list(available_record_ids),
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
        timings = {
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "llm_ms": round(llm_ms, 2),
            "total_ms": round(total_ms, 2),
        }
        return response, timings
