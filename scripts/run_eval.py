"""Evaluation runner for LearnForge AI Support Assistant.

Loads eval/golden.jsonl, benchmarks hybrid retrieval and reranking,
computes Hit@5, MRR@5, latency metrics, and writes detailed results
to eval/results/.
Usage:
    python -m scripts.run_eval
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learnforge_support.config import get_settings
from learnforge_support.logging_utils import setup_logger
from learnforge_support.reranking import rerank_candidates
from learnforge_support.retrieval import build_retrieval_query, hybrid_retrieve
from learnforge_support.schemas import ChatRequest
from learnforge_support.service import SupportService

logger = setup_logger()


def run_evaluation() -> dict[str, Any]:
    """Runs evaluation benchmark against eval/golden.jsonl."""
    settings = get_settings()
    service = SupportService(settings=settings)

    if not service.is_index_ready():
        raise RuntimeError("Qdrant index is not built. Run 'python -m scripts.build_index' first.")

    golden_file = Path("eval/golden.jsonl")
    if not golden_file.is_file():
        raise FileNotFoundError(f"Golden dataset not found at {golden_file}")

    cases: list[dict[str, Any]] = []
    with open(golden_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    total_cases = len(cases)
    retrieval_hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    retrieval_times_ms: list[float] = []
    rerank_times_ms: list[float] = []
    detailed_results: list[dict[str, Any]] = []

    # Check if live LLM evaluation is available
    has_live_llm = bool(settings.groq_api_key.strip())
    correct_decisions = 0
    valid_citations_count = 0
    total_answers = 0
    escalation_true_positives = 0
    escalation_false_positives = 0
    escalation_false_negatives = 0

    print("\n" + "=" * 80)
    print("LEARNFORGE SUPPORT RAG EVALUATION BENCHMARK")
    print(f"Total Test Cases: {total_cases} | Live LLM Enabled: {has_live_llm}")
    print("=" * 80 + "\n")

    for case in cases:
        cid = case["id"]
        query = case["query"]
        history = case.get("history", [])
        expected_decision = case.get("expected_decision")
        expected_sources = set(case.get("expected_sources", []))

        # Contextual query building
        augmented_query = build_retrieval_query(query, history)

        # First-stage hybrid retrieval
        t_ret_start = time.perf_counter()
        fused = hybrid_retrieve(
            client=service.qdrant_client,
            collection_name=settings.qdrant_collection,
            query=augmented_query,
            dense_model=service.dense_model,
            sparse_model=service.sparse_model,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
            fused_top_k=settings.fused_top_k,
        )
        ret_ms = (time.perf_counter() - t_ret_start) * 1000.0
        retrieval_times_ms.append(ret_ms)

        # Cross-encoder reranking
        top_5, rerank_ms = rerank_candidates(
            query=augmented_query,
            items=fused,
            reranker=service.reranker,
            final_top_k=5,
        )
        rerank_times_ms.append(rerank_ms)

        retrieved_ids = [item.record.record_id for item in top_5]

        # Calculate Hit@5 and MRR@5
        hit = False
        rr = 0.0
        if not expected_sources:
            # For unknown questions where no source is expected:
            hit = True
            rr = 1.0
        else:
            for rank_idx, rid in enumerate(retrieved_ids, start=1):
                if rid in expected_sources:
                    hit = True
                    if rr == 0.0:
                        rr = 1.0 / rank_idx
                    break

        if hit:
            retrieval_hits_at_5 += 1
        reciprocal_ranks.append(rr)

        result_entry: dict[str, Any] = {
            "id": cid,
            "query": query,
            "category": case.get("category"),
            "expected_decision": expected_decision,
            "expected_sources": list(expected_sources),
            "retrieved_top_5": retrieved_ids,
            "hit_at_5": hit,
            "mrr_at_5": round(rr, 4),
            "timings_ms": {
                "retrieval": round(ret_ms, 2),
                "rerank": round(rerank_ms, 2),
            },
        }

        # If live LLM is configured, test end-to-end generation
        if has_live_llm:
            chat_req = ChatRequest(message=query)
            # Replay history if multi-turn
            for h in history:
                service.conversation_store.add_turn(
                    session_id=chat_req.session_id or "eval-session",
                    role=h["role"],
                    content=h["content"],
                )
            chat_res, chat_timings = service.process_chat(chat_req)

            decision_match = chat_res.decision == expected_decision
            if decision_match:
                correct_decisions += 1

            if expected_decision == "escalate":
                if chat_res.decision == "escalate":
                    escalation_true_positives += 1
                else:
                    escalation_false_negatives += 1
            else:
                if chat_res.decision == "escalate":
                    escalation_false_positives += 1

            if chat_res.decision == "answer":
                total_answers += 1
                cited_ids = {c.record_id for c in chat_res.citations}
                if cited_ids.issubset(set(retrieved_ids)):
                    valid_citations_count += 1

            result_entry["live_llm"] = {
                "decision": chat_res.decision,
                "reason_code": chat_res.reason_code,
                "citations": [c.record_id for c in chat_res.citations],
                "decision_correct": decision_match,
                "total_ms": chat_timings.get("total_ms"),
            }

        detailed_results.append(result_entry)
        status_symbol = "✓" if hit else "✗"
        print(
            f"[{status_symbol}] {cid:<30} | MRR@5: {rr:.2f} | Ret: {ret_ms:5.1f}ms | Rerank: {rerank_ms:5.1f}ms"
        )

    # Aggregate Metrics
    hit_rate = (retrieval_hits_at_5 / total_cases) * 100.0
    mean_mrr = (sum(reciprocal_ranks) / total_cases) if total_cases else 0.0
    avg_ret_ms = sum(retrieval_times_ms) / total_cases if total_cases else 0.0
    avg_rerank_ms = sum(rerank_times_ms) / total_cases if total_cases else 0.0

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"Retrieval Hit@5:         {hit_rate:.1f}% ({retrieval_hits_at_5}/{total_cases})")
    print(f"Retrieval MRR@5:         {mean_mrr:.3f}")
    print(f"Avg Retrieval Latency:   {avg_ret_ms:.2f} ms")
    print(f"Avg Reranking Latency:   {avg_rerank_ms:.2f} ms")
    print(f"Avg Retrieval + Rerank:  {avg_ret_ms + avg_rerank_ms:.2f} ms")

    metrics: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_cases": total_cases,
        "hit_at_5_pct": round(hit_rate, 2),
        "mrr_at_5": round(mean_mrr, 4),
        "avg_retrieval_ms": round(avg_ret_ms, 2),
        "avg_rerank_ms": round(avg_rerank_ms, 2),
    }

    if has_live_llm:
        decision_acc = (correct_decisions / total_cases) * 100.0
        citation_acc = (valid_citations_count / total_answers * 100.0) if total_answers else 100.0
        esc_precision = (
            escalation_true_positives / (escalation_true_positives + escalation_false_positives)
            if (escalation_true_positives + escalation_false_positives)
            else 1.0
        ) * 100.0
        esc_recall = (
            escalation_true_positives / (escalation_true_positives + escalation_false_negatives)
            if (escalation_true_positives + escalation_false_negatives)
            else 1.0
        ) * 100.0

        print(f"Decision Accuracy:       {decision_acc:.1f}% ({correct_decisions}/{total_cases})")
        print(
            f"Citation Validity Rate:  {citation_acc:.1f}% ({valid_citations_count}/{total_answers})"
        )
        print(f"Escalation Precision:    {esc_precision:.1f}%")
        print(f"Escalation Recall:       {esc_recall:.1f}%")

        metrics.update(
            {
                "decision_accuracy_pct": round(decision_acc, 2),
                "citation_validity_pct": round(citation_acc, 2),
                "escalation_precision_pct": round(esc_precision, 2),
                "escalation_recall_pct": round(esc_recall, 2),
            }
        )

    # Save to eval/results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / f"eval_report_{int(time.time())}.json"
    report_data = {
        "metrics": metrics,
        "results": detailed_results,
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nDetailed evaluation report saved to: {report_file}")
    print("=" * 80 + "\n")

    return report_data


def main() -> None:
    try:
        run_evaluation()
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
