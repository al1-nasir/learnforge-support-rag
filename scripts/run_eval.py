"""Evaluation runner for LearnForge AI Support Assistant.

Loads eval/golden.jsonl, benchmarks hybrid retrieval and reranking,
computes Hit@5, MRR@5, latency metrics, and when enabled, runs the full
end-to-end pipeline with the live Groq LLM to verify decision accuracy,
citation validity, source authority, hallucination suppression, and capability bounds.

Usage:
    python -m scripts.run_eval          # Auto-detects GROQ_API_KEY (live if present)
    python -m scripts.run_eval --live   # Forces live LLM evaluation
    python -m scripts.run_eval --offline# Runs retrieval-only evaluation
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from learnforge_support.config import get_settings
from learnforge_support.logging_utils import setup_logger
from learnforge_support.observability import (
    flush_observability,
    initialize_observability,
    record_eval_score,
)
from learnforge_support.reliability import (
    check_citation_completeness,
    violates_capability_boundary,
)
from learnforge_support.reranking import rerank_candidates
from learnforge_support.retrieval import build_retrieval_query, hybrid_retrieve
from learnforge_support.schemas import ChatRequest, SupportDecision
from learnforge_support.service import SupportService

logger = setup_logger()


def run_evaluation(
    dataset_path: Path | None = None,
    force_live: bool | None = None,
    dataset_name: str = "golden",
) -> dict[str, Any]:
    """Runs evaluation benchmark against the specified dataset JSONL file."""
    settings = get_settings()
    initialize_observability(settings)
    service = SupportService(settings=settings)

    if not service.is_index_ready():
        raise RuntimeError("Qdrant index is not built. Run 'python -m scripts.build_index' first.")

    target_file = dataset_path or Path("eval/golden.jsonl")
    if not target_file.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {target_file}")

    cases: list[dict[str, Any]] = []
    with open(target_file, encoding="utf-8") as f:
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

    # Determine live LLM mode
    if force_live is True:
        if not settings.groq_api_key.strip():
            raise ValueError("Live LLM evaluation requested (--live), but GROQ_API_KEY is not set.")
        has_live_llm = True
    elif force_live is False:
        has_live_llm = False
    else:
        has_live_llm = bool(settings.groq_api_key.strip())

    # Live evaluation tracking accumulators
    correct_decisions = 0
    valid_citations_count = 0
    complete_citations_count = 0
    total_answers = 0
    expected_source_hits = 0
    cases_with_expected_sources = 0
    unsupported_claims_count = 0

    stale_conflict_correct = 0
    stale_conflict_total = 0

    capability_correct = 0
    capability_total = 0

    escalation_true_positives = 0
    escalation_false_positives = 0
    escalation_false_negatives = 0

    llm_times_ms: list[float] = []
    total_e2e_times_ms: list[float] = []

    print("\n" + "=" * 80)
    print(f"LEARNFORGE SUPPORT RAG EVALUATION BENCHMARK — {dataset_name.upper()} SET")
    print(f"Dataset File: {target_file} | Total Test Cases: {total_cases} | Live LLM Enabled: {has_live_llm}")
    if has_live_llm:
        print(f"LLM Model: {settings.llm_model}")
    print("=" * 80 + "\n")

    for case in cases:
        cid = case["id"]
        query = case["query"]
        history = case.get("history", [])
        expected_decision = case.get("expected_decision")
        expected_sources = set(case.get("expected_sources", []))
        category = case.get("category", "general")

        # Step A: Contextual query building
        augmented_query = build_retrieval_query(query, history)

        # Step B: First-stage hybrid retrieval
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

        # Step C: Cross-encoder reranking
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
            "category": category,
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

        # Step D: If live LLM is configured, test end-to-end generation
        if has_live_llm:
            case_session_id = f"eval-session-{cid}"
            service.conversation_store.clear(case_session_id)
            for h in history:
                service.conversation_store.add_turn(
                    session_id=case_session_id,
                    role=h["role"],
                    content=h["content"],
                )

            chat_req = ChatRequest(session_id=case_session_id, message=query)
            chat_tags = ["evaluation", dataset_name]
            chat_meta = {
                "eval_case_id": cid,
                "eval_dataset": dataset_name,
                "eval_category": category,
            }
            chat_res, chat_timings = service.process_chat(
                chat_req,
                tags=chat_tags,
                metadata=chat_meta,
                request_id=f"eval-{dataset_name}-{cid}",
            )
            time.sleep(1.0)

            llm_ms = chat_timings.get("llm_ms", 0.0)
            total_ms = chat_timings.get("total_ms", 0.0)
            llm_times_ms.append(llm_ms)
            total_e2e_times_ms.append(total_ms)

            # 1. Decision accuracy
            decision_match = chat_res.decision == expected_decision
            if decision_match:
                correct_decisions += 1

            # 2. Escalation metrics
            if expected_decision == "escalate":
                if chat_res.decision == "escalate":
                    escalation_true_positives += 1
                else:
                    escalation_false_negatives += 1
            else:
                if chat_res.decision == "escalate":
                    escalation_false_positives += 1

            # 3. Citation validity & Hallucination check
            cited_ids = {c.record_id for c in chat_res.citations}
            citations_valid = False
            if chat_res.decision == "answer":
                total_answers += 1
                # Whitelist check: all cited IDs must be in top_5 retrieved
                if cited_ids and cited_ids.issubset(set(retrieved_ids)):
                    valid_citations_count += 1
                    citations_valid = True
                else:
                    unsupported_claims_count += 1
            elif chat_res.decision in ("clarify", "escalate"):
                # Non-answers should have valid citations (empty or verified)
                if cited_ids.issubset(set(retrieved_ids)):
                    citations_valid = True
                else:
                    unsupported_claims_count += 1

            # 3b. Citation completeness check
            decision_obj = SupportDecision(
                decision=chat_res.decision,
                message=chat_res.message,
                reason_code=chat_res.reason_code,
                citations=[c.record_id for c in chat_res.citations],
                handoff_summary=chat_res.handoff_summary,
            )
            citation_complete = check_citation_completeness(decision_obj, set(retrieved_ids))
            if citation_complete:
                complete_citations_count += 1

            # For unknown questions, answering is a hallucination
            if category == "unknown" and chat_res.decision == "answer":
                unsupported_claims_count += 1

            # 4. Expected source accuracy
            if expected_sources:
                cases_with_expected_sources += 1
                if chat_res.decision == "answer":
                    if bool(cited_ids.intersection(expected_sources)):
                        expected_source_hits += 1
                else:
                    # For clarify or escalate, check if retrieved evidence contained expected source
                    if bool(set(retrieved_ids).intersection(expected_sources)):
                        expected_source_hits += 1

            # 5. Stale / Conflict handling accuracy
            if category in ("stale_data", "conflict"):
                stale_conflict_total += 1
                # Did it pick the correct decision and avoid citing forbidden/deprecated tickets over current policy?
                if decision_match:
                    stale_conflict_correct += 1

            # 6. Capability boundary accuracy
            if category == "capability_boundary":
                capability_total += 1
                claims_action = violates_capability_boundary(chat_res.message)
                if chat_res.decision == "escalate" and not claims_action:
                    capability_correct += 1

            result_entry["live_llm"] = {
                "decision": chat_res.decision,
                "reason_code": chat_res.reason_code,
                "citations": [c.record_id for c in chat_res.citations],
                "message": chat_res.message,
                "handoff_summary": chat_res.handoff_summary,
                "decision_correct": decision_match,
                "citations_valid": citations_valid,
                "citation_complete": citation_complete,
                "timings_ms": chat_timings,
            }

            trace_id = chat_timings.get("trace_id")
            if trace_id:
                record_eval_score(
                    trace_id=trace_id,
                    name="decision_accuracy",
                    value=1 if decision_match else 0,
                    comment=f"expected: {expected_decision}, actual: {chat_res.decision}",
                )
                record_eval_score(
                    trace_id=trace_id,
                    name="citation_validity",
                    value=1 if citations_valid else 0,
                )
                record_eval_score(
                    trace_id=trace_id,
                    name="citation_completeness",
                    value=1 if citation_complete else 0,
                )
                record_eval_score(
                    trace_id=trace_id,
                    name="retrieval_hit",
                    value=1 if hit else 0,
                )

            status_symbol = "✓" if decision_match else "✗"
            print(
                f"[{status_symbol}] {cid:<30} | Expected: {expected_decision:<8} | "
                f"Actual: {chat_res.decision:<8} ({chat_res.reason_code}) | "
                f"E2E: {total_ms:5.1f}ms (LLM: {llm_ms:5.1f}ms)"
            )
        else:
            status_symbol = "✓" if hit else "✗"
            print(
                f"[{status_symbol}] {cid:<30} | MRR@5: {rr:.2f} | Ret: {ret_ms:5.1f}ms | Rerank: {rerank_ms:5.1f}ms"
            )

        detailed_results.append(result_entry)

    # Aggregate Retrieval Metrics
    hit_rate = (retrieval_hits_at_5 / total_cases) * 100.0
    mean_mrr = (sum(reciprocal_ranks) / total_cases) if total_cases else 0.0
    avg_ret_ms = sum(retrieval_times_ms) / total_cases if total_cases else 0.0
    avg_rerank_ms = sum(rerank_times_ms) / total_cases if total_cases else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION BENCHMARK SUMMARY METRICS")
    print("=" * 80)
    print(
        f"Retrieval Hit@5:                   {hit_rate:.1f}% ({retrieval_hits_at_5}/{total_cases})"
    )
    print(f"Retrieval MRR@5:                   {mean_mrr:.3f}")
    print(f"Avg Retrieval Latency:             {avg_ret_ms:.2f} ms")
    print(f"Avg Reranking Latency:             {avg_rerank_ms:.2f} ms")
    print(f"Avg Retrieval + Reranking:         {avg_ret_ms + avg_rerank_ms:.2f} ms")

    metrics: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_cases": total_cases,
        "live_llm_enabled": has_live_llm,
        "llm_model": settings.llm_model if has_live_llm else None,
        "retrieval": {
            "hit_at_5_pct": round(hit_rate, 2),
            "mrr_at_5": round(mean_mrr, 4),
            "avg_retrieval_ms": round(avg_ret_ms, 2),
            "avg_rerank_ms": round(avg_rerank_ms, 2),
            "avg_combined_ms": round(avg_ret_ms + avg_rerank_ms, 2),
        },
    }

    if has_live_llm:
        decision_acc = (correct_decisions / total_cases) * 100.0
        citation_acc = (valid_citations_count / total_answers * 100.0) if total_answers else 100.0
        expected_src_acc = (
            (expected_source_hits / cases_with_expected_sources * 100.0)
            if cases_with_expected_sources
            else 100.0
        )
        hallucination_rate = (unsupported_claims_count / total_cases) * 100.0

        stale_conflict_acc = (
            (stale_conflict_correct / stale_conflict_total * 100.0)
            if stale_conflict_total
            else 100.0
        )
        capability_acc = (
            (capability_correct / capability_total * 100.0) if capability_total else 100.0
        )

        esc_denom_prec = escalation_true_positives + escalation_false_positives
        esc_precision = (
            (escalation_true_positives / esc_denom_prec * 100.0) if esc_denom_prec else 100.0
        )

        esc_denom_rec = escalation_true_positives + escalation_false_negatives
        esc_recall = (escalation_true_positives / esc_denom_rec * 100.0) if esc_denom_rec else 100.0

        avg_llm_ms = sum(llm_times_ms) / total_cases if total_cases else 0.0
        avg_total_ms = sum(total_e2e_times_ms) / total_cases if total_cases else 0.0

        citation_completeness_acc = (
            (complete_citations_count / total_cases * 100.0) if total_cases else 100.0
        )

        print(
            f"Decision Accuracy:                 {decision_acc:.1f}% ({correct_decisions}/{total_cases})"
        )
        print(
            f"Citation Validity Rate:            {citation_acc:.1f}% ({valid_citations_count}/{total_answers})"
        )
        print(
            f"Mentioned-Source Citation Completeness: {citation_completeness_acc:.1f}% ({complete_citations_count}/{total_cases})"
        )
        print(
            f"Expected Source Accuracy:          {expected_src_acc:.1f}% ({expected_source_hits}/{cases_with_expected_sources})"
        )
        print(
            f"Unsupported Claim / Hallucination: {hallucination_rate:.1f}% ({unsupported_claims_count}/{total_cases})"
        )
        print(
            f"Stale/Conflict Handling Accuracy:  {stale_conflict_acc:.1f}% ({stale_conflict_correct}/{stale_conflict_total})"
        )
        print(
            f"Capability Boundary Accuracy:      {capability_acc:.1f}% ({capability_correct}/{capability_total})"
        )
        print(f"Escalation Precision:              {esc_precision:.1f}%")
        print(f"Escalation Recall:                 {esc_recall:.1f}%")
        print(f"Avg LLM Latency:                   {avg_llm_ms:.2f} ms")
        print(f"Avg End-to-End Latency:            {avg_total_ms:.2f} ms")

        metrics["system_behavior"] = {
            "decision_accuracy_pct": round(decision_acc, 2),
            "citation_validity_pct": round(citation_acc, 2),
            "citation_completeness_pct": round(citation_completeness_acc, 2),
            "expected_source_accuracy_pct": round(expected_src_acc, 2),
            "unsupported_claim_rate_pct": round(hallucination_rate, 2),
            "stale_conflict_accuracy_pct": round(stale_conflict_acc, 2),
            "capability_accuracy_pct": round(capability_acc, 2),
            "escalation_precision_pct": round(esc_precision, 2),
            "escalation_recall_pct": round(esc_recall, 2),
            "avg_llm_ms": round(avg_llm_ms, 2),
            "avg_total_ms": round(avg_total_ms, 2),
        }

    # Save to eval/results with separate prefix and dataset name
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "live" if has_live_llm else "offline"
    report_file = results_dir / f"{prefix}_{dataset_name}_report_{int(time.time())}.json"
    report_data = {
        "dataset": dataset_name,
        "metrics": metrics,
        "results": detailed_results,
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nDetailed {dataset_name} evaluation report saved to: {report_file}")
    print("=" * 80 + "\n")

    flush_observability()
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LearnForge Support Assistant evaluation benchmark."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live LLM evaluation using configured Groq API key.",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Run retrieval-only offline evaluation."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--golden",
        action="store_true",
        help="Run evaluation on eval/golden.jsonl only.",
    )
    dataset_group.add_argument(
        "--holdout",
        action="store_true",
        help="Run evaluation on eval/holdout.jsonl only.",
    )
    dataset_group.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation on both golden and holdout sets separately.",
    )

    args = parser.parse_args()
    force_live: bool | None = None
    if args.live:
        force_live = True
    elif args.offline:
        force_live = False

    try:
        if args.golden:
            run_evaluation(
                dataset_path=Path("eval/golden.jsonl"),
                force_live=force_live,
                dataset_name="golden",
            )
        elif args.holdout:
            run_evaluation(
                dataset_path=Path("eval/holdout.jsonl"),
                force_live=force_live,
                dataset_name="holdout",
            )
        else:
            # Default: run both golden and holdout datasets separately
            print("\n" + "#" * 80)
            print("### RUNNING BENCHMARK SUITE 1 OF 2: ORIGINAL GOLDEN BENCHMARK SET ###")
            print("#" * 80)
            run_evaluation(
                dataset_path=Path("eval/golden.jsonl"),
                force_live=force_live,
                dataset_name="golden",
            )

            print("\n" + "#" * 80)
            print("### RUNNING BENCHMARK SUITE 2 OF 2: UNTOUCHED HOLDOUT BENCHMARK SET ###")
            print("#" * 80)
            run_evaluation(
                dataset_path=Path("eval/holdout.jsonl"),
                force_live=force_live,
                dataset_name="holdout",
            )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        flush_observability()


if __name__ == "__main__":
    main()

