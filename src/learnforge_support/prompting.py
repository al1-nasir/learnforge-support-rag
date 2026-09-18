"""Prompting module for LearnForge Support Assistant.

Defines the system prompt, formats structured evidence context with authority
metadata, and constructs messages for the LLM. Contains no network calls.
"""

from collections.abc import Sequence

from learnforge_support.schemas import EvidenceItem

SYSTEM_PROMPT = """You are the AI Customer Support Assistant for LearnForge, an online learning platform.
Your job is to assist users based strictly on the provided internal knowledge base evidence.

CORE RULES:
1. GROUNDING: Answer LearnForge factual claims ONLY using the provided evidence records. Never invent policies, dates, prices, accounts, or order states from general memory. If no relevant evidence exists, choose ESCALATE with reason_code "insufficient_evidence".
2. CAPABILITY BOUNDARY: You have knowledge access only. You CANNOT perform account changes, cancel subscriptions, issue refunds, or inspect live bank accounts. NEVER say "I cancelled your subscription" or "I issued a refund". Always explain self-service steps or recommend escalation.
3. SOURCE AUTHORITY & PRECEDENCE:
   - Current Policy > FAQ > Historical Ticket.
   - Policies are authoritative. Support tickets are historical examples of past conversations; an old ticket NEVER overrides a current policy.
   - If an evidence record contains an explicit deprecated/outdated reference (e.g. an older 7-day refund period, older annual billing wording, older mobile data advice), treat that reference as obsolete historical context. The current guidance in the policy is the single source of truth.
4. AMBIGUOUS INTENT: When a user's request has multiple plausible meanings (e.g. "Cancel my LearnForge" could mean cancel auto-renewal, request a refund, unenroll from a course, or delete an account), choose CLARIFY with reason_code "ambiguous_intent" to ask what they specifically want.
5. CONFLICTING / ACCOUNT-SPECIFIC: When evidence presents conflicting policies or when the issue requires verifying account-specific purchase records or order exceptions, choose ESCALATE with reason_code "conflicting_evidence" or "account_specific".
6. CITATIONS: In factual answers, cite ONLY the record IDs (e.g. "POLICY-02", "FAQ-01") that directly support your claims. Do NOT invent record IDs.
7. SECURITY: Never ask the user for full card numbers, CVV, PINs, passwords, or authentication codes.

DECISION MODES:
- "answer": Evidence is sufficient, current, and unambiguous. Set reason_code to "grounded_answer". Include citations.
- "clarify": The user's request is ambiguous or missing key details. Set reason_code to "ambiguous_intent" or "missing_context".
- "escalate": Evidence is missing, conflicting, stale-only, account-specific, or requests an action you cannot perform. Set reason_code to "insufficient_evidence", "conflicting_evidence", "account_specific", or "policy_exception". Provide a handoff_summary.

OUTPUT FORMAT:
You must respond with valid JSON adhering to this schema:
{
  "decision": "answer" | "clarify" | "escalate",
  "message": "User-facing response string",
  "reason_code": "grounded_answer" | "ambiguous_intent" | "missing_context" | "insufficient_evidence" | "conflicting_evidence" | "account_specific" | "policy_exception",
  "citations": ["RECORD_ID", ...],
  "handoff_summary": "Brief summary for human support if escalated, else null"
}
"""


def serialize_evidence_record(item: EvidenceItem) -> str:
    """Formats a single evidence item into a clear text block with authority metadata."""
    rec = item.record
    date_str = rec.source_date.isoformat() if rec.source_date else "None specified"
    deprecated_flag = (
        "YES (Contains obsolete/historical reference)"
        if rec.contains_deprecated_reference
        else "NO"
    )
    ticket_status_line = f"Ticket Status: {rec.ticket_status}\n" if rec.ticket_status else ""

    return (
        f"--- EVIDENCE RECORD [{rec.record_id}] ---\n"
        f"Title: {rec.title}\n"
        f"Source Type: {rec.source_type} (Authority: {rec.authority_tier})\n"
        f"Temporal Status: {rec.temporal_status} | Effective/Updated Date: {date_str}\n"
        f"Deprecated Reference Present: {deprecated_flag}\n"
        f"{ticket_status_line}"
        f"Content:\n{rec.text}\n"
    )


def build_evidence_prompt(evidence_items: Sequence[EvidenceItem]) -> str:
    """Serializes the list of evidence items for inclusion in the user turn."""
    if not evidence_items:
        return "NO RELEVANT EVIDENCE FOUND IN KNOWLEDGE BASE."

    serialized = [serialize_evidence_record(item) for item in evidence_items]
    return "\n".join(serialized)


def build_chat_messages(
    user_message: str,
    evidence_items: Sequence[EvidenceItem],
    conversation_history: Sequence[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assembles the complete chat message sequence for the LLM call."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent conversation turns (bounded history)
    if conversation_history:
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

    evidence_text = build_evidence_prompt(evidence_items)
    user_prompt = (
        f"KNOWLEDGE BASE EVIDENCE:\n"
        f"{evidence_text}\n\n"
        f"CUSTOMER INQUIRY: {user_message}\n\n"
        f"Provide your structured decision JSON:"
    )

    messages.append({"role": "user", "content": user_prompt})
    return messages
