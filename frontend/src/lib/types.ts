/**
 * LearnForge Support Chat types strictly matching the FastAPI backend schema
 * defined in src/learnforge_support/schemas.py.
 */

export type DecisionType = "answer" | "clarify" | "escalate";

export type ReasonCode =
  | "grounded_answer"
  | "ambiguous_intent"
  | "missing_context"
  | "insufficient_evidence"
  | "conflicting_evidence"
  | "account_specific"
  | "policy_exception"
  | (string & {});

export interface Citation {
  record_id: string;
  title: string;
}

export interface ChatRequest {
  session_id?: string;
  message: string;
}

export interface ChatResponse {
  session_id: string;
  decision: DecisionType;
  message: string;
  reason_code: ReasonCode;
  citations: Citation[];
  handoff_summary?: string | null;
}

export interface SupportMeta {
  decision: DecisionType;
  reasonCode: ReasonCode;
  handoffSummary?: string | null;
  citations?: Citation[];
}

