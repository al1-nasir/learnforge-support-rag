# LearnForge Customer Support AI Assistant

A production-minded, reliability-first customer support Retrieval-Augmented Generation (RAG) prototype for **LearnForge**, an online learning platform.

This system answers user questions strictly from internal knowledge base documents (`faqs.md`, `policies.md`, `tickets.md`), maintains bounded conversational context, suppresses hallucinations, resolves deliberate contradictions and stale guidance, and safely clarifies or escalates when it cannot answer authoritatively.

![LearnForge System Architecture](docs/system-design.png)

---

## 1. Problem

Customer support in online education requires high factual accuracy, adherence to current policy, and strict safety boundaries. Off-the-shelf vector search bots frequently fail in real-world deployments because:
1. **Relevance is not truth**: A vector search query for "refund policy" will match old tickets or deprecated articles just as strongly as current policy documents.
2. **Knowledge bases contain contradictions**: Companies update terms (e.g. changing a refund period from 7 to 14 days, or retiring laptop downloads in favor of mobile downloads) while historical support logs and archived notes persist.
3. **Conversational ambiguity**: Prompts like *"Cancel my LearnForge"* have multiple distinct operational meanings (disabling subscription renewal vs. requesting a course refund vs. deleting an account). Assuming intent leads to incorrect guidance or customer churn.
4. **Hallucinated capabilities**: LLMs often overstep their capability boundaries, falsely claiming *"I have cancelled your subscription"* or asking for sensitive credentials (passwords, CVVs).

This prototype addresses these challenges through a layered architecture designed for correctness, readability, testability, and restraint.

---

## 2. What the Prototype Does

- **Deterministic Ingestion**: Ingests all 40 natural records from the supplied Markdown files without using an LLM for parsing. Preserves exact record identifiers (`FAQ-01`, `POLICY-02`, `TICKET-08`).
- **Hybrid Retrieval (Dense + BM25)**: Runs simultaneous dense semantic vector search (`BAAI/bge-small-en-v1.5`) and BM25 sparse keyword search (`Qdrant/bm25`) on a local persistent [Qdrant](https://qdrant.tech/) collection.
- **Pure Reciprocal Rank Fusion (RRF)**: Merges dense and sparse candidate rankings in application code using standard $RRF(k=60)$ without summing incomparable raw score distributions.
- **Cross-Encoder Reranking**: Re-scores the fused candidate shortlist using a local cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`), ensuring high-precision top-5 evidence selection.
- **Source Authority & Freshness Layer**: Enforces the hierarchy $\text{Current Policy} \succ \text{FAQ} \succ \text{Historical Ticket}$, explicitly penalizing and annotating deprecated statements.
- **Constrained Decision Triad**: Every customer turn resolves to exactly one of three actions:
  - `ANSWER`: Evidence is current, authoritative, and unambiguous. Grounded answer provided with citations.
  - `CLARIFY`: Customer intent is ambiguous or missing key context. Prompt user for clarification.
  - `ESCALATE`: Evidence is conflicting, missing, account-specific, or requires actions beyond assistant capabilities. Formulates a structured handoff summary for human specialists.
- **Strict Citation Validation**: Verifies that every cited record ID exists in the retrieved evidence set. Unknown or hallucinated citations trigger an immediate escalation fallback.
- **Capability Boundary Guard**: Programmatically intercepts and rejects any message claiming real account modifications (cancellations, refunds, progress resets) or asking for prohibited credentials.

---

## 3. System Architecture

```text
Supplied Corpus (faqs.md, policies.md, tickets.md)
  ↓
Deterministic Parsing & Metadata Enrichment (source_type, authority, dates, deprecation flags)
  ↓
Qdrant Local Vector + Sparse BM25 Index (.storage/qdrant)
  ↓
Customer Query (POST /chat + Bounded History)
  ↓
Hybrid Retrieval (Dense top-8 + BM25 top-8)
  ↓
Application-level Reciprocal Rank Fusion (RRF k=60) → Top 10 Shortlist
  ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) → Top 5 Evidence Items
  ↓
Reliability Layer (Source Precedence, Freshness & Contradiction tagging)
  ↓
Constrained LLM Prompting (Groq gpt-oss-20b Structured Output)
  ↓
Pydantic Validation + Citation Verification + Capability Guard (1 Bounded Retry)
  ↓
Customer Response (ANSWER / CLARIFY / ESCALATE)
```

---

## 4. Quick Start

### Prerequisites
- Python 3.11 or 3.12
- Linux / macOS / WSL

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url>
cd edversity_assignment

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Configure Environment Variables
Copy the template configuration:
```bash
cp .env.example .env
```
*(Optional for live LLM calls)* Add your Groq API key to `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-20b
```
> [!NOTE]
> All unit, integration, and regression tests run **completely offline** with mocked LLM boundaries and require no API key or network access.

### 3. Build the Knowledge Index
Rebuild the local Qdrant collection from raw Markdown sources:
```bash
python -m scripts.build_index
```
Output:
```text
2026-09-18 19:00:31 [INFO] Starting LearnForge index build...
2026-09-18 19:00:31 [INFO] Loaded 40 knowledge records from data
2026-09-18 19:00:31 [INFO] Creating Qdrant collection 'learnforge_support' (dense=384, sparse=bm25)
2026-09-18 19:00:37 [INFO] Indexed 40 records into collection 'learnforge_support'
2026-09-18 19:00:37 [INFO] Index build completed successfully in 5.58s: 40 records indexed.
```

### 4. Run the API
Start the FastAPI development server:
```bash
uvicorn learnforge_support.api:app --reload --port 8000
```
Verify the health check endpoint:
```bash
curl -s http://127.0.0.1:8000/health | jq
```
```json
{
  "status": "healthy",
  "index_ready": true,
  "record_count": 40,
  "llm_model": "openai/gpt-oss-20b",
  "dense_model": "BAAI/bge-small-en-v1.5"
}
```

### 5. Run the Tests & Evaluation
```bash
# Run all 45 unit, integration, and regression tests
pytest -v

# Run Ruff linter
ruff check .

# Run the 25-case evaluation benchmark
python -m scripts.run_eval
```

---

## 5. API Usage Examples

### Example 1: Standard Policy Query (`ANSWER`)
**Request**:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I get a refund for a course I bought 10 days ago?"}'
```
**Response**:
```json
{
  "session_id": "8b52c035-71ec-4e78-a3f1-f8e434407b78",
  "decision": "answer",
  "message": "Yes, LearnForge's standard policy allows eligible course purchases to be refunded within 14 days of purchase, provided the course has not been substantially consumed. Please contact Support with your order number to request a refund.",
  "reason_code": "grounded_answer",
  "citations": [
    {
      "record_id": "POLICY-02",
      "title": "Cancellation and Refund Policy"
    }
  ],
  "handoff_summary": null
}
```

### Example 2: Ambiguous Query (`CLARIFY`)
**Request**:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel my LearnForge"}'
```
**Response**:
```json
{
  "session_id": "9a1bc402-d98c-4fbc-b4fa-4b82944b2569",
  "decision": "clarify",
  "message": "Could you clarify what you would like to cancel? For example, would you like to disable subscription auto-renewal, request a refund for a recent purchase, unenroll from a course, or delete your account?",
  "reason_code": "ambiguous_intent",
  "citations": [],
  "handoff_summary": null
}
```

### Example 3: Capability Boundary / Action Request (`ESCALATE`)
**Request**:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cancel my subscription right now and refund my credit card."}'
```
**Response**:
```json
{
  "session_id": "3c82d41a-7b3f-4e08-963d-24953bf681c2",
  "decision": "escalate",
  "message": "I cannot directly perform account cancellations or issue refunds. You can cancel auto-renewal anytime in Account Settings, or I can connect you with a human support specialist to process your refund request.",
  "reason_code": "account_specific",
  "citations": [],
  "handoff_summary": "User requested immediate subscription cancellation and refund execution."
}
```

### Example 4: Multi-Turn Conversation State
```bash
# Turn 1
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-sess-100", "message": "How do course refunds work?"}'

# Turn 2 (Anaphoric reference resolved using Turn 1 context)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-sess-100", "message": "What if I bought it through Apple?"}'
```

---

## 6. Knowledge & Data Model

The knowledge base consists of 40 natural records extracted without token chunking:
- **15 FAQs** (`FAQ-01` to `FAQ-15`)
- **10 Policies** (`POLICY-01` to `POLICY-10`)
- **15 Tickets** (`TICKET-01` to `TICKET-15`)

Each indexed record carries rich metadata documented in [DATA_SCHEMA.md](DATA_SCHEMA.md):
- `record_id`: Unique identifier preserved from the source Markdown.
- `source_type`: `policy` | `faq` | `ticket`.
- `temporal_status`: `current` (dated policy) | `current_unversioned` (FAQ) | `historical` (ticket).
- `authority_tier`: `policy` (tier 1) > `faq` (tier 2) > `historical_example` (tier 3).
- `contains_deprecated_reference`: Boolean flag detecting explicit mentions of retired policies (e.g. 7-day refund, monthly annual billing, Internet Explorer).
- `content_hash`: SHA-256 hash ensuring deterministic updates and change tracking.

---

## 7. Retrieval Strategy

### Dual First-Stage Retrieval
1. **Dense Semantic Retrieval**: Uses `BAAI/bge-small-en-v1.5` (384-dimensional cosine similarity) to capture conceptual intent, paraphrase variations, and conversational follow-ups (`dense_top_k = 8`).
2. **Sparse Lexical Retrieval**: Uses `Qdrant/bm25` sparse vectors to ensure exact keyword fidelity for specific policy numbers, financial terms, and error names (`sparse_top_k = 8`).

### Reciprocal Rank Fusion (RRF)
Rather than naively adding raw cosine scores (range: $[-1, 1]$) and BM25 scores (range: $[0, \infty)$), candidates are fused via standard Reciprocal Rank Fusion:
$$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + r_m(d)}$$
where $r_m(d)$ is the 1-based rank position of candidate $d$. This yields a robust 10-item candidate shortlist.

### Cross-Encoder Reranking
The 10 fused candidates are scored with `Xenova/ms-marco-MiniLM-L-6-v2`. A cross-encoder performs joint attention across query-document token pairs, achieving higher precision than bi-encoders. The top 5 evidence records are selected for generation.

---

## 8. Freshness & Source Authority

The corpus intentionally contains outdated policies and conflicting statements:
- `POLICY-02` sets the current refund period to **14 days** (effective January 2026), but explicitly notes that older archived documents referenced **7 days**.
- `POLICY-04` recommends **Wi-Fi** for mobile downloads; older articles recommended cellular data.
- `POLICY-09` explicitly notes that **Internet Explorer** support is obsolete.
- `TICKET-03` shows an agent escalating an old 30-day guarantee claim, which conflicts with standard 14-day policy.

### Authority Rules
1. **Hierarchy**: $\text{Current Policy} \succ \text{FAQ} \succ \text{Historical Ticket}$.
2. **Tickets are Examples**: Past tickets illustrate how support handled a specific historical case; they do not establish binding company policy.
3. **Deprecation Handling**: Records with `contains_deprecated_reference: true` are serialized with an explicit warning banner: `[DEPRECATED REFERENCE: Obsolete historical context]`. The LLM is strictly instructed to treat these references as historical context and not as current truth.

---

## 9. Hallucination Controls

1. **Restricted Generation Prompt**: The model is instructed to generate claims regarding LearnForge policies strictly from the retrieved evidence.
2. **Deterministic Citation Whitelisting**: Before any answer is returned to the user, the reliability layer inspects `decision.citations`. If the model cites a record ID that was not provided in the retrieved evidence set (e.g. `POLICY-99`), the response is rejected and converted to `ESCALATE` with `insufficient_evidence`.
3. **Grounded Answer Requirement**: Any `ANSWER` decision must cite at least one supporting document ID. Uncited factual answers are safely escalated.
4. **Structured JSON Validation**: Generation outputs are validated against the `SupportDecision` Pydantic model with a single bounded retry if JSON format or schema validation fails.

---

## 10. Clarification & Escalation Logic

### Clarification (`CLARIFY`)
Triggered when the user request is ambiguous, under-specified, or has multiple valid operational paths (e.g., *"Cancel my LearnForge"* or *"The course is broken"*). The assistant poses a targeted clarifying question to avoid delivering incorrect guidance.

### Escalation (`ESCALATE`)
Triggered when:
- Zero relevant documents are retrieved (`insufficient_evidence`).
- Genuine contradictions exist between authoritative sources (`conflicting_evidence`).
- The inquiry requires inspecting user account records, order IDs, or bank transactions (`account_specific`).
- The user requests a policy exception or manual override (`policy_exception`).
- The user asks the assistant to execute an account modification directly.

Escalations generate a structured `handoff_summary` briefing human agents on the user's issue and relevant context.

---

## 11. Failure Handling Matrix

| Failure Mode | Root Cause | System Response |
| :--- | :--- | :--- |
| **Missing Index** | Database not built | `GET /health` reports `degraded`; `/chat` returns safe escalation instructing admin to run index build. |
| **Empty Retrieval** | Unrecognized query / out-of-scope | System returns `ESCALATE` with `insufficient_evidence`. Zero hallucinations. |
| **LLM Timeout / API Down** | Provider outage / network issue | Bounded 1-retry fallback caught in `llm.py`; returns safe service message with `insufficient_evidence`. |
| **Schema Parse Failure** | Malformed LLM JSON | Automatically retries once with schema error context; falls back to escalation if retry fails. |
| **Hallucinated Citation** | Model invents unretrieved ID | `validate_citations()` detects missing ID; sanitizes to `ESCALATE`. |
| **Capability Overreach** | Model says *"I cancelled your account"* | Regex capability guard detects action claim; safely sanitizes to `ESCALATE` with self-service instructions. |
| **Credential Request** | User or model mentions password/CVV | Security guard blocks password/card logging or requests; escalates securely. |

---

## 12. Evaluation Framework

Evaluation is driven by [eval/golden.jsonl](eval/golden.jsonl), a manually curated benchmark of **25 high-value test cases** representing all critical operational categories:
- Direct FAQ inquiries
- Direct Policy inquiries
- Exact lexical term matching
- Stale / outdated policy traps (14 days vs. 7 days; mobile downloads vs. desktop)
- Policy vs. Ticket contradiction handling
- Ambiguous intent ("Cancel my LearnForge")
- Multi-turn anaphoric follow-ups ("What if I bought it through Apple?")
- Account-specific billing queries (declined authorization hold vs. completed charge)
- Capability boundary enforcement
- Unsupported queries with zero corpus evidence

Run the evaluation suite:
```bash
python -m scripts.run_eval
```

---

## 13. Evaluation Results

Results from the real benchmark run on the complete 25-case golden set:

```text
================================================================================
LEARNFORGE SUPPORT RAG EVALUATION BENCHMARK
Total Test Cases: 25 | Live LLM Enabled: False
================================================================================
[✓] direct-faq-access              | MRR@5: 1.00 | Ret:  17.3ms | Rerank: 538.8ms
[✓] direct-faq-password            | MRR@5: 1.00 | Ret:  20.2ms | Rerank: 502.5ms
[✓] direct-faq-certificate         | MRR@5: 1.00 | Ret:  21.1ms | Rerank: 448.3ms
[✓] direct-faq-devices             | MRR@5: 1.00 | Ret:  23.3ms | Rerank: 509.0ms
[✓] direct-faq-email-change        | MRR@5: 1.00 | Ret:  23.2ms | Rerank: 523.7ms
[✓] direct-policy-subscription     | MRR@5: 1.00 | Ret:  30.2ms | Rerank: 516.5ms
[✓] direct-policy-accessibility    | MRR@5: 1.00 | Ret:  27.0ms | Rerank: 505.5ms
[✓] direct-policy-instructor       | MRR@5: 1.00 | Ret:  26.0ms | Rerank: 685.2ms
[✓] direct-policy-progress         | MRR@5: 1.00 | Ret:  35.0ms | Rerank: 522.7ms
[✓] direct-policy-security         | MRR@5: 1.00 | Ret:  24.2ms | Rerank: 545.4ms
[✓] exact-refund-window            | MRR@5: 1.00 | Ret:  25.6ms | Rerank: 496.3ms
[✓] stale-refund-window            | MRR@5: 0.50 | Ret:  30.0ms | Rerank: 523.5ms
[✓] stale-offline-desktop          | MRR@5: 1.00 | Ret:  24.1ms | Rerank: 478.6ms
[✓] stale-annual-billing           | MRR@5: 1.00 | Ret:  22.1ms | Rerank: 514.9ms
[✓] stale-browser-ie               | MRR@5: 0.50 | Ret:  33.9ms | Rerank: 470.4ms
[✓] policy-vs-ticket-conflict      | MRR@5: 1.00 | Ret:  51.7ms | Rerank: 647.8ms
[✓] annual-subscription-ambiguity  | MRR@5: 1.00 | Ret:  55.2ms | Rerank: 555.5ms
[✓] ambiguous-cancellation         | MRR@5: 0.33 | Ret:  22.3ms | Rerank: 493.9ms
[✓] ambiguous-course-issue         | MRR@5: 1.00 | Ret:  17.9ms | Rerank: 496.0ms
[✓] multi-turn-apple-refund        | MRR@5: 1.00 | Ret:  27.1ms | Rerank: 522.2ms
[✓] multi-turn-progress-fix        | MRR@5: 1.00 | Ret:  29.1ms | Rerank: 473.1ms
[✓] payment-declined-hold          | MRR@5: 1.00 | Ret:  31.4ms | Rerank: 512.8ms
[✓] security-password-request      | MRR@5: 1.00 | Ret:  24.7ms | Rerank: 481.4ms
[✓] capability-boundary-refund     | MRR@5: 1.00 | Ret:  27.6ms | Rerank: 501.5ms
[✓] unknown-question-cafeteria     | MRR@5: 1.00 | Ret:  24.3ms | Rerank: 499.8ms

================================================================================
SUMMARY METRICS
================================================================================
Retrieval Hit@5:         100.0% (25/25)
Retrieval MRR@5:         0.933
Avg Retrieval Latency:   27.78 ms
Avg Reranking Latency:   518.61 ms
Avg Combined Retrieval:  546.39 ms
Detailed Report:         eval/results/eval_report_*.json
================================================================================
```

### Critical Regression Case Audit
All 8 required regression tests in `tests/test_regressions.py` passed:
1. **Refund Freshness**: Current 14-day policy (`POLICY-02`) successfully prioritized over archived 7-day guidance.
2. **Offline Downloads**: Current mobile app download rules beat outdated laptop download references.
3. **Ambiguous Cancellation**: `"Cancel my LearnForge"` produces `CLARIFY` with `ambiguous_intent`.
4. **Annual Subscription Conflict**: Subscription renewal dispute produces `ESCALATE`.
5. **Payment Authorization**: Pending authorization hold identified as temporary bank hold, not completed LearnForge charge.
6. **Capability Boundary**: Prohibits claims of real refund/cancellation execution; safely sanitizes to escalation.
7. **Unknown Question**: Unsupported question routes to `ESCALATE` with `insufficient_evidence` instead of hallucinating.
8. **Multi-turn Context**: Context preserved across turns ("What if I bought it through Apple?" retrieves App Store policy).

---

## 14. Performance & Latency Profile

Measured on modern commodity CPU (x86_64, 4 cores, no GPU acceleration):

| Pipeline Stage | Implementation | Measured Avg Latency | Notes |
| :--- | :--- | :--- | :--- |
| **Query Formulation** | In-memory context extraction | $< 0.1$ ms | Zero external LLM rewrite calls |
| **Hybrid Retrieval** | Qdrant Cosine (dense) + BM25 (sparse) | **27.8 ms** | FastEmbed local ONNX inference |
| **RRF Fusion** | Pure Python application-level fusion | $< 0.5$ ms | $O(N)$ rank merging for top 16 candidates |
| **Reranking** | Cross-Encoder (`MiniLM-L-6-v2`) | **518.6 ms** | Runs only on top 10 shortlist |
| **Reliability Sorting** | Authority & deprecation prior | $< 0.2$ ms | In-memory sort |
| **LLM Generation** | Groq (`openai/gpt-oss-20b`) | ~350–600 ms | Cloud LPUs with JSON mode |
| **Total Request E2E** | Full Pipeline | **~900–1200 ms** | Well within interactive support SLA |

---

## 15. Trade-Offs & Design Rationale

### 1. Hybrid Retrieval (Dense + BM25) vs. Vector-Only
*Decision*: Combined dense embeddings (`BAAI/bge-small-en-v1.5`) with sparse BM25 (`Qdrant/bm25`).
*Rationale*: Support queries contain both semantic intent (*"I can't see the lessons I bought"*) and exact lexical tokens (*"POLICY-02"*, *"7-day"*, *"declined authorization"*, *"Biology Essentials"*). Pure dense search often misses exact numerical criteria, while pure keyword search fails on synonyms. Hybrid provides the highest recall.

### 2. Local ONNX Models vs. Cloud Embedding APIs
*Decision*: Used FastEmbed with small, open ONNX models locally.
*Rationale*: Eliminates external API costs, network latency, and third-party rate limits. Cold startup is fast, models are loaded once into memory, and tests run fully self-contained.

### 3. Application-Level RRF vs. Vector DB Score Summation
*Decision*: Implemented Reciprocal Rank Fusion ($k=60$) in Python rather than summing raw scores.
*Rationale*: Raw dense cosine scores and sparse BM25 scores have incompatible scales and distributions. Normalizing and tuning linear alpha weights ($\alpha \cdot \text{dense} + (1-\alpha) \cdot \text{sparse}$) is fragile across different query types. RRF is scale-invariant, robust, and unit-testable.

### 4. Cross-Encoder Shortlist vs. Full Collection Reranking
*Decision*: Cross-encoder reranks only the top 10 fused candidates, never the full database.
*Rationale*: Full collection reranking requires $O(N)$ expensive transformer evaluations ($40 \times \text{cost}$). Shortlist reranking bounds latency to ~500ms while delivering high-precision ordering.

### 5. Natural Logical Records vs. Fixed Token Chunking
*Decision*: Maintained the natural record boundaries of the supplied corpus (individual FAQs, policy sections, and ticket transcripts).
*Rationale*: Arbitrary sliding token windows (e.g. 256 tokens with 50-token overlap) fracture logical rules across chunk boundaries, separating conditions from conclusions. For a ~40-record corpus, whole-record indexing preserves complete semantic integrity.

### 6. Single Generation Call vs. Multi-Agent Orchestration
*Decision*: Single structured LLM call with a bounded retry rather than LangChain/LangGraph multi-agent teams.
*Rationale*: Multi-agent frameworks add compounding latency, non-deterministic failure modes, and debugging opacity. A single well-constrained call with strict Pydantic schemas satisfies all product requirements in under 1 second.

---

## 16. What I Would Change in Production

If transitioning this prototype to enterprise scale:

1. **Remote Clustered Vector Storage**:
   - Transition from local embedded Qdrant (`QdrantClient(path=...)`) to a managed, distributed Qdrant cluster (`QdrantClient(url=..., api_key=...)`). No code modifications required—only configuration.
2. **Incremental Ingestion & Change Data Capture (CDC)**:
   - Replace full index rebuilds with event-driven indexing triggered by CMS/Help-Center webhooks (e.g. Zendesk or Notion).
   - Implement document versioning with tombstones for deprecated policies.
3. **Durable Distributed Session Store**:
   - Migrate `ConversationStore` from process memory to a distributed cache (e.g. Redis) with TTL expiration, ensuring session continuity across load-balanced API replicas.
4. **Adaptive Query Rewriting for Complex Follow-Ups**:
   - For conversations exceeding 5 turns, deploy a lightweight, quantized query reformulation model to rewrite ambiguous follow-ups before retrieval.
5. **Observability & Guardrails**:
   - Integrate OpenTelemetry for distributed tracing.
   - Deploy automated evaluation monitors in production to flag drifting hallucination rates or sudden spikes in human escalations.

