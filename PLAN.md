# PLAN.md
# LearnForge Applied AI/LLM Engineer Take-Home

## 0. Purpose

Build a small, production-minded customer-support RAG prototype for LearnForge using the supplied:

- `faqs.md`
- `policies.md`
- `tickets.md`

The system must answer from the supplied knowledge base, handle multi-turn conversations, reduce hallucination, recognize stale/conflicting information, and escalate when it cannot answer safely.

This is a take-home prototype, not a full SaaS product. The architecture should demonstrate production judgment without adding infrastructure that does not improve the evaluation criteria.

The system should be easy for another engineer to understand, run, test, and extend.

---

## 1. What the reviewers are evaluating

The implementation must visibly address four things:

### 1.1 Architecture soundness

The design must address:

- retrieval quality;
- scale;
- freshness;
- hallucination reduction;
- multi-turn conversations;
- ambiguity;
- escalation.

### 1.2 Reasoning and trade-offs

Every important architectural choice should have a clear reason.

The project should not claim there is one universally correct RAG architecture.

The README must explain:

- what was chosen;
- why it was chosen;
- what alternatives were considered;
- what would change with more time, data, or budget.

### 1.3 Implementation/design consistency

The prototype must actually implement the architecture shown in the system design.

Do not put components in the architecture diagram that do not exist in the repository.

Do not implement major components that are absent from the architecture and README.

### 1.4 Communication

A new engineer should be able to:

1. understand the system;
2. install it;
3. build the index;
4. start the API;
5. send a query;
6. run tests;
7. run evaluation;

without reverse-engineering the codebase.

---

## 2. Core engineering principle

> Retrieval finds relevant evidence. Reliability logic decides whether that evidence is trustworthy enough to answer.

The supplied data intentionally contains stale guidance, contradictory information, and ambiguous support conversations.

A plain vector-search chatbot is therefore insufficient.

The system will distinguish between:

- relevance;
- source authority;
- freshness;
- evidence sufficiency;
- user intent.

---

## 3. Product behavior

Every user turn results in exactly one of three decisions:

### `ANSWER`

Use when:

- useful evidence was retrieved;
- evidence is sufficiently current and authoritative;
- the question is clear enough to answer;
- no unresolved material conflict remains.

### `CLARIFY`

Use when:

- the user request has multiple plausible meanings;
- a required detail is missing;
- answering immediately could lead to incorrect support guidance.

Example:

`Cancel my LearnForge.`

This could mean:

- stop subscription renewal;
- request a refund;
- remove a course;
- delete the account.

The assistant should clarify rather than assume.

### `ESCALATE`

Use when:

- evidence is insufficient;
- strong sources materially conflict;
- the request depends on account-specific data;
- purchase-specific terms must be verified;
- the user asks the assistant to perform an action it cannot perform;
- a policy exception requires human review.

The system must prefer an honest escalation over a confident unsupported answer.

---

## 4. Non-goals

Do not build:

- authentication;
- real account access;
- real refund/cancellation tools;
- a ticketing integration;
- Kafka;
- Kubernetes;
- microservices;
- autonomous agents;
- fine-tuning;
- a complex frontend;
- distributed caches;
- a production deployment.

A minimal demo UI may be added only after the core system, tests, evaluation, README, data schema, and diagram are complete.

---

## 5. Technology choices

### Runtime

- Python 3.11+
- FastAPI
- Pydantic v2
- `pydantic-settings`
- standard Python logging

### Retrieval

- Qdrant using local persistent mode for the prototype
- `qdrant-client`
- FastEmbed for local embedding/reranking inference

### Dense embedding

Default:

`BAAI/bge-small-en-v1.5`

Reasons:

- small;
- fast;
- 384-dimensional;
- local;
- no embedding API cost;
- sufficient for a ~40-record English support corpus.

The model must be configurable.

### Sparse retrieval

Default:

`Qdrant/bm25`

Reasons:

- exact terminology matters in support questions;
- useful for refund windows, product names, billing terms, and policy wording;
- complements semantic retrieval.

### Hybrid fusion

Use Reciprocal Rank Fusion (RRF) in application code.

Why application-level RRF:

- transparent;
- easy to test;
- avoids mixing incomparable dense and BM25 raw score ranges;
- straightforward for a reviewer to understand.

### Reranker

Default:

`Xenova/ms-marco-MiniLM-L-6-v2`

Use FastEmbed `TextCrossEncoder`.

Reasons:

- lightweight;
- local;
- inexpensive on a small candidate set;
- improves ordering after broad first-stage retrieval.

Reranking must only run on the shortlist, never the full collection.

### LLM

Default provider:

Groq.

Default model:

`openai/gpt-oss-20b`

The model name must be configurable by environment variable.

Use Structured Outputs when supported.

The internal response must be validated by Pydantic even when the provider claims schema adherence.

### Framework policy

Do not introduce LangChain/LlamaIndex unless a concrete requirement cannot be implemented cleanly without them.

Direct SDK usage is preferred because:

- the corpus is tiny;
- the pipeline is simple;
- behavior remains explicit;
- reviewers can trace every step;
- debugging is easier.

---

## 6. Repository structure

```text
learnforge-support/
├── README.md
├── PLAN.md
├── AGENT.md
├── DATA_SCHEMA.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── data/
│   ├── faqs.md
│   ├── policies.md
│   └── tickets.md
│
├── docs/
│   ├── system-design.excalidraw
│   └── system-design.png
│
├── src/
│   └── learnforge_support/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── schemas.py
│       ├── ingestion.py
│       ├── indexing.py
│       ├── retrieval.py
│       ├── reranking.py
│       ├── reliability.py
│       ├── prompting.py
│       ├── llm.py
│       ├── conversation.py
│       ├── service.py
│       └── logging_utils.py
│
├── scripts/
│   ├── build_index.py
│   └── run_eval.py
│
├── eval/
│   ├── golden.jsonl
│   └── results/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_reliability.py
│   ├── test_conversation.py
│   └── test_api.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

Do not add modules merely to make the repository look sophisticated.

If two modules remain tiny and inseparable, combine them.

---

## 7. Knowledge-record design

The corpus already has useful semantic boundaries.

Do not blindly split every file into arbitrary fixed token chunks.

Primary logical records:

- one FAQ entry;
- one policy entry;
- one ticket transcript.

The parser should preserve the original record ID such as:

- `FAQ-02`
- `POLICY-02`
- `TICKET-08`

These IDs become human-readable citations.

### Initial record model

```python
KnowledgeRecord:
    record_id: str
    source_type: Literal["faq", "policy", "ticket"]
    title: str
    text: str

    source_date: date | None
    temporal_status: Literal[
        "current",
        "current_unversioned",
        "historical"
    ]

    authority_tier: Literal[
        "policy",
        "faq",
        "historical_example"
    ]

    contains_deprecated_reference: bool

    ticket_status: str | None

    content_hash: str
```

The final implemented schema must be documented in `DATA_SCHEMA.md`.

---

## 8. Authority and freshness rules

These rules are central to the assignment.

### Source precedence

When sources conflict:

```text
current policy
    >
current/current-unversioned FAQ
    >
historical support ticket
```

A support ticket is evidence of how an old case was handled.

It is not automatically current policy.

### Policy dates

Parse explicit markers such as:

- `Effective date`
- `Updated`
- `Last updated`
- `Last reviewed`
- `Reviewed`

### Tickets

All supplied support tickets are treated as historical examples.

### FAQs

FAQs without an explicit date are:

`current_unversioned`

They may support a current policy but should not silently override a newer explicit policy.

### Deprecated statements

The supplied corpus contains phrases explaining that older guidance is obsolete.

Preserve the whole record, but annotate records containing explicit deprecated/outdated references.

Do not remove the old statement from the document because its presence is useful for evaluating conflict handling.

---

## 9. Ingestion pipeline

Ingestion must be deterministic.

Do not use an LLM to parse these Markdown files.

### Ingestion steps

```text
Markdown files
    ↓
source-specific parser
    ↓
logical records
    ↓
metadata extraction
    ↓
content hash
    ↓
validation
    ↓
dense + sparse indexing
```

### Validation

Before indexing, verify:

- unique `record_id`;
- non-empty text;
- valid `source_type`;
- expected source count where appropriate;
- parseable metadata;
- no duplicated records.

### Idempotence

Running:

```bash
python -m scripts.build_index
```

multiple times must not create duplicate records.

A clean rebuild is acceptable for this small corpus.

---

## 10. Indexing

Persist the local Qdrant database under a configurable path such as:

```text
.storage/qdrant/
```

This directory must be gitignored.

Store:

- dense vector;
- sparse BM25 representation;
- payload metadata.

Do not commit generated vector-store files to Git.

The repository should rebuild the index from the supplied Markdown corpus.

---

## 11. Retrieval

### Input

The retrieval query should contain:

- the current user message;
- only the recent conversation context needed to resolve follow-ups.

Do not send the complete conversation forever.

### Conversation context

For the prototype, an in-memory conversation store is sufficient.

Keep a bounded number of recent turns.

Recommended:

- last 6 conversational turns maximum;
- prioritize user messages for retrieval context.

No database is required for session memory.

### First-stage retrieval

Run both:

```text
dense top_k = 8
BM25 top_k = 8
```

Values must be configurable.

### RRF

Fuse both rankings using Reciprocal Rank Fusion.

Recommended constant:

```text
RRF_K = 60
```

Do not add cosine and BM25 scores directly.

### Candidate set

Keep approximately:

```text
8–10 fused candidates
```

for reranking.

---

## 12. Reranking

Run the cross-encoder only on the fused shortlist.

Default:

`Xenova/ms-marco-MiniLM-L-6-v2`

After reranking, keep approximately:

```text
top 5 evidence records
```

for the reliability/generation stage.

Capture reranking latency separately.

---

## 13. Reliability layer

Do not represent confidence as a fake calibrated probability.

Use explicit evidence signals.

Each candidate should carry:

```python
EvidenceItem:
    record: KnowledgeRecord
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    reranker_score: float
```

The reliability layer should derive:

```python
EvidenceContext:
    items: list[EvidenceItem]
    has_current_policy: bool
    stale_only: bool
    contains_deprecated_reference: bool
    source_types: set[str]
```

### Evidence ordering

Relevance remains important, but source metadata must be visible to the LLM.

Current authoritative sources should be presented before historical examples when both address the same issue.

Do not silently discard lower-authority sources; they may expose conflicts.

---

## 14. Conflict handling

A fully generic semantic contradiction engine is out of scope.

The prototype should instead use a layered strategy:

### Layer 1 — metadata

The system knows:

- source authority;
- source recency;
- historical/current status;
- explicit deprecated-reference flags.

### Layer 2 — evidence presentation

Evidence passed to the LLM includes:

- source ID;
- source type;
- date/status;
- authority;
- text.

### Layer 3 — decision instruction

The LLM is explicitly instructed:

- current policy overrides old ticket behavior;
- deprecated statements are historical context;
- unresolved conflicts between strong/current sources require escalation;
- purchase/account-specific exceptions require escalation.

### Layer 4 — evaluation

Known contradiction cases are included as regression tests.

Do not claim the system solves all natural-language contradiction detection.

State this limitation honestly in the trade-offs section.

---

## 15. LLM contract

The LLM must not produce free-form uncontrolled output internally.

Use a structured model such as:

```python
class SupportDecision(BaseModel):
    decision: Literal["answer", "clarify", "escalate"]
    message: str

    reason_code: Literal[
        "grounded_answer",
        "ambiguous_intent",
        "missing_context",
        "insufficient_evidence",
        "conflicting_evidence",
        "account_specific",
        "policy_exception",
    ]

    citations: list[str]

    handoff_summary: str | None
```

If strict structured outputs are supported by the configured model, use them.

Always run the result through Pydantic validation.

---

## 16. Prompt rules

The system prompt must include these rules:

1. Answer LearnForge-specific factual claims only from supplied evidence.
2. Never invent a policy, date, price, order state, or account state.
3. Never claim that an account action was executed.
4. Historical tickets are examples, not authoritative policy.
5. Prefer current policy over FAQ and tickets when they conflict.
6. Treat explicitly deprecated statements as historical context.
7. Ask a clarification question when intent is genuinely ambiguous.
8. Escalate when evidence is insufficient or unresolved.
9. Cite the record IDs that directly support the answer.
10. Do not request passwords, CVV, PINs, full card numbers, authentication codes, or other prohibited secrets.

The prompt should be concise enough that an engineer can understand why every instruction exists.

---

## 17. Citation validation

After the model responds:

- every citation must correspond to an evidence record passed to the model;
- unknown citation IDs make the response invalid;
- factual `ANSWER` responses should contain at least one valid citation.

On invalid structured output:

1. retry once with the validation failure;
2. if it still fails, return a safe error/escalation response.

Do not create an infinite retry loop.

---

## 18. Capability boundary

The assistant has knowledge access only.

It cannot:

- cancel a subscription;
- issue a refund;
- inspect a real order;
- restore course access;
- change an email;
- edit course progress;
- investigate an actual card charge.

A response such as:

`I cancelled your subscription.`

is incorrect.

The system should say what the user can do or escalate to support.

Capability-boundary behavior must have regression tests.

---

## 19. API

### `POST /chat`

Request:

```json
{
  "session_id": "optional-session-id",
  "message": "Can I get a refund?"
}
```

Response:

```json
{
  "session_id": "generated-or-existing-id",
  "decision": "answer",
  "message": "...",
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

### `GET /health`

Return:

- API status;
- index availability;
- configured LLM provider/model.

Do not expose secrets.

---

## 20. Failure handling

### Missing index

Return a clear startup/runtime error explaining how to build the index.

### LLM timeout/provider error

- bounded retry;
- log failure category;
- return a safe service-unavailable response.

Do not fabricate an answer when the provider fails.

### Empty retrieval

Return:

`ESCALATE / insufficient_evidence`

### Stale-only retrieval

Do not present historical guidance as current fact.

Escalate unless the question is explicitly asking about historical behavior.

### Ambiguous user intent

Return:

`CLARIFY / ambiguous_intent`

### Invalid model output

- validate;
- retry once;
- safe fallback.

---

## 21. Logging

Use structured, readable logging.

Each request should have a generated `request_id`.

Log:

- request ID;
- session ID;
- selected record IDs;
- dense retrieval latency;
- sparse retrieval latency;
- reranking latency;
- LLM latency;
- final decision;
- reason code;
- errors.

Do not log:

- API keys;
- passwords;
- complete payment information;
- unnecessary user-sensitive content.

The implementation may use the standard library `logging` module.

No observability platform is required.

---

## 22. Performance

This assignment does not require internet-scale infrastructure.

Still measure latency.

Capture:

```text
retrieval_ms
rerank_ms
llm_ms
total_ms
```

Report measured values in the README after implementation.

Do not invent latency numbers.

### Speed principles

- local embeddings;
- local BM25;
- rerank only a small shortlist;
- one normal LLM generation call;
- no unnecessary query-rewrite LLM call;
- bounded conversation history.

---

## 23. Scale story

The prototype uses local Qdrant because the corpus is tiny.

The architecture should allow:

```text
local Qdrant
    →
remote Qdrant
```

through configuration rather than a rewrite.

For a larger production corpus:

- incremental indexing would replace full rebuilds;
- document-version/event ingestion would drive freshness;
- remote vector storage would replace local persistence;
- caching could be added after measuring repeated-query patterns;
- conversation state would move from process memory to a durable/shared store.

These belong in the README trade-offs section, not in the take-home implementation.

---

## 24. Freshness story

Prototype:

- source dates/status stored in metadata;
- explicit stale/deprecated references flagged;
- deterministic index rebuild;
- source precedence in generation.

Production extension:

- document version IDs;
- ingestion timestamps;
- change-event/webhook indexing;
- tombstones for removed content;
- source lifecycle/status;
- stale-index monitoring.

Again, document this without implementing unnecessary infrastructure.

---

## 25. Evaluation

Create a manually curated evaluation set:

```text
eval/golden.jsonl
```

Target:

```text
24–30 high-quality examples
```

Quality matters more than count.

### Required categories

- direct FAQ;
- direct policy;
- exact/lexical retrieval;
- stale information;
- policy vs ticket conflict;
- ambiguous intent;
- multi-turn follow-up;
- unknown question;
- account-specific request;
- payment/security handling;
- escalation;
- capability boundary.

### Critical regression cases

At minimum include:

1. current refund period vs older refund guidance;
2. desktop offline download vs current mobile behavior;
3. `Cancel my LearnForge`;
4. annual-subscription refund ambiguity;
5. pending authorization vs completed payment;
6. assistant must not claim that it performed a cancellation/refund;
7. app-store purchase/refund path;
8. unknown question with no corpus support.

---

## 26. Evaluation schema

Example:

```json
{
  "id": "refund-current-policy",
  "query": "Can I refund a course after 10 days?",
  "history": [],
  "category": "policy",
  "expected_decision": "answer",
  "expected_sources": ["POLICY-02"],
  "forbidden_authority_sources": [],
  "notes": "Current refund policy should be used."
}
```

Multi-turn example:

```json
{
  "id": "apple-follow-up",
  "query": "What if I bought it through Apple?",
  "history": [
    {
      "role": "user",
      "content": "How do refunds work?"
    }
  ],
  "category": "multi_turn",
  "expected_decision": "answer",
  "expected_sources": ["POLICY-04"]
}
```

---

## 27. Metrics

### Retrieval

Measure:

- Hit@5;
- MRR@5;
- correct authoritative source retrieved.

### System behavior

Measure:

- decision accuracy;
- citation validity;
- expected-source accuracy;
- stale-source failure rate;
- conflict-case accuracy;
- capability-boundary accuracy;
- escalation precision;
- escalation recall;
- unsupported-claim rate.

### Hallucination measurement

For this take-home, define an unsupported claim as:

> A LearnForge-specific factual claim that is not supported by any cited retrieved evidence.

Use deterministic/manual evaluation first.

An optional LLM judge may be added only as a secondary signal.

Do not reduce evaluation to a single RAGAS score.

---

## 28. Tests

### Unit tests

Test:

- FAQ parsing;
- policy parsing;
- ticket parsing;
- date extraction;
- deprecated-reference detection;
- duplicate record rejection;
- RRF;
- reranker ordering wrapper;
- evidence preparation;
- citation validation;
- conversation windowing;
- Pydantic output validation.

### Integration tests

Test:

- full index build;
- dense retrieval;
- BM25 retrieval;
- hybrid retrieval;
- current policy vs historical ticket;
- ambiguous cancellation;
- unknown question;
- multi-turn follow-up.

### LLM tests

Most automated tests must not require a paid/live API.

Mock the LLM boundary.

Add a clearly marked optional live smoke test.

---

## 29. Code quality

The governing rule:

> A strong programmer writes code a junior engineer on their first day can understand.

That means:

- descriptive names;
- small functions;
- explicit control flow;
- type hints;
- no clever one-liners;
- no unnecessary design patterns;
- no framework magic;
- comments explain `why`, not obvious `what`;
- configuration is centralized;
- public boundaries have clear docstrings;
- failures are explicit.

Code readability is part of the submission quality.

---

## 30. Dependency discipline

Prefer fewer dependencies.

Expected core dependencies:

```text
fastapi
uvicorn
pydantic
pydantic-settings
qdrant-client[fastembed]
groq
python-dotenv
```

Development:

```text
pytest
pytest-asyncio
httpx
ruff
```

Do not add a library for functionality that can be expressed clearly in a few lines of Python.

Pin sensible version ranges in `pyproject.toml`.

---

## 31. Configuration

`.env.example` should include:

```env
GROQ_API_KEY=
LLM_MODEL=openai/gpt-oss-20b

QDRANT_PATH=.storage/qdrant
QDRANT_COLLECTION=learnforge_support

DENSE_MODEL=BAAI/bge-small-en-v1.5
SPARSE_MODEL=Qdrant/bm25
RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

DENSE_TOP_K=8
SPARSE_TOP_K=8
FUSED_TOP_K=10
FINAL_TOP_K=5
RRF_K=60

MAX_CONVERSATION_TURNS=6
LOG_LEVEL=INFO
```

Never commit a real `.env`.

---

## 32. CI

Add a minimal GitHub Actions workflow.

It should run:

```text
ruff check
pytest
```

Do not create a complex CI/CD pipeline.

This demonstrates that the public repository is maintainable.

---

## 33. Implementation sequence

### Phase 1 — repository foundation

Create:

- `pyproject.toml`;
- package structure;
- config;
- schemas;
- `.env.example`;
- basic logging;
- tests directory.

Acceptance:

```text
imports work
ruff passes
pytest starts successfully
```

### Phase 2 — ingestion

Implement:

- parsers;
- metadata extraction;
- validation;
- content hashes.

Acceptance:

```text
all supplied records parsed
IDs preserved
metadata inspected manually
unit tests pass
```

### Phase 3 — indexing

Implement:

- Qdrant collection;
- dense vectors;
- sparse BM25;
- deterministic rebuild command.

Acceptance:

```text
index builds from scratch
rerun produces no duplicates
payload metadata is correct
```

### Phase 4 — retrieval

Implement:

- conversation-aware query text;
- dense retrieval;
- BM25 retrieval;
- RRF;
- shortlist.

Acceptance:

```text
known relevant records are retrieved
retrieval unit/integration tests pass
```

### Phase 5 — reranking

Implement:

- cross-encoder wrapper;
- candidate reranking;
- top evidence selection.

Acceptance:

```text
reranking is isolated and testable
latency is measured
```

### Phase 6 — reliability + LLM

Implement:

- evidence serialization;
- source precedence instructions;
- structured output;
- decision modes;
- citation validation;
- one bounded retry.

Acceptance:

```text
ANSWER works
CLARIFY works
ESCALATE works
fake actions are not claimed
citations validate
```

### Phase 7 — conversation + API

Implement:

- in-memory bounded session store;
- `POST /chat`;
- `GET /health`;
- request IDs;
- timings.

Acceptance:

```text
multi-turn follow-up works
API contracts are typed
errors are understandable
```

### Phase 8 — evaluation

Create:

- golden dataset;
- retrieval metrics;
- decision/citation metrics;
- regression cases.

Acceptance:

```text
python -m scripts.run_eval
```

produces a human-readable result summary.

### Phase 9 — documentation

Finalize:

- `README.md`;
- `DATA_SCHEMA.md`;
- system design image;
- trade-offs;
- measured evaluation results;
- measured latency;
- exact run commands.

Acceptance:

A new engineer can run the project using only the README.

### Phase 10 — final review

Compare:

```text
assignment
↔ system diagram
↔ PLAN.md
↔ implementation
↔ README
```

Remove inconsistencies.

---

## 34. README structure

The final README should contain:

```text
1. Problem
2. What the prototype does
3. Architecture
4. Quick start
5. Example
6. Knowledge/data model
7. Retrieval strategy
8. Freshness + source authority
9. Hallucination controls
10. Clarification + escalation
11. Failure handling
12. Evaluation
13. Results
14. Performance
15. Trade-offs
16. What I would change in production
```

Keep it concise enough that a reviewer actually reads it.

---

## 35. Trade-offs that must be discussed

### Hybrid vs vector-only

Hybrid was chosen because support questions contain both semantic language and exact policy/billing terminology.

### Local models vs embedding APIs

Local embedding/reranking keeps the prototype reproducible, cheap, and fast.

### Cross-encoder vs no reranker

The corpus is small enough that reranking a small shortlist is inexpensive and demonstrates higher retrieval precision.

### Qdrant local vs hosted

Local mode avoids deployment while preserving realistic vector/payload behavior.

### Logical-record chunking vs fixed token chunks

The supplied data already has meaningful record boundaries.

### One generation call vs multi-agent/multi-LLM pipeline

One normal LLM call minimizes latency, cost, and failure surface.

### Heuristic/source metadata vs generic contradiction model

Metadata solves the assignment's explicit stale/authority problem simply.

A generic semantic contradiction engine would be disproportionate to the dataset and time budget.

### In-memory session state vs Redis/database

Enough for the prototype; production scaling would externalize state.

---

## 36. Definition of done

The submission is complete only when:

### Runs

- clean install works;
- index builds;
- API starts;
- `/health` works;
- `/chat` works.

### Reliability

- current policy outranks historical ticket behavior;
- explicitly stale guidance is not presented as current;
- ambiguous intent clarifies;
- insufficient/conflicting evidence escalates;
- unsupported questions do not hallucinate;
- assistant never claims unavailable account actions;
- citations are validated.

### Evaluation

- golden cases exist;
- retrieval metrics are reported;
- decision/citation metrics are reported;
- critical regression cases pass.

### Engineering

- tests pass;
- Ruff passes;
- no secrets are committed;
- generated storage is gitignored;
- code is easy to trace.

### Communication

- README is complete;
- trade-offs are explicit;
- `DATA_SCHEMA.md` matches real code;
- diagram matches real code;
- evaluation results are real measurements.

---

## 37. Final project philosophy

Do not try to impress the reviewer with the number of technologies.

Impress them with:

- correctness;
- restraint;
- explicit reasoning;
- reliability;
- readable code;
- meaningful tests;
- honest limitations;
- consistency between design and implementation.

The project should feel like something a good engineer could safely inherit.
