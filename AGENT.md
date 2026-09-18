# AGENT.md
# Implementation Rules for Antigravity

## 0. Your role

You are the implementation engineer for the LearnForge Applied AI/LLM Engineer take-home.

Your job is to implement the system described in `PLAN.md` using the supplied:

- assignment document;
- `faqs.md`;
- `policies.md`;
- `tickets.md`;
- system design diagram.

Do not redesign the project casually.

If implementation evidence shows that a planned choice is incorrect or unnecessarily complex, change it only when you can explain why, then update the relevant documentation so code and design remain consistent.

---

## 1. Governing rule

> Great programmer code must be understandable by a junior engineer on their first day.

Optimize for:

1. correctness;
2. readability;
3. reliability;
4. testability;
5. simplicity;
6. performance where it matters.

Do not optimize for cleverness.

---

## 2. Before writing code

Read, in this order:

```text
1. assignment
2. PLAN.md
3. faqs.md
4. policies.md
5. tickets.md
6. system-design diagram
```

Understand the deliberate data problems before implementing retrieval.

The corpus contains:

- current guidance;
- old/deprecated wording;
- conflicting support history;
- ambiguous customer requests.

Do not treat every retrieved document as equally authoritative.

---

## 3. Do not improvise the architecture

The intended runtime flow is:

```text
User
  ↓
Chat API
  ↓
bounded conversation context
  ↓
dense + BM25 retrieval
  ↓
RRF fusion
  ↓
cross-encoder reranking
  ↓
reliability/source metadata
  ↓
ANSWER / CLARIFY / ESCALATE
  ↓
structured LLM response
  ↓
citation/schema validation
  ↓
User or human handoff
```

The intended ingestion flow is:

```text
supplied Markdown
  ↓
deterministic parsing
  ↓
logical records + metadata
  ↓
dense + sparse indexing
  ↓
Qdrant
```

If the implementation diverges materially from these flows, stop and document why.

---

## 4. No overengineering

Do not add:

- agents;
- graph orchestrators;
- Celery;
- Kafka;
- Redis;
- Kubernetes;
- microservices;
- dependency injection frameworks;
- repository/service/factory layers without a real need;
- event buses;
- generic plugin systems;
- custom framework abstractions.

This is a ~40-record support RAG prototype.

Production thinking means choosing the simplest architecture that still handles the failure modes correctly.

---

## 5. Framework restraint

Prefer direct, explicit code.

Do not introduce LangChain or LlamaIndex unless there is a demonstrated requirement that cannot be implemented clearly with:

- Qdrant client;
- FastEmbed;
- Groq SDK;
- Pydantic;
- FastAPI.

If a framework is introduced, document the exact benefit in README trade-offs.

---

## 6. Module boundaries

Each module should have one obvious purpose.

Expected responsibilities:

### `config.py`

Only:

- environment settings;
- defaults;
- configuration validation.

No business logic.

### `schemas.py`

Only domain/API models:

- knowledge records;
- evidence items;
- chat request/response;
- LLM structured output.

### `ingestion.py`

Only:

- Markdown parsing;
- metadata extraction;
- record validation;
- content hashing.

No vector DB calls.

### `indexing.py`

Only:

- collection setup;
- embedding/sparse indexing;
- rebuild/upsert logic.

### `retrieval.py`

Only:

- retrieval query construction;
- dense retrieval;
- BM25 retrieval;
- RRF fusion.

### `reranking.py`

Only:

- cross-encoder initialization;
- reranking shortlist.

### `reliability.py`

Only:

- source precedence;
- evidence metadata preparation;
- citation validation;
- reliability/capability rules.

Do not put LLM network calls here.

### `prompting.py`

Only:

- system prompt;
- evidence serialization;
- prompt/message construction.

No API calls.

### `llm.py`

Only:

- provider client;
- structured-output call;
- bounded retry;
- Pydantic validation.

### `conversation.py`

Only:

- bounded session history;
- adding/getting turns.

### `service.py`

Orchestrates the request.

This should be the easiest place to understand the full request lifecycle.

It should call the other modules in obvious order.

### `api.py`

HTTP only:

- endpoints;
- HTTP status translation;
- dependency initialization;
- request/response schemas.

Do not put retrieval or prompting logic in routes.

---

## 7. Readability rules

### Naming

Use names that communicate intent.

Good:

```python
retrieve_candidates()
rerank_candidates()
build_evidence_context()
validate_citations()
should_escalate_for_stale_evidence()
```

Bad:

```python
process()
handle()
run()
do_rag()
util()
manager()
helper2()
```

### Functions

Prefer functions that do one thing.

A function should normally fit on one screen.

If a function has many conceptual steps, split it.

Do not split trivial logic into ten tiny wrapper functions.

### Control flow

Prefer:

```python
if not evidence:
    return ...
```

over deeply nested conditionals.

Avoid clever comprehensions when a normal loop communicates intent better.

### Comments

Comment `why`.

Bad:

```python
# Increment i
i += 1
```

Good:

```python
# Tickets are historical examples and must not outrank a current policy.
authority_tier = AuthorityTier.HISTORICAL_EXAMPLE
```

### Docstrings

Add docstrings to:

- public classes/functions;
- non-obvious algorithms such as RRF;
- logic whose rationale matters.

Do not write repetitive docstrings that merely restate the function name.

---

## 8. Type discipline

Use type hints throughout production code.

Use Pydantic at external/data boundaries.

Prefer domain enums/literals over magic strings.

Do not pass loosely structured dictionaries through the entire application.

Good:

```python
EvidenceItem
KnowledgeRecord
SupportDecision
ChatResponse
```

instead of:

```python
dict[str, Any]
```

everywhere.

Use `Any` only when truly unavoidable.

---

## 9. Configuration discipline

All configurable values belong in settings.

Do not scatter constants such as:

```python
top_k = 8
model = "..."
```

through multiple modules.

Use one typed settings object.

Never hard-code:

- API keys;
- user machine paths;
- secrets;
- environment-specific URLs.

---

## 10. Error handling

Errors should be specific and useful.

Avoid:

```python
except Exception:
    return None
```

Do not silently swallow failures.

Catch exceptions where you can:

- add useful context;
- retry safely;
- convert them to a domain/API error.

A failure to call the LLM must never turn into a fabricated support answer.

Use bounded retries only.

No infinite retries.

---

## 11. Dependency initialization

Heavy models should not be recreated per request.

Initialize:

- dense embedding model;
- sparse/BM25 embedder;
- reranker;
- Qdrant client;
- LLM client;

once per application lifecycle where practical.

Keep initialization explicit and testable.

---

## 12. Data parsing rules

Parsing must be deterministic.

Do not ask an LLM to turn the Markdown into records.

Preserve source IDs exactly where available:

```text
FAQ-01
POLICY-02
TICKET-08
```

Preserve record text faithfully.

Do not silently “correct” the supplied corpus.

The contradictions are part of the assignment.

---

## 13. Chunking rule

Start with the supplied logical record boundaries.

Do not introduce arbitrary token chunking unless a measured retrieval problem requires it.

Before changing chunking:

1. show the failing retrieval case;
2. explain why record-level retrieval fails;
3. add a regression test;
4. then change the implementation.

---

## 14. Retrieval rules

Implement retrieval explicitly.

### Dense

Retrieve semantic candidates.

### Sparse

Retrieve BM25 candidates.

### Fusion

Use RRF.

Do not sum raw dense and BM25 scores.

RRF must be independently unit tested.

### Reranking

Rerank only the fused shortlist.

Never rerank the whole collection.

---

## 15. Source authority rules

Do not treat vector relevance as truth.

The prompt and evidence metadata must preserve:

```text
current policy
    >
FAQ
    >
historical ticket
```

A highly similar historical ticket must not override a current policy.

Explicit stale/deprecated wording should not be presented as current guidance.

---

## 16. Confidence rule

Do not invent fake calibrated confidence numbers.

The project is not trained/calibrated enough to claim:

```text
confidence = 0.93
```

Prefer explicit decisions and reason codes:

```text
ANSWER
CLARIFY
ESCALATE
```

with evidence.

If a numeric retrieval/model score is logged internally, label it accurately as a score, not probability/confidence.

---

## 17. LLM rules

The LLM is a constrained decision/generation component.

It is not the source of LearnForge facts.

It may only use the supplied evidence for LearnForge-specific claims.

Use structured output.

Validate with Pydantic.

The LLM output schema must contain:

```text
decision
message
reason_code
citations
handoff_summary
```

---

## 18. Citation rules

The model may cite only record IDs that were included in its evidence context.

Validate this in code.

If it cites:

```text
POLICY-99
```

and that was not retrieved/passed, reject the response.

Do not rely only on the prompt to enforce citations.

---

## 19. Capability boundary

The assistant cannot perform real account actions.

Never allow final responses such as:

```text
I cancelled your subscription.
I issued your refund.
I restored your course.
I checked your bank transaction.
```

unless a real tool capable of that action exists.

No such tools exist in this assignment.

The system can:

- explain;
- clarify;
- recommend next steps;
- escalate.

Add regression tests for this.

---

## 20. Security/privacy rules

Never ask users for:

- password;
- CVV;
- PIN;
- authentication code;
- full card number.

Do not log secrets.

Do not log API keys.

Do not return secrets through errors.

Keep `.env` gitignored.

Provide `.env.example`.

---

## 21. Conversation rules

Use bounded context.

Do not pass the entire session forever.

Use the recent turns needed for follow-ups.

The retrieval query should remain understandable in logs/tests.

Avoid an additional LLM query-rewrite call unless evaluation proves it is needed.

---

## 22. Logging rules

Each chat request gets a request ID.

Record useful engineering signals:

```text
request_id
session_id
retrieved_record_ids
decision
reason_code
retrieval_ms
rerank_ms
llm_ms
total_ms
error category
```

Do not dump entire user conversations by default.

Logs should help debugging, not become another data leak.

---

## 23. Testing rule

If a reliability behavior matters enough to discuss in README, it matters enough to test.

Every important bug fix must add a regression test.

Tests should be readable examples of expected system behavior.

Most tests must not require network access.

Mock the LLM boundary.

---

## 24. Required regression tests

Do not mark the project complete without tests for:

### Refund freshness

Current refund guidance must not be replaced by old seven-day language.

### Offline downloads

Current mobile-download guidance must beat outdated laptop-download guidance.

### Ambiguous cancellation

`Cancel my LearnForge`

must not immediately assume one action.

### Annual subscription ambiguity

A genuinely ambiguous policy/purchase-specific case should escalate.

### Payment authorization

Pending authorization must not be described as a completed LearnForge charge.

### Capability boundary

The assistant must not claim it performed a refund/cancellation/account action.

### Unknown question

A question unsupported by the corpus must not be answered from model memory.

### Multi-turn

A follow-up such as:

`What if I bought it through Apple?`

must use the previous context correctly.

---

## 25. Evaluation discipline

Build the golden set from the supplied corpus.

Do not create evaluation questions that require facts not present in the assignment data.

Keep expected decisions/sources explicit.

Evaluation output should be reproducible.

Do not manually cherry-pick only successful examples for README.

Report failures honestly.

---

## 26. Performance rule

Measure before optimizing.

Track:

```text
dense/sparse retrieval
reranking
LLM
total request
```

Do not add caching, concurrency machinery, or GPU complexity before a measured need exists.

The small local models should be loaded once and reused.

---

## 27. Scale rule

Do not implement distributed infrastructure to “show scale.”

Instead:

- keep storage access behind clear code;
- configure local/remote Qdrant cleanly;
- keep ingestion deterministic;
- explain production scaling in trade-offs.

The reviewer asked whether the architecture addresses scale, not whether you can add unnecessary infrastructure.

---

## 28. Freshness rule

Freshness must exist in actual code/data.

Do not merely write “we handle stale data” in README.

The implementation should preserve:

- source date when available;
- temporal status;
- authority;
- deprecated-reference flag.

The LLM must receive this metadata.

---

## 29. Commit/documentation consistency

After each major phase, compare implementation to `PLAN.md`.

If an architectural choice changes:

1. update code;
2. update tests;
3. update `PLAN.md` if needed;
4. update README/design documentation.

Never let the documentation describe a system that no longer exists.

---

## 30. Work phase by phase

Do not implement the whole project in one uncontrolled pass.

For each phase:

1. implement;
2. run tests;
3. inspect output;
4. fix obvious issues;
5. only then continue.

Follow the phases in `PLAN.md`.

---

## 31. Before adding a dependency

Ask:

```text
Can this be implemented clearly in <30 lines of ordinary Python?
```

If yes, prefer ordinary Python unless the dependency materially improves correctness.

Examples:

- RRF does not need a library.
- content hashing does not need a library.
- bounded in-memory conversation history does not need Redis.

---

## 32. Before adding an abstraction

Ask:

```text
Does this remove real duplication or isolate an external boundary for testing?
```

If no, do not add it.

Avoid:

```text
BaseRetrieverFactory
AbstractRAGManager
PipelineOrchestratorFactory
```

when a readable function/class is sufficient.

---

## 33. Code review questions

Before considering a module finished, ask:

### Readability

Can a junior engineer tell what this module does in 30 seconds?

### Necessity

Does every class/function have a real job?

### Failure

What happens when its dependency fails?

### Testability

Can this behavior be tested without a live LLM?

### Naming

Would the function names still make sense six months later?

### Documentation

Is the reason for non-obvious logic captured somewhere?

---

## 34. API review questions

Before considering `/chat` finished:

- Is the request typed?
- Is the response typed?
- Is session behavior explicit?
- Are internal scores hidden from normal users?
- Are errors understandable?
- Can the endpoint hallucinate on provider failure?
- Are citations valid?
- Can it claim account actions?

---

## 35. README rule

README is part of the assignment, not an afterthought.

Write it after the implementation is stable.

Do not fill it with generic AI/RAG explanations.

Document this exact system.

Include real:

- commands;
- architecture;
- schema;
- examples;
- evaluation results;
- failure behavior;
- trade-offs;
- measured latency.

No fake metrics.

---

## 36. DATA_SCHEMA.md rule

Generate `DATA_SCHEMA.md` from the final implemented model, not from an early guess.

It must document:

- each field;
- type;
- required/optional;
- purpose;
- example;
- vector fields;
- metadata payload;
- source precedence meaning.

The submission schema must match the real indexed payload.

---

## 37. System-design rule

The final diagram must match the implementation.

If the diagram shows:

```text
BM25
RRF
reranker
Qdrant
source authority
clarification
escalation
```

those must exist in the actual project.

If any are removed during implementation, update the diagram before submission.

---

## 38. Quality gates

Do not proceed to final documentation until:

```text
ruff check passes
pytest passes
index rebuild succeeds
API starts
health endpoint succeeds
normal answer works
clarification works
escalation works
multi-turn works
evaluation runner works
```

---

## 39. Final repository audit

Before submission, inspect the public repository as if you were the reviewer.

Check:

- no secrets;
- no local absolute paths;
- no generated Qdrant database;
- no model files committed;
- no dead code;
- no unused dependencies;
- no debug prints;
- no TODOs that undermine required functionality;
- no architecture claims unsupported by code;
- no broken commands in README.

Clone/install mentally from scratch.

A reviewer should not need your machine to make sense of the project.

---

## 40. When to stop and ask

Stop and ask the user rather than guessing when:

- a requirement from the assignment is genuinely ambiguous;
- supplied files are missing or malformed;
- the chosen LLM/model is unavailable;
- an architecture change would materially alter the submitted design;
- a dependency behaves differently from its official documentation;
- a requested feature would require fabricating capabilities/data.

Do not silently invent missing business requirements.

---

## 41. Definition of implementation success

A strong submission is not the one with the most code.

It is the one where a reviewer can see:

```text
clear problem understanding
→ deliberate architecture
→ readable implementation
→ failure-aware behavior
→ meaningful evaluation
→ honest trade-offs
```

Build that system.
