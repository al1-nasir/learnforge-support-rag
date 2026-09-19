# LearnForge Support Assistant — Data Schema Specification

This document formally specifies the canonical knowledge schema, vector indexes, and metadata payload representation implemented in the LearnForge Support Assistant prototype.

---

## 1. Overview & Storage Architecture

The knowledge base is stored in a local, persistent [Qdrant](https://qdrant.tech/) vector database configured with dual vector representations (dense semantic vectors + sparse lexical BM25 vectors) alongside a structured JSON metadata payload.

- **Storage Location**: `.storage/qdrant` (configurable via `QDRANT_PATH`)
- **Collection Name**: `learnforge_support` (configurable via `QDRANT_COLLECTION`)
- **Total Points**: Exactly 40 canonical records (15 FAQs, 10 Policies, 15 Support Tickets)
- **Point ID**: Deterministic UUIDv5 generated via `uuid5(NAMESPACE_DNS, f"learnforge:{record_id}")` ensuring idempotent rebuilds without duplication.

---

## 2. Vector Index Fields

Each point in the collection contains two named vectors:

| Vector Name | Type | Model Identifier | Dimension / Representation | Distance Metric / Modifier | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dense` | Dense Float Vector | `BAAI/bge-small-en-v1.5` | 384 dimensions | Cosine Distance | Semantic similarity matching, handling rephrased queries, intent alignment. |
| `bm25` | Sparse Vector | `Qdrant/bm25` | Token indices + IDF weights | IDF Sparse Modifier | Exact lexical matching (policy IDs, specific product terms, numbers, billing codes). |

---

## 3. Metadata Payload Schema (`KnowledgeRecord`)

The payload stored with each vector point represents the canonical `KnowledgeRecord` Pydantic model defined in `src/learnforge_support/schemas.py`.

### Field Definitions

| Field Name | Type | Required / Optional | Purpose | Example |
| :--- | :--- | :--- | :--- | :--- |
| `record_id` | `str` | **Required** | Canonical document identifier matching source files. Used for deterministic citation tracking. | `"POLICY-02"` |
| `source_type` | `Literal["faq", "policy", "ticket"]` | **Required** | Origin document category. | `"policy"` |
| `title` | `str` | **Required** | Document heading or topic title. | `"Cancellation and Refund Policy"` |
| `text` | `str` | **Required** | Unmodified textual body content of the entry. | `"LearnForge generally allows eligible course purchases to be refunded within 14 days..."` |
| `source_date` | `date \| None` (ISO 8601: `YYYY-MM-DD`) | *Optional* | Explicit publication, effective, or review date extracted from document headers. | `"2026-01-01"` |
| `temporal_status` | `Literal["current", "current_unversioned", "historical"]` | **Required** | Freshness status. Policies with dates are `current`; FAQs without dates are `current_unversioned`; tickets are `historical`. | `"current"` |
| `authority_tier` | `Literal["policy", "faq", "historical_example"]` | **Required** | Precedence hierarchy tier used to resolve conflicts between evidence sources. | `"policy"` |
| `contains_deprecated_reference` | `bool` | **Required** (default `False`) | Indicates whether the document explicitly cites obsolete or superseded platform policies. | `true` |
| `ticket_status` | `str \| None` | *Optional* | Resolution state recorded in ticket transcripts. Present only when `source_type == "ticket"`. | `"Resolved."` |
| `content_hash` | `str` | **Required** | SHA-256 digest of canonical content (`record_id`, `title`, `text`) for change detection. | `"d2e5a48b3c9f..."` |

---

## 4. Source Precedence & Authority Hierarchy

When retrieved records contain conflicting statements or differing guidance, the reliability layer enforces a strict authority prior:

$$\text{Current Policy} \succ \text{Current / Unversioned FAQ} \succ \text{Historical Ticket}$$

1. **Policy (`authority_tier: "policy"`, Weight: 1)**
   - Official institutional rules approved by LearnForge management.
   - Authoritative over FAQs and support tickets.
   - Dated policies (e.g. `POLICY-02` effective January 2026) represent the ground truth for refund windows (14 days), billing cycles, and platform terms.
2. **FAQ (`authority_tier: "faq"`, Weight: 2)**
   - General learner assistance and common operational workflows.
   - Authoritative for day-to-day user queries when unversioned or current.
   - Subordinate to explicit policy articles in the event of an unresolved conflict.
3. **Historical Ticket (`authority_tier: "historical_example"`, Weight: 3)**
   - Transcripts of individual past support interactions.
   - Treated as historical case evidence only.
   - **Never authoritative policy**: an exception granted in a past ticket (such as an agent escalating a 30-day promotional refund) does not override standard current policy.
4. **Deprecated Reference Flag (`contains_deprecated_reference: true`)**
   - Automatically detected via keywords (e.g., `"7-day refund period"`, `"billed monthly"`, `"Internet Explorer"`, `"cellular data"`).
   - Signals to the LLM that the referenced statement is obsolete historical context, preventing deprecated language from leaking into modern answers.

---

## 5. Sample Qdrant Payload Record

Below is an actual payload JSON extracted from the live collection:

```json
{
  "id": "e028bfa4-3559-5ca3-9993-455bfa07e477",
  "vector": {
    "dense": [0.0124, -0.0431, 0.0812, "... 384 dimensions total"],
    "bm25": {
      "indices": [412, 1892, 5321],
      "values": [1.452, 2.108, 0.895]
    }
  },
  "payload": {
    "record_id": "POLICY-02",
    "source_type": "policy",
    "title": "Cancellation and Refund Policy",
    "text": "You can cancel an active subscription at any time through Account Settings where self-service cancellation is available. Cancellation normally prevents the next renewal but does not automatically terminate access immediately.\n\nUnless otherwise stated at checkout, users generally retain subscription access until the end of the already-paid billing period. Canceling immediately after a renewal therefore does not necessarily result in an automatic refund.\n\nRefund requests are evaluated separately from cancellation requests. Eligible requests should include the account email and relevant order or transaction information.\n\nFor individual course purchases, LearnForge's standard refund period is generally 14 days, subject to course-consumption restrictions and applicable local law. Certain promotional products, bundles, and third-party purchases may have separate terms.\n\nWhere a technical problem materially prevented access to purchased content, Support may review an exception even when the normal refund period has passed.\n\nOlder help-center documentation referenced a 7-day refund period for all digital products. That article remains accessible in some archived search results but should not be treated as the current standard policy.\n\nRefunds are normally returned to the original payment method. Processing time depends on the payment provider.\n\nEffective date: January 2026.",
    "source_date": "2026-01-01",
    "temporal_status": "current",
    "authority_tier": "policy",
    "contains_deprecated_reference": true,
    "ticket_status": null,
    "content_hash": "629ad59d0f1b2b8cbb4ec0424fb8bb65715a31b6727c62b66236b2ae8f6b7ea5"
  }
}
```

---

## 6. Runtime Evidence & Decision Schemas

### `EvidenceItem`
```python
class EvidenceItem(BaseModel):
    record: KnowledgeRecord
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    reranker_score: float = 0.0
```

### `SupportDecision` (LLM Structured Output)
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
    handoff_summary: str | None = None
```

### `ChatResponse` (HTTP Client API)
```python
class ChatResponse(BaseModel):
    session_id: str
    decision: Literal["answer", "clarify", "escalate"]
    message: str
    reason_code: str
    citations: list[Citation]  # {"record_id": str, "title": str}
    handoff_summary: str | None = None
```

