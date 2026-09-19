# LearnForge Support Assistant — Data Schema

The index contains 40 logical knowledge records: 15 FAQs, 10 policies, and 15 historical support tickets. A logical record is one source entry, rather than an arbitrary token chunk, so policy conditions remain intact and citations remain readable.

## Indexed representation

Each Qdrant point has a deterministic UUIDv5 ID derived from `record_id`, two named vectors, and a JSON payload. Rebuilding the index is therefore idempotent.

| Part | Implementation | Purpose |
| --- | --- | --- |
| `dense` vector | `BAAI/bge-small-en-v1.5`, 384 dimensions, cosine distance | Semantic retrieval |
| `bm25` vector | `Qdrant/bm25` sparse indices and values | Exact terminology and identifiers |
| payload | `KnowledgeRecord` serialized as JSON | Citations, filtering, and reliability context |

The index text is `title + "\\n\\n" + text`. Dense and sparse rankings are fused with RRF; vector scores are never used as an authority score.

## Qdrant payload (`KnowledgeRecord`)

| Field | Type | Meaning |
| --- | --- | --- |
| `record_id` | string | Stable source ID, for example `POLICY-02` |
| `source_type` | `faq` \| `policy` \| `ticket` | Source document family |
| `title` | string | Source entry heading |
| `text` | string | Complete, unmodified entry content |
| `source_date` | ISO date or `null` | Explicit effective, updated, or reviewed date when present |
| `temporal_status` | `current` \| `current_unversioned` \| `historical` | Freshness classification |
| `authority_tier` | `policy` \| `faq` \| `historical_example` | Conflict-resolution priority |
| `contains_deprecated_reference` | boolean | The record mentions superseded guidance |
| `ticket_status` | string or `null` | Resolution state from a ticket transcript only |
| `content_hash` | SHA-256 hex string | Change detection for canonical content |

## Authority and freshness semantics

The reliability layer orders evidence as follows:

```text
current policy > FAQ > historical support ticket
```

- Policies with explicit dates are `current`; policies without a date and FAQs are `current_unversioned`.
- Every ticket is `historical` and may illustrate a prior case, but cannot establish current policy.
- `contains_deprecated_reference` preserves useful context about retired instructions, such as the former 7-day refund rule. It is a warning, not a reason to discard the entire record: a current policy can explain both the superseded and current rule.
- `content_hash` is calculated from `record_id`, `title`, and `text`. It makes changes visible during an index rebuild without treating embeddings as source data.

## Example payload

```json
{
  "record_id": "POLICY-02",
  "source_type": "policy",
  "title": "Cancellation and Refund Policy",
  "text": "...",
  "source_date": "2026-01-01",
  "temporal_status": "current",
  "authority_tier": "policy",
  "contains_deprecated_reference": true,
  "ticket_status": null,
  "content_hash": "629ad59d0f1b2b8cbb4ec0424fb8bb65715a31b6727c62b66236b2ae8f6b7ea5"
}
```

At runtime, `EvidenceItem` adds dense rank, sparse rank, RRF score, and cross-encoder score. These are transient ranking data; they are not stored in the canonical payload. The LLM receives the final evidence records with their authority and freshness metadata, and returned citations are validated against those record IDs.
