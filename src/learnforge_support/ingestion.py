"""Deterministic ingestion module for LearnForge knowledge base.

Parses Markdown files (faqs.md, policies.md, tickets.md) into strongly typed
KnowledgeRecord instances, extracts metadata (dates, authority, deprecations,
ticket status), and validates record integrity and idempotency.
Contains no vector database calls.
"""

import hashlib
import re
from datetime import date
from pathlib import Path

from learnforge_support.schemas import (
    KnowledgeRecord,
    TemporalStatus,
)

# Month string to integer mapping for date parsing
_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Regex pattern matching date markers in policy documents
# e.g., "Effective date: January 2026", "Last reviewed: February 2026", "Updated: March 2026"
_POLICY_DATE_REGEX = re.compile(
    r"(?:Effective date|Last updated|Updated|Last reviewed|Reviewed|Effective):\s*([A-Za-z]+)\s*(\d{4})",
    re.IGNORECASE,
)

# Regex pattern matching ticket resolution status
# e.g., "STATUS: Resolved.", "STATUS: Escalated due to policy ambiguity."
_TICKET_STATUS_REGEX = re.compile(r"STATUS:\s*([^\n\r]+)", re.IGNORECASE)

# Keywords indicating explicit references to outdated/deprecated guidance
_DEPRECATED_KEYWORDS: list[str] = [
    "older version",
    "older help",
    "older documentation",
    "older article",
    "older instructor guide",
    "older internal billing",
    "archived documentation",
    "previous version",
    "previous mobile help",
    "outdated",
    "obsolete",
    "retired",
    "no longer",
    "7-day refund",
    "five quizzes",
    "five-user family plan",
    "internet explorer",
    "first six and last four",
    "cellular data",
    "instantly",
]


def compute_content_hash(record_id: str, title: str, text: str) -> str:
    """Computes a SHA-256 hash of canonical content for integrity and change detection."""
    content = f"{record_id.strip()}::{title.strip()}::{text.strip()}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_deprecated_reference(text: str) -> bool:
    """Detects whether text explicitly mentions outdated or superseded guidance."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _DEPRECATED_KEYWORDS)


def parse_policy_date(text: str) -> date | None:
    """Extracts date from standard policy date lines, normalized to the 1st of the month."""
    match = _POLICY_DATE_REGEX.search(text)
    if not match:
        return None
    month_str, year_str = match.groups()
    month = _MONTH_MAP.get(month_str.lower())
    if not month:
        return None
    return date(int(year_str), month, 1)


def parse_ticket_status(text: str) -> str | None:
    """Extracts the final resolution status string from a support ticket."""
    match = _TICKET_STATUS_REGEX.search(text)
    if match:
        return match.group(1).strip()
    return None


def clean_section_artifacts(text: str) -> str:
    """Strips file-level section separator banners from text."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^SECTION\s+\d+\s+—", line.strip(), re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_faqs_markdown(content: str) -> list[KnowledgeRecord]:
    """Deterministically parses faqs.md into KnowledgeRecord objects."""
    records: list[KnowledgeRecord] = []
    # Match headers of format: # FAQ-XX — Title
    header_pattern = re.compile(r"^#\s+(FAQ-\d+)\s+[—\-]\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    for idx, match in enumerate(matches):
        record_id = match.group(1).strip()
        title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)

        raw_body = content[start_pos:end_pos].strip()
        # Remove trailing markdown dividers
        raw_body = re.sub(r"\n---\s*$", "", raw_body).strip()
        body = clean_section_artifacts(raw_body)

        has_deprecated = detect_deprecated_reference(body)
        content_hash = compute_content_hash(record_id, title, body)

        records.append(
            KnowledgeRecord(
                record_id=record_id,
                source_type="faq",
                title=title,
                text=body,
                source_date=None,
                temporal_status="current_unversioned",
                authority_tier="faq",
                contains_deprecated_reference=has_deprecated,
                ticket_status=None,
                content_hash=content_hash,
            )
        )

    return records


def parse_policies_markdown(content: str) -> list[KnowledgeRecord]:
    """Deterministically parses policies.md into KnowledgeRecord objects."""
    records: list[KnowledgeRecord] = []
    # Match headers of format: # POLICY-XX — Title
    header_pattern = re.compile(r"^#\s+(POLICY-\d+)\s+[—\-]\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    for idx, match in enumerate(matches):
        record_id = match.group(1).strip()
        title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)

        raw_body = content[start_pos:end_pos].strip()
        raw_body = re.sub(r"\n---\s*$", "", raw_body).strip()
        body = clean_section_artifacts(raw_body)

        source_date = parse_policy_date(body)
        temporal_status: TemporalStatus = "current" if source_date else "current_unversioned"
        has_deprecated = detect_deprecated_reference(body)
        content_hash = compute_content_hash(record_id, title, body)

        records.append(
            KnowledgeRecord(
                record_id=record_id,
                source_type="policy",
                title=title,
                text=body,
                source_date=source_date,
                temporal_status=temporal_status,
                authority_tier="policy",
                contains_deprecated_reference=has_deprecated,
                ticket_status=None,
                content_hash=content_hash,
            )
        )

    return records


def parse_tickets_markdown(content: str) -> list[KnowledgeRecord]:
    """Deterministically parses tickets.md into KnowledgeRecord objects."""
    records: list[KnowledgeRecord] = []
    # Match headers of format: # TICKET-XX — Title
    header_pattern = re.compile(r"^#\s+(TICKET-\d+)\s+[—\-]\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    for idx, match in enumerate(matches):
        record_id = match.group(1).strip()
        title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)

        raw_body = content[start_pos:end_pos].strip()
        raw_body = re.sub(r"\n---\s*$", "", raw_body).strip()
        body = clean_section_artifacts(raw_body)

        ticket_status = parse_ticket_status(body)
        has_deprecated = detect_deprecated_reference(body)
        content_hash = compute_content_hash(record_id, title, body)

        records.append(
            KnowledgeRecord(
                record_id=record_id,
                source_type="ticket",
                title=title,
                text=body,
                source_date=None,
                temporal_status="historical",
                authority_tier="historical_example",
                contains_deprecated_reference=has_deprecated,
                ticket_status=ticket_status,
                content_hash=content_hash,
            )
        )

    return records


def load_knowledge_base(data_dir: Path | str) -> list[KnowledgeRecord]:
    """Loads and parses all knowledge records from the supplied Markdown directory.

    Validates that:
    - Exactly 40 records are loaded (15 FAQs, 10 Policies, 15 Tickets)
    - All record_ids are unique
    - All titles and body texts are non-empty
    """
    base_path = Path(data_dir)
    faqs_path = base_path / "faqs.md"
    policies_path = base_path / "policies.md"
    tickets_path = base_path / "tickets.md"

    if not faqs_path.is_file():
        raise FileNotFoundError(f"Missing required FAQ file: {faqs_path}")
    if not policies_path.is_file():
        raise FileNotFoundError(f"Missing required policies file: {policies_path}")
    if not tickets_path.is_file():
        raise FileNotFoundError(f"Missing required tickets file: {tickets_path}")

    faqs = parse_faqs_markdown(faqs_path.read_text(encoding="utf-8"))
    policies = parse_policies_markdown(policies_path.read_text(encoding="utf-8"))
    tickets = parse_tickets_markdown(tickets_path.read_text(encoding="utf-8"))

    all_records = faqs + policies + tickets

    # Validation
    seen_ids: set[str] = set()
    for rec in all_records:
        if rec.record_id in seen_ids:
            raise ValueError(f"Duplicate record_id encountered: {rec.record_id}")
        seen_ids.add(rec.record_id)
        if not rec.text.strip():
            raise ValueError(f"Record {rec.record_id} has empty body text")
        if not rec.title.strip():
            raise ValueError(f"Record {rec.record_id} has empty title")

    if len(all_records) != 40:
        raise ValueError(
            f"Expected 40 total records (15 FAQ, 10 Policy, 15 Ticket), found {len(all_records)}"
        )

    return all_records
