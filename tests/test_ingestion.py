"""Unit tests for knowledge base ingestion.

Verifies deterministic parsing, record integrity, metadata extraction
(dates, authority, deprecations, status), and error handling.
"""

from datetime import date
from pathlib import Path

import pytest

from learnforge_support.ingestion import (
    compute_content_hash,
    load_knowledge_base,
)


def test_load_knowledge_base_full_corpus():
    """Verify all 40 records are loaded with unique IDs and expected types."""
    records = load_knowledge_base("data")
    assert len(records) == 40

    faqs = [r for r in records if r.source_type == "faq"]
    policies = [r for r in records if r.source_type == "policy"]
    tickets = [r for r in records if r.source_type == "ticket"]

    assert len(faqs) == 15
    assert len(policies) == 10
    assert len(tickets) == 15

    # Check ID patterns
    assert faqs[0].record_id == "FAQ-01"
    assert faqs[-1].record_id == "FAQ-15"
    assert policies[0].record_id == "POLICY-01"
    assert policies[-1].record_id == "POLICY-10"
    assert tickets[0].record_id == "TICKET-01"
    assert tickets[-1].record_id == "TICKET-15"


def test_authority_tiers():
    """Verify source authority tiers follow policy > faq > ticket."""
    records = load_knowledge_base("data")
    for r in records:
        if r.source_type == "policy":
            assert r.authority_tier == "policy"
        elif r.source_type == "faq":
            assert r.authority_tier == "faq"
        elif r.source_type == "ticket":
            assert r.authority_tier == "historical_example"
            assert r.temporal_status == "historical"


def test_policy_date_parsing():
    """Verify explicit policy dates are parsed accurately."""
    records = {r.record_id: r for r in load_knowledge_base("data")}

    # POLICY-01: Last reviewed: February 2026.
    assert records["POLICY-01"].source_date == date(2026, 2, 1)
    assert records["POLICY-01"].temporal_status == "current"

    # POLICY-02: Effective date: January 2026.
    assert records["POLICY-02"].source_date == date(2026, 1, 1)
    assert records["POLICY-02"].temporal_status == "current"

    # POLICY-07: Effective: December 2025.
    assert records["POLICY-07"].source_date == date(2025, 12, 1)
    assert records["POLICY-07"].temporal_status == "current"

    # POLICY-05 has no explicit date line -> None
    assert records["POLICY-05"].source_date is None
    assert records["POLICY-05"].temporal_status == "current_unversioned"


def test_deprecated_reference_detection():
    """Verify explicit deprecated/outdated statements are flagged."""
    records = {r.record_id: r for r in load_knowledge_base("data")}

    # POLICY-02 mentions older 7-day refund period
    assert records["POLICY-02"].contains_deprecated_reference is True

    # POLICY-01 mentions older annual billing wording
    assert records["POLICY-01"].contains_deprecated_reference is True

    # POLICY-09 mentions Internet Explorer
    assert records["POLICY-09"].contains_deprecated_reference is True

    # POLICY-10 mentions older instruction asking for first six and last four card digits
    assert records["POLICY-10"].contains_deprecated_reference is True

    # FAQ-07 mentions older desktop download feature
    assert records["FAQ-07"].contains_deprecated_reference is True


def test_ticket_status_extraction():
    """Verify ticket resolution status is properly extracted."""
    records = {r.record_id: r for r in load_knowledge_base("data")}

    assert records["TICKET-01"].ticket_status == "Resolved."
    assert records["TICKET-03"].ticket_status == "Escalated."
    assert records["TICKET-08"].ticket_status == "Escalated due to policy ambiguity."
    assert records["TICKET-07"].ticket_status == "Refund request opened."


def test_content_hash_integrity():
    """Verify content hash changes if body text changes."""
    h1 = compute_content_hash("FAQ-01", "Title", "Text A")
    h2 = compute_content_hash("FAQ-01", "Title", "Text B")
    assert h1 != h2
    assert len(h1) == 64


def test_duplicate_record_rejection(tmp_path: Path):
    """Verify duplicate record_ids trigger ValueError."""
    dup_faq = tmp_path / "faqs.md"
    dup_faq.write_text(
        "# FAQ-01 — First\n\nBody 1\n\n---\n\n# FAQ-01 — Second\n\nBody 2",
        encoding="utf-8",
    )
    pol = tmp_path / "policies.md"
    pol.write_text("# POLICY-01 — Pol\n\nBody", encoding="utf-8")
    tkt = tmp_path / "tickets.md"
    tkt.write_text("# TICKET-01 — Tkt\n\nBody\nSTATUS: Open.", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate record_id"):
        load_knowledge_base(tmp_path)
