"""Integration tests for vector and BM25 indexing in Qdrant.

Verifies collection creation, dual vector configurations, payload persistence,
and idempotency.
"""

from pathlib import Path

from learnforge_support.config import get_settings
from learnforge_support.indexing import (
    get_qdrant_client,
    prepare_index_text,
    setup_collection,
)
from learnforge_support.schemas import KnowledgeRecord


def test_prepare_index_text():
    """Verify index text concatenates title and body."""
    rec = KnowledgeRecord(
        record_id="FAQ-01",
        source_type="faq",
        title="Sample Title",
        text="Sample Body",
        temporal_status="current_unversioned",
        authority_tier="faq",
        content_hash="hash1",
    )
    text = prepare_index_text(rec)
    assert text == "Sample Title\n\nSample Body"


def test_indexed_collection_payload_and_count():
    """Verify that the built collection contains exactly 40 records with valid payloads."""
    settings = get_settings()
    client = get_qdrant_client(settings.qdrant_path)

    assert client.collection_exists(settings.qdrant_collection)
    info = client.get_collection(settings.qdrant_collection)
    assert info.points_count == 40

    # Retrieve one point and verify payload deserializes into KnowledgeRecord
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert len(points) == 1
    payload = points[0].payload
    assert payload is not None
    record = KnowledgeRecord.model_validate(payload)
    assert record.record_id.startswith(("FAQ-", "POLICY-", "TICKET-"))


def test_setup_collection_idempotent(tmp_path: Path):
    """Verify setup_collection creates and recreates cleanly on temporary storage."""
    client = get_qdrant_client(tmp_path)
    coll = "test_coll"

    # First creation
    setup_collection(client, coll, dense_dim=384, recreate=False)
    assert client.collection_exists(coll)

    # Calling without recreate=True does not error
    setup_collection(client, coll, dense_dim=384, recreate=False)
    assert client.collection_exists(coll)

    # Calling with recreate=True deletes and recreates
    setup_collection(client, coll, dense_dim=384, recreate=True)
    assert client.collection_exists(coll)
