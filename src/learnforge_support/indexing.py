"""Indexing module for LearnForge knowledge base.

Handles Qdrant collection lifecycle, dense vector and sparse BM25 indexing,
and deterministic, idempotent rebuilding.
"""

import atexit
import uuid
from collections.abc import Sequence
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from learnforge_support.config import Settings, get_settings
from learnforge_support.ingestion import load_knowledge_base
from learnforge_support.logging_utils import setup_logger
from learnforge_support.schemas import KnowledgeRecord

logger = setup_logger()

_QDRANT_CLIENT_CACHE: dict[str, QdrantClient] = {}


def _cleanup_qdrant_clients() -> None:
    for client in list(_QDRANT_CLIENT_CACHE.values()):
        try:
            client.close()
        except Exception:
            pass
    _QDRANT_CLIENT_CACHE.clear()


atexit.register(_cleanup_qdrant_clients)


def get_qdrant_client(storage_path: str | Path) -> QdrantClient:
    """Initializes and returns a cached local embedded QdrantClient for the path.

    Reusing the client per storage path avoids file lock contention in the same process.
    """
    path_obj = Path(storage_path).resolve()
    path_key = str(path_obj)
    if path_key not in _QDRANT_CLIENT_CACHE:
        path_obj.mkdir(parents=True, exist_ok=True)
        _QDRANT_CLIENT_CACHE[path_key] = QdrantClient(path=path_key)
    return _QDRANT_CLIENT_CACHE[path_key]


def close_qdrant_client(storage_path: str | Path) -> None:
    """Closes and removes a cached QdrantClient for cleanup."""
    path_key = str(Path(storage_path).resolve())
    client = _QDRANT_CLIENT_CACHE.pop(path_key, None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def setup_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = 384,
    recreate: bool = False,
) -> None:
    """Creates or recreates the target collection configured for hybrid search.

    Configures:
    - 'dense': Dense vectors with Cosine distance
    - 'bm25': Sparse vectors with BM25/IDF modifier
    """
    exists = client.collection_exists(collection_name)
    if exists:
        if recreate:
            logger.info("Dropping existing collection '%s' for clean rebuild", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.debug("Collection '%s' already exists", collection_name)
            return

    logger.info(
        "Creating Qdrant collection '%s' (dense=%d, sparse=bm25)", collection_name, dense_dim
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def prepare_index_text(record: KnowledgeRecord) -> str:
    """Formats record content for indexing, combining title and text."""
    return f"{record.title}\n\n{record.text}"


def index_records(
    client: QdrantClient,
    collection_name: str,
    records: Sequence[KnowledgeRecord],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
) -> int:
    """Embeds and upserts records into Qdrant using dense and sparse vectors.

    Generates deterministic UUIDs from record IDs for idempotency.
    """
    if not records:
        return 0

    texts = [prepare_index_text(r) for r in records]

    # Generate dense embeddings (shape: [N, 384])
    dense_vectors = list(dense_model.embed(texts))

    # Generate sparse BM25 embeddings
    sparse_vectors = list(sparse_model.embed(texts))

    points: list[PointStruct] = []
    for record, dense_vec, sparse_vec in zip(records, dense_vectors, sparse_vectors, strict=True):
        # Deterministic UUID per record_id guarantees idempotency across rebuilds
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"learnforge:{record.record_id}"))

        sparse_data = SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist(),
        )

        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec.tolist(),
                "bm25": sparse_data,
            },
            payload=record.model_dump(mode="json"),
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Indexed %d records into collection '%s'", len(points), collection_name)
    return len(points)


def rebuild_index(settings: Settings | None = None) -> int:
    """Executes an idempotent, complete index build from raw markdown files."""
    if settings is None:
        settings = get_settings()

    records = load_knowledge_base(settings.data_dir)
    logger.info("Loaded %d knowledge records from %s", len(records), settings.data_dir)

    client = get_qdrant_client(settings.qdrant_path)
    setup_collection(client=client, collection_name=settings.qdrant_collection, recreate=True)

    dense_model = TextEmbedding(model_name=settings.dense_model)
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    count = index_records(
        client=client,
        collection_name=settings.qdrant_collection,
        records=records,
        dense_model=dense_model,
        sparse_model=sparse_model,
    )
    return count
