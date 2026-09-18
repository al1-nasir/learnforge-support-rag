"""CLI script to build or rebuild the LearnForge Qdrant vector index.

Usage:
    python -m scripts.build_index
"""

import sys
import time

from learnforge_support.config import get_settings
from learnforge_support.indexing import rebuild_index
from learnforge_support.logging_utils import setup_logger

logger = setup_logger()


def main() -> None:
    """Rebuilds the local hybrid index from raw knowledge base files."""
    settings = get_settings()
    logger.info("Starting LearnForge index build...")
    logger.info("Storage directory: %s", settings.qdrant_path)
    logger.info("Collection: %s", settings.qdrant_collection)
    logger.info("Dense model: %s | Sparse model: %s", settings.dense_model, settings.sparse_model)

    start_time = time.perf_counter()
    try:
        count = rebuild_index(settings)
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Index build completed successfully in %.2fs: %d records indexed.",
            elapsed,
            count,
        )
    except Exception as exc:
        logger.error("Failed to build index: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
