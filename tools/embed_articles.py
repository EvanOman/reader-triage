"""Embed all scored articles into the Qdrant vector store.

Usage:
    uv run python -m tools.embed_articles [--reindex]

Options:
    --reindex   Delete existing collection and re-embed everything.
"""

import asyncio
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    reindex = "--reindex" in sys.argv

    # Import here so dotenv is loaded before anything else
    from app.services.vectorstore import (
        COLLECTION_NAME,
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        VectorStore,
    )

    store = VectorStore()

    logger.info("Model: %s (%d dims)", EMBEDDING_MODEL, EMBEDDING_DIM)

    if reindex:
        logger.info("Reindexing: deleting existing collection")
        try:
            store._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        store._ensure_collection()

    # Warm up the model (first load is slow, don't count it)
    logger.info("Loading embedding model...")
    model_load_start = time.time()
    store._get_model()
    model_load_time = time.time() - model_load_start
    logger.info("Model loaded in %.1fs", model_load_time)

    start = time.time()
    count = await store.embed_all_articles()
    elapsed = time.time() - start

    total = store.collection_count()

    # Performance summary
    logger.info("=" * 60)
    logger.info("EMBEDDING PERFORMANCE REPORT")
    logger.info("=" * 60)
    logger.info("Model:              %s", EMBEDDING_MODEL)
    logger.info("Dimensions:         %d", EMBEDDING_DIM)
    logger.info("Model load time:    %.1fs", model_load_time)
    logger.info("Articles embedded:  %d", count)
    logger.info("Total in store:     %d", total)
    logger.info("Total embed time:   %.1fs", elapsed)
    if count > 0:
        avg_per_article = elapsed / count
        articles_per_sec = count / elapsed if elapsed > 0 else 0
        logger.info("Avg time/article:   %.3fs", avg_per_article)
        logger.info("Articles/second:    %.1f", articles_per_sec)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
