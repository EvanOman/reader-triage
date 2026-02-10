"""Backfill article content from Readwise API.

Fetches full content for articles with NULL content column,
strips HTML, stores clean text, and rebuilds the FTS index.

Usage:
    uv run python -m tools.backfill_content
"""

import asyncio
import logging
import re

from readwise_sdk.exceptions import RateLimitError
from sqlalchemy import select

from app.models.article import Article, get_session_factory, rebuild_fts_index
from app.services.readwise import get_readwise_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill_content():
    """Fetch and store content for all articles missing it."""
    from app.models.article import init_db

    await init_db()

    factory = await get_session_factory()
    readwise = get_readwise_service()

    # Get all articles with NULL content
    async with factory() as session:
        result = await session.execute(select(Article.id).where(Article.content.is_(None)))
        article_ids = [row[0] for row in result.all()]

    logger.info("Found %d articles with NULL content", len(article_ids))

    updated = 0
    for i, article_id in enumerate(article_ids):
        try:
            # Fetch full content from Readwise with retry on rate limits
            doc = None
            for attempt in range(6):
                try:
                    doc = await readwise.get_document(article_id, with_content=True)
                    break
                except RateLimitError as e:
                    wait = e.retry_after if e.retry_after else 2 ** (attempt + 1)
                    logger.info(
                        "Rate limited on %s, waiting %ds (attempt %d)",
                        article_id,
                        wait,
                        attempt + 1,
                    )
                    await asyncio.sleep(wait)

            if doc is None or not doc.content:
                logger.debug("No content for %s (%d/%d)", article_id, i + 1, len(article_ids))
                continue

            # Strip HTML tags
            clean_text = re.sub(r"<[^>]+>", "", doc.content)

            # Store content and backfill content_preview
            async with factory() as session:
                article = await session.get(Article, article_id)
                if article is None:
                    continue
                article.content = clean_text
                if not article.content_preview:
                    article.content_preview = clean_text[:2000]
                await session.commit()

            updated += 1
            if updated % 10 == 0:
                logger.info(
                    "Progress: %d/%d updated (%d fetched)", updated, len(article_ids), i + 1
                )

            # Pace requests (1s between calls)
            await asyncio.sleep(1)

        except Exception:
            logger.exception("Error processing %s", article_id)
            await asyncio.sleep(2)

    logger.info("Backfilled content for %d articles", updated)

    # Rebuild FTS index with new content
    logger.info("Rebuilding FTS index...")
    await rebuild_fts_index()
    logger.info("Done")


if __name__ == "__main__":
    asyncio.run(backfill_content())
