"""Backfill v3-binary scores for archived articles.

Fetches archived articles from Readwise Reader API, scores them with
BinaryScoringStrategy, and saves to article_scores_v3 with highlighted_words
populated from Readwise v2 export API for calibration.

Usage:
    python -m tools.backfill_v3 [--limit N] [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _fetch_highlighted_words() -> dict[str, int]:
    """Fetch highlighted word counts from Readwise v2 export API.

    Returns {source_url_normalized: total_highlighted_words}.
    """
    from urllib.parse import urldefrag, urlparse, urlunparse

    import requests

    token = os.environ.get("READWISE_TOKEN", "")
    if not token:
        logger.warning("READWISE_TOKEN not set, skipping highlight fetch")
        return {}

    def _normalize_url(url: str | None) -> str | None:
        if not url:
            return None
        url, _ = urldefrag(url)
        parsed = urlparse(url)
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/") if parsed.path != "/" else parsed.path,
                parsed.params,
                parsed.query,
                "",
            )
        )

    logger.info("Fetching highlight exports from Readwise v2 API...")
    url_to_words: dict[str, int] = {}
    api_url = "https://readwise.io/api/v2/export/"
    headers = {"Authorization": f"Token {token}"}
    params: dict[str, str] = {}
    page = 0

    while True:
        page += 1
        resp = requests.get(api_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        for book in data.get("results", []):
            source_url = book.get("source_url") or book.get("url")
            norm = _normalize_url(source_url)
            if not norm:
                continue
            total_words = 0
            for hl in book.get("highlights", []):
                text = hl.get("text", "")
                total_words += len(text.split())
            if total_words > 0:
                url_to_words[norm] = url_to_words.get(norm, 0) + total_words

        next_cursor = data.get("nextPageCursor")
        if not next_cursor:
            break
        params = {"pageCursor": next_cursor}
        logger.info("  Page %d fetched (%d URLs so far)", page, len(url_to_words))

    logger.info("Fetched highlighted words for %d URLs", len(url_to_words))
    return url_to_words


async def backfill(limit: int, dry_run: bool) -> None:
    """Score archived articles with v3-binary strategy."""
    from urllib.parse import urldefrag, urlparse, urlunparse

    from anthropic import Anthropic

    from app.config import get_settings
    from app.models.article import BinaryArticleScore, get_engine, get_session_factory
    from app.services.scoring_strategy import BinaryScoringStrategy

    settings = get_settings()
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    strategy = BinaryScoringStrategy()

    # Load DB to find archived articles without v3 scores
    db_path = _PROJECT_ROOT / "reader_triage.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT a.id, a.title, a.url, a.author, a.word_count, a.content
        FROM articles a
        LEFT JOIN article_scores_v3 v3 ON v3.article_id = a.id
        WHERE a.location = 'archive'
          AND v3.id IS NULL
          AND a.content IS NOT NULL
          AND length(a.content) > 200
        ORDER BY a.readwise_created_at DESC
        LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    logger.info("Found %d archived articles to score", len(rows))

    if not rows:
        return

    # Fetch highlighted words for calibration
    highlighted_words_map: dict[str, int] = {}
    if not dry_run:
        raw_map = _fetch_highlighted_words()

        # Match by normalized URL to article IDs
        def _normalize_url(url: str | None) -> str | None:
            if not url:
                return None
            url_clean, _ = urldefrag(url)
            parsed = urlparse(url_clean)
            return urlunparse(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    parsed.path.rstrip("/") if parsed.path != "/" else parsed.path,
                    parsed.params,
                    parsed.query,
                    "",
                )
            )

        for row in rows:
            norm = _normalize_url(row["url"])
            if norm and norm in raw_map:
                highlighted_words_map[row["id"]] = raw_map[norm]

    # Initialize DB engine for async writes
    engine = await get_engine()

    from app.models.article import Base

    async with engine.begin() as conn_async:
        await conn_async.run_sync(Base.metadata.create_all)

    factory = await get_session_factory()

    scored = 0
    for i, row in enumerate(rows):
        article_id = row["id"]
        title = row["title"]
        content = row["content"]
        author = row["author"]
        word_count = row["word_count"]

        if not content or len(content) < 200:
            continue

        logger.info(
            "[%d/%d] Scoring %s: %s",
            i + 1,
            len(rows),
            article_id,
            title[:60],
        )

        if dry_run:
            logger.info("  [DRY RUN] Would score and save")
            scored += 1
            continue

        try:
            result = await strategy.score(
                title=title,
                author=author,
                content=content,
                word_count=word_count,
                content_type_hint="article",
                anthropic_client=anthropic_client,
                entity_id=article_id,
            )

            if result is None:
                logger.warning("  Scoring returned None, skipping")
                continue

            async with factory() as session:
                v3_record = BinaryArticleScore(
                    article_id=article_id,
                    info_score=result.total,
                    specificity_score=result.specificity,
                    novelty_score=result.novelty,
                    depth_score=result.depth,
                    actionability_score=result.actionability,
                    raw_responses=json.dumps(result.raw_responses)
                    if result.raw_responses
                    else None,
                    score_reasons=json.dumps(
                        [
                            result.specificity_reason,
                            result.novelty_reason,
                            result.depth_reason,
                            result.actionability_reason,
                        ]
                    ),
                    overall_assessment=result.overall_assessment,
                    content_fetch_failed=result.content_fetch_failed,
                    model_used="claude-sonnet-4-20250514",
                    scoring_version=strategy.version,
                    scored_at=datetime.now(),
                    highlighted_words=highlighted_words_map.get(article_id),
                )
                session.add(v3_record)
                await session.commit()

            scored += 1
            logger.info(
                "  Score: %d (hw=%s)", result.total, highlighted_words_map.get(article_id, "n/a")
            )

            # Rate limiting
            await asyncio.sleep(1.5)

        except Exception:
            logger.exception("  Error scoring %s", article_id)
            await asyncio.sleep(2)

    logger.info("Done: %d articles scored", scored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill v3-binary scores for archived articles")
    parser.add_argument("--limit", type=int, default=50, help="Max articles to score")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    asyncio.run(backfill(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
