"""Score all non-archived articles with v3-binary scoring.

Processes articles that have content in the DB first, then attempts
to fetch content from Readwise for articles without content.

Usage:
    python -m tools.score_inbox_v3 [--limit N] [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def score_inbox(limit: int, dry_run: bool) -> None:
    """Score non-archived articles with v3-binary strategy."""
    from anthropic import Anthropic

    from app.config import get_settings
    from app.models.article import Base, BinaryArticleScore, get_engine, get_session_factory
    from app.services.scoring_strategy import BinaryScoringStrategy

    settings = get_settings()
    anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
    strategy = BinaryScoringStrategy()

    db_path = _PROJECT_ROOT / "reader_triage.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Articles with content that need v3 scoring
    rows = conn.execute(
        """
        SELECT a.id, a.title, a.url, a.author, a.word_count, a.content
        FROM articles a
        LEFT JOIN article_scores_v3 v3 ON v3.article_id = a.id
        WHERE a.location != 'archive'
          AND v3.id IS NULL
          AND a.content IS NOT NULL
          AND length(a.content) > 200
        ORDER BY a.readwise_created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    # Articles without content
    no_content_rows = conn.execute(
        """
        SELECT a.id, a.title, a.url, a.author, a.word_count
        FROM articles a
        LEFT JOIN article_scores_v3 v3 ON v3.article_id = a.id
        WHERE a.location != 'archive'
          AND v3.id IS NULL
          AND (a.content IS NULL OR length(a.content) <= 200)
        ORDER BY a.readwise_created_at DESC
        LIMIT ?
        """,
        (max(0, limit - len(rows)),),
    ).fetchall()
    conn.close()

    logger.info(
        "Found %d articles with content, %d without content to score",
        len(rows),
        len(no_content_rows),
    )

    if not rows and not no_content_rows:
        return

    engine = await get_engine()
    async with engine.begin() as conn_async:
        await conn_async.run_sync(Base.metadata.create_all)

    factory = await get_session_factory()

    scored = 0
    failed = 0
    all_items = [(r, True) for r in rows] + [(r, False) for r in no_content_rows]

    # For articles without content, try fetching from Readwise
    readwise_service = None
    if no_content_rows:
        from app.services.readwise import get_readwise_service

        readwise_service = get_readwise_service()

    for i, (row, has_content) in enumerate(all_items):
        article_id = row["id"]
        title = row["title"]
        author = row["author"]
        word_count = row["word_count"]

        if has_content:
            content = row["content"]
        else:
            if readwise_service is None:
                continue
            logger.info("[%d/%d] Fetching content for %s", i + 1, len(all_items), title[:60])
            try:
                fetched = await readwise_service.get_document(article_id, with_content=True)
                if fetched is None or not fetched.content or len(fetched.content) < 200:
                    logger.warning("  No content available, skipping")
                    continue
                import re

                content = re.sub(r"<[^>]+>", "", fetched.content)
                # Save content back to DB
                db_conn = sqlite3.connect(str(db_path))
                db_conn.execute(
                    "UPDATE articles SET content = ? WHERE id = ?",
                    (content, article_id),
                )
                db_conn.commit()
                db_conn.close()
            except Exception:
                logger.exception("  Error fetching content for %s", article_id)
                continue

        logger.info(
            "[%d/%d] Scoring %s: %s",
            i + 1,
            len(all_items),
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
                failed += 1
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
                )
                session.add(v3_record)
                await session.commit()

            scored += 1
            logger.info("  Score: %d", result.total)

            await asyncio.sleep(1.0)

        except Exception:
            logger.exception("  Error scoring %s", article_id)
            failed += 1
            await asyncio.sleep(2)

    logger.info("Done: %d scored, %d failed", scored, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score non-archived articles with v3-binary")
    parser.add_argument("--limit", type=int, default=200, help="Max articles to score")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    asyncio.run(score_inbox(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
