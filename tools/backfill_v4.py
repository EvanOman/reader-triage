"""Backfill v4-binary scores for articles with engagement data.

Scores articles using the TieredBinaryScoringStrategy (v4) and saves
results to article_scores_v3 table (with scoring_version='v4-binary').
Prioritizes articles that already have highlight data for calibration.

Usage:
    python -m tools.backfill_v4 [--limit N] [--dry-run] [--all]
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


async def backfill(limit: int, dry_run: bool, score_all: bool) -> None:
    """Score articles with v4-binary strategy."""
    from app.models.article import V4ArticleScore, get_engine, get_session_factory
    from app.services.scoring_strategy import TieredBinaryScoringStrategy

    strategy = TieredBinaryScoringStrategy(model_id="groq/qwen/qwen3-32b")

    db_path = _PROJECT_ROOT / "reader_triage.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find articles to score with V4. Prioritize those with highlight data.
    # Skip articles that already have a v4-binary score in article_scores_v4.
    if score_all:
        query = """
            SELECT a.id, a.title, a.url, a.author, a.word_count, a.content,
                   a.highlighted_words, a.num_highlights, a.category
            FROM articles a
            LEFT JOIN article_scores_v4 v4 ON v4.article_id = a.id
            WHERE v4.id IS NULL
              AND a.content IS NOT NULL
              AND length(a.content) > 200
              AND a.category IN ('article', 'podcast')
            ORDER BY a.highlighted_words DESC NULLS LAST,
                     a.readwise_created_at DESC
            LIMIT ?
        """
    else:
        # Default: only articles with highlight data (for calibration)
        query = """
            SELECT a.id, a.title, a.url, a.author, a.word_count, a.content,
                   a.highlighted_words, a.num_highlights, a.category
            FROM articles a
            LEFT JOIN article_scores_v4 v4 ON v4.article_id = a.id
            WHERE v4.id IS NULL
              AND a.content IS NOT NULL
              AND length(a.content) > 200
              AND a.category IN ('article', 'podcast')
              AND a.highlighted_words IS NOT NULL
            ORDER BY a.highlighted_words DESC
            LIMIT ?
        """

    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    logger.info("Found %d articles to score with V4", len(rows))
    if not rows:
        return

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
        hw = row["highlighted_words"]
        category = row["category"] or "article"

        if not content or len(content) < 200:
            continue

        logger.info(
            "[%d/%d] Scoring %s: %s (hw=%s)",
            i + 1,
            len(rows),
            article_id,
            title[:60],
            hw if hw is not None else "n/a",
        )

        if dry_run:
            logger.info("  [DRY RUN] Would score and save")
            scored += 1
            continue

        try:
            content_type = "podcast" if category == "podcast" else "article"
            result = await strategy.score(
                title=title,
                author=author,
                content=content,
                word_count=word_count,
                content_type_hint=content_type,
                entity_id=article_id,
            )

            if result is None:
                logger.warning("  Scoring returned None, skipping")
                continue

            async with factory() as session:
                from sqlalchemy import select

                existing_result = await session.execute(
                    select(V4ArticleScore).where(V4ArticleScore.article_id == article_id)
                )
                existing = existing_result.scalar_one_or_none()

                if existing is not None:
                    existing.info_score = result.total
                    existing.specificity_score = result.specificity
                    existing.novelty_score = result.novelty
                    existing.depth_score = result.depth
                    existing.actionability_score = result.actionability
                    existing.raw_responses = (
                        json.dumps(result.raw_responses) if result.raw_responses else None
                    )
                    existing.score_reasons = json.dumps(
                        [
                            result.specificity_reason,
                            result.novelty_reason,
                            result.depth_reason,
                            result.actionability_reason,
                        ]
                    )
                    existing.overall_assessment = result.overall_assessment
                    existing.content_fetch_failed = result.content_fetch_failed
                    existing.model_used = "claude-sonnet-4-5-20250929"
                    existing.scoring_version = strategy.version
                    existing.scored_at = datetime.now()
                    existing.highlighted_words = hw
                else:
                    v4_record = V4ArticleScore(
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
                        model_used="claude-sonnet-4-5-20250929",
                        scoring_version=strategy.version,
                        scored_at=datetime.now(),
                        highlighted_words=hw,
                    )
                    session.add(v4_record)
                await session.commit()

            scored += 1
            logger.info("  Score: %d", result.total)

            # Rate limiting
            await asyncio.sleep(1.5)

        except Exception:
            logger.exception("  Error scoring %s", article_id)
            await asyncio.sleep(2)

    logger.info("Done: %d articles scored with V4", scored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill v4-binary scores")
    parser.add_argument("--limit", type=int, default=50, help="Max articles to score")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all articles, not just those with highlight data",
    )
    args = parser.parse_args()

    asyncio.run(backfill(args.limit, args.dry_run, args.all))


if __name__ == "__main__":
    main()
