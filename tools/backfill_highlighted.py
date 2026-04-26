"""Backfill V2 scores for highlighted articles that are missing scores.

Targets the 100+ archived/highlighted articles that were never scored by V2,
so calibration analysis has a proper dataset.

Usage:
    uv run python -m tools.backfill_highlighted [--limit N] [--dry-run]
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

DB_PATH = _PROJECT_ROOT / "reader_triage.db"


def get_unscored_highlighted_ids() -> list[str]:
    """Find article IDs that have highlights but no V2 score."""
    from tools.cal_data import fetch_highlights

    highlights = fetch_highlights()
    has_hl = {aid for aid, d in highlights.items() if d["count"] > 0}

    conn = sqlite3.connect(str(DB_PATH))
    scored = {r[0] for r in conn.execute("SELECT article_id FROM article_scores").fetchall()}
    conn.close()

    unscored = has_hl - scored
    logger.info("Found %d highlighted articles without V2 scores", len(unscored))
    return sorted(unscored)


async def backfill(limit: int, dry_run: bool) -> None:
    from app.models.article import ArticleScore, get_session_factory, init_db
    from app.services.readwise import get_readwise_service
    from app.services.scorer import _get_default_strategy

    await init_db()
    strategy = _get_default_strategy()
    readwise = get_readwise_service()

    ids = get_unscored_highlighted_ids()
    if limit:
        ids = ids[:limit]

    logger.info("Will score %d articles (dry_run=%s)", len(ids), dry_run)
    if dry_run:
        for aid in ids:
            logger.info("  Would score: %s", aid)
        return

    factory = await get_session_factory()
    scored = 0
    failed = 0

    for i, article_id in enumerate(ids):
        logger.info("[%d/%d] Scoring %s...", i + 1, len(ids), article_id)

        # Fetch full content from Readwise
        doc = await readwise.get_document(article_id, with_content=True)
        if doc is None or not doc.content:
            logger.warning("  Could not fetch content, skipping")
            failed += 1
            continue

        # Score with V2 strategy
        try:
            result = await strategy.score(
                title=doc.title,
                author=doc.author or "Unknown",
                content=doc.content[:15000],
                word_count=doc.word_count or 0,
                content_type_hint=doc.category or "article",
                entity_id=article_id,
            )
        except Exception as e:
            logger.error("  Scoring failed: %s", e)
            failed += 1
            continue

        if result is None:
            logger.warning("  Strategy returned None, skipping")
            failed += 1
            continue

        # Save to DB
        async with factory() as session:
            article_score = ArticleScore(
                article_id=article_id,
                info_score=result.total,
                specificity_score=result.specificity,
                novelty_score=result.novelty,
                depth_score=result.depth,
                actionability_score=result.actionability,
                score_reasons=json.dumps(
                    [
                        result.specificity_reason,
                        result.novelty_reason,
                        result.depth_reason,
                        result.actionability_reason,
                    ]
                ),
                overall_assessment=result.overall_assessment,
                priority_score=float(result.total),
                author_boost=0.0,
                content_fetch_failed=result.content_fetch_failed,
                skip_recommended=result.total < 30,
                skip_reason="Low information content" if result.total < 30 else None,
                model_used="claude-sonnet-4-5-20250929",
                scoring_version=strategy.version,
                scored_at=datetime.now(),
                priority_computed_at=datetime.now(),
            )
            session.add(article_score)
            await session.commit()

        scored += 1
        logger.info(
            "  Score: %d (Q=%d S=%d A=%d I=%d)",
            result.total,
            result.specificity,
            result.novelty,
            result.depth,
            result.actionability,
        )

    logger.info("Done: scored %d, failed %d out of %d", scored, failed, len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill V2 scores for highlighted articles")
    parser.add_argument("--limit", type=int, default=0, help="Max articles to score (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be scored")
    args = parser.parse_args()
    asyncio.run(backfill(args.limit, args.dry_run))


if __name__ == "__main__":
    main()
