"""Sync archived articles from Readwise: score with v2+v3 and fetch highlight counts.

Two-phase pipeline:
  1. Score phase  — fetches archived docs via ReadwiseService, creates Article
                    rows, scores with both v2 and v3 strategies (via ArticleScorer).
  2. Highlight phase — fetches highlight counts from Readwise v2 export API and
                       stores them in Article.num_highlights.

Usage:
    python -m tools.sync_archive [--limit N] [--skip-scoring] [--skip-highlights]
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
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


async def run_scoring(limit: int) -> None:
    """Phase 1: sync and score archived documents."""
    from app.models.article import init_db
    from app.services.scorer import get_article_scorer

    await init_db()
    scorer = get_article_scorer()

    logger.info("Scanning up to %d archived documents...", limit)
    result = await scorer.scan_archived_documents(limit=limit)
    logger.info(
        "Done: scanned %d, newly scored %d",
        result.total_scanned,
        result.newly_scored,
    )


def run_highlights() -> None:
    """Phase 2: fetch highlight data and update Article calibration columns."""
    from tools.cal_data import fetch_highlights

    logger.info("Fetching highlight data from Readwise v2 export API...")
    highlights = fetch_highlights(force=True)

    db_path = _PROJECT_ROOT / "reader_triage.db"
    conn = sqlite3.connect(str(db_path))

    updated = 0
    for article_id, hl_data in highlights.items():
        cur = conn.execute(
            "UPDATE articles SET num_highlights = ?, highlighted_words = ? WHERE id = ?",
            (hl_data["count"], hl_data["words"], article_id),
        )
        if cur.rowcount:
            updated += 1

    conn.commit()
    conn.close()

    with_hl = sum(1 for d in highlights.values() if d["count"] > 0)
    total_words = sum(d["words"] for d in highlights.values())
    logger.info(
        "Updated %d articles (%d with highlights, %d total highlighted words)",
        updated,
        with_hl,
        total_words,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync archived articles: score + fetch highlights")
    parser.add_argument(
        "--limit", type=int, default=500, help="Max articles to score (default 500)"
    )
    parser.add_argument("--skip-scoring", action="store_true", help="Skip the scoring phase")
    parser.add_argument("--skip-highlights", action="store_true", help="Skip the highlights phase")
    args = parser.parse_args()

    if not args.skip_scoring:
        asyncio.run(run_scoring(args.limit))

    if not args.skip_highlights:
        run_highlights()

    logger.info("All done.")


if __name__ == "__main__":
    main()
