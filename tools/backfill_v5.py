"""Convert stored v2-categorical scores to v5-reweighted, arithmetically.

v5 keeps the same four dimension subscores but totals only Quotability +
Applicable Insight (see app/services/scoring_strategy.py:reweight_total).
Because the subscores are already stored, this needs zero LLM calls: it
recomputes info_score, priority_score, and skip flags in place and stamps
scoring_version = 'v5-reweighted'.

Usage:
    python -m tools.backfill_v5 [--dry-run]
"""

import argparse
import asyncio
import logging
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


async def run(dry_run: bool) -> int:
    from sqlalchemy import select

    from app.models.article import ArticleScore, get_session_factory
    from app.services.scoring_strategy import reweight_total

    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(ArticleScore).where(ArticleScore.scoring_version == "v2-categorical")
        )
        scores = list(result.scalars().all())
        logger.info("Found %d v2-categorical scores to convert", len(scores))

        changed_tier = 0
        for score in scores:
            new_total = float(reweight_total(score.specificity_score, score.actionability_score))
            old_total = score.info_score
            if (old_total >= 60) != (new_total >= 60):
                changed_tier += 1

            score.info_score = new_total
            score.priority_score = new_total + score.author_boost
            score.skip_recommended = new_total < 30
            score.skip_reason = "Low information content" if new_total < 30 else None
            score.scoring_version = "v5-reweighted"
            score.priority_computed_at = datetime.now()

        logger.info("%d articles change High-tier membership", changed_tier)

        if dry_run:
            await session.rollback()
            logger.info("Dry run — rolled back")
            return 0

        await session.commit()
        logger.info("Converted %d scores to v5-reweighted", len(scores))
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.dry_run)))


if __name__ == "__main__":
    main()
