"""Send the daily digest or weekly roundup to Telegram via nanobot.

Selects high-value articles, records exposure events, and sends a MarkdownV2
message. Exposure events are only committed after a successful send, so a
failed send leaves no trace and the articles remain eligible next run.

Usage:
    python -m tools.send_digest daily [--dry-run]
    python -m tools.send_digest weekly [--dry-run]
"""

import argparse
import asyncio
import logging
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


async def run(kind: str, dry_run: bool) -> int:
    """Build and send one digest. Returns a process exit code."""
    from app.models.article import Base, get_engine, get_session_factory
    from app.services.digest import (
        check_nanobot_health,
        compute_pipeline_health,
        deliver_digest,
        format_daily_message,
        format_weekly_message,
        record_exposures,
        select_daily_articles,
        select_weekly_articles,
    )

    # Pre-flight health check (logged, non-fatal — delivery will still be attempted)
    health_err = check_nanobot_health()
    if health_err:
        logger.warning("Pre-flight: %s", health_err)

    # Ensure the exposure_events table exists without touching FTS
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = await get_session_factory()
    async with factory() as session:
        if kind == "daily":
            pairs = await select_daily_articles(session)
        else:
            pairs = await select_weekly_articles(session)

        if not pairs and kind == "daily":
            logger.info("No qualifying articles — no digest sent")
            return 0

        items = await record_exposures(session, pairs, kind)
        if kind == "daily":
            message = format_daily_message(items)
        else:
            health = await compute_pipeline_health(session)
            message = format_weekly_message(items, health)

        if dry_run:
            await session.rollback()
            print("--- DRY RUN (no send, no exposure records) ---")
            print(message)
            return 0

        if not await deliver_digest(message):
            await session.rollback()
            logger.error("All delivery channels failed — exposure events rolled back")
            return 1

        await session.commit()
        logger.info("%s digest sent: %d article(s), exposures logged", kind, len(items))
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["daily", "weekly"], help="Which digest to send")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message without sending or recording exposures",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.kind, args.dry_run)))


if __name__ == "__main__":
    main()
