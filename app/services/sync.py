"""Background sync service for periodic article scanning."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from app.models.article import Article, get_session_factory
from app.services.readwise import get_readwise_service
from app.services.scorer import CURRENT_SCORING_VERSION, get_article_scorer
from app.services.summarizer import get_summarizer
from app.services.tagger import get_tagger

logger = logging.getLogger(__name__)

# Quick refresh: update article locations every 30 seconds
QUICK_REFRESH_SECONDS = 30

# Full sync: score/summarize/tag new articles every 10 minutes
FULL_SYNC_SECONDS = 10 * 60


@dataclass
class SyncStatus:
    """Tracks the state of background sync operations."""

    is_syncing: bool = False
    last_sync_at: datetime | None = None
    last_refresh_at: datetime | None = None
    articles_processed: int = 0
    newly_scored: int = 0
    newly_tagged: int = 0
    newly_embedded: int = 0
    last_error: str | None = None
    scoring_version: str = field(default_factory=lambda: CURRENT_SCORING_VERSION)


class BackgroundSync:
    """Manages periodic background sync of Readwise articles.

    Two sync modes:
    - Quick refresh (every 30s): Updates article locations only.
      This ensures archived articles disappear promptly from the dashboard.
    - Full sync (every 10m or on-demand): Scores, summarizes, and tags new articles.
    """

    def __init__(self):
        self._status = SyncStatus()
        self._full_sync_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._quick_task: asyncio.Task | None = None
        self._full_task: asyncio.Task | None = None

    @property
    def status(self) -> SyncStatus:
        """Get current sync status."""
        return self._status

    async def quick_refresh(self) -> int:
        """Lightweight location refresh - updates article locations from Readwise.

        Returns the number of articles whose location changed.
        """
        if self._refresh_lock.locked():
            return 0

        async with self._refresh_lock:
            try:
                readwise = get_readwise_service()
                documents = await readwise.get_all_documents(
                    limit=300, with_content=False, exclude_archived=False
                )

                # Build a lookup of doc_id -> (location, reading_progress)
                doc_updates: dict[str, tuple[str | None, float | None]] = {}
                for doc in documents:
                    doc_updates[doc.id] = (doc.location, doc.reading_progress)

                factory = await get_session_factory()
                updated = 0
                async with factory() as session:
                    for doc_id, (location, progress) in doc_updates.items():
                        article = await session.get(Article, doc_id)
                        if article is None:
                            continue

                        if article.location != location:
                            logger.info(
                                "Article %s location: %s -> %s",
                                doc_id,
                                article.location,
                                location,
                            )
                            article.location = location
                            article.last_synced_at = datetime.now()
                            updated += 1

                        if progress is not None and article.reading_progress != progress:
                            article.reading_progress = progress

                    await session.commit()
                    if updated > 0:
                        logger.info("Quick refresh: updated %d article locations", updated)

                self._status.last_refresh_at = datetime.now()
                return updated

            except Exception as e:
                logger.exception("Quick refresh failed: %s", e)
                return 0

    async def run_sync(self) -> None:
        """Run a full sync cycle. Uses a lock to prevent concurrent syncs."""
        if self._full_sync_lock.locked():
            logger.info("Full sync already in progress, skipping")
            return

        async with self._full_sync_lock:
            self._status.is_syncing = True
            self._status.last_error = None
            try:
                logger.info("Starting full background sync")
                scorer = get_article_scorer()

                # Scan all non-archived documents
                result = await scorer.scan_all_documents(limit=200)

                self._status.articles_processed = result.total_scanned
                self._status.newly_scored = result.newly_scored
                self._status.last_sync_at = datetime.now()

                logger.info(
                    "Full sync complete: scanned=%d, newly_scored=%d",
                    result.total_scanned,
                    result.newly_scored,
                )

                # Generate summaries for low-info articles
                if result.newly_scored > 0:
                    summarizer = get_summarizer()
                    summaries = await summarizer.summarize_low_info_articles()
                    logger.info("Generated %d new summaries", len(summaries))

                # Tag untagged articles
                tagger = get_tagger()
                tag_results = await tagger.tag_untagged_articles()
                self._status.newly_tagged = len(tag_results)
                logger.info("Tagged %d articles", len(tag_results))

                # Embed new articles into vector store
                try:
                    from app.services.vectorstore import get_vectorstore

                    vectorstore = get_vectorstore()
                    newly_embedded = await vectorstore.embed_all_articles()
                    self._status.newly_embedded = newly_embedded
                    logger.info("Embedded %d new articles", newly_embedded)
                except Exception as e:
                    logger.warning("Vector embedding failed (non-fatal): %s", e)

            except Exception as e:
                self._status.last_error = str(e)
                logger.exception("Background sync failed: %s", e)
            finally:
                self._status.is_syncing = False

    async def _quick_refresh_loop(self) -> None:
        """Run quick location refresh on a fast schedule."""
        while True:
            try:
                await self.quick_refresh()
            except Exception:
                logger.exception("Unexpected error in quick refresh loop")
            await asyncio.sleep(QUICK_REFRESH_SECONDS)

    async def _full_sync_loop(self) -> None:
        """Run full sync on a slower schedule."""
        while True:
            try:
                await self.run_sync()
            except Exception:
                logger.exception("Unexpected error in full sync loop")
            await asyncio.sleep(FULL_SYNC_SECONDS)

    def start_periodic(self) -> None:
        """Start both periodic sync tasks."""
        if self._quick_task is None or self._quick_task.done():
            self._quick_task = asyncio.create_task(self._quick_refresh_loop())
            logger.info("Started quick refresh loop (interval=%ds)", QUICK_REFRESH_SECONDS)

        if self._full_task is None or self._full_task.done():
            self._full_task = asyncio.create_task(self._full_sync_loop())
            logger.info("Started full sync loop (interval=%ds)", FULL_SYNC_SECONDS)

    def stop_periodic(self) -> None:
        """Stop all periodic sync tasks."""
        for task, name in [
            (self._quick_task, "quick refresh"),
            (self._full_task, "full sync"),
        ]:
            if task is not None and not task.done():
                task.cancel()
                logger.info("Stopped %s loop", name)
        self._quick_task = None
        self._full_task = None

    def trigger_sync(self) -> None:
        """Trigger an immediate (non-blocking) full sync.

        If a sync is already running, this is a no-op.
        """
        if self._full_sync_lock.locked():
            logger.info("Full sync already in progress, trigger ignored")
            return
        asyncio.create_task(self.run_sync())


# Singleton instance
_background_sync: BackgroundSync | None = None


def get_background_sync() -> BackgroundSync:
    """Get or create the background sync singleton."""
    global _background_sync
    if _background_sync is None:
        _background_sync = BackgroundSync()
    return _background_sync
