"""Background sync service for periodic article scanning."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

import httpx

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

# Alert when sync fails this many times in a row
FAILURE_ALERT_THRESHOLD = 3


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks: log exceptions that would otherwise be lost."""
    if not task.cancelled() and task.exception():
        logger.error("Background task %s died: %s", task.get_name(), task.exception())


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
    consecutive_failures: int = 0
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
        self._failure_alert_sent = False

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

    async def _send_failure_alert(self) -> None:
        """Alert when consecutive sync failures cross the threshold."""
        from app.config import get_settings

        count = self._status.consecutive_failures
        error = self._status.last_error or "unknown error"
        title = f"Sync failing: {count} consecutive failures"
        body = f"Last error: {error}"
        severity = "critical" if count >= 6 else "warning"

        settings = get_settings()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    settings.alert_webhook_url,
                    json={
                        "source": "reader-triage",
                        "severity": severity,
                        "title": title,
                        "body": body,
                        "alert_id": "reader-triage-sync-failure",
                    },
                )
                data = resp.json()
                if data.get("ok"):
                    logger.info("Sync failure alert sent via webhook (streak=%d)", count)
                    self._failure_alert_sent = True
                    return
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)

        # Fallback to ntfy
        from app.services.digest import send_ntfy

        if await send_ntfy(f"{title}\n\n{body}", title="Reader Triage: sync failures"):
            self._failure_alert_sent = True
        else:
            logger.error("All alert channels failed for sync failure notification")

    async def _send_failure_recovery(self) -> None:
        """Send recovery notice when sync succeeds after prior failures."""
        from app.config import get_settings

        settings = get_settings()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    settings.alert_webhook_url,
                    json={
                        "source": "reader-triage",
                        "severity": "info",
                        "title": "Sync recovered",
                        "body": "Background sync is working again",
                        "alert_id": "reader-triage-sync-failure",
                        "resolved": True,
                    },
                )
                data = resp.json()
                if data.get("ok"):
                    logger.info("Sync recovery notice sent via webhook")
                    return
        except Exception as e:
            logger.warning("Webhook recovery failed: %s", e)

        from app.services.digest import send_ntfy

        await send_ntfy(
            "Background sync recovered and is working again",
            title="Reader Triage: sync recovered",
        )

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

                # Send recovery if we were in a failure streak
                if self._failure_alert_sent:
                    await self._send_failure_recovery()
                    self._failure_alert_sent = False
                self._status.consecutive_failures = 0

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
                    logger.warning("Vector embedding failed (non-fatal): %s", e, exc_info=True)

            except Exception as e:
                self._status.consecutive_failures += 1
                self._status.last_error = str(e)
                logger.exception(
                    "Background sync failed (streak=%d): %s", self._status.consecutive_failures, e
                )
                if self._status.consecutive_failures >= FAILURE_ALERT_THRESHOLD:
                    await self._send_failure_alert()
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
            self._quick_task = asyncio.create_task(
                self._quick_refresh_loop(), name="quick_refresh_loop"
            )
            self._quick_task.add_done_callback(_log_task_exception)
            logger.info("Started quick refresh loop (interval=%ds)", QUICK_REFRESH_SECONDS)

        if self._full_task is None or self._full_task.done():
            self._full_task = asyncio.create_task(self._full_sync_loop(), name="full_sync_loop")
            self._full_task.add_done_callback(_log_task_exception)
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
        task = asyncio.create_task(self.run_sync(), name="triggered_sync")
        task.add_done_callback(_log_task_exception)


# Singleton instance
_background_sync: BackgroundSync | None = None


def get_background_sync() -> BackgroundSync:
    """Get or create the background sync singleton."""
    global _background_sync
    if _background_sync is None:
        _background_sync = BackgroundSync()
    return _background_sync
