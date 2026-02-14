"""Tests for the background sync service."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.article import Article
from app.services.sync import (
    FULL_SYNC_SECONDS,
    QUICK_REFRESH_SECONDS,
    BackgroundSync,
    SyncStatus,
    get_background_sync,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_readwise_doc(
    *,
    id: str = "doc-1",
    location: str | None = "new",
    reading_progress: float | None = 0.0,
) -> MagicMock:
    """Create a mock Readwise document with the fields used by quick_refresh."""
    doc = MagicMock()
    doc.id = id
    doc.location = location
    doc.reading_progress = reading_progress
    return doc


def _make_scan_result(
    *, total_scanned: int = 10, newly_scored: int = 3
) -> MagicMock:
    """Create a mock ScanResult returned by scorer.scan_all_documents."""
    result = MagicMock()
    result.total_scanned = total_scanned
    result.newly_scored = newly_scored
    result.top_5 = []
    return result


# ---------------------------------------------------------------------------
# 1. SyncStatus dataclass
# ---------------------------------------------------------------------------


class TestSyncStatus:
    """Test the SyncStatus dataclass defaults."""

    async def test_default_values(self):
        status = SyncStatus()
        assert status.is_syncing is False
        assert status.last_sync_at is None
        assert status.last_refresh_at is None
        assert status.articles_processed == 0
        assert status.newly_scored == 0
        assert status.newly_tagged == 0
        assert status.newly_embedded == 0
        assert status.last_error is None

    async def test_scoring_version_set(self):
        from app.services.scorer import CURRENT_SCORING_VERSION

        status = SyncStatus()
        assert status.scoring_version == CURRENT_SCORING_VERSION

    async def test_mutable_fields(self):
        status = SyncStatus()
        status.is_syncing = True
        status.articles_processed = 42
        status.last_error = "something broke"
        assert status.is_syncing is True
        assert status.articles_processed == 42
        assert status.last_error == "something broke"


# ---------------------------------------------------------------------------
# 2. BackgroundSync initialization and status
# ---------------------------------------------------------------------------


class TestBackgroundSyncInit:
    """Test BackgroundSync construction and status property."""

    async def test_initial_status(self):
        sync = BackgroundSync()
        assert isinstance(sync.status, SyncStatus)
        assert sync.status.is_syncing is False
        assert sync._quick_task is None
        assert sync._full_task is None

    async def test_status_property_returns_same_instance(self):
        sync = BackgroundSync()
        s1 = sync.status
        s2 = sync.status
        assert s1 is s2


# ---------------------------------------------------------------------------
# 3. Quick refresh
# ---------------------------------------------------------------------------


class TestQuickRefresh:
    """Test the quick_refresh method."""

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_updates_article_location(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """When a document's location changed in Readwise, it should be updated in DB."""
        # Set up an article in the DB with location="new"
        async with session_factory() as session:
            article = Article(
                id="doc-1",
                title="Test Article",
                url="https://example.com/test",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        # Mock readwise returning the article with location="archive"
        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(
            return_value=[_make_readwise_doc(id="doc-1", location="archive")]
        )
        mock_get_readwise.return_value = mock_readwise

        # Mock session factory
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        updated = await sync.quick_refresh()

        assert updated == 1

        # Verify the article location was updated in the DB
        async with session_factory() as session:
            article = await session.get(Article, "doc-1")
            assert article is not None
            assert article.location == "archive"

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_no_update_when_location_unchanged(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """No update counter when location stays the same."""
        async with session_factory() as session:
            article = Article(
                id="doc-2",
                title="Unchanged Article",
                url="https://example.com/unchanged",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(
            return_value=[_make_readwise_doc(id="doc-2", location="new")]
        )
        mock_get_readwise.return_value = mock_readwise
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        updated = await sync.quick_refresh()

        assert updated == 0

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_updates_reading_progress(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Reading progress should be updated when it changes."""
        async with session_factory() as session:
            article = Article(
                id="doc-3",
                title="Progress Article",
                url="https://example.com/progress",
                location="new",
                category="article",
                reading_progress=0.0,
            )
            session.add(article)
            await session.commit()

        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(
            return_value=[_make_readwise_doc(id="doc-3", location="new", reading_progress=0.5)]
        )
        mock_get_readwise.return_value = mock_readwise
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        await sync.quick_refresh()

        async with session_factory() as session:
            article = await session.get(Article, "doc-3")
            assert article is not None
            assert article.reading_progress == 0.5

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_skips_unknown_documents(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Documents not in the DB should be silently skipped."""
        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(
            return_value=[_make_readwise_doc(id="unknown-doc", location="new")]
        )
        mock_get_readwise.return_value = mock_readwise
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        updated = await sync.quick_refresh()
        assert updated == 0

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_sets_last_refresh_at(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """last_refresh_at should be set after a successful refresh."""
        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(return_value=[])
        mock_get_readwise.return_value = mock_readwise
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        assert sync.status.last_refresh_at is None

        await sync.quick_refresh()

        assert sync.status.last_refresh_at is not None
        assert isinstance(sync.status.last_refresh_at, datetime)

    @patch("app.services.sync.get_readwise_service")
    async def test_returns_zero_on_exception(self, mock_get_readwise: MagicMock):
        """If readwise throws, quick_refresh returns 0 and does not raise."""
        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(side_effect=RuntimeError("API down"))
        mock_get_readwise.return_value = mock_readwise

        sync = BackgroundSync()
        result = await sync.quick_refresh()
        assert result == 0

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_skips_when_lock_held(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock
    ):
        """If the refresh lock is already held, returns 0 immediately."""
        sync = BackgroundSync()

        # Acquire the lock externally
        async with sync._refresh_lock:
            result = await sync.quick_refresh()
            assert result == 0

        # Readwise service should NOT have been called
        mock_get_readwise.assert_not_called()

    @patch("app.services.sync.get_session_factory")
    @patch("app.services.sync.get_readwise_service")
    async def test_multiple_documents(
        self, mock_get_readwise: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should handle multiple documents, updating only those that changed."""
        async with session_factory() as session:
            for i in range(3):
                article = Article(
                    id=f"multi-{i}",
                    title=f"Article {i}",
                    url=f"https://example.com/{i}",
                    location="new",
                    category="article",
                )
                session.add(article)
            await session.commit()

        mock_readwise = AsyncMock()
        mock_readwise.get_all_documents = AsyncMock(
            return_value=[
                _make_readwise_doc(id="multi-0", location="new"),       # unchanged
                _make_readwise_doc(id="multi-1", location="later"),     # changed
                _make_readwise_doc(id="multi-2", location="archive"),   # changed
            ]
        )
        mock_get_readwise.return_value = mock_readwise
        mock_get_factory.return_value = session_factory

        sync = BackgroundSync()
        updated = await sync.quick_refresh()
        assert updated == 2


# ---------------------------------------------------------------------------
# 4. Full sync (run_sync)
# ---------------------------------------------------------------------------


class TestRunSync:
    """Test the run_sync method."""

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_summarizer")
    @patch("app.services.sync.get_article_scorer")
    async def test_successful_full_sync(
        self,
        mock_get_scorer: MagicMock,
        mock_get_summarizer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """A successful sync should update status fields."""
        # Mock scorer
        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(
            return_value=_make_scan_result(total_scanned=15, newly_scored=5)
        )
        mock_get_scorer.return_value = mock_scorer

        # Mock summarizer (called because newly_scored > 0)
        mock_summarizer = AsyncMock()
        mock_summarizer.summarize_low_info_articles = AsyncMock(return_value=["s1", "s2"])
        mock_get_summarizer.return_value = mock_summarizer

        # Mock tagger
        mock_tagger = AsyncMock()
        mock_tagger.tag_untagged_articles = AsyncMock(return_value={"art-1": ["tag1"]})
        mock_get_tagger.return_value = mock_tagger

        sync = BackgroundSync()
        assert sync.status.is_syncing is False

        # Patch the dynamic import of get_vectorstore inside run_sync
        mock_vs = AsyncMock()
        mock_vs.embed_all_articles = AsyncMock(return_value=3)
        with patch("app.services.vectorstore.get_vectorstore", return_value=mock_vs):
            await sync.run_sync()

        assert sync.status.is_syncing is False
        assert sync.status.articles_processed == 15
        assert sync.status.newly_scored == 5
        assert sync.status.last_sync_at is not None
        assert sync.status.newly_tagged == 1
        assert sync.status.last_error is None

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_article_scorer")
    async def test_no_summarization_when_zero_newly_scored(
        self,
        mock_get_scorer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """Summarizer should NOT be called when newly_scored is 0."""
        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(
            return_value=_make_scan_result(total_scanned=10, newly_scored=0)
        )
        mock_get_scorer.return_value = mock_scorer

        mock_tagger = AsyncMock()
        mock_tagger.tag_untagged_articles = AsyncMock(return_value={})
        mock_get_tagger.return_value = mock_tagger

        sync = BackgroundSync()

        with patch("app.services.sync.get_summarizer") as mock_get_summarizer:
            import app.services.vectorstore as vs_mod

            mock_vs = AsyncMock()
            mock_vs.embed_all_articles = AsyncMock(return_value=0)
            with patch.object(vs_mod, "get_vectorstore", return_value=mock_vs):
                await sync.run_sync()
            mock_get_summarizer.assert_not_called()

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_article_scorer")
    async def test_sync_error_sets_last_error(
        self,
        mock_get_scorer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """If scanner throws, status.last_error should be set."""
        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(
            side_effect=RuntimeError("Readwise API timeout")
        )
        mock_get_scorer.return_value = mock_scorer

        sync = BackgroundSync()
        await sync.run_sync()

        assert sync.status.is_syncing is False
        assert sync.status.last_error == "Readwise API timeout"

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_article_scorer")
    async def test_sync_clears_previous_error(
        self,
        mock_get_scorer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """A successful sync should clear any previous error."""
        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(
            return_value=_make_scan_result(total_scanned=5, newly_scored=0)
        )
        mock_get_scorer.return_value = mock_scorer

        mock_tagger = AsyncMock()
        mock_tagger.tag_untagged_articles = AsyncMock(return_value={})
        mock_get_tagger.return_value = mock_tagger

        sync = BackgroundSync()
        sync._status.last_error = "previous error"

        import app.services.vectorstore as vs_mod

        mock_vs = AsyncMock()
        mock_vs.embed_all_articles = AsyncMock(return_value=0)
        with patch.object(vs_mod, "get_vectorstore", return_value=mock_vs):
            await sync.run_sync()

        assert sync.status.last_error is None

    @patch("app.services.sync.get_article_scorer")
    async def test_sync_skipped_when_lock_held(self, mock_get_scorer: MagicMock):
        """If the sync lock is already held, run_sync returns immediately."""
        sync = BackgroundSync()

        async with sync._full_sync_lock:
            await sync.run_sync()

        # Scorer should never have been called
        mock_get_scorer.assert_not_called()

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_article_scorer")
    async def test_is_syncing_set_during_sync(
        self,
        mock_get_scorer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """is_syncing should be True during execution, False after."""
        observed_states: list[bool] = []

        async def capture_state(limit: int = 200) -> MagicMock:
            observed_states.append(sync.status.is_syncing)
            return _make_scan_result(total_scanned=0, newly_scored=0)

        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(side_effect=capture_state)
        mock_get_scorer.return_value = mock_scorer

        mock_tagger = AsyncMock()
        mock_tagger.tag_untagged_articles = AsyncMock(return_value={})
        mock_get_tagger.return_value = mock_tagger

        sync = BackgroundSync()

        import app.services.vectorstore as vs_mod

        mock_vs = AsyncMock()
        mock_vs.embed_all_articles = AsyncMock(return_value=0)
        with patch.object(vs_mod, "get_vectorstore", return_value=mock_vs):
            await sync.run_sync()

        # During scan, is_syncing should have been True
        assert observed_states == [True]
        # After sync, it should be False
        assert sync.status.is_syncing is False

    @patch("app.services.sync.get_tagger")
    @patch("app.services.sync.get_summarizer")
    @patch("app.services.sync.get_article_scorer")
    async def test_vectorstore_failure_is_nonfatal(
        self,
        mock_get_scorer: MagicMock,
        mock_get_summarizer: MagicMock,
        mock_get_tagger: MagicMock,
    ):
        """Vectorstore embedding failure should not fail the overall sync."""
        mock_scorer = AsyncMock()
        mock_scorer.scan_all_documents = AsyncMock(
            return_value=_make_scan_result(total_scanned=5, newly_scored=2)
        )
        mock_get_scorer.return_value = mock_scorer

        mock_summarizer = AsyncMock()
        mock_summarizer.summarize_low_info_articles = AsyncMock(return_value=[])
        mock_get_summarizer.return_value = mock_summarizer

        mock_tagger = AsyncMock()
        mock_tagger.tag_untagged_articles = AsyncMock(return_value={})
        mock_get_tagger.return_value = mock_tagger

        sync = BackgroundSync()

        import app.services.vectorstore as vs_mod

        mock_vs = AsyncMock()
        mock_vs.embed_all_articles = AsyncMock(side_effect=RuntimeError("Qdrant connection error"))
        with patch.object(vs_mod, "get_vectorstore", return_value=mock_vs):
            await sync.run_sync()

        # Sync should still succeed
        assert sync.status.is_syncing is False
        assert sync.status.last_error is None
        assert sync.status.articles_processed == 5


# ---------------------------------------------------------------------------
# 5. Lifecycle: start/stop periodic
# ---------------------------------------------------------------------------


class TestPeriodicLifecycle:
    """Test start_periodic and stop_periodic."""

    async def test_start_creates_tasks(self):
        sync = BackgroundSync()
        assert sync._quick_task is None
        assert sync._full_task is None

        # Patch the loops so they don't actually run
        with (
            patch.object(sync, "_quick_refresh_loop", new_callable=AsyncMock) as mock_quick,
            patch.object(sync, "_full_sync_loop", new_callable=AsyncMock) as mock_full,
        ):
            sync.start_periodic()

            assert sync._quick_task is not None
            assert sync._full_task is not None

            # Clean up the tasks
            sync.stop_periodic()

    async def test_stop_cancels_tasks(self):
        sync = BackgroundSync()

        with (
            patch.object(sync, "_quick_refresh_loop", new_callable=AsyncMock),
            patch.object(sync, "_full_sync_loop", new_callable=AsyncMock),
        ):
            sync.start_periodic()
            quick_task = sync._quick_task
            full_task = sync._full_task

            sync.stop_periodic()

            # Let the event loop process the cancellations
            await asyncio.sleep(0)

            assert sync._quick_task is None
            assert sync._full_task is None
            assert quick_task is not None
            assert quick_task.cancelled()
            assert full_task is not None
            assert full_task.cancelled()

    async def test_stop_when_no_tasks(self):
        """stop_periodic when no tasks are running should not raise."""
        sync = BackgroundSync()
        sync.stop_periodic()
        assert sync._quick_task is None
        assert sync._full_task is None

    async def test_start_idempotent_when_tasks_running(self):
        """Calling start_periodic again should not create duplicate tasks."""
        sync = BackgroundSync()

        with (
            patch.object(sync, "_quick_refresh_loop", new_callable=AsyncMock),
            patch.object(sync, "_full_sync_loop", new_callable=AsyncMock),
        ):
            sync.start_periodic()
            quick1 = sync._quick_task
            full1 = sync._full_task

            sync.start_periodic()
            # Same tasks should be reused
            assert sync._quick_task is quick1
            assert sync._full_task is full1

            sync.stop_periodic()

    async def test_start_replaces_done_tasks(self):
        """If a task has finished (done), start_periodic should create a new one."""
        sync = BackgroundSync()

        with (
            patch.object(sync, "_quick_refresh_loop", new_callable=AsyncMock),
            patch.object(sync, "_full_sync_loop", new_callable=AsyncMock),
        ):
            sync.start_periodic()
            old_quick = sync._quick_task
            old_full = sync._full_task

            # Cancel to simulate done tasks
            sync.stop_periodic()

            # Now start again - since tasks are None, new ones should be created
            sync.start_periodic()
            assert sync._quick_task is not None
            assert sync._full_task is not None

            sync.stop_periodic()


# ---------------------------------------------------------------------------
# 6. Trigger sync
# ---------------------------------------------------------------------------


class TestTriggerSync:
    """Test the trigger_sync method."""

    async def test_trigger_creates_task(self):
        """trigger_sync should create a background task for run_sync."""
        sync = BackgroundSync()

        with patch.object(sync, "run_sync", new_callable=AsyncMock) as mock_run:
            sync.trigger_sync()
            # Give the event loop a chance to start the task
            await asyncio.sleep(0.01)
            mock_run.assert_called_once()

    async def test_trigger_ignored_when_lock_held(self):
        """trigger_sync should be a no-op when sync is already running."""
        sync = BackgroundSync()

        async with sync._full_sync_lock:
            with patch.object(sync, "run_sync", new_callable=AsyncMock) as mock_run:
                sync.trigger_sync()
                await asyncio.sleep(0.01)
                mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Singleton accessor
# ---------------------------------------------------------------------------


class TestGetBackgroundSync:
    """Test the singleton accessor."""

    async def test_returns_instance(self):
        import app.services.sync as sync_mod

        original = sync_mod._background_sync
        try:
            sync_mod._background_sync = None
            instance = get_background_sync()
            assert isinstance(instance, BackgroundSync)
        finally:
            sync_mod._background_sync = original

    async def test_returns_same_instance(self):
        import app.services.sync as sync_mod

        original = sync_mod._background_sync
        try:
            sync_mod._background_sync = None
            instance1 = get_background_sync()
            instance2 = get_background_sync()
            assert instance1 is instance2
        finally:
            sync_mod._background_sync = original


# ---------------------------------------------------------------------------
# 8. Constants
# ---------------------------------------------------------------------------


class TestSyncConstants:
    """Test sync interval constants."""

    async def test_quick_refresh_interval(self):
        assert QUICK_REFRESH_SECONDS == 30

    async def test_full_sync_interval(self):
        assert FULL_SYNC_SECONDS == 600  # 10 minutes


# ---------------------------------------------------------------------------
# 9. Loop methods
# ---------------------------------------------------------------------------


class TestSyncLoops:
    """Test the internal loop methods."""

    async def test_quick_refresh_loop_calls_refresh(self):
        """_quick_refresh_loop should call quick_refresh and sleep."""
        sync = BackgroundSync()
        call_count = 0

        async def mock_refresh() -> int:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()
            return 0

        with (
            patch.object(sync, "quick_refresh", side_effect=mock_refresh),
            patch("app.services.sync.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(asyncio.CancelledError):
                await sync._quick_refresh_loop()

            assert call_count == 2
            mock_sleep.assert_called_with(QUICK_REFRESH_SECONDS)

    async def test_full_sync_loop_calls_run_sync(self):
        """_full_sync_loop should call run_sync and sleep."""
        sync = BackgroundSync()
        call_count = 0

        async def mock_sync() -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch.object(sync, "run_sync", side_effect=mock_sync),
            patch("app.services.sync.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(asyncio.CancelledError):
                await sync._full_sync_loop()

            assert call_count == 2
            mock_sleep.assert_called_with(FULL_SYNC_SECONDS)

    async def test_quick_refresh_loop_handles_exceptions(self):
        """Exceptions in quick_refresh should be caught and loop continues."""
        sync = BackgroundSync()
        call_count = 0

        async def mock_refresh() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient error")
            if call_count >= 2:
                raise asyncio.CancelledError()
            return 0

        with (
            patch.object(sync, "quick_refresh", side_effect=mock_refresh),
            patch("app.services.sync.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(asyncio.CancelledError):
                await sync._quick_refresh_loop()

            # Should have looped at least twice (first error, second cancel)
            assert call_count == 2

    async def test_full_sync_loop_handles_exceptions(self):
        """Exceptions in run_sync should be caught and loop continues."""
        sync = BackgroundSync()
        call_count = 0

        async def mock_sync() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient error")
            if call_count >= 2:
                raise asyncio.CancelledError()

        with (
            patch.object(sync, "run_sync", side_effect=mock_sync),
            patch("app.services.sync.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(asyncio.CancelledError):
                await sync._full_sync_loop()

            assert call_count == 2
