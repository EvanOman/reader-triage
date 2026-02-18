"""Tests for podcast API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.article import Base
from app.models.podcast import (
    Podcast,
    PodcastEpisode,
    PodcastEpisodeScore,
    PodcastEpisodeTag,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_engine():
    """Create a fresh in-memory SQLite engine for API integration tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts "
                "USING fts5(article_id UNINDEXED, title, author, content)"
            )
        )
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session_factory(test_engine):
    """Create a session factory bound to the test engine."""
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture
async def patched_app(test_engine, test_session_factory):
    """Patch the app module globals so all routes use the test DB."""
    import app.models.article as article_mod

    original_engine = article_mod._engine
    original_factory = article_mod._session_factory

    article_mod._engine = test_engine
    article_mod._session_factory = test_session_factory

    mock_sync = MagicMock()
    mock_sync.start_periodic = MagicMock()
    mock_sync.stop_periodic = MagicMock()
    mock_sync.status = MagicMock(
        is_syncing=False,
        last_sync_at=None,
        articles_processed=0,
        newly_scored=0,
        newly_tagged=0,
        scoring_version="test",
        last_error=None,
    )

    with (
        patch("app.main.init_db", new_callable=AsyncMock),
        patch("app.main.rebuild_fts_index", new_callable=AsyncMock),
        patch("app.main.get_background_sync", return_value=mock_sync),
    ):
        from app.main import app

        yield app

    article_mod._engine = original_engine
    article_mod._session_factory = original_factory


@pytest.fixture
async def client(patched_app):
    """Async HTTP test client using ASGI transport."""
    transport = httpx.ASGITransport(app=patched_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def populated_podcast_db(test_session_factory):
    """Insert sample podcast data into the test DB."""
    async with test_session_factory() as session:
        podcast = Podcast(
            title="Test Podcast",
            feed_url="https://example.com/feed.rss",
            youtube_channel_id="UC123",
            youtube_channel_name="Test Channel",
            mapping_confirmed=True,
        )
        session.add(podcast)
        await session.flush()

        episode = PodcastEpisode(
            podcast_id=podcast.id,
            guid="ep-001",
            title="Great Episode",
            audio_url="https://example.com/ep.mp3",
            duration_seconds=3600,
            transcript="Test transcript content.",
            transcript_source="youtube",
            youtube_video_id="vid123",
            status="scored",
        )
        session.add(episode)
        await session.flush()

        score = PodcastEpisodeScore(
            episode_id=episode.id,
            specificity_score=20,
            novelty_score=18,
            depth_score=22,
            actionability_score=15,
            score_reasons='["Good quotes", "Novel framing"]',
            overall_assessment="Excellent episode.",
            model_used="claude-sonnet-4-20250514",
            scoring_version="v2-categorical",
        )
        tag = PodcastEpisodeTag(
            episode_id=episode.id,
            tag_slug="ai-safety",
            tagging_version="v1",
        )
        session.add(score)
        session.add(tag)
        await session.commit()


# ---------------------------------------------------------------------------
# 1. List podcasts
# ---------------------------------------------------------------------------


class TestListPodcasts:
    async def test_list_empty(self, client):
        response = await client.get("/api/podcasts/")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_with_data(self, client, populated_podcast_db):
        response = await client.get("/api/podcasts/")
        assert response.status_code == 200
        podcasts = response.json()
        assert len(podcasts) == 1
        assert podcasts[0]["title"] == "Test Podcast"
        assert podcasts[0]["episode_count"] == 1


# ---------------------------------------------------------------------------
# 2. Podcast stats
# ---------------------------------------------------------------------------


class TestPodcastStats:
    async def test_stats_empty(self, client):
        response = await client.get("/api/podcasts/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_episodes"] == 0
        assert data["scored_episodes"] == 0

    async def test_stats_with_data(self, client, populated_podcast_db):
        response = await client.get("/api/podcasts/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_episodes"] == 1
        assert data["scored_episodes"] == 1
        assert data["high_value_count"] == 1  # 20+18+22+15 = 75 >= 60


# ---------------------------------------------------------------------------
# 3. List episodes
# ---------------------------------------------------------------------------


class TestListEpisodes:
    async def test_list_empty(self, client):
        response = await client.get("/api/podcasts/episodes")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_with_data(self, client, populated_podcast_db):
        response = await client.get("/api/podcasts/episodes")
        assert response.status_code == 200
        episodes = response.json()
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["title"] == "Great Episode"
        assert ep["info_score"] == 75
        assert "ai-safety" in ep["tags"]


# ---------------------------------------------------------------------------
# 4. Sync status
# ---------------------------------------------------------------------------


class TestSyncStatus:
    async def test_sync_status_empty(self, client):
        response = await client.get("/api/podcasts/sync-status")
        assert response.status_code == 200
        data = response.json()
        assert data["total_episodes"] == 0

    async def test_sync_status_with_data(self, client, populated_podcast_db):
        response = await client.get("/api/podcasts/sync-status")
        assert response.status_code == 200
        data = response.json()
        assert data["total_episodes"] == 1
        assert data["scored"] == 1


# ---------------------------------------------------------------------------
# 5. Get episode detail
# ---------------------------------------------------------------------------


class TestGetEpisode:
    async def test_episode_not_found(self, client):
        response = await client.get("/api/podcasts/episodes/9999")
        assert response.status_code == 404

    async def test_episode_found(self, client, populated_podcast_db):
        # First, get the episode list to find the ID
        list_resp = await client.get("/api/podcasts/episodes")
        episodes = list_resp.json()
        assert len(episodes) > 0
        episode_id = episodes[0]["id"]

        response = await client.get(f"/api/podcasts/episodes/{episode_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Great Episode"
        assert "transcript_preview" in data


# ---------------------------------------------------------------------------
# 6. Delete podcast
# ---------------------------------------------------------------------------


class TestDeletePodcast:
    async def test_delete_not_found(self, client):
        response = await client.delete("/api/podcasts/9999")
        assert response.status_code == 404

    async def test_delete_success(self, client, test_session_factory):
        """Delete a podcast with no scored episodes (no cascade FK issues)."""
        async with test_session_factory() as session:
            podcast = Podcast(
                title="Delete Me",
                feed_url="https://example.com/delete.rss",
            )
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-delete",
                title="Deletable Episode",
                status="pending",
            )
            session.add(episode)
            await session.commit()
            podcast_id = podcast.id

        response = await client.delete(f"/api/podcasts/{podcast_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify it's gone
        list_resp = await client.get("/api/podcasts/")
        # Should not contain the deleted podcast
        assert all(p["title"] != "Delete Me" for p in list_resp.json())
