"""Tests for podcast tagger service."""

from unittest.mock import AsyncMock, patch

from app.models.podcast import Podcast, PodcastEpisode, PodcastEpisodeTag
from app.services.podcast_tagger import PodcastTagger
from app.services.tagger import CURRENT_TAGGING_VERSION

# ---------------------------------------------------------------------------
# 1. tag_episode
# ---------------------------------------------------------------------------


class TestTagEpisode:
    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_success(self, mock_factory, session_factory):
        """Test tagging an episode inserts tags into the DB."""
        mock_factory.return_value = session_factory

        # Insert test data
        async with session_factory() as session:
            podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed")
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-1",
                title="AI Safety Discussion",
                transcript="A discussion about AI safety and alignment research.",
                status="scored",
            )
            session.add(episode)
            await session.commit()
            episode_id = episode.id

        with (
            patch("app.services.podcast_tagger.Anthropic"),
            patch(
                "app.services.podcast_tagger.classify_content",
                return_value=(
                    ["ai-safety", "agi-scaling"],
                    ("claude-sonnet-4-20250514", 500, 50),
                ),
            ),
            patch("app.services.podcast_tagger.log_usage", new_callable=AsyncMock),
        ):
            tagger = PodcastTagger()
            result = await tagger.tag_episode(episode_id)

        assert "ai-safety" in result
        assert "agi-scaling" in result

    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_not_found(self, mock_factory, session_factory):
        """Test tagging non-existent episode returns empty list."""
        mock_factory.return_value = session_factory

        with patch("app.services.podcast_tagger.Anthropic"):
            tagger = PodcastTagger()
        result = await tagger.tag_episode(9999)
        assert result == []

    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_no_transcript(self, mock_factory, session_factory):
        """Episode without transcript returns empty list."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed2")
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-no-transcript",
                title="No Transcript Episode",
                transcript=None,
                status="scored",
            )
            session.add(episode)
            await session.commit()
            episode_id = episode.id

        with patch("app.services.podcast_tagger.Anthropic"):
            tagger = PodcastTagger()
        result = await tagger.tag_episode(episode_id)
        assert result == []

    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_skips_already_tagged(self, mock_factory, session_factory):
        """Episodes already tagged with current version are skipped (force=False)."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed3")
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-already-tagged",
                title="Already Tagged",
                transcript="Some transcript.",
                status="scored",
            )
            session.add(episode)
            await session.flush()
            tag = PodcastEpisodeTag(
                episode_id=episode.id,
                tag_slug="ai-safety",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add(tag)
            await session.commit()
            episode_id = episode.id

        with patch("app.services.podcast_tagger.Anthropic"):
            tagger = PodcastTagger()
        result = await tagger.tag_episode(episode_id, force=False)
        assert result == []

    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_force_retags(self, mock_factory, session_factory):
        """Force=True re-tags even if already tagged."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed4")
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-force-retag",
                title="Force Retag",
                transcript="A transcript about software engineering.",
                status="scored",
            )
            session.add(episode)
            await session.flush()
            tag = PodcastEpisodeTag(
                episode_id=episode.id,
                tag_slug="ai-safety",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add(tag)
            await session.commit()
            episode_id = episode.id

        with (
            patch("app.services.podcast_tagger.Anthropic"),
            patch(
                "app.services.podcast_tagger.classify_content",
                return_value=(
                    ["software-eng"],
                    ("claude-sonnet-4-20250514", 500, 50),
                ),
            ),
            patch("app.services.podcast_tagger.log_usage", new_callable=AsyncMock),
        ):
            tagger = PodcastTagger()
            result = await tagger.tag_episode(episode_id, force=True)

        assert result == ["software-eng"]

        # Verify old tag was replaced
        async with session_factory() as session:
            from sqlalchemy import select

            tags_result = await session.execute(
                select(PodcastEpisodeTag).where(PodcastEpisodeTag.episode_id == episode_id)
            )
            tags = list(tags_result.scalars().all())
            assert len(tags) == 1
            assert tags[0].tag_slug == "software-eng"

    @patch("app.services.podcast_tagger.get_session_factory")
    async def test_tag_episode_classify_returns_none(self, mock_factory, session_factory):
        """When classify_content returns None, tag_episode returns empty list."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            podcast = Podcast(title="Test Podcast", feed_url="https://example.com/feed5")
            session.add(podcast)
            await session.flush()
            episode = PodcastEpisode(
                podcast_id=podcast.id,
                guid="ep-classify-none",
                title="Classify None",
                transcript="A transcript.",
                status="scored",
            )
            session.add(episode)
            await session.commit()
            episode_id = episode.id

        with (
            patch("app.services.podcast_tagger.Anthropic"),
            patch(
                "app.services.podcast_tagger.classify_content",
                return_value=(None, None),
            ),
        ):
            tagger = PodcastTagger()
            result = await tagger.tag_episode(episode_id)

        assert result == []
