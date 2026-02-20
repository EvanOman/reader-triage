"""Tests for podcast scorer service."""

from unittest.mock import AsyncMock, MagicMock, patch

from app.services.podcast_scorer import PodcastScorer
from app.services.scorer import InfoScore
from tests.factories import make_episode

# ---------------------------------------------------------------------------
# 1. score_episode
# ---------------------------------------------------------------------------


def _make_mock_strategy(return_value: InfoScore | None = None) -> MagicMock:
    """Create a mock ScoringStrategy with configurable score() return value."""
    strategy = MagicMock()
    strategy.version = "v2-categorical"
    strategy.score = AsyncMock(return_value=return_value)
    return strategy


class TestPodcastScorer:
    async def test_score_episode_no_transcript(self):
        """Episodes without transcript should return None."""
        strategy = _make_mock_strategy()
        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)
        episode = make_episode(transcript=None)
        result = await scorer.score_episode(episode)
        assert result is None

    async def test_score_episode_empty_transcript(self):
        """Episodes with empty string transcript should return None."""
        strategy = _make_mock_strategy()
        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)
        episode = make_episode(transcript="")
        result = await scorer.score_episode(episode)
        assert result is None

    async def test_score_episode_success(self):
        """Test scoring an episode with a transcript."""
        mock_score = InfoScore(
            specificity=20,
            specificity_reason="Several quotable passages",
            novelty=18,
            novelty_reason="Novel framing of AI safety",
            depth=22,
            depth_reason="Strong practitioner voice",
            actionability=15,
            actionability_reason="Applicable frameworks",
            overall_assessment="Excellent episode on AI scaling.",
        )
        strategy = _make_mock_strategy(return_value=mock_score)

        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)

        episode = make_episode(transcript="A deep technical discussion about AI." * 50)
        # Set up the podcast relationship
        mock_podcast = MagicMock()
        mock_podcast.title = "Test Podcast"
        episode.podcast = mock_podcast

        result = await scorer.score_episode(episode)

        assert result is not None
        assert result.specificity_score == 20
        assert result.novelty_score == 18
        assert result.depth_score == 22
        assert result.actionability_score == 15
        assert result.info_score == 75

    async def test_score_episode_uses_podcast_title_as_author(self):
        """The podcast's title is passed as 'author' to the strategy's score()."""
        mock_score = InfoScore(
            specificity=10,
            specificity_reason="r",
            novelty=10,
            novelty_reason="r",
            depth=10,
            depth_reason="r",
            actionability=10,
            actionability_reason="r",
            overall_assessment="ok",
        )
        strategy = _make_mock_strategy(return_value=mock_score)

        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)

        episode = make_episode(transcript="Content here. " * 50)
        mock_podcast = MagicMock()
        mock_podcast.title = "Dwarkesh Podcast"
        episode.podcast = mock_podcast

        await scorer.score_episode(episode)

        # Check that author kwarg was the podcast title
        call_kwargs = strategy.score.call_args.kwargs
        assert call_kwargs["author"] == "Dwarkesh Podcast"
        assert call_kwargs["content_type_hint"] == "podcast"

    async def test_score_episode_score_content_returns_none(self):
        """When strategy.score() returns None, score_episode returns None."""
        strategy = _make_mock_strategy(return_value=None)

        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)

        episode = make_episode(transcript="Some transcript. " * 50)
        mock_podcast = MagicMock()
        mock_podcast.title = "Test"
        episode.podcast = mock_podcast

        result = await scorer.score_episode(episode)
        assert result is None

    async def test_score_episode_no_podcast_relationship(self):
        """Episode with no podcast relationship uses None as author."""
        mock_score = InfoScore(
            specificity=10,
            specificity_reason="r",
            novelty=10,
            novelty_reason="r",
            depth=10,
            depth_reason="r",
            actionability=10,
            actionability_reason="r",
            overall_assessment="ok",
        )
        strategy = _make_mock_strategy(return_value=mock_score)

        with patch("app.services.podcast_scorer.Anthropic"):
            scorer = PodcastScorer(strategy=strategy)

        episode = make_episode(transcript="Content here. " * 50)
        episode.podcast = None  # type: ignore[assignment]

        result = await scorer.score_episode(episode)

        assert result is not None
        assert strategy.score.call_args.kwargs["author"] is None
