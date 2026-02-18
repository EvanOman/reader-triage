"""Tests for YouTube channel mapper."""

from app.services.youtube_mapper import YouTubeChannelMapper, YouTubeVideo

# ---------------------------------------------------------------------------
# 1. _normalize_title (instance method)
# ---------------------------------------------------------------------------


class TestNormalizeTitle:
    def test_lowercases_and_strips(self):
        mapper = YouTubeChannelMapper()
        assert mapper._normalize_title("  Hello World  ") == "hello world"

    def test_removes_episode_number_ep_prefix(self):
        mapper = YouTubeChannelMapper()
        result = mapper._normalize_title("Ep. 123 - Guest Name")
        assert "123" not in result
        assert "guest name" in result

    def test_removes_hash_episode_number(self):
        mapper = YouTubeChannelMapper()
        result = mapper._normalize_title("#42 - The Answer")
        assert "42" not in result
        assert "the answer" in result

    def test_removes_pipe_suffix(self):
        mapper = YouTubeChannelMapper()
        result = mapper._normalize_title("Episode Title | Podcast Name")
        assert "podcast name" not in result
        assert "episode title" in result

    def test_empty_string(self):
        mapper = YouTubeChannelMapper()
        assert mapper._normalize_title("") == ""

    def test_collapses_whitespace(self):
        mapper = YouTubeChannelMapper()
        result = mapper._normalize_title("too   many   spaces")
        assert result == "too many spaces"


# ---------------------------------------------------------------------------
# 2. match_episode_to_video (fuzzy matching)
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def test_exact_match(self):
        mapper = YouTubeChannelMapper()
        videos = [YouTubeVideo(video_id="v1", title="Dario Amodei -- Scaling", published="")]
        result = mapper.match_episode_to_video("Dario Amodei -- Scaling", videos)
        assert result is not None
        assert result.video_id == "v1"

    def test_close_match(self):
        mapper = YouTubeChannelMapper()
        videos = [
            YouTubeVideo(
                video_id="v1",
                title="Dario Amodei - Scaling and Safety",
                published="",
            )
        ]
        result = mapper.match_episode_to_video("Dario Amodei -- Scaling & Safety", videos)
        assert result is not None

    def test_no_match(self):
        mapper = YouTubeChannelMapper()
        videos = [YouTubeVideo(video_id="v1", title="Completely Different Title", published="")]
        result = mapper.match_episode_to_video("Something Else Entirely", videos)
        assert result is None

    def test_empty_video_list(self):
        mapper = YouTubeChannelMapper()
        result = mapper.match_episode_to_video("Any Title", [])
        assert result is None

    def test_selects_best_match(self):
        mapper = YouTubeChannelMapper()
        videos = [
            YouTubeVideo(video_id="v1", title="Unrelated Content", published=""),
            YouTubeVideo(video_id="v2", title="AI Safety Discussion Episode", published=""),
        ]
        result = mapper.match_episode_to_video("AI Safety Discussion", videos)
        assert result is not None
        assert result.video_id == "v2"

    def test_threshold_respected(self):
        """Matches below the threshold return None."""
        mapper = YouTubeChannelMapper()
        videos = [
            YouTubeVideo(video_id="v1", title="AAAA", published=""),
        ]
        result = mapper.match_episode_to_video("ZZZZ", videos, threshold=0.9)
        assert result is None
