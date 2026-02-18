"""Tests for podcast feed service."""

from app.services.podcast_feed import _extract_youtube_video_id, _parse_duration

# ---------------------------------------------------------------------------
# 1. _parse_duration
# ---------------------------------------------------------------------------


class TestParseDuration:
    def test_hh_mm_ss(self):
        assert _parse_duration("01:30:00") == 5400

    def test_mm_ss(self):
        assert _parse_duration("45:30") == 2730

    def test_seconds_only(self):
        assert _parse_duration("3600") == 3600

    def test_none(self):
        assert _parse_duration(None) is None

    def test_empty(self):
        assert _parse_duration("") is None

    def test_single_digit_parts(self):
        assert _parse_duration("1:2:3") == 3723

    def test_large_hour_value(self):
        assert _parse_duration("10:00:00") == 36000

    def test_non_numeric_returns_none(self):
        assert _parse_duration("abc") is None

    def test_four_parts_returns_none(self):
        """Four colon-separated parts are not handled."""
        assert _parse_duration("1:2:3:4") is None


# ---------------------------------------------------------------------------
# 2. _extract_youtube_video_id
# ---------------------------------------------------------------------------


class TestExtractYoutubeVideoId:
    def test_standard_url(self):
        assert (
            _extract_youtube_video_id("Check out https://www.youtube.com/watch?v=abc123def45")
            == "abc123def45"
        )

    def test_short_url(self):
        assert _extract_youtube_video_id("https://youtu.be/abc123def45") == "abc123def45"

    def test_embed_url(self):
        assert (
            _extract_youtube_video_id("https://www.youtube.com/embed/abc123def45") == "abc123def45"
        )

    def test_no_url(self):
        assert _extract_youtube_video_id("No youtube here") is None

    def test_empty_string(self):
        assert _extract_youtube_video_id("") is None

    def test_id_in_longer_text(self):
        text = "Episode at https://www.youtube.com/watch?v=xYz_1234567 was great"
        result = _extract_youtube_video_id(text)
        assert result == "xYz_1234567"
