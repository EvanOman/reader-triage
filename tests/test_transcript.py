"""Tests for transcript service."""

from unittest.mock import MagicMock, patch

from app.services.transcript import TranscriptService

# ---------------------------------------------------------------------------
# 1. _fetch_youtube_transcript
# ---------------------------------------------------------------------------


class TestFetchYoutubeTranscript:
    async def test_success(self):
        """Test successful YouTube transcript fetch."""
        service = TranscriptService()

        mock_snippet = MagicMock()
        mock_snippet.text = "Hello world. This is a test."

        mock_api_instance = MagicMock()
        mock_api_instance.fetch.return_value = [mock_snippet]

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi",
            return_value=mock_api_instance,
        ):
            result = await service._fetch_youtube_transcript("test123")

        assert result is not None
        assert "Hello world" in result

    async def test_multiple_snippets_joined(self):
        """Multiple transcript snippets should be joined with spaces."""
        service = TranscriptService()

        snippet1 = MagicMock()
        snippet1.text = "First part."
        snippet2 = MagicMock()
        snippet2.text = "Second part."

        mock_api_instance = MagicMock()
        mock_api_instance.fetch.return_value = [snippet1, snippet2]

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi",
            return_value=mock_api_instance,
        ):
            result = await service._fetch_youtube_transcript("test123")

        assert result == "First part. Second part."

    async def test_api_error_returns_none(self):
        """When the YouTube API raises an error, returns None."""
        service = TranscriptService()

        mock_api_instance = MagicMock()
        mock_api_instance.fetch.side_effect = Exception("Transcript unavailable")

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi",
            return_value=mock_api_instance,
        ):
            result = await service._fetch_youtube_transcript("test123")

        assert result is None

    async def test_empty_transcript_returns_none(self):
        """Empty transcript list returns None."""
        service = TranscriptService()

        mock_api_instance = MagicMock()
        mock_api_instance.fetch.return_value = []

        with patch(
            "youtube_transcript_api.YouTubeTranscriptApi",
            return_value=mock_api_instance,
        ):
            result = await service._fetch_youtube_transcript("test123")

        assert result is None
