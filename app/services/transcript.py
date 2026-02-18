"""Transcript fetching service for podcast episodes."""

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.models.article import get_session_factory
from app.models.podcast import PodcastEpisode
from app.services.usage import log_usage

logger = logging.getLogger(__name__)


class TranscriptService:
    """Fetches transcripts for podcast episodes via YouTube or Groq Whisper."""

    async def fetch_transcript(self, episode: PodcastEpisode) -> bool:
        """Fetch transcript for a single episode. Returns True on success."""
        # Try YouTube first
        if episode.youtube_video_id:
            transcript = await self._fetch_youtube_transcript(episode.youtube_video_id)
            if transcript:
                factory = await get_session_factory()
                async with factory() as session:
                    ep = await session.get(PodcastEpisode, episode.id)
                    if ep:
                        ep.transcript = transcript
                        ep.transcript_source = "youtube"
                        ep.status = "transcript_ready"
                        await session.commit()
                logger.info(
                    "YouTube transcript fetched for '%s' (%d chars)",
                    episode.title,
                    len(transcript),
                )
                return True

        # Groq Whisper fallback
        if episode.audio_url:
            transcript = await self._fetch_groq_transcript(episode)
            if transcript:
                factory = await get_session_factory()
                async with factory() as session:
                    ep = await session.get(PodcastEpisode, episode.id)
                    if ep:
                        ep.transcript = transcript
                        ep.transcript_source = "groq_whisper"
                        ep.status = "transcript_ready"
                        await session.commit()
                logger.info(
                    "Groq transcript fetched for '%s' (%d chars)",
                    episode.title,
                    len(transcript),
                )
                return True

        logger.warning(
            "No transcript source available for '%s' (video_id=%s, audio_url=%s)",
            episode.title,
            episode.youtube_video_id,
            bool(episode.audio_url),
        )
        return False

    async def _fetch_youtube_transcript(self, video_id: str) -> str | None:
        """Fetch transcript from YouTube (free). Returns joined text or None."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            api = YouTubeTranscriptApi()
            # youtube_transcript_api is synchronous, run in thread
            transcript_data = await asyncio.to_thread(api.fetch, video_id)
            # Join all snippet texts
            parts: list[str] = []
            for snippet in transcript_data:
                parts.append(snippet.text)
            return " ".join(parts) if parts else None
        except Exception:
            logger.info("YouTube transcript not available for %s", video_id)
            return None

    async def _fetch_groq_transcript(self, episode: PodcastEpisode) -> str | None:
        """Fetch transcript via Groq Whisper API. Returns text or None."""
        settings = get_settings()
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set, cannot use Groq Whisper")
            return None

        if not episode.audio_url:
            return None

        try:
            from groq import AsyncGroq

            # Download audio to temp file
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                response = await client.get(episode.audio_url)
                response.raise_for_status()

            # Write to temp file
            suffix = ".mp3" if ".mp3" in episode.audio_url else ".m4a"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(response.content)
                temp_path = Path(f.name)

            try:
                groq_client = AsyncGroq(api_key=settings.groq_api_key)
                with open(temp_path, "rb") as audio_file:
                    transcription = await groq_client.audio.transcriptions.create(
                        file=(temp_path.name, audio_file.read()),
                        model="whisper-large-v3-turbo",
                    )

                # Log usage
                duration_secs = episode.duration_seconds or 0

                await log_usage(
                    service="groq_whisper",
                    model="whisper-large-v3-turbo",
                    input_tokens=duration_secs,  # Use seconds as proxy for input
                    output_tokens=0,
                    article_id=str(episode.id),
                )

                return transcription.text if transcription.text else None
            finally:
                temp_path.unlink(missing_ok=True)

        except Exception:
            logger.exception("Groq Whisper failed for '%s'", episode.title)
            return None

    async def fetch_all_pending(self) -> dict[str, int]:
        """Process all pending episodes that have a transcript source available.

        Returns dict with 'success' and 'failed' counts.
        """
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(PodcastEpisode).where(PodcastEpisode.status == "pending")
            )
            episodes = list(result.scalars().all())

        success = 0
        failed = 0

        for episode in episodes:
            if not episode.youtube_video_id and not episode.audio_url:
                continue

            ok = await self.fetch_transcript(episode)
            if ok:
                success += 1
            else:
                # Mark as failed
                factory = await get_session_factory()
                async with factory() as session:
                    ep = await session.get(PodcastEpisode, episode.id)
                    if ep:
                        ep.status = "failed"
                        await session.commit()
                failed += 1

        logger.info("Transcript fetch: %d success, %d failed", success, failed)
        return {"success": success, "failed": failed}


# Singleton
_transcript_service: TranscriptService | None = None


def get_transcript_service() -> TranscriptService:
    """Get or create the transcript service singleton."""
    global _transcript_service
    if _transcript_service is None:
        _transcript_service = TranscriptService()
    return _transcript_service
