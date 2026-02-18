"""RSS feed parsing and episode discovery for podcasts."""

import logging
import re
from datetime import UTC, datetime

import feedparser
import httpx
from sqlalchemy import select

from app.models.article import get_session_factory
from app.models.podcast import Podcast, PodcastEpisode

logger = logging.getLogger(__name__)

# Only import episodes from this date forward
CUTOFF_DATE = datetime(2026, 1, 1, tzinfo=UTC)


def _parse_duration(value: str | None) -> int | None:
    """Parse iTunes duration (HH:MM:SS or seconds) to total seconds."""
    if not value:
        return None

    # Already numeric (seconds)
    if value.isdigit():
        return int(value)

    # HH:MM:SS or MM:SS
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        pass
    return None


def _extract_youtube_video_id(text: str) -> str | None:
    """Extract YouTube video ID from text (description, links)."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


class PodcastFeedService:
    """Fetches and parses podcast RSS feeds to discover episodes."""

    async def sync_feed(self, podcast: Podcast) -> int:
        """Sync a single podcast feed. Returns number of new episodes."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(podcast.feed_url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to fetch feed for %s", podcast.title)
            return 0

        feed = feedparser.parse(response.text)
        new_episodes = 0

        factory = await get_session_factory()
        async with factory() as session:
            for entry in feed.entries:
                guid = entry.get("id") or entry.get("link") or entry.get("title", "")
                if not guid:
                    continue

                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=UTC)
                    except (TypeError, ValueError):
                        pass

                # Skip episodes before cutoff
                if published and published < CUTOFF_DATE:
                    continue

                # Check for duplicate
                result = await session.execute(
                    select(PodcastEpisode).where(PodcastEpisode.guid == guid)
                )
                if result.scalar_one_or_none():
                    continue

                # Extract audio URL from enclosures
                audio_url = None
                enclosures = entry.get("enclosures", [])
                for enc in enclosures:
                    enc_type = enc.get("type", "")
                    if "audio" in enc_type or enc.get("href", "").endswith((".mp3", ".m4a")):
                        audio_url = enc.get("href")
                        break

                # Parse duration
                duration = _parse_duration(entry.get("itunes_duration") or entry.get("duration"))

                # Try to find YouTube video ID in description
                description = entry.get("summary", "") or entry.get("description", "")
                content = entry.get("content", [{}])
                if content and isinstance(content, list):
                    content_value = content[0].get("value", "")
                else:
                    content_value = ""
                all_text = f"{description} {content_value}"

                youtube_video_id = _extract_youtube_video_id(all_text)

                # Also check links
                for link in entry.get("links", []):
                    href = link.get("href", "")
                    vid = _extract_youtube_video_id(href)
                    if vid:
                        youtube_video_id = vid
                        break

                episode = PodcastEpisode(
                    podcast_id=podcast.id,
                    guid=guid,
                    title=entry.get("title", "Untitled"),
                    audio_url=audio_url,
                    duration_seconds=duration,
                    published_at=published,
                    youtube_video_id=youtube_video_id,
                    status="pending",
                )
                session.add(episode)
                new_episodes += 1

            await session.commit()

        logger.info("Synced %s: %d new episodes", podcast.title, new_episodes)
        return new_episodes

    async def sync_all_feeds(self) -> dict[str, int]:
        """Sync all podcast feeds. Returns dict of podcast_title -> new episode count."""
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(select(Podcast))
            podcasts = list(result.scalars().all())

        results: dict[str, int] = {}
        for podcast in podcasts:
            count = await self.sync_feed(podcast)
            results[podcast.title] = count

        return results


# Singleton
_feed_service: PodcastFeedService | None = None


def get_podcast_feed_service() -> PodcastFeedService:
    """Get or create the podcast feed service singleton."""
    global _feed_service
    if _feed_service is None:
        _feed_service = PodcastFeedService()
    return _feed_service
