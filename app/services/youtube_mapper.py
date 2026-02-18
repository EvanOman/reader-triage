"""YouTube channel mapping for podcasts."""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import httpx
from sqlalchemy import select

from app.models.article import get_session_factory
from app.models.podcast import Podcast, PodcastEpisode

logger = logging.getLogger(__name__)


@dataclass
class YouTubeVideo:
    """A YouTube video from a channel's RSS feed."""

    video_id: str
    title: str
    published: str


class YouTubeChannelMapper:
    """Maps podcasts to their YouTube channels and matches episodes to videos."""

    async def fetch_recent_videos(self, channel_id: str) -> list[YouTubeVideo]:
        """Fetch recent videos from a YouTube channel's RSS feed (no API key needed).

        Returns up to 15 most recent videos.
        """
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to fetch YouTube feed for channel %s", channel_id)
            return []

        # Parse Atom XML
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            logger.error("Failed to parse YouTube feed XML for channel %s", channel_id)
            return []

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
        }

        videos: list[YouTubeVideo] = []
        for entry in root.findall("atom:entry", ns):
            video_id_el = entry.find("yt:videoId", ns)
            title_el = entry.find("atom:title", ns)
            published_el = entry.find("atom:published", ns)

            if video_id_el is not None and title_el is not None:
                videos.append(
                    YouTubeVideo(
                        video_id=video_id_el.text or "",
                        title=title_el.text or "",
                        published=(published_el.text or "") if published_el is not None else "",
                    )
                )

        return videos

    def match_episode_to_video(
        self, episode_title: str, videos: list[YouTubeVideo], threshold: float = 0.5
    ) -> YouTubeVideo | None:
        """Fuzzy match an episode title to a YouTube video.

        Uses SequenceMatcher with a configurable similarity threshold.
        """
        best_match: YouTubeVideo | None = None
        best_ratio = 0.0

        # Normalize for comparison
        ep_clean = self._normalize_title(episode_title)

        for video in videos:
            vid_clean = self._normalize_title(video.title)
            ratio = SequenceMatcher(None, ep_clean, vid_clean).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = video

        if best_ratio >= threshold and best_match is not None:
            return best_match
        return None

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for fuzzy matching."""
        # Remove common prefixes/suffixes and extra whitespace
        title = title.lower().strip()
        # Remove episode numbers like "Ep. 123" or "#123"
        title = re.sub(r"(?:ep\.?\s*|#)\d+\s*[-\u2013\u2014:]?\s*", "", title)
        # Remove pipe/dash separators with podcast name
        title = re.sub(r"\s*[|]\s*.*$", "", title)
        # Collapse whitespace
        title = re.sub(r"\s+", " ", title)
        return title

    async def confirm_mapping(self, podcast_id: int, channel_id: str, channel_name: str) -> None:
        """Store a confirmed YouTube channel mapping on a Podcast record."""
        factory = await get_session_factory()
        async with factory() as session:
            podcast = await session.get(Podcast, podcast_id)
            if podcast:
                podcast.youtube_channel_id = channel_id
                podcast.youtube_channel_name = channel_name
                podcast.mapping_confirmed = True
                await session.commit()
                logger.info(
                    "Confirmed YouTube mapping for %s: %s (%s)",
                    podcast.title,
                    channel_name,
                    channel_id,
                )

    async def match_episodes_to_videos(self, podcast_id: int) -> int:
        """Match unmatched episodes to YouTube videos for a podcast.

        Returns number of episodes matched.
        """
        factory = await get_session_factory()
        async with factory() as session:
            podcast = await session.get(Podcast, podcast_id)
            if not podcast or not podcast.youtube_channel_id:
                return 0

            # Fetch recent videos
            videos = await self.fetch_recent_videos(podcast.youtube_channel_id)
            if not videos:
                return 0

            # Find episodes without YouTube video IDs
            result = await session.execute(
                select(PodcastEpisode).where(
                    PodcastEpisode.podcast_id == podcast_id,
                    PodcastEpisode.youtube_video_id.is_(None),
                )
            )
            episodes = list(result.scalars().all())

            matched = 0
            for episode in episodes:
                video = self.match_episode_to_video(episode.title, videos)
                if video:
                    episode.youtube_video_id = video.video_id
                    matched += 1
                    logger.info(
                        "Matched '%s' -> '%s' (%s)",
                        episode.title,
                        video.title,
                        video.video_id,
                    )

            await session.commit()

        return matched


# Singleton
_mapper: YouTubeChannelMapper | None = None


def get_youtube_mapper() -> YouTubeChannelMapper:
    """Get or create the YouTube channel mapper singleton."""
    global _mapper
    if _mapper is None:
        _mapper = YouTubeChannelMapper()
    return _mapper
