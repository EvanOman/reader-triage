"""Topic tagging service for podcast episodes using predefined tags."""

import logging

from anthropic import Anthropic
from sqlalchemy import delete, select

from app.config import get_settings
from app.models.article import get_session_factory
from app.models.podcast import PodcastEpisode, PodcastEpisodeTag
from app.services.tagger import CURRENT_TAGGING_VERSION, classify_content
from app.services.usage import log_usage

logger = logging.getLogger(__name__)


class PodcastTagger:
    """Tags podcast episodes with predefined topic tags using Claude."""

    def __init__(self):
        settings = get_settings()
        self._anthropic = Anthropic(api_key=settings.anthropic_api_key)

    async def tag_episode(self, episode_id: int, force: bool = False) -> list[str]:
        """Tag a single episode. Returns list of tag slugs assigned.

        If force=False, skips episodes already tagged with current version.
        If force=True, re-tags regardless.
        """
        factory = await get_session_factory()
        async with factory() as session:
            episode = await session.get(PodcastEpisode, episode_id)
            if not episode:
                logger.warning("Episode %d not found", episode_id)
                return []

            # Check if already tagged with current version
            if not force:
                existing = await session.execute(
                    select(PodcastEpisodeTag)
                    .where(PodcastEpisodeTag.episode_id == episode_id)
                    .where(PodcastEpisodeTag.tagging_version == CURRENT_TAGGING_VERSION)
                    .limit(1)
                )
                if existing.scalar_one_or_none():
                    return []

            # Use transcript as content
            content = episode.transcript or ""
            if not content:
                logger.warning("No transcript for episode %d", episode_id)
                return []

            # Load podcast for author info
            await session.refresh(episode, ["podcast"])
            author = episode.podcast.title if episode.podcast else None

            # Classify
            tag_slugs, usage_info = classify_content(
                title=episode.title,
                author=author,
                content=content,
                anthropic_client=self._anthropic,
                content_type_hint="podcast",
            )
            if tag_slugs is None:
                return []

            # Log API usage
            if usage_info:
                model, in_tok, out_tok = usage_info
                await log_usage(
                    "podcast_tagger", model, in_tok, out_tok, article_id=str(episode_id)
                )

            # Delete existing tags and insert new ones
            await session.execute(
                delete(PodcastEpisodeTag).where(PodcastEpisodeTag.episode_id == episode_id)
            )
            for slug in tag_slugs:
                session.add(
                    PodcastEpisodeTag(
                        episode_id=episode_id,
                        tag_slug=slug,
                        tagging_version=CURRENT_TAGGING_VERSION,
                    )
                )
            await session.commit()

            logger.info("Tagged episode %d with %s", episode_id, tag_slugs)
            return tag_slugs

    async def tag_untagged_episodes(self) -> dict[int, list[str]]:
        """Tag all scored episodes that haven't been tagged with current version.

        Returns dict of episode_id -> list of tag slugs.
        """
        factory = await get_session_factory()
        results: dict[int, list[str]] = {}

        # Find episodes that have been scored but not tagged with current version
        async with factory() as session:
            tagged_ids = (
                select(PodcastEpisodeTag.episode_id)
                .where(PodcastEpisodeTag.tagging_version == CURRENT_TAGGING_VERSION)
                .distinct()
                .scalar_subquery()
            )

            result = await session.execute(
                select(PodcastEpisode.id)
                .where(PodcastEpisode.status == "scored")
                .where(PodcastEpisode.id.not_in(tagged_ids))
            )
            episode_ids = [row[0] for row in result.all()]

        logger.info("Found %d episodes to tag", len(episode_ids))

        for episode_id in episode_ids:
            tags = await self.tag_episode(episode_id, force=False)
            if tags:
                results[episode_id] = tags

        return results

    async def retag_all_episodes(self) -> dict[int, list[str]]:
        """Force re-tag every scored episode.

        Returns dict of episode_id -> list of tag slugs.
        """
        factory = await get_session_factory()
        results: dict[int, list[str]] = {}

        async with factory() as session:
            result = await session.execute(
                select(PodcastEpisode.id).where(PodcastEpisode.status == "scored")
            )
            episode_ids = [row[0] for row in result.all()]

        logger.info("Re-tagging %d episodes", len(episode_ids))

        for episode_id in episode_ids:
            tags = await self.tag_episode(episode_id, force=True)
            if tags:
                results[episode_id] = tags

        return results


# Singleton instance
_podcast_tagger: PodcastTagger | None = None


def get_podcast_tagger() -> PodcastTagger:
    """Get or create the podcast tagger singleton."""
    global _podcast_tagger
    if _podcast_tagger is None:
        _podcast_tagger = PodcastTagger()
    return _podcast_tagger
