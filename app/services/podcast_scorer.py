"""Podcast episode scoring service using Claude."""

import json
import logging
from datetime import datetime

from anthropic import Anthropic
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.article import get_session_factory
from app.models.podcast import PodcastEpisode, PodcastEpisodeScore
from app.services.scorer import CURRENT_SCORING_VERSION, InfoScore, score_content

logger = logging.getLogger(__name__)


class PodcastScorer:
    """Scores podcast episodes by capture value using Claude."""

    def __init__(self):
        settings = get_settings()
        self._anthropic = Anthropic(api_key=settings.anthropic_api_key)

    async def score_episode(self, episode: PodcastEpisode) -> PodcastEpisodeScore | None:
        """Score a single episode that has a transcript.

        Args:
            episode: The podcast episode to score. Must have a transcript.

        Returns:
            PodcastEpisodeScore or None if scoring failed.
        """
        if not episode.transcript:
            logger.warning("Episode %d has no transcript, skipping scoring", episode.id)
            return None

        # Use the podcast's title as the author/show context
        show_name = episode.podcast.title if episode.podcast else None

        # Estimate word count from transcript
        word_count = len(episode.transcript.split())

        result: InfoScore | None = await score_content(
            title=episode.title,
            author=show_name,
            content=episode.transcript,
            word_count=word_count,
            content_type_hint="podcast",
            anthropic_client=self._anthropic,
            entity_id=str(episode.id),
        )

        if result is None:
            return None

        return PodcastEpisodeScore(
            episode_id=episode.id,
            specificity_score=result.specificity,
            novelty_score=result.novelty,
            depth_score=result.depth,
            actionability_score=result.actionability,
            score_reasons=json.dumps(
                [
                    result.specificity_reason,
                    result.novelty_reason,
                    result.depth_reason,
                    result.actionability_reason,
                ]
            ),
            overall_assessment=result.overall_assessment,
            model_used="claude-sonnet-4-20250514",
            scoring_version=CURRENT_SCORING_VERSION,
            scored_at=datetime.now(),
        )

    async def _score_episode_from_data(
        self, ep_info: dict[str, object]
    ) -> PodcastEpisodeScore | None:
        """Score an episode from pre-extracted data (avoids session issues)."""
        transcript = ep_info["transcript"]
        if not transcript or not isinstance(transcript, str):
            return None

        word_count = len(transcript.split())
        title = str(ep_info["title"])
        podcast_title = ep_info.get("podcast_title")
        author = str(podcast_title) if podcast_title else None

        result: InfoScore | None = await score_content(
            title=title,
            author=author,
            content=transcript,
            word_count=word_count,
            content_type_hint="podcast",
            anthropic_client=self._anthropic,
            entity_id=str(ep_info["id"]),
        )

        if result is None:
            return None

        return PodcastEpisodeScore(
            episode_id=int(str(ep_info["id"])),
            specificity_score=result.specificity,
            novelty_score=result.novelty,
            depth_score=result.depth,
            actionability_score=result.actionability,
            score_reasons=json.dumps(
                [
                    result.specificity_reason,
                    result.novelty_reason,
                    result.depth_reason,
                    result.actionability_reason,
                ]
            ),
            overall_assessment=result.overall_assessment,
            model_used="claude-sonnet-4-20250514",
            scoring_version=CURRENT_SCORING_VERSION,
            scored_at=datetime.now(),
        )

    async def score_pending_episodes(self) -> int:
        """Score all episodes with status='transcript_ready'.

        Returns:
            Number of episodes scored.
        """
        factory = await get_session_factory()
        scored_count = 0

        async with factory() as session:
            result = await session.execute(
                select(PodcastEpisode)
                .where(PodcastEpisode.status == "transcript_ready")
                .options(selectinload(PodcastEpisode.podcast))
            )
            episodes = list(result.scalars().all())
            # Extract data we need before session closes (avoid lazy load errors)
            episode_data = []
            for ep in episodes:
                episode_data.append(
                    {
                        "id": ep.id,
                        "title": ep.title,
                        "transcript": ep.transcript,
                        "podcast_title": ep.podcast.title if ep.podcast else None,
                    }
                )

        logger.info("Found %d episodes to score", len(episode_data))

        for ep_info in episode_data:
            try:
                score = await self._score_episode_from_data(ep_info)
                if score is None:
                    logger.warning("Scoring returned None for episode %d, skipping", ep_info["id"])
                    continue

                async with factory() as session:
                    session.add(score)
                    ep = await session.get(PodcastEpisode, ep_info["id"])
                    if ep is not None:
                        ep.status = "scored"
                    await session.commit()

                scored_count += 1
                logger.info(
                    "Scored episode %d (%s): %d",
                    ep_info["id"],
                    ep_info["title"][:60],
                    score.info_score,
                )
            except Exception:
                logger.exception("Error scoring episode %d", ep_info["id"])

        return scored_count


# Singleton instance
_podcast_scorer: PodcastScorer | None = None


def get_podcast_scorer() -> PodcastScorer:
    """Get or create the podcast scorer singleton."""
    global _podcast_scorer
    if _podcast_scorer is None:
        _podcast_scorer = PodcastScorer()
    return _podcast_scorer
