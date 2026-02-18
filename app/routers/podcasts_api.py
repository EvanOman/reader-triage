"""REST API endpoints for podcast triage."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.article import get_session_factory
from app.models.podcast import (
    Podcast,
    PodcastEpisode,
    PodcastEpisodeScore,
    PodcastEpisodeTag,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/podcasts", tags=["podcasts-api"])


# --- Pydantic response models ---


class PodcastResponse(BaseModel):
    """API response for a podcast subscription."""

    id: int
    title: str
    feed_url: str
    youtube_channel_id: str | None
    youtube_channel_name: str | None
    mapping_confirmed: bool
    episode_count: int


class EpisodeResponse(BaseModel):
    """API response for a scored podcast episode."""

    id: int
    podcast_id: int
    podcast_title: str
    title: str
    duration_seconds: int | None
    published_at: datetime | None
    status: str
    transcript_source: str | None
    youtube_video_id: str | None
    listened: bool
    info_score: int
    specificity_score: int
    novelty_score: int
    depth_score: int
    actionability_score: int
    score_reasons: list[str]
    overall_assessment: str | None
    tags: list[str]


class EpisodeDetailResponse(EpisodeResponse):
    """Detailed episode response with transcript preview."""

    transcript_preview: str | None  # First 500 chars of transcript


class PodcastStatsResponse(BaseModel):
    """Episode count statistics by tier."""

    total_episodes: int
    scored_episodes: int
    high_value_count: int  # >= 60
    medium_value_count: int  # 30-59
    low_value_count: int  # < 30
    average_score: float


# --- Helper functions ---


def _episode_to_response(
    episode: PodcastEpisode,
    score: PodcastEpisodeScore | None,
    podcast_title: str,
    tag_slugs: list[str],
) -> EpisodeResponse:
    """Convert a PodcastEpisode + score to API response."""
    return EpisodeResponse(
        id=episode.id,
        podcast_id=episode.podcast_id,
        podcast_title=podcast_title,
        title=episode.title,
        duration_seconds=episode.duration_seconds,
        published_at=episode.published_at,
        status=episode.status,
        transcript_source=episode.transcript_source,
        youtube_video_id=episode.youtube_video_id,
        listened=episode.listened,
        info_score=score.info_score if score else 0,
        specificity_score=score.specificity_score if score else 0,
        novelty_score=score.novelty_score if score else 0,
        depth_score=score.depth_score if score else 0,
        actionability_score=score.actionability_score if score else 0,
        score_reasons=json.loads(score.score_reasons) if score and score.score_reasons else [],
        overall_assessment=score.overall_assessment if score else None,
        tags=tag_slugs,
    )


# --- Endpoints ---


@router.post("/import")
async def import_opml(file: UploadFile):
    """Upload OPML file to import podcast subscriptions."""
    from app.services.opml import get_opml_importer

    content = await file.read()
    opml_text = content.decode("utf-8")

    importer = get_opml_importer()
    result = importer.parse_opml(opml_text)  # validate first
    if not result:
        raise HTTPException(status_code=400, detail="No podcast feeds found in OPML")

    counts = await importer.import_opml(opml_text)
    return {"status": "ok", "podcasts_imported": counts["imported"]}


@router.get("/", response_model=list[PodcastResponse])
async def list_podcasts():
    """List all subscribed podcasts with episode counts."""
    factory = await get_session_factory()
    async with factory() as session:
        # Get podcasts with episode counts via subquery
        episode_count_sq = (
            select(
                PodcastEpisode.podcast_id,
                func.count(PodcastEpisode.id).label("episode_count"),
            )
            .group_by(PodcastEpisode.podcast_id)
            .subquery()
        )

        result = await session.execute(
            select(Podcast, episode_count_sq.c.episode_count).outerjoin(
                episode_count_sq, Podcast.id == episode_count_sq.c.podcast_id
            )
        )
        rows = result.all()

        return [
            PodcastResponse(
                id=podcast.id,
                title=podcast.title,
                feed_url=podcast.feed_url,
                youtube_channel_id=podcast.youtube_channel_id,
                youtube_channel_name=podcast.youtube_channel_name,
                mapping_confirmed=podcast.mapping_confirmed,
                episode_count=count or 0,
            )
            for podcast, count in rows
        ]


@router.delete("/{podcast_id}")
async def delete_podcast(podcast_id: int):
    """Remove a podcast subscription (cascade deletes episodes)."""
    factory = await get_session_factory()
    async with factory() as session:
        podcast = await session.get(Podcast, podcast_id)
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")
        await session.delete(podcast)
        await session.commit()
    return {"status": "ok"}


@router.get("/{podcast_id}/youtube-suggestions")
async def youtube_suggestions(podcast_id: int):
    """Get YouTube channel candidates for a podcast."""
    from app.services.youtube_mapper import get_youtube_mapper

    factory = await get_session_factory()
    async with factory() as session:
        podcast = await session.get(Podcast, podcast_id)
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")

    if podcast.youtube_channel_id and podcast.mapping_confirmed:
        mapper = get_youtube_mapper()
        videos = await mapper.fetch_recent_videos(podcast.youtube_channel_id)
        return {
            "channel_id": podcast.youtube_channel_id,
            "channel_name": podcast.youtube_channel_name,
            "confirmed": True,
            "recent_videos": [
                {"video_id": v.video_id, "title": v.title, "published": v.published} for v in videos
            ],
        }

    return {"channel_id": None, "channel_name": None, "confirmed": False, "recent_videos": []}


@router.post("/{podcast_id}/youtube-mapping")
async def set_youtube_mapping(podcast_id: int, channel_id: str, channel_name: str = ""):
    """Confirm YouTube channel mapping for a podcast."""
    from app.services.youtube_mapper import get_youtube_mapper

    factory = await get_session_factory()
    async with factory() as session:
        podcast = await session.get(Podcast, podcast_id)
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")

    mapper = get_youtube_mapper()
    await mapper.confirm_mapping(podcast_id, channel_id, channel_name)
    return {"status": "ok"}


@router.get("/episodes", response_model=list[EpisodeResponse])
async def list_episodes(
    skip: int = 0,
    limit: int = 30,
    podcast_id: int | None = None,
    tier: str | None = None,
    tag: str | None = None,
    sort: str | None = None,
    q: str | None = None,
    include_listened: bool = False,
):
    """List scored episodes with filtering and sorting."""
    factory = await get_session_factory()
    async with factory() as session:
        query = (
            select(PodcastEpisode)
            .outerjoin(PodcastEpisodeScore)
            .options(selectinload(PodcastEpisode.podcast))
            .options(selectinload(PodcastEpisode.score))
            .options(selectinload(PodcastEpisode.tags))
        )

        if not include_listened:
            query = query.where(PodcastEpisode.listened == False)  # noqa: E712

        if podcast_id is not None:
            query = query.where(PodcastEpisode.podcast_id == podcast_id)

        score_total = PodcastEpisodeScore.score_total_expr()
        if tier == "high":
            query = query.where(score_total >= 60)
        elif tier == "medium":
            query = query.where(score_total >= 30, score_total < 60)
        elif tier == "low":
            query = query.where(score_total < 30)

        if tag:
            query = query.join(PodcastEpisodeTag).where(PodcastEpisodeTag.tag_slug == tag)

        if q and q.strip():
            query = query.where(PodcastEpisode.title.ilike(f"%{q.strip()}%"))

        if sort == "published":
            query = query.order_by(PodcastEpisode.published_at.desc().nulls_last())
        elif sort == "duration":
            query = query.order_by(PodcastEpisode.duration_seconds.desc().nulls_last())
        else:
            query = query.order_by(score_total.desc().nulls_last())

        query = query.offset(skip).limit(limit)
        result = await session.execute(query)
        episodes = result.scalars().unique().all()

        responses = []
        for ep in episodes:
            podcast_title = ep.podcast.title if ep.podcast else ""
            tag_slugs = [t.tag_slug for t in ep.tags]
            responses.append(_episode_to_response(ep, ep.score, podcast_title, tag_slugs))

    return responses


@router.get("/episodes/{episode_id}", response_model=EpisodeDetailResponse)
async def get_episode(episode_id: int):
    """Get episode detail with score and transcript preview."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PodcastEpisode)
            .where(PodcastEpisode.id == episode_id)
            .options(selectinload(PodcastEpisode.podcast))
            .options(selectinload(PodcastEpisode.score))
            .options(selectinload(PodcastEpisode.tags))
        )
        episode = result.scalar_one_or_none()
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")

        podcast_title = episode.podcast.title if episode.podcast else ""
        tag_slugs = [t.tag_slug for t in episode.tags]
        base = _episode_to_response(episode, episode.score, podcast_title, tag_slugs)

        return EpisodeDetailResponse(
            **base.model_dump(),
            transcript_preview=episode.transcript[:500] if episode.transcript else None,
        )


@router.post("/episodes/{episode_id}/listened")
async def toggle_listened(episode_id: int, listened: bool = True):
    """Mark/unmark an episode as listened."""
    factory = await get_session_factory()
    async with factory() as session:
        episode = await session.get(PodcastEpisode, episode_id)
        if not episode:
            raise HTTPException(status_code=404, detail="Episode not found")
        episode.listened = listened
        await session.commit()
    return {"status": "ok", "listened": listened}


@router.post("/process")
async def trigger_processing():
    """Full pipeline: sync feeds, fetch transcripts, score, tag.

    Runs synchronously and returns results.
    """
    from app.services.podcast_feed import get_podcast_feed_service
    from app.services.podcast_scorer import get_podcast_scorer
    from app.services.podcast_tagger import get_podcast_tagger
    from app.services.transcript import get_transcript_service

    results: dict[str, object] = {}

    # 1. Sync feeds
    feed_service = get_podcast_feed_service()
    sync_results = await feed_service.sync_all_feeds()
    results["feeds_synced"] = sync_results

    # 2. Fetch transcripts
    transcript_service = get_transcript_service()
    transcript_results = await transcript_service.fetch_all_pending()
    results["transcripts"] = transcript_results

    # 3. Score episodes
    scorer = get_podcast_scorer()
    scored_count = await scorer.score_pending_episodes()
    results["episodes_scored"] = scored_count

    # 4. Tag episodes
    tagger = get_podcast_tagger()
    tag_results = await tagger.tag_untagged_episodes()
    results["episodes_tagged"] = len(tag_results)

    return {"status": "completed", "results": results}


@router.get("/sync-status")
async def podcast_sync_status():
    """Get podcast processing status (basic counts)."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PodcastEpisode.status, func.count(PodcastEpisode.id)).group_by(
                PodcastEpisode.status
            )
        )
        counts = dict(result.all())

    total = sum(counts.values())
    return {
        "total_episodes": total,
        "pending": counts.get("pending", 0),
        "transcript_ready": counts.get("transcript_ready", 0),
        "scored": counts.get("scored", 0),
        "failed": counts.get("failed", 0),
    }


@router.get("/stats", response_model=PodcastStatsResponse)
async def podcast_stats():
    """Episode counts by tier."""
    score_total = PodcastEpisodeScore.score_total_expr()
    factory = await get_session_factory()
    async with factory() as session:
        total_result = await session.execute(select(func.count(PodcastEpisode.id)))
        total = total_result.scalar() or 0

        scored_result = await session.execute(
            select(func.count(PodcastEpisode.id)).where(PodcastEpisode.status == "scored")
        )
        scored = scored_result.scalar() or 0

        high_result = await session.execute(
            select(func.count(PodcastEpisodeScore.id)).where(score_total >= 60)
        )
        high = high_result.scalar() or 0

        medium_result = await session.execute(
            select(func.count(PodcastEpisodeScore.id)).where(score_total >= 30, score_total < 60)
        )
        medium = medium_result.scalar() or 0

        low_result = await session.execute(
            select(func.count(PodcastEpisodeScore.id)).where(score_total < 30)
        )
        low = low_result.scalar() or 0

        avg_result = await session.execute(select(func.avg(score_total)))
        avg = avg_result.scalar() or 0.0

    return PodcastStatsResponse(
        total_episodes=total,
        scored_episodes=scored,
        high_value_count=high,
        medium_value_count=medium,
        low_value_count=low,
        average_score=round(float(avg), 1),
    )
