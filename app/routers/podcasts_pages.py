"""HTML page routes for podcast triage."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.article import get_session_factory
from app.models.podcast import (
    Podcast,
    PodcastEpisode,
    PodcastEpisodeScore,
    PodcastEpisodeTag,
)
from app.services.tagger import get_all_tags, get_tag_colors, get_tag_names, get_tag_styles

router = APIRouter(tags=["podcast-pages"])

# Templates directory
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _shortdate(value: datetime | None) -> str:
    """Format a datetime as 'Jan 5' or 'Jan 5, 2025'."""
    if not value:
        return ""
    now = datetime.now()
    if value.year == now.year:
        return value.strftime("%b %-d")
    return value.strftime("%b %-d, %Y")


def _format_duration(seconds: int | None) -> str:
    """Format seconds as '1h 23m' or '45m'."""
    if not seconds:
        return ""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


templates.env.filters["shortdate"] = _shortdate
templates.env.filters["duration"] = _format_duration


@router.get("/podcasts", response_class=HTMLResponse)
async def podcasts_dashboard(request: Request):
    """Render the podcast episodes dashboard."""
    active_tag = request.query_params.get("tag")
    active_tier = request.query_params.get("tier")
    active_sort = request.query_params.get("sort")

    score_total = PodcastEpisodeScore.score_total_expr()
    factory = await get_session_factory()
    async with factory() as session:
        # Get stats
        total_result = await session.execute(select(func.count(PodcastEpisodeScore.id)))
        total = total_result.scalar() or 0

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

        # Build episodes query
        episodes_query = (
            select(PodcastEpisode)
            .outerjoin(PodcastEpisodeScore)
            .options(selectinload(PodcastEpisode.podcast))
            .options(selectinload(PodcastEpisode.score))
            .options(selectinload(PodcastEpisode.tags))
        )

        # Only show scored episodes by default, exclude listened
        episodes_query = episodes_query.where(PodcastEpisode.status == "scored")
        episodes_query = episodes_query.where(PodcastEpisode.listened == False)  # noqa: E712

        if active_tag:
            episodes_query = episodes_query.join(PodcastEpisodeTag).where(
                PodcastEpisodeTag.tag_slug == active_tag
            )

        if active_tier == "high":
            episodes_query = episodes_query.where(score_total >= 60)
        elif active_tier == "medium":
            episodes_query = episodes_query.where(score_total >= 30, score_total < 60)
        elif active_tier == "low":
            episodes_query = episodes_query.where(score_total < 30)

        if active_sort == "published":
            episodes_query = episodes_query.order_by(
                PodcastEpisode.published_at.desc().nulls_last()
            )
        elif active_sort == "duration":
            episodes_query = episodes_query.order_by(
                PodcastEpisode.duration_seconds.desc().nulls_last()
            )
        else:
            episodes_query = episodes_query.order_by(score_total.desc().nulls_last())

        episodes_query = episodes_query.limit(30)
        result = await session.execute(episodes_query)
        episode_rows = result.scalars().unique().all()

        # Build episode dicts for template
        episodes = []
        for ep in episode_rows:
            score = ep.score
            tag_slugs = [t.tag_slug for t in ep.tags]
            episodes.append(
                {
                    "id": ep.id,
                    "title": ep.title,
                    "podcast_title": ep.podcast.title if ep.podcast else "",
                    "duration_seconds": ep.duration_seconds,
                    "published_at": ep.published_at,
                    "status": ep.status,
                    "listened": ep.listened,
                    "info_score": score.info_score if score else 0,
                    "specificity_score": score.specificity_score if score else 0,
                    "novelty_score": score.novelty_score if score else 0,
                    "depth_score": score.depth_score if score else 0,
                    "actionability_score": score.actionability_score if score else 0,
                    "overall_assessment": score.overall_assessment if score else None,
                    "tags": tag_slugs,
                }
            )

        # Get tag counts for filter bar
        tag_counts_result = await session.execute(
            select(PodcastEpisodeTag.tag_slug, func.count(PodcastEpisodeTag.id))
            .join(PodcastEpisode)
            .where(PodcastEpisode.status == "scored")
            .group_by(PodcastEpisodeTag.tag_slug)
        )
        tag_counts = dict(tag_counts_result.all())

    # Build available tags list (only tags with episodes)
    available_tags = [
        {"slug": t.slug, "name": t.name, "count": tag_counts.get(t.slug, 0)}
        for t in get_all_tags()
        if tag_counts.get(t.slug, 0) > 0
    ]

    tag_names = get_tag_names()
    tag_styles = get_tag_styles()
    tag_colors = get_tag_colors()

    return templates.TemplateResponse(
        "podcasts.html",
        {
            "request": request,
            "stats": {
                "total_scored": total,
                "high_value_count": high,
                "medium_value_count": medium,
                "low_value_count": low,
            },
            "episodes": episodes,
            "available_tags": available_tags,
            "active_tag": active_tag,
            "active_tier": active_tier,
            "active_sort": active_sort,
            "tag_names": tag_names,
            "tag_styles": tag_styles,
            "tag_names_json": json.dumps(tag_names),
            "tag_colors_json": json.dumps(tag_colors),
        },
    )


@router.get("/podcasts/import", response_class=HTMLResponse)
async def podcast_import_page(request: Request):
    """Render the OPML import page."""
    factory = await get_session_factory()
    async with factory() as session:
        # Get episode counts per podcast via subquery
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

        podcasts = [
            {
                "id": podcast.id,
                "title": podcast.title,
                "feed_url": podcast.feed_url,
                "youtube_channel_id": podcast.youtube_channel_id,
                "mapping_confirmed": podcast.mapping_confirmed,
                "episode_count": count or 0,
            }
            for podcast, count in rows
        ]

    return templates.TemplateResponse(
        "podcast_import.html",
        {
            "request": request,
            "podcasts": podcasts,
        },
    )


@router.get("/podcasts/episodes/{episode_id}", response_class=HTMLResponse)
async def episode_detail(request: Request, episode_id: int):
    """Render episode detail page."""
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

        score = episode.score
        tag_slugs = [t.tag_slug for t in episode.tags]

        episode_data = {
            "id": episode.id,
            "title": episode.title,
            "podcast_title": episode.podcast.title if episode.podcast else "",
            "duration_seconds": episode.duration_seconds,
            "published_at": episode.published_at,
            "status": episode.status,
            "transcript_source": episode.transcript_source,
            "youtube_video_id": episode.youtube_video_id,
            "info_score": score.info_score if score else 0,
            "specificity_score": score.specificity_score if score else 0,
            "novelty_score": score.novelty_score if score else 0,
            "depth_score": score.depth_score if score else 0,
            "actionability_score": score.actionability_score if score else 0,
            "score_reasons": (
                json.loads(score.score_reasons) if score and score.score_reasons else []
            ),
            "overall_assessment": score.overall_assessment if score else None,
            "transcript_preview": (episode.transcript[:500] if episode.transcript else None),
            "tags": tag_slugs,
        }

    tag_names = get_tag_names()
    tag_styles = get_tag_styles()

    return templates.TemplateResponse(
        "episode.html",
        {
            "request": request,
            "episode": episode_data,
            "tag_names": tag_names,
            "tag_styles": tag_styles,
        },
    )
