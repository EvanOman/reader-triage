"""HTML page routes for reader triage."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select

from app.models.article import (
    ApiUsageLog,
    Article,
    ArticleScore,
    ArticleTag,
    Author,
    BinaryArticleScore,
    Summary,
    get_session_factory,
)
from app.services.tagger import get_all_tags, get_tag_colors, get_tag_names, get_tag_styles

router = APIRouter(tags=["pages"])

# Templates directory
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _shortdate(value):
    """Format a datetime as 'Jan 5' or 'Jan 5, 2025'."""
    if not value:
        return ""
    from datetime import datetime

    now = datetime.now()
    if value.year == now.year:
        return value.strftime("%b %-d")
    return value.strftime("%b %-d, %Y")


templates.env.filters["shortdate"] = _shortdate


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard with top articles."""
    active_tag = request.query_params.get("tag")
    active_sort = request.query_params.get("sort")  # v2_score, v3_score, added, published

    factory = await get_session_factory()
    async with factory() as session:
        not_archived = Article.location != "archive"

        # Get total article count (excluding archived)
        total_result = await session.execute(
            select(func.count(ArticleScore.id))
            .join(Article, ArticleScore.article_id == Article.id)
            .where(not_archived)
        )
        total = total_result.scalar() or 0

        # Get articles with both v2 and v3 scores
        articles_query = (
            select(Article, ArticleScore, BinaryArticleScore)
            .join(ArticleScore)
            .outerjoin(BinaryArticleScore)
            .where(not_archived)
        )
        if active_sort == "v3_score":
            articles_query = articles_query.order_by(
                BinaryArticleScore.info_score.desc().nulls_last()
            )
        elif active_sort == "added":
            articles_query = articles_query.order_by(
                Article.readwise_created_at.desc().nulls_last()
            )
        elif active_sort == "published":
            articles_query = articles_query.order_by(Article.published_date.desc().nulls_last())
        else:
            # Default: v2 score
            articles_query = articles_query.order_by(
                ArticleScore.info_score.desc(), ArticleScore.priority_score.desc()
            )
        if active_tag:
            articles_query = articles_query.join(ArticleTag).where(
                ArticleTag.tag_slug == active_tag
            )
        articles_query = articles_query.limit(30)

        top_articles_result = await session.execute(articles_query)
        top_articles_rows = top_articles_result.all()

        # Build response
        articles = []
        for article, score, v3_score in top_articles_rows:
            summary_result = await session.execute(
                select(Summary).where(Summary.article_id == article.id)
            )
            has_summary = summary_result.scalar_one_or_none() is not None

            tag_result = await session.execute(
                select(ArticleTag.tag_slug).where(ArticleTag.article_id == article.id)
            )
            tag_slugs = [row[0] for row in tag_result.all()]

            articles.append(
                {
                    "id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "author": article.author,
                    "word_count": article.word_count,
                    "location": article.location,
                    "info_score": score.info_score,
                    "priority_score": score.priority_score,
                    "author_boost": score.author_boost,
                    "specificity_score": score.specificity_score,
                    "novelty_score": score.novelty_score,
                    "depth_score": score.depth_score,
                    "actionability_score": score.actionability_score,
                    "overall_assessment": score.overall_assessment,
                    "skip_recommended": score.skip_recommended,
                    "has_summary": has_summary,
                    "tags": tag_slugs,
                    "added_at": article.readwise_created_at,
                    "published_date": article.published_date,
                    "v3_info_score": v3_score.info_score if v3_score else None,
                    "v3_specificity_score": v3_score.specificity_score if v3_score else None,
                    "v3_novelty_score": v3_score.novelty_score if v3_score else None,
                    "v3_depth_score": v3_score.depth_score if v3_score else None,
                    "v3_actionability_score": v3_score.actionability_score if v3_score else None,
                    "v3_overall_assessment": v3_score.overall_assessment if v3_score else None,
                }
            )

        # Get tag counts for filter bar
        tag_counts_result = await session.execute(
            select(ArticleTag.tag_slug, func.count(ArticleTag.id))
            .join(Article)
            .where(not_archived)
            .group_by(ArticleTag.tag_slug)
        )
        tag_counts = dict(tag_counts_result.all())

    # Build available tags list (only tags with articles)
    available_tags = [
        {"slug": t.slug, "name": t.name, "count": tag_counts.get(t.slug, 0)}
        for t in get_all_tags()
        if tag_counts.get(t.slug, 0) > 0
    ]

    tag_names = get_tag_names()
    tag_colors = get_tag_colors()
    tag_styles = get_tag_styles()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": {
                "total_articles": total,
            },
            "articles": articles,
            "available_tags": available_tags,
            "active_tag": active_tag,
            "active_sort": active_sort,
            "tag_names": tag_names,
            "tag_colors": tag_colors,
            "tag_styles": tag_styles,
            "tag_names_json": json.dumps(tag_names),
            "tag_colors_json": json.dumps(tag_colors),
        },
    )


@router.get("/articles/{article_id}", response_class=HTMLResponse)
async def article_detail(request: Request, article_id: str):
    """Render single article detail page."""
    factory = await get_session_factory()
    async with factory() as session:
        article = await session.get(Article, article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        score_result = await session.execute(
            select(ArticleScore).where(ArticleScore.article_id == article_id)
        )
        score = score_result.scalar_one_or_none()

        v3_result = await session.execute(
            select(BinaryArticleScore).where(BinaryArticleScore.article_id == article_id)
        )
        v3_score = v3_result.scalar_one_or_none()

        summary_result = await session.execute(
            select(Summary).where(Summary.article_id == article_id)
        )
        summary = summary_result.scalar_one_or_none()

        # Get author info if available
        author_info = None
        if article.author:
            author_result = await session.execute(
                select(Author).where(Author.normalized_name == article.author.lower().strip())
            )
            author_obj = author_result.scalar_one_or_none()
            if author_obj:
                author_info = {
                    "total_highlights": author_obj.total_highlights,
                    "total_books": author_obj.total_books,
                    "is_favorite": author_obj.is_favorite,
                }

        # Get tags
        tag_result = await session.execute(
            select(ArticleTag.tag_slug).where(ArticleTag.article_id == article.id)
        )
        tag_slugs = [row[0] for row in tag_result.all()]

        # Parse v3 raw responses for individual question display
        v3_raw = None
        if v3_score and v3_score.raw_responses:
            try:
                v3_raw = json.loads(v3_score.raw_responses)
            except (json.JSONDecodeError, TypeError):
                pass

        article_data = {
            "id": article.id,
            "title": article.title,
            "url": article.url,
            "author": article.author,
            "word_count": article.word_count,
            "location": article.location,
            "info_score": score.info_score if score else 0,
            "priority_score": score.priority_score if score else None,
            "author_boost": score.author_boost if score else 0,
            "specificity_score": score.specificity_score if score else 0,
            "novelty_score": score.novelty_score if score else 0,
            "depth_score": score.depth_score if score else 0,
            "actionability_score": score.actionability_score if score else 0,
            "score_reasons": json.loads(score.score_reasons)
            if score and score.score_reasons
            else [],
            "overall_assessment": score.overall_assessment if score else None,
            "skip_recommended": score.skip_recommended if score else False,
            "summary_text": summary.summary_text if summary else None,
            "key_points": json.loads(summary.key_points) if summary else None,
            "author_info": author_info,
            "tags": tag_slugs,
            "added_at": article.readwise_created_at,
            "published_date": article.published_date,
            "v3_info_score": v3_score.info_score if v3_score else None,
            "v3_specificity_score": v3_score.specificity_score if v3_score else None,
            "v3_novelty_score": v3_score.novelty_score if v3_score else None,
            "v3_depth_score": v3_score.depth_score if v3_score else None,
            "v3_actionability_score": v3_score.actionability_score if v3_score else None,
            "v3_overall_assessment": v3_score.overall_assessment if v3_score else None,
            "v3_raw_responses": v3_raw,
        }

    tag_names = get_tag_names()
    tag_styles = get_tag_styles()

    return templates.TemplateResponse(
        "article.html",
        {
            "request": request,
            "article": article_data,
            "tag_names": tag_names,
            "tag_styles": tag_styles,
        },
    )


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Render the chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@router.get("/usage", response_class=HTMLResponse)
async def usage_dashboard(request: Request):
    """Render API usage dashboard."""
    factory = await get_session_factory()
    async with factory() as session:
        # Total usage
        total_result = await session.execute(
            select(
                func.count(ApiUsageLog.id),
                func.coalesce(func.sum(ApiUsageLog.input_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.output_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.cost_usd), 0.0),
            )
        )
        total_row = total_result.one()
        total_calls = total_row[0]
        total_input = total_row[1]
        total_output = total_row[2]
        total_cost = total_row[3]

        # Usage by service
        service_result = await session.execute(
            select(
                ApiUsageLog.service,
                func.count(ApiUsageLog.id),
                func.coalesce(func.sum(ApiUsageLog.input_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.output_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.cost_usd), 0.0),
            ).group_by(ApiUsageLog.service)
        )
        by_service = [
            {
                "service": row[0],
                "calls": row[1],
                "input_tokens": row[2],
                "output_tokens": row[3],
                "cost": round(row[4], 4),
            }
            for row in service_result.all()
        ]

        # Daily usage for the last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        daily_result = await session.execute(
            select(
                func.date(ApiUsageLog.timestamp).label("day"),
                func.count(ApiUsageLog.id),
                func.coalesce(func.sum(ApiUsageLog.input_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.output_tokens), 0),
                func.coalesce(func.sum(ApiUsageLog.cost_usd), 0.0),
            )
            .where(ApiUsageLog.timestamp >= thirty_days_ago)
            .group_by(func.date(ApiUsageLog.timestamp))
            .order_by(func.date(ApiUsageLog.timestamp))
        )
        daily_usage = [
            {
                "date": str(row[0]),
                "calls": row[1],
                "input_tokens": row[2],
                "output_tokens": row[3],
                "cost": round(row[4], 4),
            }
            for row in daily_result.all()
        ]

        # Daily breakdown by service (for stacked chart)
        daily_service_result = await session.execute(
            select(
                func.date(ApiUsageLog.timestamp).label("day"),
                ApiUsageLog.service,
                func.coalesce(func.sum(ApiUsageLog.cost_usd), 0.0),
            )
            .where(ApiUsageLog.timestamp >= thirty_days_ago)
            .group_by(func.date(ApiUsageLog.timestamp), ApiUsageLog.service)
            .order_by(func.date(ApiUsageLog.timestamp))
        )
        daily_by_service_raw = daily_service_result.all()

        # Reorganize into {date: {service: cost}}
        daily_by_service: dict[str, dict[str, float]] = {}
        for row in daily_by_service_raw:
            day = str(row[0])
            if day not in daily_by_service:
                daily_by_service[day] = {}
            daily_by_service[day][row[1]] = round(row[2], 4)

        # Recent calls (last 20)
        recent_result = await session.execute(
            select(ApiUsageLog).order_by(ApiUsageLog.timestamp.desc()).limit(20)
        )
        recent_calls = [
            {
                "timestamp": row.timestamp.isoformat() if row.timestamp else "",
                "service": row.service,
                "model": row.model,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "cost": round(row.cost_usd, 6),
                "article_id": row.article_id,
            }
            for row in recent_result.scalars().all()
        ]

    return templates.TemplateResponse(
        "usage.html",
        {
            "request": request,
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 4),
            "by_service": by_service,
            "daily_usage": daily_usage,
            "daily_by_service": daily_by_service,
            "recent_calls": recent_calls,
            "daily_usage_json": json.dumps(daily_usage),
            "daily_by_service_json": json.dumps(daily_by_service),
            "by_service_json": json.dumps(by_service),
        },
    )
