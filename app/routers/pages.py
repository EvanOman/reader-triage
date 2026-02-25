"""HTML page routes for reader triage."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

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
    V4ArticleScore,
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

        # Get articles with v2, v3, and v4 scores
        articles_query = (
            select(Article, ArticleScore, BinaryArticleScore, V4ArticleScore)
            .join(ArticleScore)
            .outerjoin(BinaryArticleScore)
            .outerjoin(V4ArticleScore)
            .where(not_archived)
        )
        if active_sort == "v4_score":
            articles_query = articles_query.order_by(V4ArticleScore.info_score.desc().nulls_last())
        elif active_sort == "v3_score":
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
        for article, score, v3_score, v4_score in top_articles_rows:
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
                    "v4_info_score": v4_score.info_score if v4_score else None,
                    "v4_specificity_score": v4_score.specificity_score if v4_score else None,
                    "v4_novelty_score": v4_score.novelty_score if v4_score else None,
                    "v4_depth_score": v4_score.depth_score if v4_score else None,
                    "v4_actionability_score": v4_score.actionability_score if v4_score else None,
                    "v4_overall_assessment": v4_score.overall_assessment if v4_score else None,
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

        v4_result = await session.execute(
            select(V4ArticleScore).where(V4ArticleScore.article_id == article_id)
        )
        v4_score = v4_result.scalar_one_or_none()

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

        # Parse v4 raw responses for individual question display
        v4_raw = None
        if v4_score and v4_score.raw_responses:
            try:
                v4_raw = json.loads(v4_score.raw_responses)
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
            "v4_info_score": v4_score.info_score if v4_score else None,
            "v4_specificity_score": v4_score.specificity_score if v4_score else None,
            "v4_novelty_score": v4_score.novelty_score if v4_score else None,
            "v4_depth_score": v4_score.depth_score if v4_score else None,
            "v4_actionability_score": v4_score.actionability_score if v4_score else None,
            "v4_overall_assessment": v4_score.overall_assessment if v4_score else None,
            "v4_raw_responses": v4_raw,
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


def _histogram_bins(scores: list[float], bin_size: int = 10, max_val: int = 100) -> list[int]:
    """Bucket scores into fixed-width bins."""
    n_bins = max_val // bin_size
    bins = [0] * n_bins
    for s in scores:
        idx = min(int(s // bin_size), n_bins - 1)
        bins[idx] += 1
    return bins


def _score_stats(scores: list[float]) -> dict[str, Any]:
    """Compute mean, median, count for a list of scores."""
    if not scores:
        return {"mean": 0, "median": 0, "count": 0}
    return {
        "mean": round(sum(scores) / len(scores), 1),
        "median": round(median(scores), 1),
        "count": len(scores),
    }


def _calibration_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Spearman rho and summary stats for calibration data."""
    n = len(items)
    with_hl = sum(1 for d in items if d["highlights"] > 0)
    if n < 3:
        return {"n": n, "with_hl": with_hl, "rho": None, "p": None}

    from scipy.stats import spearmanr

    scores = [d["score"] for d in items]
    highlights = [d["highlights"] for d in items]
    rho, p = spearmanr(scores, highlights)
    return {
        "n": n,
        "with_hl": with_hl,
        "rho": round(float(rho), 3),
        "p": round(float(p), 4),
    }


@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    """Render score distribution analytics page."""
    # Exclude non-article categories (highlights, notes) from analytics
    _article_categories = ("article", "tweet", "rss", "pdf")

    factory = await get_session_factory()
    async with factory() as session:
        # V2 scores (articles only)
        v2_result = await session.execute(
            select(
                ArticleScore.info_score,
                ArticleScore.specificity_score,
                ArticleScore.novelty_score,
                ArticleScore.depth_score,
                ArticleScore.actionability_score,
            )
            .join(Article, Article.id == ArticleScore.article_id)
            .where(Article.category.in_(_article_categories))
        )
        v2_rows = v2_result.all()
        v2_totals = [float(r[0]) for r in v2_rows]
        v2_dims = {
            "quotability": [int(r[1]) for r in v2_rows],
            "surprise": [int(r[2]) for r in v2_rows],
            "argument": [int(r[3]) for r in v2_rows],
            "insight": [int(r[4]) for r in v2_rows],
        }

        # V3 scores (articles only)
        v3_result = await session.execute(
            select(
                BinaryArticleScore.info_score,
                BinaryArticleScore.specificity_score,
                BinaryArticleScore.novelty_score,
                BinaryArticleScore.depth_score,
                BinaryArticleScore.actionability_score,
            )
            .join(Article, Article.id == BinaryArticleScore.article_id)
            .where(Article.category.in_(_article_categories))
        )
        v3_rows = v3_result.all()
        v3_totals = [float(r[0]) for r in v3_rows]
        v3_dims = {
            "quotability": [int(r[1]) for r in v3_rows],
            "surprise": [int(r[2]) for r in v3_rows],
            "argument": [int(r[3]) for r in v3_rows],
            "insight": [int(r[4]) for r in v3_rows],
        }

        # V4 scores (articles only)
        v4_result = await session.execute(
            select(
                V4ArticleScore.info_score,
                V4ArticleScore.specificity_score,
                V4ArticleScore.novelty_score,
                V4ArticleScore.depth_score,
                V4ArticleScore.actionability_score,
            )
            .join(Article, Article.id == V4ArticleScore.article_id)
            .where(Article.category.in_(_article_categories))
        )
        v4_rows = v4_result.all()
        v4_totals = [float(r[0]) for r in v4_rows]
        v4_dims = {
            "quotability": [int(r[1]) for r in v4_rows],
            "surprise": [int(r[2]) for r in v4_rows],
            "argument": [int(r[3]) for r in v4_rows],
            "insight": [int(r[4]) for r in v4_rows],
        }

        # Paired scores: v2 vs v3, v2 vs v4
        paired_v3_result = await session.execute(
            select(
                Article.title,
                ArticleScore.info_score,
                BinaryArticleScore.info_score,
            )
            .join(ArticleScore, Article.id == ArticleScore.article_id)
            .join(BinaryArticleScore, Article.id == BinaryArticleScore.article_id)
            .where(Article.category.in_(_article_categories))
        )
        paired_v3_rows = paired_v3_result.all()
        scatter_v2_v3 = [
            {"title": r[0], "v2": float(r[1]), "v3": float(r[2])} for r in paired_v3_rows
        ]

        paired_v4_result = await session.execute(
            select(
                Article.title,
                ArticleScore.info_score,
                V4ArticleScore.info_score,
            )
            .join(ArticleScore, Article.id == ArticleScore.article_id)
            .join(V4ArticleScore, Article.id == V4ArticleScore.article_id)
            .where(Article.category.in_(_article_categories))
        )
        paired_v4_rows = paired_v4_result.all()
        scatter_v2_v4 = [
            {"title": r[0], "v2": float(r[1]), "v4": float(r[2])} for r in paired_v4_rows
        ]

        # Calibration: articles with highlighted_words populated
        cal_v2_result = await session.execute(
            select(
                Article.title,
                ArticleScore.info_score,
                Article.highlighted_words,
                Article.word_count,
            )
            .join(ArticleScore, Article.id == ArticleScore.article_id)
            .where(Article.highlighted_words.is_not(None))
            .where(Article.category.in_(_article_categories))
        )
        cal_v2_rows = cal_v2_result.all()
        cal_v2 = [
            {
                "title": r[0],
                "score": float(r[1]),
                "highlights": int(r[2]),
                "density": round(int(r[2]) / int(r[3]) * 1000, 1) if r[3] and int(r[3]) > 0 else 0,
            }
            for r in cal_v2_rows
        ]

        cal_v3_result = await session.execute(
            select(
                Article.title,
                BinaryArticleScore.info_score,
                Article.highlighted_words,
                Article.word_count,
            )
            .join(BinaryArticleScore, Article.id == BinaryArticleScore.article_id)
            .where(Article.highlighted_words.is_not(None))
            .where(Article.category.in_(_article_categories))
        )
        cal_v3_rows = cal_v3_result.all()
        cal_v3 = [
            {
                "title": r[0],
                "score": float(r[1]),
                "highlights": int(r[2]),
                "density": round(int(r[2]) / int(r[3]) * 1000, 1) if r[3] and int(r[3]) > 0 else 0,
            }
            for r in cal_v3_rows
        ]

        cal_v4_result = await session.execute(
            select(
                Article.title,
                V4ArticleScore.info_score,
                Article.highlighted_words,
                Article.word_count,
            )
            .join(V4ArticleScore, Article.id == V4ArticleScore.article_id)
            .where(Article.highlighted_words.is_not(None))
            .where(Article.category.in_(_article_categories))
        )
        cal_v4_rows = cal_v4_result.all()
        cal_v4 = [
            {
                "title": r[0],
                "score": float(r[1]),
                "highlights": int(r[2]),
                "density": round(int(r[2]) / int(r[3]) * 1000, 1) if r[3] and int(r[3]) > 0 else 0,
            }
            for r in cal_v4_rows
        ]

    # Compute Spearman rho for calibration data (raw highlighted words)
    cal_stats_v2 = _calibration_stats(cal_v2)
    cal_stats_v3 = _calibration_stats(cal_v3)
    cal_stats_v4 = _calibration_stats(cal_v4)

    # Compute Spearman rho for density (highlighted words per 1k words)
    density_stats_v2 = _calibration_stats(
        [{"score": d["score"], "highlights": d["density"]} for d in cal_v2]
    )
    density_stats_v3 = _calibration_stats(
        [{"score": d["score"], "highlights": d["density"]} for d in cal_v3]
    )
    density_stats_v4 = _calibration_stats(
        [{"score": d["score"], "highlights": d["density"]} for d in cal_v4]
    )

    data = {
        "v2_histogram": _histogram_bins(v2_totals),
        "v3_histogram": _histogram_bins(v3_totals),
        "v4_histogram": _histogram_bins(v4_totals),
        "v2_stats": _score_stats(v2_totals),
        "v3_stats": _score_stats(v3_totals),
        "v4_stats": _score_stats(v4_totals),
        "scatter_v2_v3": scatter_v2_v3,
        "scatter_v2_v4": scatter_v2_v4,
        "dimensions": {
            dim: {
                "v2": _histogram_bins([float(s) for s in v2_dims[dim]], bin_size=5, max_val=25),
                "v3": _histogram_bins([float(s) for s in v3_dims[dim]], bin_size=5, max_val=25),
                "v4": _histogram_bins([float(s) for s in v4_dims[dim]], bin_size=5, max_val=25),
            }
            for dim in ["quotability", "surprise", "argument", "insight"]
        },
        "cal_v2": cal_v2,
        "cal_v3": cal_v3,
        "cal_v4": cal_v4,
        "cal_stats_v2": cal_stats_v2,
        "cal_stats_v3": cal_stats_v3,
        "cal_stats_v4": cal_stats_v4,
    }

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "data_json": json.dumps(data),
            "v2_stats": data["v2_stats"],
            "v3_stats": data["v3_stats"],
            "v4_stats": data["v4_stats"],
            "cal_stats_v2": cal_stats_v2,
            "cal_stats_v3": cal_stats_v3,
            "cal_stats_v4": cal_stats_v4,
            "density_stats_v2": density_stats_v2,
            "density_stats_v3": density_stats_v3,
            "density_stats_v4": density_stats_v4,
        },
    )
