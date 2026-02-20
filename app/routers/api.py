"""REST API endpoints for reader triage."""

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.models.article import (
    Article,
    ArticleScore,
    ArticleTag,
    Author,
    BinaryArticleScore,
    Summary,
    get_session_factory,
    search_articles_fts,
)
from app.services.authors import get_author_service
from app.services.scorer import get_article_scorer
from app.services.sync import get_background_sync
from app.services.tagger import get_all_tags, get_tagger

router = APIRouter(prefix="/api", tags=["api"])


class ArticleResponse(BaseModel):
    """API response for an article with its score."""

    id: str
    title: str
    url: str
    author: str | None
    word_count: int | None
    location: str | None
    category: str | None
    info_score: float
    priority_score: float | None
    author_boost: float
    specificity_score: int
    novelty_score: int
    depth_score: int
    actionability_score: int
    score_reasons: list[str]
    overall_assessment: str | None
    skip_recommended: bool
    skip_reason: str | None
    has_summary: bool
    tags: list[str] = []
    added_at: datetime | None
    published_date: datetime | None
    # v3-binary scores (optional, None if not yet scored)
    v3_info_score: float | None = None
    v3_specificity_score: int | None = None
    v3_novelty_score: int | None = None
    v3_depth_score: int | None = None
    v3_actionability_score: int | None = None
    v3_overall_assessment: str | None = None


class ArticleDetailResponse(ArticleResponse):
    """Detailed article response with summary if available."""

    summary_text: str | None
    key_points: list[str] | None


class StatsResponse(BaseModel):
    """Statistics response."""

    total_articles: int
    high_value_count: int  # >= 60
    medium_value_count: int  # 30-59
    low_value_count: int  # < 30
    summarized_count: int
    average_score: float
    authors_synced: int
    liked_authors_count: int  # Authors with 2+ highlights


class AuthorResponse(BaseModel):
    """API response for an author."""

    id: int
    name: str
    total_highlights: int
    total_books: int
    is_favorite: bool


class SyncStatusResponse(BaseModel):
    """Response for sync status."""

    is_syncing: bool
    last_sync_at: datetime | None
    articles_processed: int
    newly_scored: int
    newly_tagged: int
    scoring_version: str
    last_error: str | None


class AuthorSyncResponse(BaseModel):
    """Response from syncing authors."""

    total_authors: int
    new_authors: int
    updated_authors: int
    total_books: int
    priorities_updated: int


async def _article_to_response(
    article: Article,
    score: ArticleScore | None,
    has_summary: bool = False,
    tags: list[str] | None = None,
    v3_score: BinaryArticleScore | None = None,
) -> ArticleResponse:
    """Convert Article and ArticleScore to API response."""
    v3_info_score = v3_score.info_score if v3_score else None
    v3_specificity_score = v3_score.specificity_score if v3_score else None
    v3_novelty_score = v3_score.novelty_score if v3_score else None
    v3_depth_score = v3_score.depth_score if v3_score else None
    v3_actionability_score = v3_score.actionability_score if v3_score else None
    v3_overall_assessment = v3_score.overall_assessment if v3_score else None

    if score is None:
        return ArticleResponse(
            id=article.id,
            title=article.title,
            url=article.url,
            author=article.author,
            word_count=article.word_count,
            location=article.location,
            category=article.category,
            info_score=0,
            priority_score=None,
            author_boost=0,
            specificity_score=0,
            novelty_score=0,
            depth_score=0,
            actionability_score=0,
            score_reasons=[],
            overall_assessment=None,
            skip_recommended=False,
            skip_reason=None,
            has_summary=has_summary,
            tags=tags or [],
            added_at=article.readwise_created_at,
            published_date=article.published_date,
            v3_info_score=v3_info_score,
            v3_specificity_score=v3_specificity_score,
            v3_novelty_score=v3_novelty_score,
            v3_depth_score=v3_depth_score,
            v3_actionability_score=v3_actionability_score,
            v3_overall_assessment=v3_overall_assessment,
        )

    return ArticleResponse(
        id=article.id,
        title=article.title,
        url=article.url,
        author=article.author,
        word_count=article.word_count,
        location=article.location,
        category=article.category,
        info_score=score.info_score,
        priority_score=score.priority_score,
        author_boost=score.author_boost,
        specificity_score=score.specificity_score,
        novelty_score=score.novelty_score,
        depth_score=score.depth_score,
        actionability_score=score.actionability_score,
        score_reasons=json.loads(score.score_reasons) if score.score_reasons else [],
        overall_assessment=score.overall_assessment,
        skip_recommended=score.skip_recommended,
        skip_reason=score.skip_reason,
        has_summary=has_summary,
        tags=tags or [],
        added_at=article.readwise_created_at,
        published_date=article.published_date,
        v3_info_score=v3_info_score,
        v3_specificity_score=v3_specificity_score,
        v3_novelty_score=v3_novelty_score,
        v3_depth_score=v3_depth_score,
        v3_actionability_score=v3_actionability_score,
        v3_overall_assessment=v3_overall_assessment,
    )


async def _get_article_tags(session, article_id: str) -> list[str]:
    """Get tag slugs for an article."""
    result = await session.execute(
        select(ArticleTag.tag_slug).where(ArticleTag.article_id == article_id)
    )
    return [row[0] for row in result.all()]


@router.get("/top5", response_model=list[ArticleResponse])
async def get_top5():
    """Get top 5 articles by priority score (excludes archived)."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Article, ArticleScore, BinaryArticleScore)
            .join(ArticleScore)
            .outerjoin(BinaryArticleScore)
            .where(Article.location != "archive")
            .order_by(ArticleScore.info_score.desc(), ArticleScore.priority_score.desc())
            .limit(5)
        )
        rows = result.all()

        responses = []
        for article, score, v3_score in rows:
            summary_result = await session.execute(
                select(Summary).where(Summary.article_id == article.id)
            )
            has_summary = summary_result.scalar_one_or_none() is not None
            tags = await _get_article_tags(session, article.id)
            responses.append(
                await _article_to_response(article, score, has_summary, tags, v3_score)
            )

    return responses


@router.get("/articles", response_model=list[ArticleResponse])
async def list_articles(
    skip: int = 0,
    limit: int = 20,
    location: str | None = None,
    tag: str | None = None,
    tier: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    include_archived: bool = False,
):
    """List all scored articles with pagination.

    By default, archived articles are excluded. Pass include_archived=true to include them.
    Optionally filter by tag slug, tier (high/medium/low), or search query.
    """
    factory = await get_session_factory()
    async with factory() as session:
        # If searching, get matching IDs from FTS first
        search_ids: list[str] | None = None
        if q and q.strip():
            search_ids = await search_articles_fts(q, limit=200)
            if not search_ids:
                return []  # No FTS matches

        query = (
            select(Article, ArticleScore, BinaryArticleScore)
            .join(ArticleScore)
            .outerjoin(BinaryArticleScore)
        )
        if sort == "v3_score":
            query = query.order_by(BinaryArticleScore.info_score.desc().nulls_last())
        elif sort == "added":
            query = query.order_by(Article.readwise_created_at.desc().nulls_last())
        elif sort == "published":
            query = query.order_by(Article.published_date.desc().nulls_last())
        else:
            query = query.order_by(
                ArticleScore.info_score.desc(), ArticleScore.priority_score.desc()
            )

        if search_ids is not None:
            query = query.where(Article.id.in_(search_ids))

        if tag:
            query = query.join(ArticleTag).where(ArticleTag.tag_slug == tag)

        if tier == "high":
            query = query.where(ArticleScore.info_score >= 60)
        elif tier == "medium":
            query = query.where(ArticleScore.info_score >= 30, ArticleScore.info_score < 60)
        elif tier == "low":
            query = query.where(ArticleScore.info_score < 30)

        if location:
            query = query.where(Article.location == location)
        elif not include_archived:
            query = query.where(Article.location != "archive")

        query = query.offset(skip).limit(limit)
        result = await session.execute(query)
        rows = result.all()

        responses = []
        for article, score, v3_score in rows:
            summary_result = await session.execute(
                select(Summary).where(Summary.article_id == article.id)
            )
            has_summary = summary_result.scalar_one_or_none() is not None
            tags = await _get_article_tags(session, article.id)
            responses.append(
                await _article_to_response(article, score, has_summary, tags, v3_score)
            )

    return responses


@router.get("/articles/skip", response_model=list[ArticleResponse])
async def get_skip_recommended():
    """Get articles recommended to skip, excluding archived."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Article, ArticleScore, BinaryArticleScore)
            .join(ArticleScore)
            .outerjoin(BinaryArticleScore)
            .where(ArticleScore.skip_recommended == True)  # noqa: E712
            .where(Article.location != "archive")
            .order_by(ArticleScore.info_score.asc())
        )
        rows = result.all()

        responses = []
        for article, score, v3_score in rows:
            summary_result = await session.execute(
                select(Summary).where(Summary.article_id == article.id)
            )
            has_summary = summary_result.scalar_one_or_none() is not None
            tags = await _get_article_tags(session, article.id)
            responses.append(
                await _article_to_response(article, score, has_summary, tags, v3_score)
            )

    return responses


@router.get("/articles/{article_id}", response_model=ArticleDetailResponse)
async def get_article(article_id: str):
    """Get a single article with its summary if available."""
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

        tags = await _get_article_tags(session, article.id)
        base = await _article_to_response(article, score, summary is not None, tags, v3_score)

        return ArticleDetailResponse(
            **base.model_dump(),
            summary_text=summary.summary_text if summary else None,
            key_points=json.loads(summary.key_points) if summary else None,
        )


@router.post("/rescore-failed")
async def rescore_failed():
    """Re-score articles that had content fetch failures.

    Fetches full content individually and re-scores.
    """
    scorer = get_article_scorer()
    rescored = await scorer.rescore_failed_articles()
    return {"status": "completed", "articles_rescored": rescored}


@router.post("/scan")
async def trigger_scan():
    """Trigger a background sync to scan and score new articles.

    Returns immediately. Use GET /api/sync-status to check progress.
    """
    sync = get_background_sync()
    sync.trigger_sync()
    return {"status": "started", "message": "Sync triggered. Check /api/sync-status for progress."}


@router.get("/sync-status", response_model=SyncStatusResponse)
async def get_sync_status():
    """Get the current background sync status."""
    sync = get_background_sync()
    status = sync.status
    return SyncStatusResponse(
        is_syncing=status.is_syncing,
        last_sync_at=status.last_sync_at,
        articles_processed=status.articles_processed,
        newly_scored=status.newly_scored,
        newly_tagged=status.newly_tagged,
        scoring_version=status.scoring_version,
        last_error=status.last_error,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics (excludes archived articles)."""
    factory = await get_session_factory()
    async with factory() as session:
        # All stats exclude archived articles
        not_archived = Article.location != "archive"

        # Total articles with scores
        total_result = await session.execute(
            select(func.count(ArticleScore.id)).join(Article).where(not_archived)
        )
        total = total_result.scalar() or 0

        # High value (>= 60)
        high_result = await session.execute(
            select(func.count(ArticleScore.id))
            .join(Article)
            .where(not_archived)
            .where(ArticleScore.info_score >= 60)
        )
        high = high_result.scalar() or 0

        # Medium value (30-59)
        medium_result = await session.execute(
            select(func.count(ArticleScore.id))
            .join(Article)
            .where(not_archived)
            .where(ArticleScore.info_score >= 30, ArticleScore.info_score < 60)
        )
        medium = medium_result.scalar() or 0

        # Low value (< 30)
        low_result = await session.execute(
            select(func.count(ArticleScore.id))
            .join(Article)
            .where(not_archived)
            .where(ArticleScore.info_score < 30)
        )
        low = low_result.scalar() or 0

        # Summarized count (excluding archived)
        summarized_result = await session.execute(
            select(func.count(Summary.id)).join(Article).where(not_archived)
        )
        summarized = summarized_result.scalar() or 0

        # Average score (excluding archived)
        avg_result = await session.execute(
            select(func.avg(ArticleScore.info_score)).join(Article).where(not_archived)
        )
        avg = avg_result.scalar() or 0.0

        # Authors synced
        authors_result = await session.execute(select(func.count(Author.id)))
        authors_synced = authors_result.scalar() or 0

        # Liked authors (2+ highlights)
        liked_result = await session.execute(
            select(func.count(Author.id)).where(Author.total_highlights >= 2)
        )
        liked_authors = liked_result.scalar() or 0

    return StatsResponse(
        total_articles=total,
        high_value_count=high,
        medium_value_count=medium,
        low_value_count=low,
        summarized_count=summarized,
        average_score=round(avg, 1),
        authors_synced=authors_synced,
        liked_authors_count=liked_authors,
    )


# Tag endpoints


@router.post("/tag")
async def trigger_tagging(force: bool = False):
    """Tag all untagged articles (or re-tag all if force=true)."""
    tagger = get_tagger()
    if force:
        results = await tagger.retag_all_articles()
    else:
        results = await tagger.tag_untagged_articles()
    return {
        "status": "completed",
        "articles_tagged": len(results),
        "results": results,
    }


@router.get("/tags")
async def list_tags():
    """List all available tags with article counts."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(ArticleTag.tag_slug, func.count(ArticleTag.id)).group_by(ArticleTag.tag_slug)
        )
        counts = dict(result.all())

    return [
        {
            "slug": tag.slug,
            "name": tag.name,
            "description": tag.description,
            "article_count": counts.get(tag.slug, 0),
        }
        for tag in get_all_tags()
    ]


# Author endpoints


@router.get("/authors", response_model=list[AuthorResponse])
async def list_authors(min_highlights: int = 1, limit: int = 50):
    """List authors sorted by highlight count."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Author)
            .where(Author.total_highlights >= min_highlights)
            .order_by(Author.total_highlights.desc())
            .limit(limit)
        )
        authors = result.scalars().all()

        return [
            AuthorResponse(
                id=a.id,
                name=a.name,
                total_highlights=a.total_highlights,
                total_books=a.total_books,
                is_favorite=a.is_favorite,
            )
            for a in authors
        ]


@router.post("/authors/sync", response_model=AuthorSyncResponse)
async def sync_authors():
    """Sync authors from Readwise and recompute priorities."""
    author_service = get_author_service()
    sync_result = await author_service.sync_authors_from_readwise()

    # Recompute priorities with new author data
    scorer = get_article_scorer()
    priorities_updated = await scorer.recompute_priorities()

    return AuthorSyncResponse(
        total_authors=sync_result.total_authors,
        new_authors=sync_result.new_authors,
        updated_authors=sync_result.updated_authors,
        total_books=sync_result.total_books,
        priorities_updated=priorities_updated,
    )


@router.post("/authors/{author_id}/favorite")
async def toggle_author_favorite(author_id: int, is_favorite: bool = True):
    """Mark an author as favorite."""
    author_service = get_author_service()
    await author_service.mark_favorite(author_id, is_favorite)
    return {"status": "ok"}
