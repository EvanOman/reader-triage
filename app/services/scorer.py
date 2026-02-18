"""Information content scoring service using Claude."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from anthropic import Anthropic
from anthropic.types import TextBlock
from readwise_sdk.exceptions import RateLimitError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.article import Article, ArticleScore, Author, get_session_factory, upsert_fts_entry
from app.services.readwise import ReaderDocument, get_readwise_service
from app.services.usage import log_usage

logger = logging.getLogger(__name__)

# Current scoring version - bump this when the scoring rubric changes
# to trigger re-scoring of all articles
CURRENT_SCORING_VERSION = "v2-categorical"

# Point mappings for categorical responses → numeric scores
# Quotability bucket (→ specificity_score, 0-25)
STANDALONE_SCORES = {"none": 0, "a_few": 9, "several": 17, "many": 25}

# Surprise bucket (→ novelty_score, 0-25)
NOVEL_FRAMING_POINTS = 15
CONTENT_TYPE_SCORES = {
    "original_analysis": 10,
    "opinion_with_evidence": 8,
    "informational_summary": 3,
    "product_review": 2,
    "news_or_roundup": 0,
}
PODCAST_CONTENT_TYPE_SCORES = {
    "deep_dive_interview": 10,
    "experienced_practitioner_sharing": 10,
    "news_commentary": 3,
    "casual_conversation": 2,
    "solo_essay": 8,
}

# Argument bucket (→ depth_score, 0-25)
AUTHOR_CONVICTION_POINTS = 12
PRACTITIONER_VOICE_POINTS = 8
COMPLETENESS_SCORES = {"complete": 5, "appears_truncated": 2, "summary_or_excerpt": 0}
PODCAST_COMPLETENESS_SCORES = {"complete": 5, "partial": 2, "low_quality": 0}

# Insight bucket (→ actionability_score, 0-25)
NAMED_FRAMEWORK_POINTS = 12
APPLICABLE_SCORES = {"broadly": 13, "narrowly": 7, "not_really": 0}

# ---------------------------------------------------------------------------
# Parameterized scoring prompt: Q3 (content type) and Q6 (completeness)
# vary by content type (article vs podcast).
# ---------------------------------------------------------------------------

Q3_ARTICLE = (
    "3. CONTENT TYPE: What best describes this content?\n"
    "   Options: original_analysis / opinion_with_evidence"
    " / informational_summary / product_review / news_or_roundup"
)
Q3_PODCAST = (
    "3. CONTENT TYPE: What best describes this podcast episode?\n"
    "   Options: deep_dive_interview / experienced_practitioner_sharing"
    " / news_commentary / casual_conversation / solo_essay"
)

Q6_ARTICLE = (
    "6. CONTENT COMPLETENESS: Does the available text appear to be a complete piece?\n"
    "   Options: complete / appears_truncated / summary_or_excerpt"
)
Q6_PODCAST = (
    "6. TRANSCRIPT QUALITY: Does the transcript appear to be a complete episode?\n"
    "   Options: complete / partial / low_quality"
)

# JSON response snippets for Q3 and Q6 (embedded in the prompt's response template)
_Q3_JSON_ARTICLE = (
    '"content_type": "<original_analysis|opinion_with_evidence'
    '|informational_summary|product_review|news_or_roundup>"'
)
_Q3_JSON_PODCAST = (
    '"content_type": "<deep_dive_interview|experienced_practitioner_sharing'
    '|news_commentary|casual_conversation|solo_essay>"'
)
_Q6_JSON_ARTICLE = '"content_completeness": "<complete|appears_truncated|summary_or_excerpt>"'
_Q6_JSON_PODCAST = '"content_completeness": "<complete|partial|low_quality>"'

_SCORING_PROMPT_TEMPLATE = """Evaluate this {content_label} for capture value — how likely a reader is to want to save and highlight passages.

Answer each question by selecting from the provided options.

1. STANDALONE PASSAGES: How many passages could stand alone as a saved note — a memorable phrasing, crisp claim, or striking example worth revisiting?
   Options: none / a_few / several / many

2. NOVEL FRAMING: Does it reframe a familiar topic or present a surprising, unexpected lens for understanding something?
   Options: true / false

{q3}

4. AUTHOR CONVICTION: Does the author argue for a clear position with conviction, rather than just reporting or summarizing?
   Options: true / false

5. PRACTITIONER VOICE: Is this written from first-person practitioner experience sharing hard-won opinions?
   Options: true / false

{q6}

7. NAMED FRAMEWORK: Does it introduce or organize around a named concept, framework, or mental model?
   Options: true / false

8. APPLICABLE IDEAS: Could a reader apply ideas from this in their own work or thinking?
   Options: broadly / narrowly / not_really

{{content_warning}}Article Title: {{title}}
Author: {{author}}
Word Count: {{word_count}}

Content:
{{content}}

Respond with ONLY a JSON object (no markdown, no extra text):
{{{{"standalone_passages": "<none|a_few|several|many>", "quotability_reason": "<brief reason>", "novel_framing": <true or false>, {q3_json}, "surprise_reason": "<brief reason>", "author_conviction": <true or false>, "practitioner_voice": <true or false>, {q6_json}, "argument_reason": "<brief reason>", "named_framework": <true or false>, "applicable_ideas": "<broadly|narrowly|not_really>", "insight_reason": "<brief reason>", "overall_assessment": "<1-2 sentence summary>"}}}}"""

# Pre-built prompt templates for each content type
_ARTICLE_SCORING_PROMPT = _SCORING_PROMPT_TEMPLATE.format(
    content_label="article",
    q3=Q3_ARTICLE,
    q6=Q6_ARTICLE,
    q3_json=_Q3_JSON_ARTICLE,
    q6_json=_Q6_JSON_ARTICLE,
)

_PODCAST_SCORING_PROMPT = _SCORING_PROMPT_TEMPLATE.format(
    content_label="podcast episode transcript",
    q3=Q3_PODCAST,
    q6=Q6_PODCAST,
    q3_json=_Q3_JSON_PODCAST,
    q6_json=_Q6_JSON_PODCAST,
)

# Max content truncation lengths per content type
_MAX_CONTENT_LENGTH = {"article": 15000, "podcast": 50000}


@dataclass
class InfoScore:
    """Capture-value score breakdown."""

    specificity: int  # 0-25 (quotability)
    specificity_reason: str
    novelty: int  # 0-25 (surprise)
    novelty_reason: str
    depth: int  # 0-25 (argument quality)
    depth_reason: str
    actionability: int  # 0-25 (applicable insight)
    actionability_reason: str
    overall_assessment: str
    content_fetch_failed: bool = False

    @property
    def total(self) -> int:
        """Total score (0-100)."""
        return self.specificity + self.novelty + self.depth + self.actionability


@dataclass
class ScanResult:
    """Result of scanning documents."""

    total_scanned: int
    newly_scored: int
    top_5: list[Article]


def normalize_author_name(name: str) -> str:
    """Normalize author name for matching."""
    return name.lower().strip()


# Patterns in overall_assessment that indicate the scoring was based on
# truncated/incomplete content and should be flagged for rescore.
# These are intentionally specific to content-fetch failures, not article quality.
_BAD_ASSESSMENT_PATTERNS = [
    "truncated content",
    "truncated article",
    "severely limited by",
    "prevents full eval",
    "cannot assess",
    "cannot properly",
    "only a summary was returned",
    "only a trivial fragment",
    "content not available",
]


def _assessment_indicates_bad_content(assessment: str) -> bool:
    """Check if an assessment text indicates scoring was done on bad content."""
    text = assessment.lower()
    return any(p in text for p in _BAD_ASSESSMENT_PATTERNS)


async def score_content(
    *,
    title: str,
    author: str | None,
    content: str,
    word_count: int | None,
    content_type_hint: str,
    anthropic_client: Anthropic,
    entity_id: str,
    content_warning: str = "",
) -> InfoScore | None:
    """Score content for capture value using Claude.

    Shared scoring logic used by both ArticleScorer and PodcastScorer.
    Does NOT handle stub detection — callers must handle that article-specific
    concern before calling this function.

    Args:
        title: Title of the content.
        author: Author or host name (may be None).
        content: The text content to score (already extracted by caller).
        word_count: Reported word count (may be None).
        content_type_hint: "article" or "podcast" — selects prompt variant and mappings.
        anthropic_client: Anthropic client instance.
        entity_id: ID string for usage logging (article ID or episode ID).
        content_warning: Optional warning text prepended to the article metadata in the
            prompt (used for truncated article content). Empty string by default.

    Returns:
        InfoScore or None if scoring failed.
    """
    if not content:
        return None

    # Select prompt template and score mappings based on content type
    if content_type_hint == "podcast":
        scoring_prompt = _PODCAST_SCORING_PROMPT
        content_type_scores = PODCAST_CONTENT_TYPE_SCORES
        completeness_scores = PODCAST_COMPLETENESS_SCORES
    else:
        scoring_prompt = _ARTICLE_SCORING_PROMPT
        content_type_scores = CONTENT_TYPE_SCORES
        completeness_scores = COMPLETENESS_SCORES

    # Truncate content if too long
    max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
    if len(content) > max_content_length:
        content = content[:max_content_length] + "... [truncated]"

    prompt = scoring_prompt.format(
        title=title,
        author=author or "Unknown",
        word_count=word_count or "Unknown",
        content=content,
        content_warning=content_warning,
    )

    try:
        model = "claude-sonnet-4-20250514"
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )

        # Log usage
        service = "podcast_scorer" if content_type_hint == "podcast" else "scorer"
        await log_usage(
            service=service,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            article_id=entity_id,
        )

        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        text = first_block.text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()

        data = json.loads(text)

        # Map categorical responses to numeric scores
        quotability = STANDALONE_SCORES.get(data.get("standalone_passages", "none"), 0)
        surprise = (NOVEL_FRAMING_POINTS if data.get("novel_framing") else 0) + (
            content_type_scores.get(data.get("content_type", ""), 0)
        )
        argument = (
            (AUTHOR_CONVICTION_POINTS if data.get("author_conviction") else 0)
            + (PRACTITIONER_VOICE_POINTS if data.get("practitioner_voice") else 0)
            + completeness_scores.get(data.get("content_completeness", ""), 0)
        )
        insight = (NAMED_FRAMEWORK_POINTS if data.get("named_framework") else 0) + (
            APPLICABLE_SCORES.get(data.get("applicable_ideas", ""), 0)
        )

        return InfoScore(
            specificity=min(25, max(0, quotability)),
            specificity_reason=data.get("quotability_reason", ""),
            novelty=min(25, max(0, surprise)),
            novelty_reason=data.get("surprise_reason", ""),
            depth=min(25, max(0, argument)),
            depth_reason=data.get("argument_reason", ""),
            actionability=min(25, max(0, insight)),
            actionability_reason=data.get("insight_reason", ""),
            overall_assessment=data.get("overall_assessment", ""),
        )
    except Exception as e:
        logger.error("Error scoring content %s: %s", entity_id, e)
        return None


class ArticleScorer:
    """Scores articles by capture value using Claude."""

    # Keep SCORING_PROMPT as a class attribute for backward compatibility with tests
    SCORING_PROMPT = _ARTICLE_SCORING_PROMPT

    # Author boost: points added to priority score based on author highlight count
    AUTHOR_BOOST_THRESHOLDS = [
        (50, 15),  # 50+ highlights: +15 priority
        (20, 10),  # 20+ highlights: +10 priority
        (10, 7),  # 10+ highlights: +7 priority
        (5, 5),  # 5+ highlights: +5 priority
        (2, 3),  # 2+ highlights: +3 priority
    ]

    def __init__(self):
        settings = get_settings()
        self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
        self._readwise = get_readwise_service()

    async def scan_all_documents(self, limit: int = 100) -> ScanResult:
        """Scan all non-archived documents and score unscored ones.

        Args:
            limit: Maximum number of documents to process.

        Returns:
            ScanResult with statistics and top 5.
        """
        # Fetch document list WITHOUT content (fast metadata-only call)
        documents = await self._readwise.get_all_documents(limit=limit, with_content=False)
        return await self._process_documents(documents)

    async def scan_inbox(self, limit: int = 50) -> ScanResult:
        """Scan inbox only and score unscored articles.

        Args:
            limit: Maximum number of inbox items to process.

        Returns:
            ScanResult with statistics and top 5.
        """
        documents = await self._readwise.get_inbox_documents(limit=limit, with_content=False)
        return await self._process_documents(documents)

    async def _process_documents(self, documents: list[ReaderDocument]) -> ScanResult:
        """Process a list of documents, scoring unscored ones."""
        factory = await get_session_factory()
        new_articles: list[Article] = []
        async with factory() as session:
            newly_scored = 0

            for doc in documents:
                # Check if article exists
                article = await session.get(Article, doc.id)

                if article is None:
                    # Create new article
                    article = Article(
                        id=doc.id,
                        title=doc.title,
                        url=doc.url,
                        author=doc.author,
                        word_count=doc.word_count,
                        content_preview=doc.content[:2000] if doc.content else None,
                        location=doc.location,
                        category=doc.category,
                        site_name=doc.site_name,
                        reading_progress=doc.reading_progress,
                        readwise_created_at=doc.created_at,
                        readwise_updated_at=doc.updated_at,
                        published_date=doc.published_date,
                        last_synced_at=datetime.now(),
                    )
                    session.add(article)
                    new_articles.append(article)
                else:
                    # Update existing article metadata
                    article.location = doc.location
                    article.reading_progress = doc.reading_progress
                    article.readwise_updated_at = doc.updated_at
                    article.last_synced_at = datetime.now()

                # Check if already scored with current version
                score_result = await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == doc.id)
                )
                existing_score = score_result.scalar_one_or_none()

                needs_scoring = existing_score is None
                if (
                    existing_score is not None
                    and existing_score.scoring_version != CURRENT_SCORING_VERSION
                ):
                    needs_scoring = True

                if needs_scoring:
                    # Fetch full content individually (batch API often returns
                    # only summaries; single-fetch is reliable)
                    full_doc = await self._readwise.get_document(doc.id, with_content=True)
                    if full_doc is None:
                        logger.warning("Could not fetch full content for %s", doc.id)
                        full_doc = doc  # fall back to whatever we have

                    # Store clean content on article for FTS
                    if full_doc.content:
                        clean_text = re.sub(r"<[^>]+>", "", full_doc.content)
                        article.content = clean_text
                        if not article.content_preview:
                            article.content_preview = clean_text[:2000]

                    # Score the article
                    score = await self._score_document(full_doc)
                    if score is None:
                        continue

                    # Get author boost
                    author_boost = await self._get_author_boost(session, doc.author)

                    # Calculate priority score
                    priority_score = score.total + author_boost

                    # Determine if skip recommended
                    skip_recommended = score.total < 30
                    skip_reason = "Low information content" if skip_recommended else None

                    if existing_score is not None:
                        # Update existing score (re-scoring due to version change)
                        existing_score.info_score = score.total
                        existing_score.specificity_score = score.specificity
                        existing_score.novelty_score = score.novelty
                        existing_score.depth_score = score.depth
                        existing_score.actionability_score = score.actionability
                        existing_score.score_reasons = json.dumps(
                            [
                                score.specificity_reason,
                                score.novelty_reason,
                                score.depth_reason,
                                score.actionability_reason,
                            ]
                        )
                        existing_score.overall_assessment = score.overall_assessment
                        existing_score.priority_score = priority_score
                        existing_score.author_boost = author_boost
                        existing_score.priority_signals = json.dumps(
                            {"author_highlights": author_boost > 0}
                        )
                        existing_score.content_fetch_failed = score.content_fetch_failed
                        existing_score.skip_recommended = skip_recommended
                        existing_score.skip_reason = skip_reason
                        existing_score.model_used = "claude-sonnet-4-20250514"
                        existing_score.scoring_version = CURRENT_SCORING_VERSION
                        existing_score.scored_at = datetime.now()
                        existing_score.priority_computed_at = datetime.now()
                    else:
                        # Save new score
                        article_score = ArticleScore(
                            article_id=doc.id,
                            info_score=score.total,
                            specificity_score=score.specificity,
                            novelty_score=score.novelty,
                            depth_score=score.depth,
                            actionability_score=score.actionability,
                            score_reasons=json.dumps(
                                [
                                    score.specificity_reason,
                                    score.novelty_reason,
                                    score.depth_reason,
                                    score.actionability_reason,
                                ]
                            ),
                            overall_assessment=score.overall_assessment,
                            priority_score=priority_score,
                            author_boost=author_boost,
                            priority_signals=json.dumps({"author_highlights": author_boost > 0}),
                            content_fetch_failed=score.content_fetch_failed,
                            skip_recommended=skip_recommended,
                            skip_reason=skip_reason,
                            model_used="claude-sonnet-4-20250514",
                            scoring_version=CURRENT_SCORING_VERSION,
                            scored_at=datetime.now(),
                            priority_computed_at=datetime.now(),
                        )
                        session.add(article_score)
                    newly_scored += 1

            await session.commit()

            # Update FTS index for new articles
            for art in new_articles:
                await upsert_fts_entry(art.id, art.title, art.author, art.content)

            # Get top 5 by priority score
            top_5 = await self._get_top_n_from_session(session, 5)

        return ScanResult(
            total_scanned=len(documents),
            newly_scored=newly_scored,
            top_5=top_5,
        )

    async def _get_author_boost(self, session: AsyncSession, author_name: str | None) -> float:
        """Calculate author boost based on their highlight count."""
        if not author_name:
            return 0.0

        normalized = normalize_author_name(author_name)
        result = await session.execute(select(Author).where(Author.normalized_name == normalized))
        author = result.scalar_one_or_none()

        if not author:
            return 0.0

        # Check thresholds
        for threshold, boost in self.AUTHOR_BOOST_THRESHOLDS:
            if author.total_highlights >= threshold:
                return float(boost)

        return 0.0

    async def score_article(self, doc: ReaderDocument) -> InfoScore | None:
        """Score a single article.

        Args:
            doc: The reader document to score.

        Returns:
            InfoScore or None if scoring failed.
        """
        return await self._score_document(doc)

    async def get_top_n(self, n: int = 5) -> list[Article]:
        """Get top N articles by priority score.

        Args:
            n: Number of articles to return.

        Returns:
            List of top N Articles.
        """
        factory = await get_session_factory()
        async with factory() as session:
            return await self._get_top_n_from_session(session, n)

    async def _get_top_n_from_session(self, session: AsyncSession, n: int) -> list[Article]:
        """Get top N articles from a session (excludes archived)."""
        result = await session.execute(
            select(Article)
            .join(ArticleScore)
            .where(Article.location != "archive")
            .order_by(ArticleScore.info_score.desc(), ArticleScore.priority_score.desc())
            .limit(n)
        )
        return list(result.scalars().all())

    async def rescore_failed_articles(self) -> int:
        """Re-score articles that had content fetch failures.

        Fetches full content individually and re-scores. Each article is
        processed in its own short DB session to avoid locking conflicts
        with the background sync.

        Returns:
            Number of articles re-scored.
        """
        factory = await get_session_factory()

        # Step 1: Collect article IDs to rescore (short-lived query session)
        # Rescore articles with:
        #   - explicit content fetch failures, OR
        #   - bad assessment patterns (truncated/incomplete mentions), OR
        #   - missing DB content despite significant reported word count
        #     (articles scored on summary text rather than full content)
        # Only rescore low-scoring articles (< 60) to avoid re-evaluating
        # articles that scored well despite bad patterns.
        async with factory() as session:
            result = await session.execute(
                select(ArticleScore.article_id)
                .join(Article, Article.id == ArticleScore.article_id)
                .where(
                    or_(
                        ArticleScore.content_fetch_failed == True,  # noqa: E712
                        *(
                            ArticleScore.overall_assessment.like(f"%{p}%")
                            for p in _BAD_ASSESSMENT_PATTERNS
                        ),
                        # Catch articles scored on summary text: no stored
                        # content but Readwise reports significant word count
                        and_(
                            or_(
                                Article.content.is_(None),
                                func.length(Article.content) < 200,
                            ),
                            Article.word_count > 200,
                        ),
                    ),
                    # Don't rescore articles that already have good scores
                    ArticleScore.info_score < 60,
                )
            )
            article_ids = [row[0] for row in result.all()]

        logger.info("Found %d articles to rescore", len(article_ids))

        # Step 2: Process each article individually with rate limit awareness
        rescored = 0
        for i, article_id in enumerate(article_ids):
            try:
                # Fetch full content from Readwise with retry on rate limits
                full_doc = None
                for attempt in range(6):
                    try:
                        full_doc = await self._readwise.get_document(article_id, with_content=True)
                        break
                    except RateLimitError as e:
                        wait = e.retry_after if e.retry_after else 2 ** (attempt + 1)
                        logger.info(
                            "Rate limited on %s, waiting %ds (attempt %d)",
                            article_id,
                            wait,
                            attempt + 1,
                        )
                        await asyncio.sleep(wait)
                    except Exception:
                        raise
                if full_doc is None:
                    logger.warning(
                        "Could not fetch content for %s after retries, skipping", article_id
                    )
                    continue

                # If Readwise still has no content, don't re-score but
                # ensure the article is properly flagged so the UI can
                # distinguish "scored on summary" from "genuinely low value".
                if not full_doc.content:
                    logger.info(
                        "Flagging %s (%d/%d) as content_fetch_failed: Readwise has no content",
                        article_id,
                        i + 1,
                        len(article_ids),
                    )
                    async with factory() as session:
                        result = await session.execute(
                            select(ArticleScore).where(ArticleScore.article_id == article_id)
                        )
                        existing = result.scalar_one_or_none()
                        if existing and not existing.content_fetch_failed:
                            existing.content_fetch_failed = True
                            await session.commit()
                    continue

                # Score with Claude
                score = await self._score_document(full_doc)
                if score is None:
                    continue

                # Don't overwrite existing score with a stub detection 0
                if score.content_fetch_failed and score.total == 0:
                    logger.info(
                        "Skipping rescore of %s (%d/%d): still no usable content",
                        article_id,
                        i + 1,
                        len(article_ids),
                    )
                    continue

                # Update DB in a short session
                async with factory() as session:
                    existing_score = await session.execute(
                        select(ArticleScore).where(ArticleScore.article_id == article_id)
                    )
                    existing = existing_score.scalar_one_or_none()
                    if existing is None:
                        continue

                    article = await session.get(Article, article_id)
                    # Save content to article if we got real content this time
                    if article and full_doc.content and not article.content:
                        clean_text = re.sub(r"<[^>]+>", "", full_doc.content)
                        if clean_text.strip():
                            article.content = clean_text
                            if not article.content_preview:
                                article.content_preview = clean_text[:2000]

                    author_boost = await self._get_author_boost(
                        session, article.author if article else None
                    )
                    priority_score = score.total + author_boost
                    skip_recommended = score.total < 30

                    existing.info_score = score.total
                    existing.specificity_score = score.specificity
                    existing.novelty_score = score.novelty
                    existing.depth_score = score.depth
                    existing.actionability_score = score.actionability
                    existing.score_reasons = json.dumps(
                        [
                            score.specificity_reason,
                            score.novelty_reason,
                            score.depth_reason,
                            score.actionability_reason,
                        ]
                    )
                    existing.overall_assessment = score.overall_assessment
                    existing.priority_score = priority_score
                    existing.author_boost = author_boost
                    existing.content_fetch_failed = score.content_fetch_failed
                    existing.skip_recommended = skip_recommended
                    existing.skip_reason = "Low information content" if skip_recommended else None
                    existing.model_used = "claude-sonnet-4-20250514"
                    existing.scoring_version = CURRENT_SCORING_VERSION
                    existing.scored_at = datetime.now()
                    existing.priority_computed_at = datetime.now()
                    await session.commit()

                rescored += 1
                logger.info(
                    "Re-scored %s (%d/%d): %d -> %s",
                    article_id,
                    i + 1,
                    len(article_ids),
                    score.total,
                    "OK" if not score.content_fetch_failed else "still failed",
                )
                # Pace requests to avoid Readwise rate limits
                await asyncio.sleep(1.5)
            except Exception:
                logger.exception("Error rescoring %s", article_id)
                await asyncio.sleep(2)

        return rescored

    async def recompute_priorities(self) -> int:
        """Recompute priority scores for all articles (e.g., after author sync).

        Returns:
            Number of articles updated.
        """
        factory = await get_session_factory()
        async with factory() as session:
            # Get all scores
            result = await session.execute(select(ArticleScore).join(Article))
            scores = list(result.scalars().all())

            updated = 0
            for score in scores:
                article = await session.get(Article, score.article_id)
                if not article:
                    continue

                author_boost = await self._get_author_boost(session, article.author)
                new_priority = score.info_score + author_boost

                if score.priority_score != new_priority or score.author_boost != author_boost:
                    score.priority_score = new_priority
                    score.author_boost = author_boost
                    score.priority_computed_at = datetime.now()
                    updated += 1

            await session.commit()

        return updated

    def _content_is_stub(self, doc: ReaderDocument) -> bool:
        """Check if fetched content is too short relative to reported word count."""
        content = doc.content or ""
        # Strip HTML tags before counting — raw HTML inflates word count
        # and can hide that the actual text is just a short description.
        clean = re.sub(r"<[^>]+>", "", content)
        actual_words = len(clean.split())
        if doc.word_count and doc.word_count > 500 and actual_words < doc.word_count * 0.15:
            return True
        return False

    async def _score_document(self, doc: ReaderDocument) -> InfoScore | None:
        """Score a document using Claude.

        Handles article-specific concerns (stub detection, content_warning,
        content_fetch_failed) and delegates to the shared score_content().
        """
        content = doc.content or doc.summary or ""
        if not content:
            return None

        # Pre-check: if content is a tiny stub compared to reported word count,
        # flag as content fetch failure immediately rather than wasting an API call
        is_stub = self._content_is_stub(doc)
        if is_stub and not doc.content:
            # Only have summary, no real content — skip scoring
            logger.info(
                "Skipping %s: only summary available (%s words reported)", doc.id, doc.word_count
            )
            return InfoScore(
                specificity=0,
                specificity_reason="Content not available",
                novelty=0,
                novelty_reason="Content not available",
                depth=0,
                depth_reason="Content not available",
                actionability=0,
                actionability_reason="Content not available",
                overall_assessment="Content not available from Readwise — only a brief summary was returned.",
                content_fetch_failed=True,
            )

        # Detect likely truncated/incomplete content from API
        content_warning = ""
        if is_stub:
            content_warning = (
                "NOTE: The content below appears incomplete (much shorter than the "
                f"reported {doc.word_count} words). Score based only on what is "
                "available, but note this limitation in your assessment.\n\n"
            )

        result = await score_content(
            title=doc.title,
            author=doc.author,
            content=content,
            word_count=doc.word_count,
            content_type_hint="article",
            anthropic_client=self._anthropic,
            entity_id=doc.id,
            content_warning=content_warning,
        )

        if result is None:
            return None

        # Detect genuine content fetch failure — only when the content
        # we got from Readwise was incomplete, NOT when the scorer
        # intentionally truncated long content for prompt size limits,
        # and NOT when the content is short by nature (highlights/notes).
        if is_stub:
            result.content_fetch_failed = True

        return result


# Singleton instance
_scorer: ArticleScorer | None = None


def get_article_scorer() -> ArticleScorer:
    """Get or create the article scorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = ArticleScorer()
    return _scorer
