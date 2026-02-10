"""Summarizer service for low-information content articles."""

import json
import logging
import re
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import TextBlock
from sqlalchemy import select

from app.config import get_settings
from app.models.article import Article, ArticleScore, Summary, get_session_factory
from app.services.readwise import get_readwise_service
from app.services.usage import log_usage

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Generated summary for an article."""

    summary_text: str
    key_points: list[str]


class Summarizer:
    """Generates summaries for low-information content articles."""

    SUMMARY_PROMPT = """Summarize this article concisely. The article was scored as low-information content, so extract the key points efficiently.

Article Title: {title}
Author: {author}

Content:
{content}

Provide:
1. A 2-3 sentence summary of the main point
2. 3-5 key takeaways as bullet points

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"summary": "<2-3 sentence summary>", "key_points": ["<point 1>", "<point 2>", "<point 3>"]}}"""

    LOW_INFO_THRESHOLD = 30  # Articles scoring below this get summarized

    def __init__(self):
        settings = get_settings()
        self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
        self._readwise = get_readwise_service()

    async def summarize(self, article: Article) -> Summary | None:
        """Generate a summary for an article.

        Args:
            article: The article to summarize.

        Returns:
            Summary object or None if summarization failed.
        """
        # Fetch full content from Readwise
        doc = await self._readwise.get_document(article.id, with_content=True)
        if not doc:
            return None

        content = doc.content or doc.summary or ""
        if not content:
            return None

        result = await self._generate_summary(article.title, article.author, content, article.id)
        if result is None:
            return None

        factory = await get_session_factory()
        async with factory() as session:
            summary = Summary(
                article_id=article.id,
                summary_text=result.summary_text,
                key_points=json.dumps(result.key_points),
            )
            session.add(summary)
            await session.commit()
            await session.refresh(summary)

        return summary

    async def summarize_low_info_articles(self) -> list[Summary]:
        """Generate summaries for all low-info articles that don't have one.

        Returns:
            List of newly generated summaries.
        """
        factory = await get_session_factory()
        summaries: list[Summary] = []

        async with factory() as session:
            # Find low-info articles without summaries
            result = await session.execute(
                select(Article)
                .join(ArticleScore)
                .outerjoin(Summary)
                .where(ArticleScore.info_score < self.LOW_INFO_THRESHOLD)
                .where(Summary.id.is_(None))
            )
            articles = list(result.scalars().all())

        # Generate summaries (outside the session to avoid long transactions)
        for article in articles:
            summary = await self.summarize(article)
            if summary:
                summaries.append(summary)

        return summaries

    async def get_summary(self, article_id: str) -> Summary | None:
        """Get the summary for an article if it exists.

        Args:
            article_id: The article ID.

        Returns:
            Summary or None.
        """
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(select(Summary).where(Summary.article_id == article_id))
            return result.scalar_one_or_none()

    async def _generate_summary(
        self, title: str, author: str | None, content: str, article_id: str | None = None
    ) -> SummaryResult | None:
        """Generate summary using Claude."""
        # Truncate content if too long
        max_content_length = 15000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"

        prompt = self.SUMMARY_PROMPT.format(
            title=title,
            author=author or "Unknown",
            content=content,
        )

        try:
            model = "claude-sonnet-4-20250514"
            response = self._anthropic.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            # Log usage
            await log_usage(
                service="summarizer",
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                article_id=article_id,
            )

            first_block = response.content[0]
            assert isinstance(first_block, TextBlock)
            text = first_block.text.strip()
            # Remove any markdown code blocks if present
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\n?", "", text)
                text = text.strip()

            data = json.loads(text)

            return SummaryResult(
                summary_text=data.get("summary", ""),
                key_points=data.get("key_points", []),
            )
        except Exception as e:
            logger.error("Error summarizing article: %s", e)
            return None


# Singleton instance
_summarizer: Summarizer | None = None


def get_summarizer() -> Summarizer:
    """Get or create the summarizer singleton."""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer
