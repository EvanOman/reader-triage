"""Topic tagging service for articles using predefined tags."""

import json
import logging
import re
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import TextBlock
from sqlalchemy import delete, select

from app.config import get_settings
from app.models.article import Article, ArticleScore, ArticleTag, get_session_factory
from app.services.readwise import get_readwise_service
from app.services.usage import log_usage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagDefinition:
    """A predefined topic tag."""

    slug: str
    name: str
    description: str
    color: str  # Tailwind color name (e.g., "blue", "emerald")


TAG_DEFINITIONS: list[TagDefinition] = [
    TagDefinition(
        slug="ai-dev-tools",
        name="AI Dev Tools",
        description="AI-assisted programming, Claude Code, Cursor, Copilot, vibe coding, coding agents",
        color="blue",
    ),
    TagDefinition(
        slug="ai-agents",
        name="AI Agents",
        description="Agentic systems, MCP, tool use, orchestration, agent frameworks, multi-agent workflows",
        color="indigo",
    ),
    TagDefinition(
        slug="ai-safety",
        name="AI Safety",
        description="Alignment research, interpretability, scheming, control, misalignment risks, AI ethics",
        color="red",
    ),
    TagDefinition(
        slug="llm-engineering",
        name="LLM Engineering",
        description="LLM evals, production deployment, guardrails, LLMOps, prompt engineering, data quality",
        color="cyan",
    ),
    TagDefinition(
        slug="agi-scaling",
        name="AGI & Scaling",
        description="AGI timelines, scaling laws, intelligence explosion, capability trajectories, compute economics",
        color="violet",
    ),
    TagDefinition(
        slug="ai-industry",
        name="AI Industry",
        description="AI business models, moats, competitive landscape, AI funding, market dynamics, AI products",
        color="sky",
    ),
    TagDefinition(
        slug="software-eng",
        name="Software Engineering",
        description="Craft of software engineering, future of the developer role, programming practices, developer identity",
        color="teal",
    ),
    TagDefinition(
        slug="eng-leadership",
        name="Engineering Leadership",
        description="Engineering management, staff/principal path, team org, hiring, organizational design",
        color="amber",
    ),
    TagDefinition(
        slug="data-eng",
        name="Data Engineering",
        description="Data pipelines, data architecture, analytics, data modeling, observability",
        color="emerald",
    ),
    TagDefinition(
        slug="startups",
        name="Startups",
        description="Founding companies, product-market fit, fundraising, company building, solo builders, micro-SaaS",
        color="orange",
    ),
    TagDefinition(
        slug="mental-models",
        name="Mental Models",
        description="Decision frameworks, cognitive tools, rationality, second-order thinking, structured reasoning",
        color="lime",
    ),
    TagDefinition(
        slug="behavioral-econ",
        name="Behavioral Econ",
        description="Cognitive biases, nudges, irrational behavior, prospect theory, market psychology",
        color="yellow",
    ),
    TagDefinition(
        slug="productivity",
        name="Productivity",
        description="Deep work, habits, focus, time management, personal systems, attention management",
        color="sky",
    ),
    TagDefinition(
        slug="career",
        name="Career",
        description="Career strategy, professional growth, ambition, job market, career transitions",
        color="rose",
    ),
    TagDefinition(
        slug="personal-finance",
        name="Personal Finance",
        description="Investing, money management, markets, financial literacy, retirement",
        color="green",
    ),
    TagDefinition(
        slug="geopolitics",
        name="Geopolitics",
        description="US-China tech competition, AI governance, energy policy, semiconductors, global strategy",
        color="slate",
    ),
    TagDefinition(
        slug="science-history",
        name="Science & History",
        description="Biographies of innovators, computing history, math, physics, history of technology",
        color="fuchsia",
    ),
    TagDefinition(
        slug="parenting",
        name="Parenting",
        description="Child development, evidence-based parenting, family life",
        color="pink",
    ),
]

TAGS_BY_SLUG: dict[str, TagDefinition] = {t.slug: t for t in TAG_DEFINITIONS}


def get_tag(slug: str) -> TagDefinition | None:
    return TAGS_BY_SLUG.get(slug)


def get_all_tags() -> list[TagDefinition]:
    return list(TAG_DEFINITIONS)


def get_tag_names() -> dict[str, str]:
    """Return slug -> display name mapping."""
    return {t.slug: t.name for t in TAG_DEFINITIONS}


def get_tag_colors() -> dict[str, str]:
    """Return slug -> Tailwind color name mapping."""
    return {t.slug: t.color for t in TAG_DEFINITIONS}


# Tailwind color name → (bg rgba, text hex) for inline styles on dark backgrounds
_COLOR_VALUES: dict[str, tuple[str, str]] = {
    "blue": ("rgba(59,130,246,0.15)", "#93c5fd"),
    "indigo": ("rgba(99,102,241,0.15)", "#a5b4fc"),
    "red": ("rgba(239,68,68,0.15)", "#fca5a5"),
    "cyan": ("rgba(6,182,212,0.15)", "#67e8f9"),
    "violet": ("rgba(139,92,246,0.15)", "#c4b5fd"),
    "sky": ("rgba(14,165,233,0.15)", "#7dd3fc"),
    "teal": ("rgba(20,184,166,0.15)", "#5eead4"),
    "amber": ("rgba(245,158,11,0.15)", "#fcd34d"),
    "emerald": ("rgba(16,185,129,0.15)", "#6ee7b7"),
    "orange": ("rgba(249,115,22,0.15)", "#fdba74"),
    "lime": ("rgba(132,204,22,0.15)", "#bef264"),
    "yellow": ("rgba(234,179,8,0.15)", "#fde047"),
    "rose": ("rgba(244,63,94,0.15)", "#fda4af"),
    "green": ("rgba(34,197,94,0.15)", "#86efac"),
    "slate": ("rgba(148,163,184,0.15)", "#cbd5e1"),
    "fuchsia": ("rgba(217,70,239,0.15)", "#f0abfc"),
    "pink": ("rgba(236,72,153,0.15)", "#f9a8d4"),
}
_FALLBACK = ("rgba(124,107,245,0.15)", "#a5a0f3")


def get_tag_styles() -> dict[str, str]:
    """Return slug -> inline CSS style string for tag badges."""
    styles = {}
    for t in TAG_DEFINITIONS:
        bg, text = _COLOR_VALUES.get(t.color, _FALLBACK)
        styles[t.slug] = f"background:{bg}; color:{text}"
    return styles


CURRENT_TAGGING_VERSION = "v1"

_TAGGING_PROMPT = """Classify this {content_type} into topic tags from the predefined list below.
Multiple tags can and should be applied when content covers multiple topics.
For example, content about building AI agents for startups should get both "ai-agents" and "startups".

Available tags:
{tag_catalog}

Title: {title}
Author: {author}

Content:
{content}

Select ALL tags that meaningfully apply to the primary topics.
Content typically gets 1-3 tags, sometimes more for cross-disciplinary pieces.
Only select tags where the content substantially engages with that topic.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"tags": ["tag-slug-1", "tag-slug-2"]}}"""

_MAX_CONTENT_LENGTH: dict[str, int] = {
    "article": 15000,
    "podcast": 50000,
}


def _build_tag_catalog() -> str:
    """Build the tag catalog string from TAG_DEFINITIONS."""
    lines = []
    for tag in TAG_DEFINITIONS:
        lines.append(f"- {tag.slug}: {tag.description}")
    return "\n".join(lines)


_TAG_CATALOG = _build_tag_catalog()


def classify_content(
    title: str,
    author: str | None,
    content: str,
    anthropic_client: Anthropic,
    content_type_hint: str = "article",
) -> tuple[list[str] | None, tuple[str, int, int] | None]:
    """Classify content into topic tags using Claude.

    Parameters:
        title: Content title.
        author: Content author (or None).
        content: Text content to classify.
        anthropic_client: Anthropic client instance.
        content_type_hint: "article" or "podcast" — adjusts truncation limit and prompt.

    Returns:
        (validated_tag_slugs, (model, input_tokens, output_tokens)) or (None, None).
    """
    max_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
    if len(content) > max_length:
        content = content[:max_length] + "... [truncated]"

    content_type_label = (
        "podcast episode transcript" if content_type_hint == "podcast" else "article"
    )
    prompt = _TAGGING_PROMPT.format(
        content_type=content_type_label,
        tag_catalog=_TAG_CATALOG,
        title=title,
        author=author or "Unknown",
        content=content,
    )

    try:
        model = "claude-sonnet-4-20250514"
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        usage_info = (model, response.usage.input_tokens, response.usage.output_tokens)

        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        text = first_block.text.strip()
        # Remove any markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()

        data = json.loads(text)
        raw_tags = data.get("tags", [])

        # Validate slugs against registry
        valid_tags = [slug for slug in raw_tags if slug in TAGS_BY_SLUG]
        if len(valid_tags) != len(raw_tags):
            invalid = set(raw_tags) - set(valid_tags)
            logger.warning("Dropped invalid tag slugs: %s", invalid)

        return valid_tags, usage_info
    except Exception as e:
        logger.exception("Error classifying content '%s': %s", title, e)
        return None, None


class ArticleTagger:
    """Tags articles with predefined topic tags using Claude."""

    def __init__(self):
        settings = get_settings()
        self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
        self._readwise = get_readwise_service()

    async def tag_article(self, article_id: str, force: bool = False) -> list[str]:
        """Tag a single article. Returns list of tag slugs assigned.

        If force=False, skips articles already tagged with current version.
        If force=True, re-tags regardless.
        """
        factory = await get_session_factory()
        async with factory() as session:
            article = await session.get(Article, article_id)
            if not article:
                logger.warning("Article %s not found", article_id)
                return []

            # Check if already tagged with current version
            if not force:
                existing = await session.execute(
                    select(ArticleTag)
                    .where(ArticleTag.article_id == article_id)
                    .where(ArticleTag.tagging_version == CURRENT_TAGGING_VERSION)
                    .limit(1)
                )
                if existing.scalar_one_or_none():
                    return []

            # Fetch content
            content = article.content_preview or ""
            doc = await self._readwise.get_document(article_id, with_content=True)
            if doc and (doc.content or doc.summary):
                content = doc.content or doc.summary or content

            if not content:
                logger.warning("No content for article %s", article_id)
                return []

            # Classify
            tag_slugs, usage_info = self._classify_article(article.title, article.author, content)
            if tag_slugs is None:
                return []

            # Log API usage
            if usage_info:
                model, in_tok, out_tok = usage_info
                await log_usage("tagger", model, in_tok, out_tok, article_id=article_id)

            # Delete existing tags and insert new ones
            await session.execute(delete(ArticleTag).where(ArticleTag.article_id == article_id))
            for slug in tag_slugs:
                session.add(
                    ArticleTag(
                        article_id=article_id,
                        tag_slug=slug,
                        tagging_version=CURRENT_TAGGING_VERSION,
                    )
                )
            await session.commit()

            logger.info("Tagged article %s with %s", article_id, tag_slugs)
            return tag_slugs

    async def tag_untagged_articles(self) -> dict[str, list[str]]:
        """Tag all scored articles that haven't been tagged with current version.

        Returns dict of article_id -> list of tag slugs.
        """
        factory = await get_session_factory()
        results: dict[str, list[str]] = {}

        # Find articles that have scores but no tags with current version
        async with factory() as session:
            # Subquery: article IDs already tagged with current version
            tagged_ids = (
                select(ArticleTag.article_id)
                .where(ArticleTag.tagging_version == CURRENT_TAGGING_VERSION)
                .distinct()
                .scalar_subquery()
            )

            result = await session.execute(
                select(Article.id).join(ArticleScore).where(Article.id.not_in(tagged_ids))
            )
            article_ids = [row[0] for row in result.all()]

        logger.info("Found %d articles to tag", len(article_ids))

        for article_id in article_ids:
            tags = await self.tag_article(article_id, force=False)
            if tags:
                results[article_id] = tags

        return results

    async def retag_all_articles(self) -> dict[str, list[str]]:
        """Force re-tag every scored article.

        Returns dict of article_id -> list of tag slugs.
        """
        factory = await get_session_factory()
        results: dict[str, list[str]] = {}

        async with factory() as session:
            result = await session.execute(select(Article.id).join(ArticleScore))
            article_ids = [row[0] for row in result.all()]

        logger.info("Re-tagging %d articles", len(article_ids))

        for article_id in article_ids:
            tags = await self.tag_article(article_id, force=True)
            if tags:
                results[article_id] = tags

        return results

    def _classify_article(
        self, title: str, author: str | None, content: str
    ) -> tuple[list[str] | None, tuple[str, int, int] | None]:
        """Call Claude to classify an article.

        Returns (validated_tag_slugs, (model, input_tokens, output_tokens)) or (None, None).
        """
        return classify_content(
            title=title,
            author=author,
            content=content,
            anthropic_client=self._anthropic,
            content_type_hint="article",
        )


# Singleton instance
_tagger: ArticleTagger | None = None


def get_tagger() -> ArticleTagger:
    """Get or create the tagger singleton."""
    global _tagger
    if _tagger is None:
        _tagger = ArticleTagger()
    return _tagger
