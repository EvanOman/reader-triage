"""Tests for the article tagger service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import TextBlock

from app.models.article import Article, ArticleScore, ArticleTag
from app.services.tagger import (
    _COLOR_VALUES,
    _FALLBACK,
    _TAG_CATALOG,
    CURRENT_TAGGING_VERSION,
    TAG_DEFINITIONS,
    TAGS_BY_SLUG,
    ArticleTagger,
    TagDefinition,
    get_all_tags,
    get_tag,
    get_tag_colors,
    get_tag_names,
    get_tag_styles,
)
from tests.factories import mock_anthropic_response


def _build_tagger(claude_response_data: dict[str, object]) -> ArticleTagger:
    """Build an ArticleTagger with mocked dependencies."""
    with (
        patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
        patch("app.services.tagger.get_readwise_service"),
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            claude_response_data, input_tokens=300, output_tokens=50
        )
        mock_anthropic_cls.return_value = mock_client
        tagger = ArticleTagger()
    return tagger


def _build_tagger_with_readwise(
    claude_response_data: dict[str, object], readwise_doc: object
) -> ArticleTagger:
    """Build an ArticleTagger with mocked Anthropic and Readwise."""
    with (
        patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
        patch("app.services.tagger.get_readwise_service") as mock_rw,
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(
            claude_response_data, input_tokens=300, output_tokens=50
        )
        mock_anthropic_cls.return_value = mock_client

        mock_rw_service = AsyncMock()
        mock_rw_service.get_document.return_value = readwise_doc
        mock_rw.return_value = mock_rw_service

        tagger = ArticleTagger()
    return tagger


# ---------------------------------------------------------------------------
# 1. Tag definitions and lookup
# ---------------------------------------------------------------------------


class TestTagDefinitions:
    """Test tag definition data structures and lookups."""

    async def test_tag_definitions_not_empty(self):
        assert len(TAG_DEFINITIONS) > 0

    async def test_all_tags_have_required_fields(self):
        for tag in TAG_DEFINITIONS:
            assert tag.slug, f"Tag missing slug: {tag}"
            assert tag.name, f"Tag missing name: {tag}"
            assert tag.description, f"Tag missing description: {tag}"
            assert tag.color, f"Tag missing color: {tag}"

    async def test_all_slugs_unique(self):
        slugs = [t.slug for t in TAG_DEFINITIONS]
        assert len(slugs) == len(set(slugs))

    async def test_tags_by_slug_has_all_tags(self):
        assert len(TAGS_BY_SLUG) == len(TAG_DEFINITIONS)

    async def test_get_tag_returns_definition(self):
        tag = get_tag("ai-dev-tools")
        assert tag is not None
        assert tag.name == "AI Dev Tools"

    async def test_get_tag_returns_none_for_unknown(self):
        assert get_tag("nonexistent-tag") is None

    async def test_get_all_tags_returns_copy(self):
        tags = get_all_tags()
        assert len(tags) == len(TAG_DEFINITIONS)
        # Should be a new list, not the same object
        assert tags is not TAG_DEFINITIONS

    async def test_get_tag_names_returns_slug_to_name(self):
        names = get_tag_names()
        assert isinstance(names, dict)
        assert names["ai-dev-tools"] == "AI Dev Tools"
        assert names["software-eng"] == "Software Engineering"
        assert len(names) == len(TAG_DEFINITIONS)

    async def test_get_tag_colors_returns_slug_to_color(self):
        colors = get_tag_colors()
        assert isinstance(colors, dict)
        assert colors["ai-dev-tools"] == "blue"
        assert colors["ai-safety"] == "red"
        assert len(colors) == len(TAG_DEFINITIONS)

    async def test_tag_definition_is_frozen(self):
        tag = TAG_DEFINITIONS[0]
        with pytest.raises(AttributeError):
            tag.slug = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Tag styles
# ---------------------------------------------------------------------------


class TestTagStyles:
    """Test inline CSS style generation for tag badges."""

    async def test_get_tag_styles_returns_all_tags(self):
        styles = get_tag_styles()
        assert len(styles) == len(TAG_DEFINITIONS)

    async def test_style_contains_background_and_color(self):
        styles = get_tag_styles()
        for slug, style in styles.items():
            assert "background:" in style, f"Missing background for {slug}"
            assert "color:" in style, f"Missing color for {slug}"

    async def test_known_tag_uses_color_values(self):
        styles = get_tag_styles()
        # ai-dev-tools uses "blue"
        bg, text_color = _COLOR_VALUES["blue"]
        expected = f"background:{bg}; color:{text_color}"
        assert styles["ai-dev-tools"] == expected

    async def test_fallback_for_unknown_color(self):
        """A tag with a color not in _COLOR_VALUES uses the fallback."""
        # Create a temporary tag with unknown color
        bg, text_color = _FALLBACK
        expected_style = f"background:{bg}; color:{text_color}"

        # Manually test the logic with a tag that has an unknown color
        tag = TagDefinition(slug="test", name="Test", description="Test", color="nonexistent-color")
        bg_val, text_val = _COLOR_VALUES.get(tag.color, _FALLBACK)
        style = f"background:{bg_val}; color:{text_val}"
        assert style == expected_style


# ---------------------------------------------------------------------------
# 3. _classify_article (Claude API interaction)
# ---------------------------------------------------------------------------


class TestClassifyArticle:
    """Test the _classify_article method that calls Claude."""

    async def test_valid_tags_returned(self):
        data = {"tags": ["ai-dev-tools", "software-eng"]}
        tagger = _build_tagger(data)
        tags, usage_info = tagger._classify_article("Test Title", "Author", "Content here")

        assert tags == ["ai-dev-tools", "software-eng"]
        assert usage_info is not None
        model, in_tok, out_tok = usage_info
        assert model == "claude-sonnet-4-20250514"
        assert in_tok == 300
        assert out_tok == 50

    async def test_invalid_slugs_filtered_out(self):
        data = {"tags": ["ai-dev-tools", "invalid-slug", "software-eng"]}
        tagger = _build_tagger(data)
        tags, usage_info = tagger._classify_article("Test Title", "Author", "Content here")

        assert tags == ["ai-dev-tools", "software-eng"]
        assert usage_info is not None

    async def test_all_invalid_slugs_returns_empty_list(self):
        data = {"tags": ["bad-1", "bad-2"]}
        tagger = _build_tagger(data)
        tags, usage_info = tagger._classify_article("Test Title", "Author", "Content here")

        assert tags == []
        assert usage_info is not None

    async def test_empty_tags_list(self):
        data = {"tags": []}
        tagger = _build_tagger(data)
        tags, usage_info = tagger._classify_article("Test Title", "Author", "Content here")

        assert tags == []
        assert usage_info is not None

    async def test_missing_tags_key_returns_empty(self):
        data = {"other_key": "value"}
        tagger = _build_tagger(data)
        tags, usage_info = tagger._classify_article("Test Title", "Author", "Content here")

        assert tags == []
        assert usage_info is not None

    async def test_content_truncated_at_15000_chars(self):
        """Content longer than 15000 chars should be truncated."""
        data = {"tags": ["ai-dev-tools"]}
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response(
                data, input_tokens=300, output_tokens=50
            )
            mock_anthropic_cls.return_value = mock_client
            tagger = ArticleTagger()

        long_content = "A" * 20000
        tags, _ = tagger._classify_article("Test", "Author", long_content)

        # Verify the call was made with truncated content
        call_args = mock_client.messages.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "... [truncated]" in prompt_content

    async def test_author_defaults_to_unknown_when_none(self):
        """When author is None, prompt should use 'Unknown'."""
        data = {"tags": ["ai-dev-tools"]}
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response(
                data, input_tokens=300, output_tokens=50
            )
            mock_anthropic_cls.return_value = mock_client
            tagger = ArticleTagger()

        tagger._classify_article("Test", None, "Content")
        call_args = mock_client.messages.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Author: Unknown" in prompt_content

    async def test_markdown_code_blocks_stripped(self):
        """Claude sometimes wraps JSON in markdown code blocks."""
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service"),
        ):
            mock_client = MagicMock()
            content_block = TextBlock(
                type="text",
                text='```json\n{"tags": ["ai-agents", "startups"]}\n```',
            )
            usage = MagicMock()
            usage.input_tokens = 300
            usage.output_tokens = 50
            response = MagicMock()
            response.content = [content_block]
            response.usage = usage
            mock_client.messages.create.return_value = response
            mock_anthropic_cls.return_value = mock_client
            tagger = ArticleTagger()

        tags, _ = tagger._classify_article("Test", "Author", "Content")
        assert tags == ["ai-agents", "startups"]

    async def test_api_error_returns_none(self):
        """When Claude API raises an exception, returns (None, None)."""
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError("API Error")
            mock_anthropic_cls.return_value = mock_client
            tagger = ArticleTagger()

        tags, usage_info = tagger._classify_article("Test", "Author", "Content")
        assert tags is None
        assert usage_info is None

    async def test_invalid_json_returns_none(self):
        """When Claude returns invalid JSON, returns (None, None)."""
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service"),
        ):
            mock_client = MagicMock()
            content_block = TextBlock(type="text", text="not valid json at all")
            usage = MagicMock()
            usage.input_tokens = 300
            usage.output_tokens = 50
            response = MagicMock()
            response.content = [content_block]
            response.usage = usage
            mock_client.messages.create.return_value = response
            mock_anthropic_cls.return_value = mock_client
            tagger = ArticleTagger()

        tags, usage_info = tagger._classify_article("Test", "Author", "Content")
        assert tags is None
        assert usage_info is None


# ---------------------------------------------------------------------------
# 4. _build_tag_catalog
# ---------------------------------------------------------------------------


class TestBuildTagCatalog:
    """Test the tag catalog builder."""

    async def test_catalog_contains_all_slugs(self):
        for tag in TAG_DEFINITIONS:
            assert tag.slug in _TAG_CATALOG
            assert tag.description in _TAG_CATALOG

    async def test_catalog_format(self):
        lines = _TAG_CATALOG.split("\n")
        assert len(lines) == len(TAG_DEFINITIONS)
        for line in lines:
            assert line.startswith("- ")


# ---------------------------------------------------------------------------
# 5. tag_article (integration with DB)
# ---------------------------------------------------------------------------


class TestTagArticle:
    """Test tag_article method with mocked DB sessions."""

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_not_found(self, mock_factory, mock_log_usage, session_factory):
        """Article not in DB returns empty list."""
        mock_factory.return_value = session_factory
        tagger = _build_tagger({"tags": ["ai-dev-tools"]})

        result = await tagger.tag_article("nonexistent-id")
        assert result == []

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_skips_already_tagged(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Article already tagged with current version is skipped."""
        mock_factory.return_value = session_factory

        # Insert article and existing tag
        async with session_factory() as session:
            article = Article(
                id="tagged-1",
                title="Already Tagged",
                url="https://example.com/tagged",
                author="Author",
                location="new",
                category="article",
            )
            tag = ArticleTag(
                article_id="tagged-1",
                tag_slug="ai-dev-tools",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add(article)
            session.add(tag)
            await session.commit()

        tagger = _build_tagger({"tags": ["software-eng"]})
        result = await tagger.tag_article("tagged-1", force=False)
        assert result == []

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_force_retags(self, mock_factory, mock_log_usage, session_factory):
        """Force=True re-tags even if already tagged."""
        mock_factory.return_value = session_factory

        # Insert article with existing tag and content
        async with session_factory() as session:
            article = Article(
                id="retag-1",
                title="Retag Me",
                url="https://example.com/retag",
                author="Author",
                content_preview="Some preview content for tagging.",
                location="new",
                category="article",
            )
            tag = ArticleTag(
                article_id="retag-1",
                tag_slug="ai-dev-tools",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add(article)
            session.add(tag)
            await session.commit()

        # Build tagger with readwise returning content
        readwise_doc = MagicMock()
        readwise_doc.content = "Full article content about software engineering."
        readwise_doc.summary = None
        tagger = _build_tagger_with_readwise({"tags": ["software-eng"]}, readwise_doc)

        result = await tagger.tag_article("retag-1", force=True)
        assert result == ["software-eng"]

        # Verify old tag was replaced
        async with session_factory() as session:
            from sqlalchemy import select

            tags_result = await session.execute(
                select(ArticleTag).where(ArticleTag.article_id == "retag-1")
            )
            tags = list(tags_result.scalars().all())
            assert len(tags) == 1
            assert tags[0].tag_slug == "software-eng"
            assert tags[0].tagging_version == CURRENT_TAGGING_VERSION

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_no_content_returns_empty(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Article with no content (local or remote) returns empty list."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            article = Article(
                id="no-content-1",
                title="No Content",
                url="https://example.com/empty",
                author="Author",
                content_preview=None,
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        # Readwise returns no content either
        tagger = _build_tagger_with_readwise({"tags": ["ai-dev-tools"]}, None)

        result = await tagger.tag_article("no-content-1")
        assert result == []

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_uses_content_preview_fallback(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """When readwise returns no content, falls back to content_preview."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            article = Article(
                id="preview-1",
                title="Preview Only",
                url="https://example.com/preview",
                author="Author",
                content_preview="This article is about AI agents and how they work.",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        # Readwise returns doc with no content or summary
        readwise_doc = MagicMock()
        readwise_doc.content = None
        readwise_doc.summary = None
        tagger = _build_tagger_with_readwise({"tags": ["ai-agents"]}, readwise_doc)

        result = await tagger.tag_article("preview-1")
        assert result == ["ai-agents"]

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_classify_returns_none(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """When _classify_article returns None (API error), returns empty list."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            article = Article(
                id="api-fail-1",
                title="API Fail",
                url="https://example.com/fail",
                author="Author",
                content_preview="Some content here.",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        # Build tagger that will fail on classify
        with (
            patch("app.services.tagger.Anthropic") as mock_anthropic_cls,
            patch("app.services.tagger.get_readwise_service") as mock_rw,
        ):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError("API down")
            mock_anthropic_cls.return_value = mock_client

            mock_rw_service = AsyncMock()
            mock_rw_service.get_document.return_value = None
            mock_rw.return_value = mock_rw_service

            tagger = ArticleTagger()

        result = await tagger.tag_article("api-fail-1")
        assert result == []

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_uses_readwise_content(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Readwise document content is preferred over content_preview."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            article = Article(
                id="rw-content-1",
                title="Readwise Content",
                url="https://example.com/rw",
                author="Author",
                content_preview="Short preview.",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = "Full article about AI safety research and alignment."
        readwise_doc.summary = "Summary about AI safety."
        tagger = _build_tagger_with_readwise({"tags": ["ai-safety"]}, readwise_doc)

        result = await tagger.tag_article("rw-content-1")
        assert result == ["ai-safety"]

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tag_article_uses_readwise_summary_when_no_content(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """When readwise doc has summary but no content, summary is used."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            article = Article(
                id="rw-summary-1",
                title="Readwise Summary",
                url="https://example.com/summary",
                author="Author",
                content_preview="Short preview.",
                location="new",
                category="article",
            )
            session.add(article)
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = None
        readwise_doc.summary = "This article discusses startup funding strategies."
        tagger = _build_tagger_with_readwise({"tags": ["startups"]}, readwise_doc)

        result = await tagger.tag_article("rw-summary-1")
        assert result == ["startups"]


# ---------------------------------------------------------------------------
# 6. tag_untagged_articles
# ---------------------------------------------------------------------------


class TestTagUntaggedArticles:
    """Test batch tagging of untagged articles."""

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_tags_scored_untagged_articles(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Only scored articles without current-version tags get tagged."""
        mock_factory.return_value = session_factory

        # Create two scored articles, one already tagged
        async with session_factory() as session:
            art1 = Article(
                id="untagged-1",
                title="Untagged Article",
                url="https://example.com/untagged",
                author="Author",
                content_preview="Content about AI agents.",
                location="new",
                category="article",
            )
            score1 = ArticleScore(
                article_id="untagged-1",
                info_score=80,
                specificity_score=20,
                novelty_score=20,
                depth_score=20,
                actionability_score=20,
            )
            art2 = Article(
                id="tagged-already",
                title="Tagged Article",
                url="https://example.com/tagged",
                author="Author",
                content_preview="Content about software.",
                location="new",
                category="article",
            )
            score2 = ArticleScore(
                article_id="tagged-already",
                info_score=75,
                specificity_score=18,
                novelty_score=18,
                depth_score=19,
                actionability_score=20,
            )
            tag2 = ArticleTag(
                article_id="tagged-already",
                tag_slug="software-eng",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add_all([art1, score1, art2, score2, tag2])
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = "Full article content about AI agents."
        readwise_doc.summary = None
        tagger = _build_tagger_with_readwise({"tags": ["ai-agents"]}, readwise_doc)

        results = await tagger.tag_untagged_articles()

        # Only the untagged article should be in results
        assert "untagged-1" in results
        assert "tagged-already" not in results

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_no_scored_articles_returns_empty(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """No scored articles means nothing to tag."""
        mock_factory.return_value = session_factory
        tagger = _build_tagger({"tags": ["ai-dev-tools"]})

        results = await tagger.tag_untagged_articles()
        assert results == {}


# ---------------------------------------------------------------------------
# 7. retag_all_articles
# ---------------------------------------------------------------------------


class TestRetagAllArticles:
    """Test force re-tagging all scored articles."""

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_retags_all_scored_articles(self, mock_factory, mock_log_usage, session_factory):
        """All scored articles get re-tagged regardless of existing tags."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art1 = Article(
                id="retag-all-1",
                title="Article One",
                url="https://example.com/1",
                author="Author",
                content_preview="AI agent content.",
                location="new",
                category="article",
            )
            score1 = ArticleScore(
                article_id="retag-all-1",
                info_score=80,
                specificity_score=20,
                novelty_score=20,
                depth_score=20,
                actionability_score=20,
            )
            tag1 = ArticleTag(
                article_id="retag-all-1",
                tag_slug="ai-dev-tools",
                tagging_version=CURRENT_TAGGING_VERSION,
            )
            session.add_all([art1, score1, tag1])
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = "Updated article about software engineering."
        readwise_doc.summary = None
        tagger = _build_tagger_with_readwise({"tags": ["software-eng"]}, readwise_doc)

        results = await tagger.retag_all_articles()

        assert "retag-all-1" in results
        assert results["retag-all-1"] == ["software-eng"]

    @patch("app.services.tagger.log_usage", new_callable=AsyncMock)
    @patch("app.services.tagger.get_session_factory")
    async def test_retag_empty_db(self, mock_factory, mock_log_usage, session_factory):
        """No scored articles means nothing to retag."""
        mock_factory.return_value = session_factory
        tagger = _build_tagger({"tags": ["ai-dev-tools"]})

        results = await tagger.retag_all_articles()
        assert results == {}


# ---------------------------------------------------------------------------
# 8. Singleton
# ---------------------------------------------------------------------------


class TestGetTagger:
    """Test the singleton factory."""

    async def test_get_tagger_returns_instance(self):
        with (
            patch("app.services.tagger.Anthropic"),
            patch("app.services.tagger.get_readwise_service"),
            patch("app.services.tagger._tagger", None),
        ):
            from app.services.tagger import get_tagger

            tagger = get_tagger()
            assert isinstance(tagger, ArticleTagger)

    async def test_get_tagger_returns_same_instance(self):
        with (
            patch("app.services.tagger.Anthropic"),
            patch("app.services.tagger.get_readwise_service"),
            patch("app.services.tagger._tagger", None),
        ):
            from app.services.tagger import get_tagger

            tagger1 = get_tagger()
            tagger2 = get_tagger()
            assert tagger1 is tagger2
