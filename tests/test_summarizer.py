"""Tests for the article summarizer service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.article import Article, ArticleScore, Summary
from app.services.summarizer import (
    Summarizer,
    SummaryResult,
)
from tests.factories import mock_litellm_response


@pytest.fixture(autouse=True)
def _stop_lingering_patches():
    """Stop any patches started via .start() after each test."""
    yield
    patch.stopall()


# Module-level handle to the most recently installed acompletion mock.
# Tests that need to inspect call args read this directly.
_last_acompletion_mock: AsyncMock | None = None


def _mock_acompletion(
    claude_response_data: dict[str, object] | str,
    *,
    input_tokens: int = 400,
    output_tokens: int = 150,
) -> AsyncMock:
    """Patch litellm.acompletion and return the AsyncMock.

    Patch is auto-stopped by the autouse fixture at test teardown.
    """
    global _last_acompletion_mock
    mock = AsyncMock(
        return_value=mock_litellm_response(
            claude_response_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    )
    patch("app.services.summarizer.litellm.acompletion", mock).start()
    _last_acompletion_mock = mock
    return mock


def _build_summarizer(claude_response_data: dict[str, object]) -> Summarizer:
    """Build a Summarizer with litellm.acompletion mocked."""
    _mock_acompletion(claude_response_data)
    patch("app.services.summarizer.get_readwise_service").start()
    return Summarizer()


def _build_summarizer_with_readwise(
    claude_response_data: dict[str, object], readwise_doc: object
) -> Summarizer:
    """Build a Summarizer with litellm and Readwise mocked."""
    _mock_acompletion(claude_response_data)
    rw_service = AsyncMock()
    rw_service.get_document.return_value = readwise_doc
    patch("app.services.summarizer.get_readwise_service", return_value=rw_service).start()
    return Summarizer()


def _make_article(
    *,
    article_id: str = "art-1",
    title: str = "Test Article",
    author: str | None = "Author",
) -> Article:
    """Create a minimal Article instance for testing."""
    return Article(
        id=article_id,
        title=title,
        url="https://example.com/test",
        author=author,
        word_count=1000,
        location="new",
        category="article",
    )


# ---------------------------------------------------------------------------
# 1. SummaryResult dataclass
# ---------------------------------------------------------------------------


class TestSummaryResult:
    """Test the SummaryResult dataclass."""

    async def test_summary_result_fields(self):
        result = SummaryResult(
            summary_text="A summary.",
            key_points=["Point 1", "Point 2"],
        )
        assert result.summary_text == "A summary."
        assert result.key_points == ["Point 1", "Point 2"]

    async def test_summary_result_empty(self):
        result = SummaryResult(summary_text="", key_points=[])
        assert result.summary_text == ""
        assert result.key_points == []


# ---------------------------------------------------------------------------
# 2. _generate_summary (Claude API interaction)
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    """Test the _generate_summary method."""

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_generates_summary_from_content(self, mock_log_usage):
        data = {
            "summary": "This article discusses AI safety.",
            "key_points": ["Point 1", "Point 2", "Point 3"],
        }
        summarizer = _build_summarizer(data)

        result = await summarizer._generate_summary("AI Safety", "Author", "Full content here.")
        assert result is not None
        assert result.summary_text == "This article discusses AI safety."
        assert result.key_points == ["Point 1", "Point 2", "Point 3"]

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_logs_usage(self, mock_log_usage):
        data = {"summary": "Summary text.", "key_points": ["Point 1"]}
        summarizer = _build_summarizer(data)

        await summarizer._generate_summary("Title", "Author", "Content", article_id="art-1")
        mock_log_usage.assert_awaited_once()
        kwargs = mock_log_usage.call_args.kwargs
        assert kwargs["service"] == "summarizer"
        assert kwargs["input_tokens"] == 400
        assert kwargs["output_tokens"] == 150
        assert kwargs["article_id"] == "art-1"
        # Model name varies by config but must not include the provider prefix
        assert "/" not in kwargs["model"]

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_truncates_long_content(self, mock_log_usage):
        """Content longer than 15000 chars is truncated."""
        data = {"summary": "Summary.", "key_points": ["P1"]}
        summarizer = _build_summarizer(data)
        mock_acompletion = _last_acompletion_mock
        assert mock_acompletion is not None

        long_content = "B" * 20000
        result = await summarizer._generate_summary("Title", "Author", long_content)

        prompt_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
        assert "... [truncated]" in prompt_content
        assert result is not None

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_author_defaults_to_unknown(self, mock_log_usage):
        """When author is None, the prompt uses 'Unknown'."""
        data = {"summary": "Summary.", "key_points": ["P1"]}
        summarizer = _build_summarizer(data)
        mock_acompletion = _last_acompletion_mock
        assert mock_acompletion is not None

        await summarizer._generate_summary("Title", None, "Content")
        prompt_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
        assert "Author: Unknown" in prompt_content

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_markdown_code_blocks_stripped(self, mock_log_usage):
        """LLMs sometimes wrap JSON in markdown code blocks."""
        json_data = {"summary": "Summary from markdown.", "key_points": ["P1"]}
        _mock_acompletion(f"```json\n{json.dumps(json_data)}\n```")
        patch("app.services.summarizer.get_readwise_service").start()
        summarizer = Summarizer()

        result = await summarizer._generate_summary("Title", "Author", "Content")
        assert result is not None
        assert result.summary_text == "Summary from markdown."

    async def test_api_error_returns_none(self):
        """When the LLM API raises an exception, returns None."""
        patch(
            "app.services.summarizer.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API down"),
        ).start()
        patch("app.services.summarizer.get_readwise_service").start()
        summarizer = Summarizer()

        result = await summarizer._generate_summary("Title", "Author", "Content")
        assert result is None

    async def test_invalid_json_returns_none(self):
        """When the LLM returns invalid JSON, returns None."""
        _mock_acompletion("This is not JSON")
        patch("app.services.summarizer.get_readwise_service").start()
        summarizer = Summarizer()

        result = await summarizer._generate_summary("Title", "Author", "Content")
        assert result is None

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_missing_keys_default_to_empty(self, mock_log_usage):
        """When JSON response is missing keys, they default to empty."""
        data = {"other": "value"}  # no "summary" or "key_points" keys
        summarizer = _build_summarizer(data)

        result = await summarizer._generate_summary("Title", "Author", "Content")
        assert result is not None
        assert result.summary_text == ""
        assert result.key_points == []

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    async def test_content_not_truncated_under_limit(self, mock_log_usage):
        """Content under 15000 chars is not truncated."""
        data = {"summary": "Summary.", "key_points": ["P1"]}
        summarizer = _build_summarizer(data)
        mock_acompletion = _last_acompletion_mock
        assert mock_acompletion is not None

        short_content = "C" * 10000
        await summarizer._generate_summary("Title", "Author", short_content)
        prompt_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
        assert "... [truncated]" not in prompt_content


# ---------------------------------------------------------------------------
# 3. summarize (full flow with DB)
# ---------------------------------------------------------------------------


class TestSummarize:
    """Test the summarize method with mocked DB and Readwise."""

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_success(self, mock_factory, mock_log_usage, session_factory):
        """Successfully summarize an article and store in DB."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-1")

        readwise_doc = MagicMock()
        readwise_doc.content = "Full article content for summarization."
        readwise_doc.summary = None

        data = {
            "summary": "The article covers key topics.",
            "key_points": ["Point A", "Point B"],
        }
        summarizer = _build_summarizer_with_readwise(data, readwise_doc)

        result = await summarizer.summarize(article)
        assert result is not None
        assert isinstance(result, Summary)
        assert result.summary_text == "The article covers key topics."
        assert json.loads(result.key_points) == ["Point A", "Point B"]
        assert result.article_id == "sum-1"

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_no_readwise_doc(self, mock_factory, mock_log_usage, session_factory):
        """Returns None when Readwise returns no document."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-2")
        summarizer = _build_summarizer_with_readwise({"summary": "Unused.", "key_points": []}, None)

        result = await summarizer.summarize(article)
        assert result is None

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_empty_content(self, mock_factory, mock_log_usage, session_factory):
        """Returns None when Readwise doc has no content or summary."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-3")
        readwise_doc = MagicMock()
        readwise_doc.content = None
        readwise_doc.summary = None
        summarizer = _build_summarizer_with_readwise(
            {"summary": "Unused.", "key_points": []}, readwise_doc
        )

        result = await summarizer.summarize(article)
        assert result is None

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_empty_string_content(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Returns None when Readwise doc has empty string content."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-4")
        readwise_doc = MagicMock()
        readwise_doc.content = ""
        readwise_doc.summary = ""
        summarizer = _build_summarizer_with_readwise(
            {"summary": "Unused.", "key_points": []}, readwise_doc
        )

        result = await summarizer.summarize(article)
        assert result is None

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_uses_summary_when_no_content(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Falls back to readwise doc.summary when content is None."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-5")
        readwise_doc = MagicMock()
        readwise_doc.content = None
        readwise_doc.summary = "A summary from readwise about the article topic."

        data = {
            "summary": "Generated from the readwise summary.",
            "key_points": ["Key point"],
        }
        summarizer = _build_summarizer_with_readwise(data, readwise_doc)

        result = await summarizer.summarize(article)
        assert result is not None
        assert result.summary_text == "Generated from the readwise summary."

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarize_api_failure_returns_none(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Returns None when _generate_summary fails (API error)."""
        mock_factory.return_value = session_factory

        article = _make_article(article_id="sum-6")
        readwise_doc = MagicMock()
        readwise_doc.content = "Some content to summarize."
        readwise_doc.summary = None

        patch(
            "app.services.summarizer.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API failure"),
        ).start()
        rw_service = AsyncMock()
        rw_service.get_document.return_value = readwise_doc
        patch("app.services.summarizer.get_readwise_service", return_value=rw_service).start()
        summarizer = Summarizer()

        result = await summarizer.summarize(article)
        assert result is None


# ---------------------------------------------------------------------------
# 4. get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Test retrieving existing summaries from DB."""

    @patch("app.services.summarizer.get_session_factory")
    async def test_get_summary_exists(self, mock_factory, session_factory):
        """Returns existing summary for an article."""
        mock_factory.return_value = session_factory

        # Insert article and summary
        async with session_factory() as session:
            article = Article(
                id="get-sum-1",
                title="Article",
                url="https://example.com/1",
                author="Author",
                location="new",
                category="article",
            )
            summary = Summary(
                article_id="get-sum-1",
                summary_text="Existing summary.",
                key_points='["Point 1"]',
            )
            session.add(article)
            session.add(summary)
            await session.commit()

        patch("app.services.summarizer.get_readwise_service").start()
        summarizer = Summarizer()

        result = await summarizer.get_summary("get-sum-1")
        assert result is not None
        assert result.summary_text == "Existing summary."

    @patch("app.services.summarizer.get_session_factory")
    async def test_get_summary_not_found(self, mock_factory, session_factory):
        """Returns None when no summary exists."""
        mock_factory.return_value = session_factory

        patch("app.services.summarizer.get_readwise_service").start()
        summarizer = Summarizer()

        result = await summarizer.get_summary("nonexistent-article")
        assert result is None


# ---------------------------------------------------------------------------
# 5. summarize_low_info_articles
# ---------------------------------------------------------------------------


class TestSummarizeLowInfoArticles:
    """Test batch summarization of low-info articles."""

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_summarizes_low_score_articles(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Articles below LOW_INFO_THRESHOLD that lack summaries get summarized."""
        mock_factory.return_value = session_factory

        # Create a low-score article without a summary
        async with session_factory() as session:
            art = Article(
                id="low-1",
                title="Low Info Article",
                url="https://example.com/low",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="low-1",
                info_score=20,  # Below threshold of 30
                specificity_score=5,
                novelty_score=5,
                depth_score=5,
                actionability_score=5,
            )
            session.add(art)
            session.add(score)
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = "Content to summarize."
        readwise_doc.summary = None

        data = {
            "summary": "A low-info article summarized.",
            "key_points": ["Key point 1", "Key point 2"],
        }
        summarizer = _build_summarizer_with_readwise(data, readwise_doc)

        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 1
        assert results[0].article_id == "low-1"

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_skips_high_score_articles(self, mock_factory, mock_log_usage, session_factory):
        """Articles above threshold are not summarized."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art = Article(
                id="high-1",
                title="High Info Article",
                url="https://example.com/high",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="high-1",
                info_score=80,  # Above threshold
                specificity_score=20,
                novelty_score=20,
                depth_score=20,
                actionability_score=20,
            )
            session.add(art)
            session.add(score)
            await session.commit()

        summarizer = _build_summarizer({"summary": "Unused.", "key_points": []})
        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 0

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_skips_already_summarized(self, mock_factory, mock_log_usage, session_factory):
        """Low-info articles that already have a summary are skipped."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art = Article(
                id="already-sum-1",
                title="Already Summarized",
                url="https://example.com/already",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="already-sum-1",
                info_score=15,  # Below threshold
                specificity_score=3,
                novelty_score=3,
                depth_score=5,
                actionability_score=4,
            )
            existing_summary = Summary(
                article_id="already-sum-1",
                summary_text="Existing summary.",
                key_points='["Existing point"]',
            )
            session.add(art)
            session.add(score)
            session.add(existing_summary)
            await session.commit()

        summarizer = _build_summarizer({"summary": "Unused.", "key_points": []})
        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 0

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_threshold_boundary_exactly_30(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """Article with info_score exactly at threshold (30) is NOT summarized."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art = Article(
                id="boundary-1",
                title="Boundary Article",
                url="https://example.com/boundary",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="boundary-1",
                info_score=30,  # Exactly at threshold (< 30 means below)
                specificity_score=7,
                novelty_score=8,
                depth_score=8,
                actionability_score=7,
            )
            session.add(art)
            session.add(score)
            await session.commit()

        summarizer = _build_summarizer({"summary": "Unused.", "key_points": []})
        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 0

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_threshold_boundary_29(self, mock_factory, mock_log_usage, session_factory):
        """Article with info_score=29 IS summarized."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art = Article(
                id="boundary-29",
                title="Below Boundary",
                url="https://example.com/boundary29",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="boundary-29",
                info_score=29,  # Just below threshold
                specificity_score=7,
                novelty_score=7,
                depth_score=8,
                actionability_score=7,
            )
            session.add(art)
            session.add(score)
            await session.commit()

        readwise_doc = MagicMock()
        readwise_doc.content = "Content for summarization."
        readwise_doc.summary = None

        data = {"summary": "Below boundary summary.", "key_points": ["P1"]}
        summarizer = _build_summarizer_with_readwise(data, readwise_doc)

        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 1
        assert results[0].article_id == "boundary-29"

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_empty_db_returns_empty(self, mock_factory, mock_log_usage, session_factory):
        """No articles in DB means no summaries generated."""
        mock_factory.return_value = session_factory
        summarizer = _build_summarizer({"summary": "Unused.", "key_points": []})

        results = await summarizer.summarize_low_info_articles()
        assert results == []

    @patch("app.services.summarizer.log_usage", new_callable=AsyncMock)
    @patch("app.services.summarizer.get_session_factory")
    async def test_handles_summarize_failure_gracefully(
        self, mock_factory, mock_log_usage, session_factory
    ):
        """When individual summarize fails, it's skipped (not added to results)."""
        mock_factory.return_value = session_factory

        async with session_factory() as session:
            art = Article(
                id="fail-sum-1",
                title="Fail to Summarize",
                url="https://example.com/fail",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="fail-sum-1",
                info_score=10,
                specificity_score=2,
                novelty_score=3,
                depth_score=3,
                actionability_score=2,
            )
            session.add(art)
            session.add(score)
            await session.commit()

        # Readwise returns no document, so summarize returns None
        summarizer = _build_summarizer_with_readwise({"summary": "Unused.", "key_points": []}, None)

        results = await summarizer.summarize_low_info_articles()
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 6. Singleton
# ---------------------------------------------------------------------------


class TestGetSummarizer:
    """Test the singleton factory."""

    async def test_get_summarizer_returns_instance(self):
        with (
            patch("app.services.summarizer.get_readwise_service"),
            patch("app.services.summarizer._summarizer", None),
        ):
            from app.services.summarizer import get_summarizer

            summarizer = get_summarizer()
            assert isinstance(summarizer, Summarizer)

    async def test_get_summarizer_returns_same_instance(self):
        with (
            patch("app.services.summarizer.get_readwise_service"),
            patch("app.services.summarizer._summarizer", None),
        ):
            from app.services.summarizer import get_summarizer

            s1 = get_summarizer()
            s2 = get_summarizer()
            assert s1 is s2

    async def test_low_info_threshold(self):
        """Verify the threshold constant."""
        assert Summarizer.LOW_INFO_THRESHOLD == 30
