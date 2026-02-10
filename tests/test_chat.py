"""Tests for the chat service, focusing on unified search and tool dispatch."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat import ChatService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> ChatService:
    """Create a ChatService with a mocked Anthropic client."""
    svc = ChatService.__new__(ChatService)
    svc._client = MagicMock()
    svc._chat_model = "test-model"
    svc.tool_messages = []
    return svc


# ---------------------------------------------------------------------------
# 1. Unified search tool (_tool_search)
# ---------------------------------------------------------------------------


class TestToolSearchKeywordOnly:
    """Keyword-only search: modes=["keyword"]."""

    @pytest.mark.usefixtures("populated_db")
    async def test_keyword_only_returns_results(self, session_factory):
        svc = _make_service()
        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=["test-article-001", "test-article-002"],
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "AI agents", "modes": ["keyword"]})

        assert result["modes_used"] == ["keyword"]
        assert len(result["results"]) == 2
        ids = [r["id"] for r in result["results"]]
        assert "test-article-001" in ids
        assert "test-article-002" in ids

    @pytest.mark.usefixtures("populated_db")
    async def test_keyword_only_does_not_call_semantic(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(return_value=[])

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=["test-article-001"],
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ) as mock_get_vs,
        ):
            result = await svc._tool_search({"query": "AI", "modes": ["keyword"]})

        # get_vectorstore should never have been called
        mock_get_vs.assert_not_called()
        assert result["modes_used"] == ["keyword"]


class TestToolSearchSemanticOnly:
    """Semantic-only search: modes=["semantic"]."""

    @pytest.mark.usefixtures("populated_db")
    async def test_semantic_only_returns_results(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(
            return_value=[
                {"article_id": "test-article-003", "similarity": 0.9},
                {"article_id": "test-article-001", "similarity": 0.7},
            ]
        )

        with (
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "software abundance", "modes": ["semantic"]})

        assert result["modes_used"] == ["semantic"]
        assert len(result["results"]) == 2
        # First result should be test-article-003 (rank 0 in semantic)
        assert result["results"][0]["id"] == "test-article-003"

    @pytest.mark.usefixtures("populated_db")
    async def test_semantic_only_does_not_call_fts(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(
            return_value=[{"article_id": "test-article-001", "similarity": 0.8}]
        )

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
            ) as mock_fts,
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "agents", "modes": ["semantic"]})

        mock_fts.assert_not_called()
        assert result["modes_used"] == ["semantic"]


class TestToolSearchHybrid:
    """Hybrid (default) search: both keyword + semantic, merged via RRF."""

    @pytest.mark.usefixtures("populated_db")
    async def test_hybrid_uses_both_modes(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(
            return_value=[
                {"article_id": "test-article-001", "similarity": 0.9},
                {"article_id": "test-article-003", "similarity": 0.7},
            ]
        )

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=["test-article-001", "test-article-002"],
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "AI agents"})

        assert sorted(result["modes_used"]) == ["keyword", "semantic"]
        # article-001 appears in both => should be first (highest RRF)
        assert result["results"][0]["id"] == "test-article-001"

    @pytest.mark.usefixtures("populated_db")
    async def test_hybrid_default_modes(self, session_factory):
        """When modes not supplied, defaults to both keyword + semantic."""
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(
            return_value=[{"article_id": "test-article-001", "similarity": 0.8}]
        )

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=["test-article-001"],
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "AI"})

        assert "keyword" in result["modes_used"]
        assert "semantic" in result["modes_used"]


class TestToolSearchLimit:
    """The limit parameter is respected."""

    @pytest.mark.usefixtures("populated_db")
    async def test_limit_caps_results(self, session_factory):
        svc = _make_service()
        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=[
                    "test-article-001",
                    "test-article-002",
                    "test-article-003",
                    "test-article-004",
                ],
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "test", "modes": ["keyword"], "limit": 2})

        assert len(result["results"]) <= 2


class TestToolSearchEmptyResults:
    """Empty results when no matches found."""

    async def test_no_matches_returns_empty(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.search_articles_fts",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await svc._tool_search({"query": "nonexistent", "modes": ["keyword"]})

        assert result["results"] == []
        assert result["modes_used"] == []

    async def test_both_modes_empty(self):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(return_value=[])

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
        ):
            result = await svc._tool_search({"query": "xyz"})

        assert result["results"] == []
        assert result["modes_used"] == []


class TestToolSearchGracefulDegradation:
    """If one search mode fails, the other still returns results."""

    @pytest.mark.usefixtures("populated_db")
    async def test_semantic_fails_keyword_still_works(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(side_effect=RuntimeError("Qdrant unavailable"))

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                return_value=["test-article-001"],
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "AI"})

        assert result["modes_used"] == ["keyword"]
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "test-article-001"

    @pytest.mark.usefixtures("populated_db")
    async def test_keyword_fails_semantic_still_works(self, session_factory):
        svc = _make_service()
        mock_vs = MagicMock()
        mock_vs.search = MagicMock(
            return_value=[{"article_id": "test-article-002", "similarity": 0.85}]
        )

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                side_effect=RuntimeError("FTS table missing"),
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "Obsidian"})

        assert result["modes_used"] == ["semantic"]
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "test-article-002"


# ---------------------------------------------------------------------------
# 2. RRF algorithm
# ---------------------------------------------------------------------------


class TestRRFAlgorithm:
    """Reciprocal Rank Fusion score calculation."""

    @pytest.mark.usefixtures("populated_db")
    async def test_article_in_both_rankings_scores_higher(self, session_factory):
        """An article appearing in both keyword and semantic gets a higher
        RRF score than one appearing in only one ranking."""
        svc = _make_service()
        mock_vs = MagicMock()
        # Semantic returns article-001 (rank 0) and article-003 (rank 1)
        mock_vs.search = MagicMock(
            return_value=[
                {"article_id": "test-article-001", "similarity": 0.9},
                {"article_id": "test-article-003", "similarity": 0.8},
            ]
        )

        with (
            patch(
                "app.services.chat.search_articles_fts",
                new_callable=AsyncMock,
                # Keyword returns article-001 (rank 0), article-002 (rank 1)
                return_value=["test-article-001", "test-article-002"],
            ),
            patch(
                "app.services.vectorstore.get_vectorstore",
                return_value=mock_vs,
            ),
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
        ):
            result = await svc._tool_search({"query": "AI"})

        # article-001 is in both => must be ranked first
        assert result["results"][0]["id"] == "test-article-001"

        # articles appearing in only one ranking should come after
        single_mode_ids = {r["id"] for r in result["results"][1:]}
        assert "test-article-002" in single_mode_ids
        assert "test-article-003" in single_mode_ids

    async def test_rrf_score_values(self):
        """Verify the exact RRF score arithmetic:
        - rank 0 in both modes  => 2 * 1/(60+0) = 2/60
        - rank 0 in one mode    => 1/(60+0)     = 1/60
        """
        k = 60  # the constant used in the code

        # Rank 0 in a single mode
        single_score = 1.0 / (k + 0)
        assert single_score == pytest.approx(1 / 60)
        assert single_score == pytest.approx(0.016667, rel=1e-3)

        # Rank 0 in both modes
        double_score = 2.0 / (k + 0)
        assert double_score == pytest.approx(2 / 60)
        assert double_score == pytest.approx(0.03333, rel=1e-3)

        # The dual-ranking score must be exactly twice the single
        assert double_score == pytest.approx(2 * single_score)

    async def test_rrf_later_ranks_score_lower(self):
        """Higher rank indices yield lower RRF contributions."""
        k = 60
        scores = [1.0 / (k + r) for r in range(5)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


# ---------------------------------------------------------------------------
# 3. read_article_content optimization
# ---------------------------------------------------------------------------


class TestReadArticleContent:
    """Tests for _tool_read_article_content."""

    @pytest.mark.usefixtures("populated_db")
    async def test_returns_db_content_when_available(self, session_factory):
        """Article with content in DB returns it directly (no API call)."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_read_article_content({"article_id": "test-article-001"})

        assert "error" not in result
        assert result["article_id"] == "test-article-001"
        assert "AI agents are transforming" in result["content"]
        assert result["title"] == "The Future of AI Agents"

    @pytest.mark.usefixtures("populated_db")
    async def test_no_readwise_call_when_content_exists(self, session_factory):
        """When content is in the DB, the Readwise API should NOT be called."""
        svc = _make_service()
        with (
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
            patch(
                "app.services.readwise.get_readwise_service",
            ) as mock_readwise,
        ):
            await svc._tool_read_article_content({"article_id": "test-article-001"})

        mock_readwise.assert_not_called()

    @pytest.mark.usefixtures("populated_db")
    async def test_falls_back_to_readwise_when_no_content(self, session_factory):
        """Article with no stored content falls back to Readwise API."""
        svc = _make_service()

        mock_doc = MagicMock()
        mock_doc.title = "Software Abundance in the Age of AI"
        mock_doc.content = "<p>When software becomes abundant, taste matters.</p>"

        mock_readwise_svc = MagicMock()
        mock_readwise_svc.get_document = AsyncMock(return_value=mock_doc)

        with (
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
            patch(
                "app.services.readwise.get_readwise_service",
                return_value=mock_readwise_svc,
            ),
        ):
            result = await svc._tool_read_article_content({"article_id": "test-article-003"})

        assert "error" not in result
        assert result["article_id"] == "test-article-003"
        # HTML tags should be stripped
        assert "<p>" not in result["content"]
        assert "taste matters" in result["content"]

    @pytest.mark.usefixtures("populated_db")
    async def test_content_truncated_at_10000_chars(self, session_factory):
        """Content longer than 10,000 chars is truncated."""
        svc = _make_service()

        # Temporarily make article-001's content very long
        long_content = "A" * 15000
        mock_doc = MagicMock()
        mock_doc.title = "Long Article"
        mock_doc.content = long_content  # raw content, no HTML for simplicity

        mock_readwise_svc = MagicMock()
        mock_readwise_svc.get_document = AsyncMock(return_value=mock_doc)

        with (
            patch(
                "app.services.chat.get_session_factory",
                new_callable=AsyncMock,
                return_value=session_factory,
            ),
            patch(
                "app.services.readwise.get_readwise_service",
                return_value=mock_readwise_svc,
            ),
        ):
            # article-003 has no content in DB, so it falls back to Readwise
            result = await svc._tool_read_article_content({"article_id": "test-article-003"})

        assert "truncated" in result["content"].lower()
        # The first 10000 chars should be present, plus the truncation notice
        assert len(result["content"]) < 15000

    @pytest.mark.usefixtures("populated_db")
    async def test_content_truncated_from_db(self, session, session_factory):
        """DB-stored content longer than 10,000 chars is also truncated."""
        # Update article-002 to have very long content
        from app.models.article import Article

        article = await session.get(Article, "test-article-002")
        article.content = "B" * 15000
        await session.commit()

        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_read_article_content({"article_id": "test-article-002"})

        assert "truncated" in result["content"].lower()
        # 10000 chars of content + truncation message
        assert result["content"].startswith("B" * 100)

    @pytest.mark.usefixtures("populated_db")
    async def test_article_not_found_returns_error(self, session_factory):
        """Non-existent article ID returns an error dict."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_read_article_content({"article_id": "nonexistent-id"})

        assert "error" in result
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# 4. Tool result summaries (_tool_result_summary)
# ---------------------------------------------------------------------------


class TestToolResultSummary:
    """Test _tool_result_summary for various tool types."""

    def test_search_summary_with_count_and_modes(self):
        result = {
            "results": [{"id": "1"}, {"id": "2"}, {"id": "3"}],
            "modes_used": ["keyword", "semantic"],
        }
        summary = ChatService._tool_result_summary("search", result)
        assert "3" in summary
        assert "keyword" in summary
        assert "semantic" in summary

    def test_search_summary_single_result(self):
        result = {
            "results": [{"id": "1"}],
            "modes_used": ["keyword"],
        }
        summary = ChatService._tool_result_summary("search", result)
        assert "1 article" in summary
        # singular, not plural
        assert "articles" not in summary

    def test_search_summary_zero_results(self):
        result = {"results": [], "modes_used": []}
        summary = ChatService._tool_result_summary("search", result)
        assert "0" in summary

    def test_get_article_shows_truncated_title(self):
        result = {"title": "A" * 80}
        summary = ChatService._tool_result_summary("get_article", result)
        assert summary.startswith("Loaded: ")
        # Title truncated at 50 chars
        assert len(summary) <= len("Loaded: ") + 50

    def test_get_article_short_title(self):
        result = {"title": "Short Title"}
        summary = ChatService._tool_result_summary("get_article", result)
        assert summary == "Loaded: Short Title"

    def test_read_article_content_summary(self):
        result = {"content": "x" * 5000}
        summary = ChatService._tool_result_summary("read_article_content", result)
        assert "5000" in summary
        assert "chars" in summary.lower()

    def test_browse_by_tag_summary(self):
        result = {"tag": "ai-agents", "articles": [{"id": "1"}, {"id": "2"}]}
        summary = ChatService._tool_result_summary("browse_by_tag", result)
        assert "2" in summary
        assert "ai-agents" in summary

    def test_list_tags_summary(self):
        result = {"tags": [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]}
        summary = ChatService._tool_result_summary("list_tags", result)
        assert "3" in summary

    def test_browse_top_articles_summary(self):
        result = {"articles": [{"id": "1"}]}
        summary = ChatService._tool_result_summary("browse_top_articles", result)
        assert "1" in summary

    def test_error_result(self):
        result = {"error": "Something went wrong"}
        summary = ChatService._tool_result_summary("search", result)
        assert summary == "Something went wrong"

    def test_unknown_tool_returns_done(self):
        result = {"some": "data"}
        summary = ChatService._tool_result_summary("unknown_tool", result)
        assert summary == "Done"


# ---------------------------------------------------------------------------
# 5. Tool dispatch (_execute_tool)
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Test that _execute_tool routes to the correct handler."""

    async def test_routes_to_search(self):
        svc = _make_service()
        svc._tool_search = AsyncMock(return_value={"results": []})
        result = await svc._execute_tool("search", {"query": "test"})
        svc._tool_search.assert_awaited_once_with({"query": "test"})
        assert result == {"results": []}

    async def test_routes_to_get_article(self):
        svc = _make_service()
        svc._tool_get_article = AsyncMock(return_value={"id": "123"})
        result = await svc._execute_tool("get_article", {"article_id": "123"})
        svc._tool_get_article.assert_awaited_once_with({"article_id": "123"})
        assert result == {"id": "123"}

    async def test_routes_to_read_article_content(self):
        svc = _make_service()
        svc._tool_read_article_content = AsyncMock(return_value={"content": "hello"})
        result = await svc._execute_tool("read_article_content", {"article_id": "123"})
        svc._tool_read_article_content.assert_awaited_once()
        assert result == {"content": "hello"}

    async def test_routes_to_browse_by_tag(self):
        svc = _make_service()
        svc._tool_browse_by_tag = AsyncMock(return_value={"articles": []})
        result = await svc._execute_tool("browse_by_tag", {"tag": "ai"})
        svc._tool_browse_by_tag.assert_awaited_once_with({"tag": "ai"})
        assert result == {"articles": []}

    async def test_routes_to_list_tags(self):
        svc = _make_service()
        svc._tool_list_tags = AsyncMock(return_value={"tags": []})
        result = await svc._execute_tool("list_tags", {})
        svc._tool_list_tags.assert_awaited_once()
        assert result == {"tags": []}

    async def test_routes_to_browse_top_articles(self):
        svc = _make_service()
        svc._tool_browse_top_articles = AsyncMock(return_value={"articles": []})
        result = await svc._execute_tool("browse_top_articles", {"limit": 5})
        svc._tool_browse_top_articles.assert_awaited_once_with({"limit": 5})
        assert result == {"articles": []}

    async def test_unknown_tool_returns_error(self):
        svc = _make_service()
        result = await svc._execute_tool("nonexistent_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    async def test_tool_exception_returns_error(self):
        svc = _make_service()
        svc._tool_search = AsyncMock(side_effect=ValueError("boom"))
        result = await svc._execute_tool("search", {"query": "x"})
        assert "error" in result
        assert "boom" in result["error"]
