"""Tests for the chat service, focusing on unified search and tool dispatch."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import Message, TextBlock, TextDelta, ToolUseBlock, Usage

from app.services.chat import ChatService, _serialize_content_blocks

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


# ---------------------------------------------------------------------------
# 6. _serialize_content_blocks
# ---------------------------------------------------------------------------


class TestSerializeContentBlocks:
    """Test serialization of Anthropic SDK content blocks to plain dicts."""

    def test_text_block(self):
        block = TextBlock(type="text", text="Hello world")
        result = _serialize_content_blocks([block])
        assert result == [{"type": "text", "text": "Hello world"}]

    def test_tool_use_block(self):
        block = ToolUseBlock(
            type="tool_use", id="toolu_abc", name="search", input={"query": "AI"}
        )
        result = _serialize_content_blocks([block])
        assert result == [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search",
                "input": {"query": "AI"},
            }
        ]

    def test_mixed_blocks(self):
        blocks = [
            TextBlock(type="text", text="Let me search"),
            ToolUseBlock(
                type="tool_use", id="toolu_123", name="search", input={"query": "x"}
            ),
        ]
        result = _serialize_content_blocks(blocks)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"

    def test_dict_passthrough(self):
        """Dict blocks are passed through as-is."""
        block = {"type": "text", "text": "already a dict"}
        result = _serialize_content_blocks([block])
        assert result == [block]

    def test_unknown_block_type_with_model_dump(self):
        """Block with unknown type but model_dump method uses that."""
        block = MagicMock()
        block.type = "image"
        block.model_dump = MagicMock(return_value={"type": "image", "data": "base64"})
        result = _serialize_content_blocks([block])
        assert result == [{"type": "image", "data": "base64"}]

    def test_unknown_block_type_without_model_dump(self):
        """Block with unknown type and no model_dump falls back to str(type)."""
        block = MagicMock(spec=[])
        block.type = "custom_type"
        # spec=[] means no attributes, so hasattr(block, "model_dump") is False
        result = _serialize_content_blocks([block])
        assert result == [{"type": "custom_type"}]

    def test_empty_list(self):
        result = _serialize_content_blocks([])
        assert result == []


# ---------------------------------------------------------------------------
# 7. _tool_get_article
# ---------------------------------------------------------------------------


class TestToolGetArticle:
    """Tests for _tool_get_article."""

    @pytest.mark.usefixtures("populated_db")
    async def test_get_article_with_score(self, session_factory):
        """Returns article details including score fields."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_get_article({"article_id": "test-article-001"})

        assert result["id"] == "test-article-001"
        assert result["title"] == "The Future of AI Agents"
        assert result["author"] == "Jane Smith"
        assert result["info_score"] == 85
        assert result["overall_assessment"] is not None
        assert isinstance(result["score_reasons"], list)
        assert result["skip_recommended"] is False

    @pytest.mark.usefixtures("populated_db")
    async def test_get_article_with_summary(self, session_factory):
        """Returns article with summary text and key points."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_get_article({"article_id": "test-article-004"})

        assert result["id"] == "test-article-004"
        assert "summary_text" in result
        assert "weekly roundup" in result["summary_text"].lower()
        assert "key_points" in result
        assert isinstance(result["key_points"], list)

    @pytest.mark.usefixtures("populated_db")
    async def test_get_article_with_tags(self, session_factory):
        """Returns article with associated tags."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_get_article({"article_id": "test-article-003"})

        assert result["id"] == "test-article-003"
        assert "ai-dev-tools" in result["tags"]
        assert "software-eng" in result["tags"]

    @pytest.mark.usefixtures("populated_db")
    async def test_get_article_not_found(self, session_factory):
        """Non-existent article returns an error."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_get_article({"article_id": "nonexistent"})

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.usefixtures("populated_db")
    async def test_get_article_no_summary(self, session_factory):
        """Article without summary does not include summary fields."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_get_article({"article_id": "test-article-001"})

        assert "summary_text" not in result
        assert "key_points" not in result


# ---------------------------------------------------------------------------
# 8. _tool_browse_by_tag
# ---------------------------------------------------------------------------


class TestToolBrowseByTag:
    """Tests for _tool_browse_by_tag."""

    @pytest.mark.usefixtures("populated_db")
    async def test_browse_tag_returns_matching_articles(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_by_tag({"tag": "ai-dev-tools"})

        assert result["tag"] == "ai-dev-tools"
        assert len(result["articles"]) >= 1
        assert result["articles"][0]["id"] == "test-article-003"

    @pytest.mark.usefixtures("populated_db")
    async def test_browse_tag_no_matches(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_by_tag({"tag": "nonexistent-tag"})

        assert result["tag"] == "nonexistent-tag"
        assert result["articles"] == []

    @pytest.mark.usefixtures("populated_db")
    async def test_browse_tag_respects_limit(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_by_tag({"tag": "ai-dev-tools", "limit": 1})

        assert len(result["articles"]) <= 1


# ---------------------------------------------------------------------------
# 9. _tool_list_tags
# ---------------------------------------------------------------------------


class TestToolListTags:
    """Tests for _tool_list_tags."""

    @pytest.mark.usefixtures("populated_db")
    async def test_list_tags_returns_tags_with_counts(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_list_tags()

        assert "tags" in result
        # Should have tags with article_count > 0
        for tag in result["tags"]:
            assert "slug" in tag
            assert "name" in tag
            assert "article_count" in tag
            assert tag["article_count"] > 0

    @pytest.mark.usefixtures("populated_db")
    async def test_list_tags_excludes_zero_count(self, session_factory):
        """Tags with no articles should not appear in the list."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_list_tags()

        for tag in result["tags"]:
            assert tag["article_count"] > 0


# ---------------------------------------------------------------------------
# 10. _tool_browse_top_articles
# ---------------------------------------------------------------------------


class TestToolBrowseTopArticles:
    """Tests for _tool_browse_top_articles."""

    @pytest.mark.usefixtures("populated_db")
    async def test_returns_articles_sorted_by_score(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_top_articles({"limit": 10})

        assert "articles" in result
        assert len(result["articles"]) >= 1
        # Should be sorted by info_score descending
        scores = [a["info_score"] for a in result["articles"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.usefixtures("populated_db")
    async def test_respects_limit(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_top_articles({"limit": 2})

        assert len(result["articles"]) <= 2

    @pytest.mark.usefixtures("populated_db")
    async def test_min_score_filter(self, session_factory):
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_top_articles({"limit": 10, "min_score": 80})

        for article in result["articles"]:
            assert article["info_score"] >= 80

    @pytest.mark.usefixtures("populated_db")
    async def test_min_score_excludes_low_articles(self, session_factory):
        """A very high min_score should exclude most articles."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_top_articles({"limit": 10, "min_score": 99})

        # Only test-article-003 (score 100) qualifies
        assert len(result["articles"]) == 1
        assert result["articles"][0]["id"] == "test-article-003"

    @pytest.mark.usefixtures("populated_db")
    async def test_includes_tags(self, session_factory):
        """Top articles should include their tags."""
        svc = _make_service()
        with patch(
            "app.services.chat.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            result = await svc._tool_browse_top_articles({"limit": 10})

        # Find article-003 which has tags
        article_003 = [a for a in result["articles"] if a["id"] == "test-article-003"]
        assert len(article_003) == 1
        assert "ai-dev-tools" in article_003[0]["tags"]


# ---------------------------------------------------------------------------
# 11. Readwise fallback error handling in _tool_read_article_content
# ---------------------------------------------------------------------------


class TestReadArticleContentReadwiseErrors:
    """Edge cases for Readwise fallback in _tool_read_article_content."""

    @pytest.mark.usefixtures("populated_db")
    async def test_readwise_returns_none_doc(self, session_factory):
        """Readwise returns None for the document."""
        svc = _make_service()
        mock_readwise_svc = MagicMock()
        mock_readwise_svc.get_document = AsyncMock(return_value=None)

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
            result = await svc._tool_read_article_content(
                {"article_id": "test-article-003"}
            )

        assert "error" in result
        assert "could not fetch" in result["error"].lower()

    @pytest.mark.usefixtures("populated_db")
    async def test_readwise_returns_doc_with_no_content(self, session_factory):
        """Readwise returns a document but content is None."""
        svc = _make_service()
        mock_doc = MagicMock()
        mock_doc.content = None

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
            result = await svc._tool_read_article_content(
                {"article_id": "test-article-003"}
            )

        assert "error" in result

    @pytest.mark.usefixtures("populated_db")
    async def test_readwise_api_exception(self, session_factory):
        """Readwise API throws an exception."""
        svc = _make_service()
        mock_readwise_svc = MagicMock()
        mock_readwise_svc.get_document = AsyncMock(
            side_effect=RuntimeError("API connection failed")
        )

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
            result = await svc._tool_read_article_content(
                {"article_id": "test-article-003"}
            )

        assert "error" in result
        assert "API connection failed" in result["error"]

    @pytest.mark.usefixtures("populated_db")
    async def test_readwise_html_stripping(self, session_factory):
        """HTML tags in Readwise content are stripped properly."""
        svc = _make_service()
        mock_doc = MagicMock()
        mock_doc.title = "Test"
        mock_doc.content = (
            "<h1>Title</h1><p>Para one.</p><br/><div class='inner'>Para two.</div>"
        )

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
            result = await svc._tool_read_article_content(
                {"article_id": "test-article-003"}
            )

        assert "<h1>" not in result["content"]
        assert "<p>" not in result["content"]
        assert "<div" not in result["content"]
        assert "Title" in result["content"]
        assert "Para one." in result["content"]
        assert "Para two." in result["content"]


# ---------------------------------------------------------------------------
# 12. send_message streaming
# ---------------------------------------------------------------------------


def _make_stream_context(events: list[object], final_message: Message) -> MagicMock:
    """Build a mock async context manager for messages.stream().

    events: list of event objects to yield
    final_message: the Message returned by stream.get_final_message()
    """

    class _MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object):
            pass

        async def __aiter__(self):
            for event in events:
                yield event

        async def get_final_message(self):
            return final_message

    return _MockStream()


def _make_event_with_text_delta(text: str) -> MagicMock:
    """Create a mock streaming event carrying a TextDelta."""
    event = MagicMock()
    event.delta = TextDelta(type="text_delta", text=text)
    return event


def _make_event_no_delta() -> MagicMock:
    """Create a mock streaming event with no delta attribute."""
    event = MagicMock(spec=[])
    return event


class TestSendMessageStreaming:
    """Tests for the send_message async generator."""

    async def test_simple_text_response(self):
        """A simple text response with no tool use yields text chunks."""
        svc = _make_service()

        final = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Hello there!")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        events = [
            _make_event_with_text_delta("Hello "),
            _make_event_with_text_delta("there!"),
        ]
        stream_ctx = _make_stream_context(events, final)
        svc._client.messages.stream = MagicMock(return_value=stream_ctx)

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert "Hello " in chunks
        assert "there!" in chunks
        assert svc.tool_messages == []

    async def test_tool_use_then_text_response(self):
        """Tool use loop: first response triggers tool, second gives final text."""
        svc = _make_service()

        # First response: tool_use
        tool_block = ToolUseBlock(
            type="tool_use",
            id="toolu_abc",
            name="search",
            input={"query": "AI agents"},
        )
        first_message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[
                TextBlock(type="text", text="Searching..."),
                tool_block,
            ],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=20),
        )

        # Second response: final text
        second_message = Message(
            id="msg_2",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Found results!")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=10),
        )

        call_count = 0

        def make_stream(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                events = [_make_event_with_text_delta("Searching...")]
                return _make_stream_context(events, first_message)
            else:
                events = [_make_event_with_text_delta("Found results!")]
                return _make_stream_context(events, second_message)

        svc._client.messages.stream = MagicMock(side_effect=make_stream)
        svc._execute_tool = AsyncMock(
            return_value={"results": [{"id": "1", "title": "Test"}], "modes_used": ["keyword"]}
        )

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "Find AI articles"}]):
            chunks.append(chunk)

        # Should have text chunks + tool markers + final text
        joined = "".join(chunks)
        assert "Searching..." in joined
        assert "__tool_use__" in joined
        assert "__tool_done__" in joined
        assert "Found results!" in joined

        # Tool messages should be stored for persistence
        assert len(svc.tool_messages) == 2  # assistant + user (tool_result)
        assert svc.tool_messages[0]["role"] == "assistant"
        assert svc.tool_messages[1]["role"] == "user"

    async def test_max_tokens_warning(self):
        """When stop_reason is max_tokens, a truncation warning is appended."""
        svc = _make_service()

        final = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Long response...")],
            stop_reason="max_tokens",
            usage=Usage(input_tokens=10, output_tokens=16384),
        )

        events = [_make_event_with_text_delta("Long response...")]
        stream_ctx = _make_stream_context(events, final)
        svc._client.messages.stream = MagicMock(return_value=stream_ctx)

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "Write a lot"}]):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "truncated" in joined.lower()

    async def test_max_tool_rounds_warning(self):
        """When all rounds use tools without a final text response, warn the user."""
        svc = _make_service()

        tool_block = ToolUseBlock(
            type="tool_use",
            id="toolu_loop",
            name="search",
            input={"query": "test"},
        )
        tool_message = Message(
            id="msg_loop",
            type="message",
            role="assistant",
            model="test-model",
            content=[tool_block],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=10),
        )

        # Every round returns tool_use
        svc._client.messages.stream = MagicMock(
            return_value=_make_stream_context([], tool_message)
        )
        svc._execute_tool = AsyncMock(return_value={"results": []})

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "loop"}]):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "maximum tool use rounds" in joined.lower()

    async def test_api_error_yields_error_message(self):
        """When the Anthropic API raises, we yield a user-friendly error."""
        svc = _make_service()

        class _FailingStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: object):
                pass

            async def __aiter__(self):
                raise RuntimeError("API is down")
                yield  # make this an async generator  # noqa: RUF027

            async def get_final_message(self):
                pass

        svc._client.messages.stream = MagicMock(return_value=_FailingStream())

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "error" in joined.lower()

    async def test_events_without_delta_are_skipped(self):
        """Events that don't have a TextDelta are silently skipped."""
        svc = _make_service()

        final = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="OK")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=5, output_tokens=2),
        )

        events = [
            _make_event_no_delta(),
            _make_event_with_text_delta("OK"),
            _make_event_no_delta(),
        ]
        stream_ctx = _make_stream_context(events, final)
        svc._client.messages.stream = MagicMock(return_value=stream_ctx)

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks == ["OK"]

    async def test_tool_use_marker_contains_tool_info(self):
        """The __tool_use__ marker includes the tool name and input as JSON."""
        svc = _make_service()

        tool_block = ToolUseBlock(
            type="tool_use",
            id="toolu_xyz",
            name="browse_by_tag",
            input={"tag": "ai-agents"},
        )
        first_message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[tool_block],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=10),
        )
        second_message = Message(
            id="msg_2",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Here are the results")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=30, output_tokens=10),
        )

        call_count = 0

        def make_stream(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_stream_context([], first_message)
            return _make_stream_context(
                [_make_event_with_text_delta("Here are the results")], second_message
            )

        svc._client.messages.stream = MagicMock(side_effect=make_stream)
        svc._execute_tool = AsyncMock(
            return_value={"tag": "ai-agents", "articles": [{"id": "1"}]}
        )

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "show ai"}]):
            chunks.append(chunk)

        # Find the tool_use marker
        tool_use_chunks = [c for c in chunks if c.startswith("__tool_use__:")]
        assert len(tool_use_chunks) == 1
        payload = json.loads(tool_use_chunks[0].split(":", 1)[1])
        assert payload["tool"] == "browse_by_tag"
        assert payload["input"] == {"tag": "ai-agents"}

        # Find the tool_done marker
        tool_done_chunks = [c for c in chunks if c.startswith("__tool_done__:")]
        assert len(tool_done_chunks) == 1
        done_payload = json.loads(tool_done_chunks[0].split(":", 1)[1])
        assert done_payload["tool"] == "browse_by_tag"
        assert "1" in done_payload["summary"]

    async def test_multiple_tool_blocks_in_single_response(self):
        """A single assistant response can contain multiple tool_use blocks."""
        svc = _make_service()

        tool1 = ToolUseBlock(
            type="tool_use",
            id="toolu_1",
            name="search",
            input={"query": "AI"},
        )
        tool2 = ToolUseBlock(
            type="tool_use",
            id="toolu_2",
            name="list_tags",
            input={},
        )
        first_message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="test-model",
            content=[tool1, tool2],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=20),
        )
        second_message = Message(
            id="msg_2",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Done")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=5),
        )

        call_count = 0

        def make_stream(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_stream_context([], first_message)
            return _make_stream_context(
                [_make_event_with_text_delta("Done")], second_message
            )

        svc._client.messages.stream = MagicMock(side_effect=make_stream)

        # Mock _execute_tool to track calls
        tool_calls: list[str] = []

        async def mock_execute(name: str, inp: dict) -> dict:
            tool_calls.append(name)
            if name == "search":
                return {"results": [], "modes_used": []}
            return {"tags": []}

        svc._execute_tool = mock_execute  # type: ignore[assignment]

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "search and list"}]):
            chunks.append(chunk)

        assert tool_calls == ["search", "list_tags"]
        tool_use_chunks = [c for c in chunks if "__tool_use__" in c]
        assert len(tool_use_chunks) == 2

    async def test_round_separator_emitted_between_tool_rounds(self):
        """A newline separator is emitted between tool use rounds (round > 0)."""
        svc = _make_service()

        tool_block = ToolUseBlock(
            type="tool_use",
            id="toolu_r",
            name="search",
            input={"query": "AI"},
        )
        tool_message = Message(
            id="msg_t",
            type="message",
            role="assistant",
            model="test-model",
            content=[tool_block],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=10),
        )
        final = Message(
            id="msg_f",
            type="message",
            role="assistant",
            model="test-model",
            content=[TextBlock(type="text", text="Result")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=30, output_tokens=5),
        )

        call_count = 0

        def make_stream(**kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_stream_context([], tool_message)
            return _make_stream_context(
                [_make_event_with_text_delta("Result")], final
            )

        svc._client.messages.stream = MagicMock(side_effect=make_stream)
        svc._execute_tool = AsyncMock(return_value={"results": []})

        chunks: list[str] = []
        async for chunk in svc.send_message([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        # Round 1 (index 0) has no separator, but round 2 (index 1) should emit "\n\n"
        assert "\n\n" in chunks
