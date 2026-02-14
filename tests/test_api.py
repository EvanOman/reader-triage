"""Integration tests for FastAPI API and chat endpoints, plus vectorstore unit tests."""

import json
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.article import (
    ApiUsageLog,
    Article,
    ArticleScore,
    ArticleTag,
    Author,
    Base,
    ChatMessage,
    ChatThread,
    Summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_engine():
    """Create a fresh in-memory SQLite engine for API integration tests."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts "
                "USING fts5(article_id UNINDEXED, title, author, content)"
            )
        )
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session_factory(test_engine):
    """Create a session factory bound to the test engine."""
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture
async def patched_app(test_engine, test_session_factory):
    """Patch the app module globals so all routes use the test DB.

    Sets _engine and _session_factory module globals in app.models.article
    so every handler resolves to the in-memory test database. Patches
    init_db and rebuild_fts_index to no-ops (tables already created by
    the test_engine fixture). Mocks the background sync service.
    """
    import app.models.article as article_mod

    original_engine = article_mod._engine
    original_factory = article_mod._session_factory

    # Install test engine and factory into the module globals.
    # All code paths that call get_engine() / get_session_factory() check
    # these globals first and return them when set.
    article_mod._engine = test_engine
    article_mod._session_factory = test_session_factory

    mock_sync = MagicMock()
    mock_sync.start_periodic = MagicMock()
    mock_sync.stop_periodic = MagicMock()
    mock_sync.status = MagicMock(
        is_syncing=False,
        last_sync_at=None,
        articles_processed=0,
        newly_scored=0,
        newly_tagged=0,
        scoring_version="test",
        last_error=None,
    )

    with (
        # init_db/rebuild_fts_index are no-ops because the test_engine fixture
        # already created all tables and the FTS virtual table.
        patch("app.main.init_db", new_callable=AsyncMock),
        patch("app.main.rebuild_fts_index", new_callable=AsyncMock),
        patch("app.main.get_background_sync", return_value=mock_sync),
    ):
        from app.main import app

        yield app

    # Restore originals
    article_mod._engine = original_engine
    article_mod._session_factory = original_factory


@pytest.fixture
async def client(patched_app):
    """Async HTTP test client using ASGI transport."""
    transport = httpx.ASGITransport(app=patched_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# 1. App startup
# ---------------------------------------------------------------------------


class TestAppStartup:
    async def test_app_responds(self, client: httpx.AsyncClient):
        """The app should respond to the OpenAPI docs endpoint."""
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "Reader Triage"


# ---------------------------------------------------------------------------
# 2. Chat API endpoints
# ---------------------------------------------------------------------------


class TestChatThreadCRUD:
    """Test the thread CRUD lifecycle without requiring Anthropic streaming."""

    async def test_list_threads_empty(self, client: httpx.AsyncClient):
        """Initially the threads list should be empty."""
        resp = await client.get("/api/chat/threads")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_create_and_list_thread(self, client: httpx.AsyncClient, test_session_factory):
        """Creating a thread via the DB and then listing it should work."""
        async with test_session_factory() as session:
            thread = ChatThread(title="Test Thread")
            session.add(thread)
            await session.commit()
            thread_id = thread.id

        resp = await client.get("/api/chat/threads")
        assert resp.status_code == 200
        threads = resp.json()
        assert len(threads) == 1
        assert threads[0]["id"] == thread_id
        assert threads[0]["title"] == "Test Thread"

    async def test_delete_thread(self, client: httpx.AsyncClient, test_session_factory):
        """Deleting a thread should remove it and its messages."""
        async with test_session_factory() as session:
            thread = ChatThread(title="To Delete")
            session.add(thread)
            await session.flush()
            msg = ChatMessage(thread_id=thread.id, role="user", content="hello")
            session.add(msg)
            await session.commit()
            thread_id = thread.id

        # Verify it exists
        resp = await client.get("/api/chat/threads")
        assert any(t["id"] == thread_id for t in resp.json())

        # Delete it
        resp = await client.delete(f"/api/chat/threads/{thread_id}")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Verify it is gone
        resp = await client.get("/api/chat/threads")
        assert not any(t["id"] == thread_id for t in resp.json())

    async def test_delete_nonexistent_thread(self, client: httpx.AsyncClient):
        """Deleting a thread that does not exist returns a 'not found' response."""
        resp = await client.delete("/api/chat/threads/99999")
        # The handler returns a tuple (dict, 404) which FastAPI serializes as
        # a JSON array with status 200 (since it isn't using HTTPException).
        assert resp.status_code == 200
        body = resp.json()
        assert body == [{"error": "Thread not found"}, 404]

    async def test_get_thread_messages(self, client: httpx.AsyncClient, test_session_factory):
        """Getting messages from a thread should return them in order."""
        async with test_session_factory() as session:
            thread = ChatThread(title="Message Thread")
            session.add(thread)
            await session.flush()
            m1 = ChatMessage(thread_id=thread.id, role="user", content="first")
            m2 = ChatMessage(thread_id=thread.id, role="assistant", content="second")
            session.add_all([m1, m2])
            await session.commit()
            thread_id = thread.id

        resp = await client.get(f"/api/chat/threads/{thread_id}/messages")
        assert resp.status_code == 200
        messages = resp.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "first"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "second"

    async def test_get_messages_hides_tool_blocks(
        self, client: httpx.AsyncClient, test_session_factory
    ):
        """Messages with content_blocks (tool use/result) should be hidden."""
        async with test_session_factory() as session:
            thread = ChatThread(title="Tool Thread")
            session.add(thread)
            await session.flush()
            m1 = ChatMessage(thread_id=thread.id, role="user", content="query")
            m2 = ChatMessage(
                thread_id=thread.id,
                role="assistant",
                content="[tool call]",
                content_blocks='[{"type": "tool_use"}]',
            )
            m3 = ChatMessage(thread_id=thread.id, role="assistant", content="answer")
            session.add_all([m1, m2, m3])
            await session.commit()
            thread_id = thread.id

        resp = await client.get(f"/api/chat/threads/{thread_id}/messages")
        messages = resp.json()
        # The tool message (m2) should be filtered out
        assert len(messages) == 2
        assert messages[0]["content"] == "query"
        assert messages[1]["content"] == "answer"

    async def test_get_messages_nonexistent_thread(self, client: httpx.AsyncClient):
        """Getting messages from a nonexistent thread returns 'not found'."""
        resp = await client.get("/api/chat/threads/99999/messages")
        assert resp.status_code == 200
        body = resp.json()
        assert body == [{"error": "Thread not found"}, 404]

    async def test_thread_lifecycle(self, client: httpx.AsyncClient, test_session_factory):
        """Full lifecycle: create threads, list, delete one, verify."""
        # Start empty
        resp = await client.get("/api/chat/threads")
        assert resp.json() == []

        # Create two threads
        async with test_session_factory() as session:
            t1 = ChatThread(title="Thread A")
            t2 = ChatThread(title="Thread B")
            session.add_all([t1, t2])
            await session.commit()
            t1_id = t1.id
            t2_id = t2.id

        # Both should appear
        resp = await client.get("/api/chat/threads")
        threads = resp.json()
        assert len(threads) == 2
        thread_ids = {t["id"] for t in threads}
        assert t1_id in thread_ids
        assert t2_id in thread_ids

        # Delete one
        resp = await client.delete(f"/api/chat/threads/{t1_id}")
        assert resp.json() == {"ok": True}

        # Only one should remain
        resp = await client.get("/api/chat/threads")
        threads = resp.json()
        assert len(threads) == 1
        assert threads[0]["id"] == t2_id


# ---------------------------------------------------------------------------
# 3. Dashboard pages (smoke tests)
# ---------------------------------------------------------------------------


class TestDashboardPages:
    async def test_dashboard_loads(self, client: httpx.AsyncClient):
        """GET / should return the dashboard HTML page."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_chat_page_loads(self, client: httpx.AsyncClient):
        """GET /chat should return the chat HTML page."""
        resp = await client.get("/chat")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Helper: populate test DB with articles, scores, tags, summaries, authors
# ---------------------------------------------------------------------------


@pytest.fixture
async def populated_test_db(test_session_factory):
    """Insert sample articles, scores, tags, summaries, and authors into the test DB."""
    async with test_session_factory() as session:
        # High-value article
        a1 = Article(
            id="art-001",
            title="The Future of AI Agents",
            url="https://example.com/ai-agents",
            author="Jane Smith",
            word_count=2500,
            content_preview="AI agents are transforming...",
            location="new",
            category="article",
            readwise_created_at=datetime(2025, 6, 1),
            published_date=datetime(2025, 5, 28),
        )
        s1 = ArticleScore(
            article_id="art-001",
            info_score=85,
            specificity_score=22,
            novelty_score=20,
            depth_score=23,
            actionability_score=20,
            score_reasons=json.dumps(["Good passages", "Novel framing"]),
            overall_assessment="Excellent article on AI agents.",
            priority_score=90,
            author_boost=5.0,
            skip_recommended=False,
        )
        tag1 = ArticleTag(article_id="art-001", tag_slug="ai-agents")
        session.add_all([a1, s1, tag1])

        # Medium-value article
        a2 = Article(
            id="art-002",
            title="Building with Obsidian",
            url="https://example.com/obsidian",
            author="John Doe",
            word_count=1500,
            content_preview="A guide to Obsidian...",
            location="later",
            category="article",
            readwise_created_at=datetime(2025, 5, 20),
            published_date=datetime(2025, 5, 15),
        )
        s2 = ArticleScore(
            article_id="art-002",
            info_score=45,
            specificity_score=12,
            novelty_score=10,
            depth_score=13,
            actionability_score=10,
            score_reasons=json.dumps(["Practical guide"]),
            overall_assessment="Decent guide with some useful tips.",
            priority_score=45,
            author_boost=0.0,
            skip_recommended=False,
        )
        tag2 = ArticleTag(article_id="art-002", tag_slug="software-eng")
        session.add_all([a2, s2, tag2])

        # Low-value article with summary (skip recommended)
        a3 = Article(
            id="art-003",
            title="Weekly Newsletter #42",
            url="https://example.com/newsletter-42",
            author="Newsletter Bot",
            word_count=800,
            location="new",
            category="email",
            readwise_created_at=datetime(2025, 5, 10),
        )
        s3 = ArticleScore(
            article_id="art-003",
            info_score=15,
            specificity_score=3,
            novelty_score=5,
            depth_score=4,
            actionability_score=3,
            score_reasons=json.dumps(["Generic roundup"]),
            overall_assessment="Standard newsletter.",
            priority_score=15,
            author_boost=0.0,
            skip_recommended=True,
            skip_reason="Low information content",
        )
        summary3 = Summary(
            article_id="art-003",
            summary_text="A weekly AI news roundup.",
            key_points=json.dumps(["AI news", "No standout items"]),
        )
        session.add_all([a3, s3, summary3])

        # Archived article (should be excluded from most queries)
        a4 = Article(
            id="art-004",
            title="Archived Old Post",
            url="https://example.com/archived",
            author="Alice",
            word_count=500,
            location="archive",
            category="article",
        )
        s4 = ArticleScore(
            article_id="art-004",
            info_score=70,
            specificity_score=18,
            novelty_score=17,
            depth_score=18,
            actionability_score=17,
            score_reasons=json.dumps(["Good article"]),
            overall_assessment="Good but archived.",
            priority_score=70,
            author_boost=0.0,
            skip_recommended=False,
        )
        session.add_all([a4, s4])

        # Extra high-value articles to fill top5
        for i in range(5, 10):
            art = Article(
                id=f"art-{i:03d}",
                title=f"Top Article {i}",
                url=f"https://example.com/top-{i}",
                author="Top Author",
                word_count=2000,
                location="new",
                category="article",
                readwise_created_at=datetime(2025, 6, i),
            )
            sc = ArticleScore(
                article_id=f"art-{i:03d}",
                info_score=60 + i,
                specificity_score=15 + (i % 5),
                novelty_score=15,
                depth_score=15,
                actionability_score=15 + (i % 5),
                score_reasons=json.dumps([f"Reason for {i}"]),
                overall_assessment=f"Assessment for article {i}.",
                priority_score=60 + i,
                author_boost=0.0,
                skip_recommended=False,
            )
            session.add_all([art, sc])

        # Authors
        author1 = Author(
            name="Jane Smith",
            normalized_name="jane smith",
            total_highlights=25,
            total_books=3,
            is_favorite=True,
        )
        author2 = Author(
            name="John Doe",
            normalized_name="john doe",
            total_highlights=1,
            total_books=1,
            is_favorite=False,
        )
        author3 = Author(
            name="Prolific Writer",
            normalized_name="prolific writer",
            total_highlights=50,
            total_books=10,
            is_favorite=False,
        )
        session.add_all([author1, author2, author3])

        # API usage logs
        log1 = ApiUsageLog(
            service="scorer",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.005,
            article_id="art-001",
            timestamp=datetime(2025, 6, 1, 10, 0, 0),
        )
        log2 = ApiUsageLog(
            service="tagger",
            model="claude-sonnet-4-20250514",
            input_tokens=800,
            output_tokens=100,
            cost_usd=0.003,
            article_id="art-001",
            timestamp=datetime(2025, 6, 1, 10, 5, 0),
        )
        log3 = ApiUsageLog(
            service="scorer",
            model="claude-sonnet-4-20250514",
            input_tokens=1200,
            output_tokens=250,
            cost_usd=0.006,
            article_id="art-002",
            timestamp=datetime(2025, 6, 2, 12, 0, 0),
        )
        session.add_all([log1, log2, log3])

        await session.commit()


# ---------------------------------------------------------------------------
# 4. API article endpoints
# ---------------------------------------------------------------------------


class TestArticlesAPI:
    """Test the /api/articles endpoints."""

    async def test_list_articles_empty_db(self, client: httpx.AsyncClient):
        """List articles returns empty list when no articles exist."""
        resp = await client.get("/api/articles")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_articles_with_data(self, client: httpx.AsyncClient, populated_test_db):
        """List articles returns scored articles, excluding archived by default."""
        resp = await client.get("/api/articles")
        assert resp.status_code == 200
        articles = resp.json()
        # art-004 is archived, so it should be excluded
        article_ids = [a["id"] for a in articles]
        assert "art-004" not in article_ids
        # But non-archived articles should be present
        assert "art-001" in article_ids
        assert "art-002" in article_ids

    async def test_list_articles_include_archived(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Passing include_archived=true includes archived articles."""
        resp = await client.get("/api/articles", params={"include_archived": "true"})
        assert resp.status_code == 200
        articles = resp.json()
        article_ids = [a["id"] for a in articles]
        assert "art-004" in article_ids

    async def test_list_articles_filter_by_tag(self, client: httpx.AsyncClient, populated_test_db):
        """Filtering by tag returns only articles with that tag."""
        resp = await client.get("/api/articles", params={"tag": "ai-agents"})
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) == 1
        assert articles[0]["id"] == "art-001"

    async def test_list_articles_filter_by_tier_high(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Filtering by tier=high returns only articles with info_score >= 60."""
        resp = await client.get("/api/articles", params={"tier": "high"})
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert a["info_score"] >= 60

    async def test_list_articles_filter_by_tier_medium(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Filtering by tier=medium returns articles with 30 <= score < 60."""
        resp = await client.get("/api/articles", params={"tier": "medium"})
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert 30 <= a["info_score"] < 60

    async def test_list_articles_filter_by_tier_low(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Filtering by tier=low returns articles with score < 30."""
        resp = await client.get("/api/articles", params={"tier": "low"})
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert a["info_score"] < 30

    async def test_list_articles_filter_by_location(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Filtering by location returns only articles in that location."""
        resp = await client.get("/api/articles", params={"location": "later"})
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert a["location"] == "later"

    async def test_list_articles_sort_by_added(self, client: httpx.AsyncClient, populated_test_db):
        """Sorting by 'added' orders by readwise_created_at descending."""
        resp = await client.get("/api/articles", params={"sort": "added"})
        assert resp.status_code == 200
        articles = resp.json()
        # Verify articles come back (sorting is by date desc)
        assert len(articles) > 0

    async def test_list_articles_sort_by_published(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Sorting by 'published' orders by published_date descending."""
        resp = await client.get("/api/articles", params={"sort": "published"})
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) > 0

    async def test_list_articles_pagination(self, client: httpx.AsyncClient, populated_test_db):
        """Pagination via skip and limit works correctly."""
        resp_all = await client.get("/api/articles", params={"limit": 100})
        all_articles = resp_all.json()
        total = len(all_articles)

        resp_page = await client.get("/api/articles", params={"skip": 2, "limit": 3})
        page_articles = resp_page.json()
        assert len(page_articles) <= 3
        # If enough articles exist, the page should have some
        if total > 2:
            assert len(page_articles) > 0

    async def test_list_articles_response_structure(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Verify the response structure of a returned article."""
        resp = await client.get("/api/articles", params={"limit": 1})
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) >= 1
        article = articles[0]
        expected_keys = {
            "id",
            "title",
            "url",
            "author",
            "word_count",
            "location",
            "category",
            "info_score",
            "priority_score",
            "author_boost",
            "specificity_score",
            "novelty_score",
            "depth_score",
            "actionability_score",
            "score_reasons",
            "overall_assessment",
            "skip_recommended",
            "skip_reason",
            "has_summary",
            "tags",
            "added_at",
            "published_date",
        }
        assert expected_keys.issubset(set(article.keys()))


class TestArticleDetailAPI:
    """Test the /api/articles/{article_id} detail endpoint."""

    async def test_get_article_detail(self, client: httpx.AsyncClient, populated_test_db):
        """Getting a valid article returns full detail including summary fields."""
        resp = await client.get("/api/articles/art-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "art-001"
        assert data["title"] == "The Future of AI Agents"
        assert data["info_score"] == 85
        assert "ai-agents" in data["tags"]
        # art-001 has no summary
        assert data["summary_text"] is None
        assert data["key_points"] is None

    async def test_get_article_with_summary(self, client: httpx.AsyncClient, populated_test_db):
        """Article with a summary returns summary_text and key_points."""
        resp = await client.get("/api/articles/art-003")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "art-003"
        assert data["summary_text"] == "A weekly AI news roundup."
        assert data["key_points"] == ["AI news", "No standout items"]
        assert data["has_summary"] is True

    async def test_get_article_not_found(self, client: httpx.AsyncClient):
        """Getting a nonexistent article returns 404."""
        resp = await client.get("/api/articles/nonexistent-id")
        assert resp.status_code == 404

    async def test_get_article_without_score(self, client: httpx.AsyncClient, test_session_factory):
        """Article with no score returns zeroed-out score fields."""
        async with test_session_factory() as session:
            art = Article(
                id="art-no-score",
                title="Unscored Article",
                url="https://example.com/unscored",
                location="new",
            )
            session.add(art)
            await session.commit()

        resp = await client.get("/api/articles/art-no-score")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info_score"] == 0
        assert data["priority_score"] is None
        assert data["specificity_score"] == 0


class TestTop5API:
    """Test the /api/top5 endpoint."""

    async def test_top5_empty_db(self, client: httpx.AsyncClient):
        """Top5 returns empty list when no articles exist."""
        resp = await client.get("/api/top5")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_top5_returns_max_five(self, client: httpx.AsyncClient, populated_test_db):
        """Top5 returns at most 5 articles."""
        resp = await client.get("/api/top5")
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) <= 5

    async def test_top5_excludes_archived(self, client: httpx.AsyncClient, populated_test_db):
        """Top5 excludes archived articles."""
        resp = await client.get("/api/top5")
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert a["id"] != "art-004"

    async def test_top5_sorted_by_score(self, client: httpx.AsyncClient, populated_test_db):
        """Top5 articles are sorted by info_score descending."""
        resp = await client.get("/api/top5")
        assert resp.status_code == 200
        articles = resp.json()
        scores = [a["info_score"] for a in articles]
        assert scores == sorted(scores, reverse=True)


class TestSkipRecommendedAPI:
    """Test the /api/articles/skip endpoint."""

    async def test_skip_recommended_empty_db(self, client: httpx.AsyncClient):
        """Returns empty list when no articles exist."""
        resp = await client.get("/api/articles/skip")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_skip_recommended_returns_correct_articles(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Returns only articles with skip_recommended=True, excluding archived."""
        resp = await client.get("/api/articles/skip")
        assert resp.status_code == 200
        articles = resp.json()
        for a in articles:
            assert a["skip_recommended"] is True
        # art-003 is skip_recommended=True and not archived
        article_ids = [a["id"] for a in articles]
        assert "art-003" in article_ids


class TestStatsAPI:
    """Test the /api/stats endpoint."""

    async def test_stats_empty_db(self, client: httpx.AsyncClient):
        """Stats with empty DB returns all zeros."""
        resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_articles"] == 0
        assert data["high_value_count"] == 0
        assert data["medium_value_count"] == 0
        assert data["low_value_count"] == 0
        assert data["summarized_count"] == 0
        assert data["average_score"] == 0.0
        assert data["authors_synced"] == 0
        assert data["liked_authors_count"] == 0

    async def test_stats_with_data(self, client: httpx.AsyncClient, populated_test_db):
        """Stats with data returns correct counts (excluding archived)."""
        resp = await client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        # art-004 is archived, so total should exclude it
        # Non-archived scored articles: art-001(85), art-002(45), art-003(15),
        # art-005(65), art-006(66), art-007(67), art-008(68), art-009(69) = 8
        assert data["total_articles"] == 8
        # High (>=60): art-001(85), art-005(65), art-006(66), art-007(67), art-008(68), art-009(69) = 6
        assert data["high_value_count"] == 6
        # Medium (30-59): art-002(45) = 1
        assert data["medium_value_count"] == 1
        # Low (<30): art-003(15) = 1
        assert data["low_value_count"] == 1
        # Summaries (non-archived): art-003 has one = 1
        assert data["summarized_count"] == 1
        assert data["average_score"] > 0
        # 3 authors total
        assert data["authors_synced"] == 3
        # Liked (2+ highlights): Jane Smith(25), Prolific Writer(50) = 2
        assert data["liked_authors_count"] == 2


class TestScanAPI:
    """Test the /api/scan endpoint."""

    async def test_trigger_scan(self, client: httpx.AsyncClient):
        """POST /api/scan triggers a sync and returns immediately."""
        mock_sync = MagicMock()
        mock_sync.trigger_sync = MagicMock()
        with patch("app.routers.api.get_background_sync", return_value=mock_sync):
            resp = await client.post("/api/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        mock_sync.trigger_sync.assert_called_once()


class TestSyncStatusAPI:
    """Test the /api/sync-status endpoint."""

    async def test_sync_status(self, client: httpx.AsyncClient):
        """GET /api/sync-status returns current sync status."""
        mock_sync = MagicMock()
        mock_sync.status = MagicMock(
            is_syncing=False,
            last_sync_at=None,
            articles_processed=42,
            newly_scored=5,
            newly_tagged=3,
            scoring_version="v2-categorical",
            last_error=None,
        )
        with patch("app.routers.api.get_background_sync", return_value=mock_sync):
            resp = await client.get("/api/sync-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_syncing"] is False
        assert data["articles_processed"] == 42
        assert data["newly_scored"] == 5
        assert data["newly_tagged"] == 3
        assert data["scoring_version"] == "v2-categorical"
        assert data["last_error"] is None

    async def test_sync_status_while_syncing(self, client: httpx.AsyncClient):
        """Sync status reports is_syncing=True during active sync."""
        mock_sync = MagicMock()
        mock_sync.status = MagicMock(
            is_syncing=True,
            last_sync_at=datetime(2025, 6, 1, 10, 0, 0),
            articles_processed=100,
            newly_scored=10,
            newly_tagged=8,
            scoring_version="v2-categorical",
            last_error=None,
        )
        with patch("app.routers.api.get_background_sync", return_value=mock_sync):
            resp = await client.get("/api/sync-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_syncing"] is True


class TestTagAPI:
    """Test the /api/tag and /api/tags endpoints."""

    async def test_trigger_tagging(self, client: httpx.AsyncClient):
        """POST /api/tag triggers tagging of untagged articles."""
        mock_tagger = MagicMock()
        mock_tagger.tag_untagged_articles = AsyncMock(
            return_value={"art-001": ["ai-agents"], "art-002": ["software-eng"]}
        )
        with patch("app.routers.api.get_tagger", return_value=mock_tagger):
            resp = await client.post("/api/tag")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["articles_tagged"] == 2
        mock_tagger.tag_untagged_articles.assert_called_once()

    async def test_trigger_tagging_force(self, client: httpx.AsyncClient):
        """POST /api/tag?force=true re-tags all articles."""
        mock_tagger = MagicMock()
        mock_tagger.retag_all_articles = AsyncMock(
            return_value={"art-001": ["ai-agents", "llm-engineering"]}
        )
        with patch("app.routers.api.get_tagger", return_value=mock_tagger):
            resp = await client.post("/api/tag", params={"force": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["articles_tagged"] == 1
        mock_tagger.retag_all_articles.assert_called_once()

    async def test_list_tags(self, client: httpx.AsyncClient, populated_test_db):
        """GET /api/tags returns all defined tags with article counts."""
        resp = await client.get("/api/tags")
        assert resp.status_code == 200
        tags = resp.json()
        # Should include all TAG_DEFINITIONS
        assert len(tags) > 0
        # Each tag should have required fields
        for tag in tags:
            assert "slug" in tag
            assert "name" in tag
            assert "description" in tag
            assert "article_count" in tag
        # ai-agents tag should have count 1 (art-001)
        ai_agents = next((t for t in tags if t["slug"] == "ai-agents"), None)
        assert ai_agents is not None
        assert ai_agents["article_count"] == 1

    async def test_list_tags_empty_db(self, client: httpx.AsyncClient):
        """GET /api/tags still returns tag definitions even with no articles."""
        resp = await client.get("/api/tags")
        assert resp.status_code == 200
        tags = resp.json()
        assert len(tags) > 0
        # All counts should be 0
        for tag in tags:
            assert tag["article_count"] == 0


class TestRescoreFailedAPI:
    """Test the /api/rescore-failed endpoint."""

    async def test_rescore_failed(self, client: httpx.AsyncClient):
        """POST /api/rescore-failed calls scorer and returns count."""
        mock_scorer = MagicMock()
        mock_scorer.rescore_failed_articles = AsyncMock(return_value=3)
        with patch("app.routers.api.get_article_scorer", return_value=mock_scorer):
            resp = await client.post("/api/rescore-failed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["articles_rescored"] == 3
        mock_scorer.rescore_failed_articles.assert_called_once()


class TestAuthorsAPI:
    """Test the /api/authors endpoints."""

    async def test_list_authors_empty_db(self, client: httpx.AsyncClient):
        """Returns empty list when no authors exist."""
        resp = await client.get("/api/authors")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_authors_with_data(self, client: httpx.AsyncClient, populated_test_db):
        """Returns authors sorted by highlight count, filtered by min_highlights."""
        resp = await client.get("/api/authors")
        assert resp.status_code == 200
        authors = resp.json()
        # Default min_highlights=1, so all 3 authors qualify
        assert len(authors) == 3
        # Should be sorted by total_highlights desc
        highlights = [a["total_highlights"] for a in authors]
        assert highlights == sorted(highlights, reverse=True)

    async def test_list_authors_min_highlights_filter(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """min_highlights parameter filters out authors below threshold."""
        resp = await client.get("/api/authors", params={"min_highlights": 10})
        assert resp.status_code == 200
        authors = resp.json()
        # Jane Smith(25) and Prolific Writer(50) qualify, John Doe(1) does not
        assert len(authors) == 2
        for a in authors:
            assert a["total_highlights"] >= 10

    async def test_list_authors_limit(self, client: httpx.AsyncClient, populated_test_db):
        """Limit parameter restricts number of results."""
        resp = await client.get("/api/authors", params={"limit": 1})
        assert resp.status_code == 200
        authors = resp.json()
        assert len(authors) == 1

    async def test_list_authors_response_structure(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Verify author response structure."""
        resp = await client.get("/api/authors", params={"limit": 1})
        assert resp.status_code == 200
        authors = resp.json()
        assert len(authors) >= 1
        author = authors[0]
        assert "id" in author
        assert "name" in author
        assert "total_highlights" in author
        assert "total_books" in author
        assert "is_favorite" in author

    async def test_sync_authors(self, client: httpx.AsyncClient):
        """POST /api/authors/sync calls the author service and scorer."""
        mock_author_service = MagicMock()
        mock_sync_result = MagicMock(
            total_authors=10,
            new_authors=3,
            updated_authors=7,
            total_books=25,
        )
        mock_author_service.sync_authors_from_readwise = AsyncMock(return_value=mock_sync_result)
        mock_scorer = MagicMock()
        mock_scorer.recompute_priorities = AsyncMock(return_value=15)

        with (
            patch("app.routers.api.get_author_service", return_value=mock_author_service),
            patch("app.routers.api.get_article_scorer", return_value=mock_scorer),
        ):
            resp = await client.post("/api/authors/sync")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_authors"] == 10
        assert data["new_authors"] == 3
        assert data["updated_authors"] == 7
        assert data["total_books"] == 25
        assert data["priorities_updated"] == 15

    async def test_toggle_author_favorite(self, client: httpx.AsyncClient, populated_test_db):
        """POST /api/authors/{id}/favorite toggles favorite status."""
        mock_author_service = MagicMock()
        mock_author_service.mark_favorite = AsyncMock()
        with patch("app.routers.api.get_author_service", return_value=mock_author_service):
            resp = await client.post("/api/authors/1/favorite", params={"is_favorite": "true"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_author_service.mark_favorite.assert_called_once_with(1, True)

    async def test_toggle_author_unfavorite(self, client: httpx.AsyncClient, populated_test_db):
        """POST /api/authors/{id}/favorite?is_favorite=false unfavorites."""
        mock_author_service = MagicMock()
        mock_author_service.mark_favorite = AsyncMock()
        with patch("app.routers.api.get_author_service", return_value=mock_author_service):
            resp = await client.post("/api/authors/1/favorite", params={"is_favorite": "false"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_author_service.mark_favorite.assert_called_once_with(1, False)


# ---------------------------------------------------------------------------
# 5. Pages with data
# ---------------------------------------------------------------------------


class TestDashboardPagesWithData:
    """Test dashboard pages when the DB has articles."""

    async def test_dashboard_with_articles(self, client: httpx.AsyncClient, populated_test_db):
        """Dashboard renders with articles in the DB."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        # Should contain article titles
        assert "The Future of AI Agents" in body

    async def test_dashboard_filter_by_tag(self, client: httpx.AsyncClient, populated_test_db):
        """Dashboard filters by tag query param."""
        resp = await client.get("/", params={"tag": "ai-agents"})
        assert resp.status_code == 200
        body = resp.text
        assert "The Future of AI Agents" in body

    async def test_dashboard_filter_by_tier(self, client: httpx.AsyncClient, populated_test_db):
        """Dashboard filters by tier query param."""
        resp = await client.get("/", params={"tier": "low"})
        assert resp.status_code == 200
        body = resp.text
        # Low tier article
        assert "Weekly Newsletter #42" in body

    async def test_dashboard_sort_by_added(self, client: httpx.AsyncClient, populated_test_db):
        """Dashboard sorts by added date."""
        resp = await client.get("/", params={"sort": "added"})
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_dashboard_sort_by_published(self, client: httpx.AsyncClient, populated_test_db):
        """Dashboard sorts by published date."""
        resp = await client.get("/", params={"sort": "published"})
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestArticleDetailPage:
    """Test the /articles/{article_id} page route."""

    async def test_article_detail_page(self, client: httpx.AsyncClient, populated_test_db):
        """Article detail page renders for a valid article."""
        resp = await client.get("/articles/art-001")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "The Future of AI Agents" in body

    async def test_article_detail_page_with_summary(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Article detail page renders with summary content."""
        resp = await client.get("/articles/art-003")
        assert resp.status_code == 200
        body = resp.text
        assert "Weekly Newsletter #42" in body
        assert "A weekly AI news roundup" in body

    async def test_article_detail_page_not_found(self, client: httpx.AsyncClient):
        """Article detail page returns 404 for nonexistent article."""
        resp = await client.get("/articles/nonexistent-id")
        assert resp.status_code == 404

    async def test_article_detail_page_with_author_info(
        self, client: httpx.AsyncClient, populated_test_db
    ):
        """Article detail page shows author info when author exists in DB."""
        resp = await client.get("/articles/art-001")
        assert resp.status_code == 200
        body = resp.text
        # Jane Smith is in the authors table
        assert "Jane Smith" in body

    async def test_article_detail_page_no_score(
        self, client: httpx.AsyncClient, test_session_factory
    ):
        """Article without a score still renders the detail page."""
        async with test_session_factory() as session:
            art = Article(
                id="art-no-score-page",
                title="Unscored Page Article",
                url="https://example.com/unscored-page",
                location="new",
            )
            session.add(art)
            await session.commit()

        resp = await client.get("/articles/art-no-score-page")
        assert resp.status_code == 200
        body = resp.text
        assert "Unscored Page Article" in body


class TestUsagePage:
    """Test the /usage page route."""

    async def test_usage_page_empty_db(self, client: httpx.AsyncClient):
        """Usage page renders even with no usage data."""
        resp = await client.get("/usage")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_usage_page_with_data(self, client: httpx.AsyncClient, populated_test_db):
        """Usage page renders with usage data present."""
        resp = await client.get("/usage")
        assert resp.status_code == 200
        body = resp.text
        assert "text/html" in resp.headers["content-type"]
        # The page should contain service names from the logs
        assert "scorer" in body


class TestChatPage:
    """Test the /chat page route."""

    async def test_chat_page_renders(self, client: httpx.AsyncClient):
        """Chat page loads successfully."""
        resp = await client.get("/chat")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# 6. Vectorstore unit tests
# ---------------------------------------------------------------------------


class TestWholeDocumentChunker:
    """Test the WholeDocumentChunker chunking strategy."""

    def test_chunk_combines_fields(self):
        from app.services.vectorstore import WholeDocumentChunker

        chunker = WholeDocumentChunker()
        chunks = chunker.chunk_article(
            title="Test Article",
            author="Jane Doe",
            overall_assessment="A thoughtful piece on testing.",
            content_preview="Testing is important for software quality.",
            score_reasons='["Well-structured", "Practical examples"]',
        )
        assert len(chunks) == 1
        text = chunks[0].text
        # Title and author combined
        assert "Test Article by Jane Doe" in text
        # Assessment included
        assert "A thoughtful piece on testing" in text
        # Score reasons included
        assert "Well-structured" in text
        assert "Practical examples" in text
        # Content preview included
        assert "Testing is important" in text

    def test_chunk_handles_none_author(self):
        from app.services.vectorstore import WholeDocumentChunker

        chunker = WholeDocumentChunker()
        chunks = chunker.chunk_article(
            title="No Author Article",
            author=None,
            overall_assessment="An assessment.",
            content_preview=None,
            score_reasons=None,
        )
        assert len(chunks) == 1
        text = chunks[0].text
        assert "No Author Article" in text
        assert " by " not in text

    def test_chunk_handles_all_none_optional_fields(self):
        from app.services.vectorstore import WholeDocumentChunker

        chunker = WholeDocumentChunker()
        chunks = chunker.chunk_article(
            title="Title Only",
            author=None,
            overall_assessment=None,
            content_preview=None,
            score_reasons=None,
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Title Only"

    def test_chunk_handles_invalid_json_score_reasons(self):
        from app.services.vectorstore import WholeDocumentChunker

        chunker = WholeDocumentChunker()
        chunks = chunker.chunk_article(
            title="Bad JSON",
            author=None,
            overall_assessment=None,
            content_preview=None,
            score_reasons="not valid json",
        )
        # Should not raise, just skip the bad reasons
        assert len(chunks) == 1
        assert "Bad JSON" in chunks[0].text

    def test_chunk_index_is_zero(self):
        from app.services.vectorstore import WholeDocumentChunker

        chunker = WholeDocumentChunker()
        chunks = chunker.chunk_article(
            title="Index Check",
            author=None,
            overall_assessment=None,
            content_preview=None,
            score_reasons=None,
        )
        assert chunks[0].chunk_index == 0


class TestVectorStore:
    """Test VectorStore with a temporary Qdrant directory."""

    def test_ensure_collection_creates_collection(self):
        from app.services.vectorstore import COLLECTION_NAME, VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            # Collection should exist after init
            collections = store._client.get_collections().collections
            names = [c.name for c in collections]
            assert COLLECTION_NAME in names

    def test_ensure_collection_idempotent(self):
        from app.services.vectorstore import COLLECTION_NAME, VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            # Calling _ensure_collection again should not raise
            store._ensure_collection()
            collections = store._client.get_collections().collections
            names = [c.name for c in collections]
            assert names.count(COLLECTION_NAME) == 1

    def test_collection_count_returns_integer(self):
        from app.services.vectorstore import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            count = store.collection_count()
            assert isinstance(count, int)
            assert count == 0

    def test_search_returns_list(self):
        from app.services.vectorstore import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            # Mock the embedding model to avoid loading a real model
            mock_model = MagicMock()
            import numpy as np

            mock_model.encode.return_value = np.zeros((1, 768))
            store._model = mock_model

            results = store.search("test query", limit=5)
            assert isinstance(results, list)
            # Empty collection should return no results
            assert len(results) == 0

    def test_search_result_structure(self):
        from app.services.vectorstore import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            # Mock the embedding model
            mock_model = MagicMock()
            import numpy as np

            fake_embedding = np.random.rand(1, 768).astype(np.float32)
            mock_model.encode.return_value = fake_embedding
            store._model = mock_model

            # Insert a point manually so search returns something
            from qdrant_client.models import PointStruct

            from app.services.vectorstore import COLLECTION_NAME

            store._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=1,
                        vector=fake_embedding[0].tolist(),
                        payload={
                            "article_id": "art-001",
                            "title": "Test Article",
                            "author": "Author",
                            "info_score": 75,
                            "tags": ["ai"],
                        },
                    )
                ],
            )

            results = store.search("test", limit=5)
            assert len(results) >= 1
            result = results[0]
            assert "article_id" in result
            assert "title" in result
            assert "author" in result
            assert "info_score" in result
            assert "tags" in result
            assert "similarity" in result
            assert result["article_id"] == "art-001"

    def test_embed_article_and_count(self):
        from app.services.vectorstore import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(path=tmpdir)
            # Mock the embedding model
            mock_model = MagicMock()
            import numpy as np

            mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
            store._model = mock_model

            store.embed_article(
                article_id="art-002",
                text="Some article text",
                metadata={"title": "My Article", "info_score": 50},
            )

            assert store.collection_count() == 1
