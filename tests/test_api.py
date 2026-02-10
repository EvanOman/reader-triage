"""Integration tests for FastAPI API and chat endpoints, plus vectorstore unit tests."""

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.article import Base, ChatMessage, ChatThread

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
# 4. Vectorstore unit tests
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
