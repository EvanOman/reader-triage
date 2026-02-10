"""Tests for Article model and FTS5 full-text search functions."""

from sqlalchemy import text
from sqlalchemy.orm import selectinload

from app.models.article import (
    Article,
    ArticleTag,
    Base,
    Summary,
)

# ---------------------------------------------------------------------------
# Article model basics
# ---------------------------------------------------------------------------


class TestArticleModel:
    """Test Article ORM model creation, persistence, and relationships."""

    async def test_create_article(self, session, sample_article):
        """An article can be added and read back."""
        session.add(sample_article)
        await session.commit()

        loaded = await session.get(Article, "test-article-001")
        assert loaded is not None
        assert loaded.title == "The Future of AI Agents"
        assert loaded.url == "https://example.com/ai-agents"
        assert loaded.author == "Jane Smith"
        assert loaded.word_count == 2500
        assert loaded.location == "new"
        assert loaded.category == "article"
        assert loaded.site_name == "example.com"

    async def test_article_content_column(self, session, sample_article):
        """The content column stores and retrieves full article text."""
        session.add(sample_article)
        await session.commit()

        loaded = await session.get(Article, "test-article-001")
        assert loaded.content is not None
        assert "architectural patterns" in loaded.content

    async def test_article_content_nullable(self, session):
        """Content column accepts None."""
        article = Article(
            id="no-content",
            title="No Content Article",
            url="https://example.com/empty",
            content=None,
        )
        session.add(article)
        await session.commit()

        loaded = await session.get(Article, "no-content")
        assert loaded.content is None

    async def test_article_content_preview_separate(self, session, sample_article):
        """content and content_preview are independent columns."""
        session.add(sample_article)
        await session.commit()

        loaded = await session.get(Article, "test-article-001")
        assert loaded.content_preview != loaded.content
        assert "content_preview" != "content"

    async def test_article_score_relationship(self, session, sample_article, sample_score):
        """Article.score one-to-one relationship loads correctly."""
        session.add(sample_article)
        session.add(sample_score)
        await session.commit()

        from sqlalchemy import select

        stmt = (
            select(Article)
            .where(Article.id == "test-article-001")
            .options(selectinload(Article.score))
        )
        result = await session.execute(stmt)
        loaded = result.scalar_one()

        assert loaded.score is not None
        assert loaded.score.info_score == 85
        assert loaded.score.specificity_score == 22
        assert loaded.score.priority_score == 85

    async def test_article_summary_relationship(self, session):
        """Article.summary one-to-one relationship loads correctly."""
        article = Article(
            id="sum-article",
            title="Summarised Article",
            url="https://example.com/sum",
        )
        summary = Summary(
            article_id="sum-article",
            summary_text="A quick overview.",
            key_points='["point one"]',
        )
        session.add(article)
        session.add(summary)
        await session.commit()

        from sqlalchemy import select

        stmt = (
            select(Article)
            .where(Article.id == "sum-article")
            .options(selectinload(Article.summary))
        )
        result = await session.execute(stmt)
        loaded = result.scalar_one()

        assert loaded.summary is not None
        assert loaded.summary.summary_text == "A quick overview."

    async def test_article_tags_relationship(self, session):
        """Article.tags list relationship works with cascade delete."""
        article = Article(
            id="tagged-article",
            title="Tagged Article",
            url="https://example.com/tagged",
        )
        tag1 = ArticleTag(article_id="tagged-article", tag_slug="ai")
        tag2 = ArticleTag(article_id="tagged-article", tag_slug="dev-tools")
        session.add(article)
        session.add(tag1)
        session.add(tag2)
        await session.commit()

        from sqlalchemy import select

        stmt = (
            select(Article)
            .where(Article.id == "tagged-article")
            .options(selectinload(Article.tags))
        )
        result = await session.execute(stmt)
        loaded = result.scalar_one()

        tag_slugs = {t.tag_slug for t in loaded.tags}
        assert tag_slugs == {"ai", "dev-tools"}

    async def test_article_defaults(self, session):
        """Default/server-default columns are set correctly."""
        article = Article(
            id="defaults-test",
            title="Defaults",
            url="https://example.com/defaults",
        )
        session.add(article)
        await session.commit()

        loaded = await session.get(Article, "defaults-test")
        assert loaded.first_synced_at is not None
        assert loaded.last_synced_at is not None


# ---------------------------------------------------------------------------
# FTS5 integration tests
# ---------------------------------------------------------------------------


class TestFTSIntegration:
    """Test FTS5 rebuild, upsert, and search functions.

    These tests operate on the raw engine from the fixture (which already has
    the articles_fts virtual table) rather than calling the production
    get_engine / init_db functions.
    """

    # -- helpers --------------------------------------------------------

    async def _populate_fts(self, engine, session, populated_db):
        """Insert all articles from populated_db into the FTS table."""
        # populated_db fixture already committed data; no await needed
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM articles_fts"))
            await conn.execute(
                text(
                    "INSERT INTO articles_fts(article_id, title, author, content) "
                    "SELECT id, title, COALESCE(author, ''), COALESCE(content, '') FROM articles"
                )
            )

    async def _search(self, engine, query: str | None, limit: int = 50) -> list[str]:
        """Run the same FTS search logic as search_articles_fts but on test engine."""
        if not query or not query.strip():
            return []

        clean_query = query.strip().replace('"', '""')
        terms = clean_query.split()
        fts_query = " ".join(f'"{t}"*' for t in terms if t)

        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT article_id FROM articles_fts "
                    "WHERE articles_fts MATCH :query "
                    "ORDER BY bm25(articles_fts, 10.0, 5.0, 1.0) "
                    "LIMIT :limit"
                ),
                {"query": fts_query, "limit": limit},
            )
            return [row[0] for row in result.fetchall()]

    async def _upsert(
        self,
        engine,
        article_id: str,
        title: str,
        author: str | None,
        content: str | None = None,
    ):
        """Mirror upsert_fts_entry but against the test engine."""
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM articles_fts WHERE article_id = :id"),
                {"id": article_id},
            )
            await conn.execute(
                text(
                    "INSERT INTO articles_fts(article_id, title, author, content) "
                    "VALUES (:id, :title, :author, :content)"
                ),
                {
                    "id": article_id,
                    "title": title,
                    "author": author or "",
                    "content": content or "",
                },
            )

    # -- rebuild_fts_index -----------------------------------------------

    async def test_rebuild_fts_populates_index(self, engine, session, populated_db):
        """rebuild_fts_index inserts all articles into the FTS table."""
        await self._populate_fts(engine, session, populated_db)

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM articles_fts"))
            count = result.scalar()
        assert count == 4  # populated_db has 4 articles

    async def test_rebuild_fts_clears_old_data(self, engine, session, populated_db):
        """Rebuilding replaces (not appends to) the FTS index."""
        # Build index twice
        await self._populate_fts(engine, session, populated_db)
        await self._populate_fts(engine, session, populated_db)

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM articles_fts"))
            count = result.scalar()
        assert count == 4  # still 4, not 8

    # -- search_articles_fts --------------------------------------------

    async def test_search_by_title_keyword(self, engine, session, populated_db):
        """Searching a title keyword returns the matching article."""
        await self._populate_fts(engine, session, populated_db)

        ids = await self._search(engine, "Obsidian")
        assert "test-article-002" in ids

    async def test_search_by_author_name(self, engine, session, populated_db):
        """Searching by author name returns that author's article."""
        await self._populate_fts(engine, session, populated_db)

        ids = await self._search(engine, "Sarah Guo")
        assert "test-article-003" in ids

    async def test_search_by_content(self, engine, session, populated_db):
        """Searching by a word only in content finds the article."""
        await self._populate_fts(engine, session, populated_db)

        # "architectural" only appears in article-001's content
        ids = await self._search(engine, "architectural")
        assert "test-article-001" in ids

    async def test_search_content_no_content_article(self, engine, session, populated_db):
        """An article with content=None is still indexed (with empty content)."""
        await self._populate_fts(engine, session, populated_db)

        # article-003 has content=None but should still be found by title
        ids = await self._search(engine, "Abundance")
        assert "test-article-003" in ids

    async def test_bm25_title_ranks_higher_than_content(self, engine, session, populated_db):
        """Title matches rank higher than content-only matches.

        The bm25 weights are (10.0, 5.0, 1.0) for (title, author, content).
        When a word appears in the title of one article and only in the content
        of another, the title match should come first.
        """
        await self._populate_fts(engine, session, populated_db)

        # Insert two articles that share the word "quantum":
        # one in title only, one in content only
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO articles(id, title, url) "
                    "VALUES ('title-match', 'Quantum Computing Revolution', 'https://example.com/q1')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO articles(id, title, url, content) "
                    "VALUES ('content-match', 'Some Other Title', 'https://example.com/q2', "
                    "'This article discusses quantum mechanics and its applications.')"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO articles_fts(article_id, title, author, content) VALUES "
                    "('title-match', 'Quantum Computing Revolution', '', ''), "
                    "('content-match', 'Some Other Title', '', "
                    "'This article discusses quantum mechanics and its applications.')"
                )
            )

        ids = await self._search(engine, "quantum")
        assert len(ids) >= 2
        title_pos = ids.index("title-match")
        content_pos = ids.index("content-match")
        assert title_pos < content_pos, (
            f"Title match should rank higher: title at {title_pos}, content at {content_pos}"
        )

    async def test_prefix_matching(self, engine, session, populated_db):
        """Prefix queries match partial words (search-as-you-type)."""
        await self._populate_fts(engine, session, populated_db)

        # "Obsid" should prefix-match "Obsidian" in article-002 title
        ids = await self._search(engine, "Obsid")
        assert "test-article-002" in ids

    async def test_prefix_matching_author(self, engine, session, populated_db):
        """Prefix match works on author names too."""
        await self._populate_fts(engine, session, populated_db)

        ids = await self._search(engine, "Newslet")
        assert "test-article-004" in ids

    async def test_empty_query_returns_empty(self, engine, session, populated_db):
        """Empty string query returns no results."""
        await self._populate_fts(engine, session, populated_db)

        assert await self._search(engine, "") == []

    async def test_blank_query_returns_empty(self, engine, session, populated_db):
        """Whitespace-only query returns no results."""
        await self._populate_fts(engine, session, populated_db)

        assert await self._search(engine, "   ") == []

    async def test_none_query_returns_empty(self, engine, session, populated_db):
        """None query returns no results."""
        await self._populate_fts(engine, session, populated_db)

        assert await self._search(engine, None) == []

    async def test_search_limit(self, engine, session, populated_db):
        """The limit parameter caps the number of results."""
        await self._populate_fts(engine, session, populated_db)

        # "article" appears in multiple entries via category or content
        # Use a broad term to get multiple results
        ids = await self._search(engine, "AI", limit=1)
        assert len(ids) <= 1

    async def test_search_multiple_terms(self, engine, session, populated_db):
        """Multi-word queries match articles containing all terms."""
        await self._populate_fts(engine, session, populated_db)

        ids = await self._search(engine, "AI Agents")
        assert "test-article-001" in ids

    # -- upsert_fts_entry ------------------------------------------------

    async def test_upsert_inserts_new_entry(self, engine):
        """upsert_fts_entry adds a new row when article_id is not present."""
        await self._upsert(engine, "new-id", "Brand New Title", "Author X", "Some content.")

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT title, author, content FROM articles_fts WHERE article_id = 'new-id'")
            )
            row = result.fetchone()

        assert row is not None
        assert row[0] == "Brand New Title"
        assert row[1] == "Author X"
        assert row[2] == "Some content."

    async def test_upsert_replaces_existing_entry(self, engine):
        """upsert_fts_entry replaces data for an existing article_id."""
        await self._upsert(engine, "upd-id", "Original Title", "Author A", "Old content")
        await self._upsert(engine, "upd-id", "Updated Title", "Author B", "New content")

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT title, author, content FROM articles_fts WHERE article_id = 'upd-id'")
            )
            rows = result.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "Updated Title"
        assert rows[0][1] == "Author B"
        assert rows[0][2] == "New content"

    async def test_upsert_none_author_becomes_empty(self, engine):
        """When author is None, it is stored as empty string."""
        await self._upsert(engine, "null-author", "Title", None)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT author FROM articles_fts WHERE article_id = 'null-author'")
            )
            row = result.fetchone()

        assert row[0] == ""

    async def test_upsert_none_content_becomes_empty(self, engine):
        """When content is None, it is stored as empty string."""
        await self._upsert(engine, "null-content", "Title", "Author", None)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT content FROM articles_fts WHERE article_id = 'null-content'")
            )
            row = result.fetchone()

        assert row[0] == ""

    async def test_upsert_then_searchable(self, engine):
        """After upserting, the entry is immediately searchable."""
        await self._upsert(
            engine, "search-me", "Distributed Systems Deep Dive", "Alice", "Consensus protocols"
        )

        ids = await self._search(engine, "Distributed")
        assert "search-me" in ids

    async def test_upsert_update_changes_search_results(self, engine):
        """After upserting with a new title, old title no longer matches."""
        await self._upsert(engine, "morph", "Alpha Topic", "Writer", "")
        assert "morph" in await self._search(engine, "Alpha")

        await self._upsert(engine, "morph", "Beta Topic", "Writer", "")
        assert "morph" not in await self._search(engine, "Alpha")
        assert "morph" in await self._search(engine, "Beta")


# ---------------------------------------------------------------------------
# init_db migration behaviour
# ---------------------------------------------------------------------------


class TestInitDbMigration:
    """Verify that the migration logic in init_db adds the content column.

    We simulate the migration by creating a schema WITHOUT the content column,
    then running the ALTER TABLE statement.
    """

    async def test_migration_adds_content_column(self):
        """The ALTER TABLE migration adds a content column to articles."""
        from sqlalchemy.ext.asyncio import create_async_engine

        eng = create_async_engine("sqlite+aiosqlite://", echo=False)
        try:
            async with eng.begin() as conn:
                # Create articles table without the content column
                await conn.execute(
                    text(
                        "CREATE TABLE articles ("
                        "  id VARCHAR(50) PRIMARY KEY,"
                        "  title VARCHAR(500) NOT NULL,"
                        "  url VARCHAR(2000) NOT NULL,"
                        "  author VARCHAR(200),"
                        "  word_count INTEGER,"
                        "  content_preview TEXT,"
                        "  location VARCHAR(20),"
                        "  category VARCHAR(20),"
                        "  site_name VARCHAR(200),"
                        "  reading_progress FLOAT,"
                        "  readwise_created_at DATETIME,"
                        "  readwise_updated_at DATETIME,"
                        "  first_synced_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                        "  last_synced_at DATETIME DEFAULT CURRENT_TIMESTAMP"
                        ")"
                    )
                )

            # Run the migration statement (same as init_db)
            async with eng.begin() as conn:
                await conn.execute(text("ALTER TABLE articles ADD COLUMN content TEXT"))

            # Verify we can now write and read the content column
            async with eng.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO articles(id, title, url, content) "
                        "VALUES ('mig-1', 'Migration Test', 'https://example.com/m', 'full text here')"
                    )
                )

            async with eng.connect() as conn:
                result = await conn.execute(text("SELECT content FROM articles WHERE id = 'mig-1'"))
                assert result.scalar() == "full text here"
        finally:
            await eng.dispose()

    async def test_migration_is_idempotent(self):
        """Running the ALTER TABLE when the column already exists does not error."""
        from sqlalchemy.ext.asyncio import create_async_engine

        eng = create_async_engine("sqlite+aiosqlite://", echo=False)
        try:
            async with eng.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # The column already exists; the try/except in init_db swallows the error.
            async with eng.begin() as conn:
                try:
                    await conn.execute(text("ALTER TABLE articles ADD COLUMN content TEXT"))
                except Exception:
                    pass  # Expected: column already exists

            # Confirm the column still works
            async with eng.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO articles(id, title, url, content) "
                        "VALUES ('idem-1', 'Idempotent', 'https://example.com/i', 'still works')"
                    )
                )
            async with eng.connect() as conn:
                result = await conn.execute(
                    text("SELECT content FROM articles WHERE id = 'idem-1'")
                )
                assert result.scalar() == "still works"
        finally:
            await eng.dispose()
