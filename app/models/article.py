"""SQLAlchemy models for articles, scores, authors, and summaries."""

import logging
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models."""

    pass


class Article(Base):
    """A document from Readwise Reader (inbox, later, archive)."""

    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # Readwise document ID
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(2000))
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Readwise metadata
    location: Mapped[str | None] = mapped_column(String(20), nullable=True)  # new, later, archive
    category: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # article, email, rss, etc.
    site_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    reading_progress: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    readwise_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    readwise_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    published_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    first_synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    score: Mapped["ArticleScore | None"] = relationship(back_populates="article", uselist=False)
    binary_score: Mapped["BinaryArticleScore | None"] = relationship(
        back_populates="article", uselist=False
    )
    summary: Mapped["Summary | None"] = relationship(back_populates="article", uselist=False)
    tags: Mapped[list["ArticleTag"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_articles_location", "location"),
        Index("idx_articles_author", "author"),
    )


class ArticleScore(Base):
    """AI-generated score for an article's information content."""

    __tablename__ = "article_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(50), ForeignKey("articles.id"), unique=True)

    # Score components (0-25 each, total 0-100)
    info_score: Mapped[float] = mapped_column(Float, default=0.0)
    specificity_score: Mapped[int] = mapped_column(Integer, default=0)
    novelty_score: Mapped[int] = mapped_column(Integer, default=0)
    depth_score: Mapped[int] = mapped_column(Integer, default=0)
    actionability_score: Mapped[int] = mapped_column(Integer, default=0)

    # Explanations
    score_reasons: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    overall_assessment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Priority calculation (combines info_score with other signals)
    priority_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100
    author_boost: Mapped[float] = mapped_column(Float, default=0.0)  # Bonus from liked author
    priority_signals: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON for extensibility

    # Content quality flags
    content_fetch_failed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Skip recommendation
    skip_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Metadata
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scoring_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    priority_computed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    article: Mapped["Article"] = relationship(back_populates="score")

    __table_args__ = (
        Index("idx_scores_priority", "priority_score"),
        Index("idx_scores_info", "info_score"),
    )


class BinaryArticleScore(Base):
    """v3-binary score for an article using 20 weighted binary questions."""

    __tablename__ = "article_scores_v3"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(50), ForeignKey("articles.id"), unique=True)

    # Score components (0-25 each, total 0-100)
    info_score: Mapped[float] = mapped_column(Float, default=0.0)
    specificity_score: Mapped[int] = mapped_column(Integer, default=0)
    novelty_score: Mapped[int] = mapped_column(Integer, default=0)
    depth_score: Mapped[int] = mapped_column(Integer, default=0)
    actionability_score: Mapped[int] = mapped_column(Integer, default=0)

    # Full binary responses (JSON text with q1-q20 booleans + reasons)
    raw_responses: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Explanations
    score_reasons: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    overall_assessment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content quality flags
    content_fetch_failed: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scoring_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Calibration data: total highlighted words across all highlights for this article
    highlighted_words: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    article: Mapped["Article"] = relationship(back_populates="binary_score")

    __table_args__ = (
        Index("idx_v3_scores_info", "info_score"),
        Index("idx_v3_scores_article", "article_id"),
    )


class Summary(Base):
    """Generated summary for a low-value article."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(50), ForeignKey("articles.id"), unique=True)
    summary_text: Mapped[str] = mapped_column(Text)
    key_points: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    article: Mapped["Article"] = relationship(back_populates="summary")


class ArticleTag(Base):
    """Association between an article and a topic tag."""

    __tablename__ = "article_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(50), ForeignKey("articles.id"))
    tag_slug: Mapped[str] = mapped_column(String(50))
    tagged_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    tagging_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    article: Mapped["Article"] = relationship(back_populates="tags")

    __table_args__ = (
        Index("idx_article_tags_article", "article_id"),
        Index("idx_article_tags_tag", "tag_slug"),
    )


class ApiUsageLog(Base):
    """Log of Anthropic API usage for cost tracking."""

    __tablename__ = "api_usage_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    service: Mapped[str] = mapped_column(String(30))  # scorer, tagger, summarizer
    model: Mapped[str] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    article_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("idx_usage_timestamp", "timestamp"),
        Index("idx_usage_service", "service"),
    )


class Author(Base):
    """Author statistics from Readwise highlights."""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    normalized_name: Mapped[str] = mapped_column(String(200), index=True)  # lowercase, trimmed

    # Aggregated stats from Readwise
    total_highlights: Mapped[int] = mapped_column(Integer, default=0)
    total_books: Mapped[int] = mapped_column(Integer, default=0)
    first_highlighted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_highlighted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # User preferences
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Sync metadata
    last_synced_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    books: Mapped[list["AuthorBook"]] = relationship(back_populates="author")

    __table_args__ = (Index("idx_authors_highlights", "total_highlights"),)


class AuthorBook(Base):
    """Books/articles with highlights, grouped by author."""

    __tablename__ = "author_books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("authors.id"))

    # Readwise book data
    readwise_book_id: Mapped[int] = mapped_column(Integer, unique=True)
    title: Mapped[str] = mapped_column(String(500))
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)  # books, articles, etc.
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    cover_image_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    # Stats
    num_highlights: Mapped[int] = mapped_column(Integer, default=0)
    last_highlight_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    author: Mapped["Author"] = relationship(back_populates="books")

    __table_args__ = (Index("idx_author_books_author", "author_id"),)


class ChatThread(Base):
    """A chat conversation thread."""

    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at.asc()",
    )

    def __repr__(self) -> str:
        return f"<ChatThread(id={self.id}, title='{self.title}')>"


class ChatMessage(Base):
    """A message within a chat thread."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_threads.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, tool_use, tool_result
    content: Mapped[str] = mapped_column(Text, default="")
    content_blocks: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON for tool use/result
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    thread: Mapped["ChatThread"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, role='{self.role}', thread_id={self.thread_id})>"


# Keep old model name for backwards compatibility during migration
ScoredArticle = Article


# Database engine and session factory
_engine = None
_session_factory = None


async def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # Set busy timeout via connect_args so SQLite waits for locks
        # instead of immediately raising "database is locked".
        _engine = create_async_engine(
            settings.database_url, echo=False, connect_args={"timeout": 30}
        )

        # Instrument for OTel tracing
        try:
            from app.tracing import instrument_engine

            instrument_engine(_engine)
        except Exception:
            pass  # Tracing is optional
    return _engine


async def get_session_factory():
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def init_db():
    """Initialize the database, creating all tables.

    Also handles lightweight schema migrations for new columns.
    """
    engine = await get_engine()

    # Enable WAL mode for better concurrent read/write performance
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Add new columns if they don't exist (migration-safe)
    async with engine.begin() as conn:
        for stmt, desc in [
            (
                "ALTER TABLE article_scores ADD COLUMN scoring_version VARCHAR(50)",
                "scoring_version",
            ),
            ("ALTER TABLE articles ADD COLUMN published_date DATETIME", "published_date"),
            (
                "ALTER TABLE article_scores ADD COLUMN content_fetch_failed BOOLEAN DEFAULT 0",
                "content_fetch_failed",
            ),
            ("ALTER TABLE articles ADD COLUMN content TEXT", "content"),
        ]:
            try:
                await conn.execute(text(stmt))
                logger.info("Added %s column", desc)
            except Exception:
                pass

    # Create FTS5 virtual table for full-text search
    async with engine.begin() as conn:
        try:
            # Drop and recreate on every startup to avoid corruption
            await conn.execute(text("DROP TABLE IF EXISTS articles_fts"))
            await conn.execute(
                text(
                    "CREATE VIRTUAL TABLE articles_fts "
                    "USING fts5(article_id UNINDEXED, title, author, content)"
                )
            )
            logger.info("FTS5 table created")
        except Exception:
            logger.warning("Could not create FTS5 table", exc_info=True)


async def rebuild_fts_index():
    """Rebuild the FTS5 index from all articles."""
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM articles_fts"))
        await conn.execute(
            text(
                "INSERT INTO articles_fts(article_id, title, author, content) "
                "SELECT id, title, COALESCE(author, ''), COALESCE(content, '') FROM articles"
            )
        )
        logger.info("FTS5 index rebuilt")


async def search_articles_fts(query: str, limit: int = 50) -> list[str]:
    """Search articles by title/author using FTS5 BM25 ranking.

    Returns list of article IDs ordered by relevance.
    """
    if not query or not query.strip():
        return []

    engine = await get_engine()
    # Escape quotes in query and add prefix matching
    clean_query = query.strip().replace('"', '""')
    # Add * for prefix matching (search-as-you-type)
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


async def upsert_fts_entry(
    article_id: str, title: str, author: str | None, content: str | None = None
):
    """Insert or update an article's FTS index entry."""
    engine = await get_engine()
    async with engine.begin() as conn:
        # Delete existing entry if any
        await conn.execute(
            text("DELETE FROM articles_fts WHERE article_id = :id"),
            {"id": article_id},
        )
        # Insert new entry
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


async def get_session():
    """Get a database session (dependency injection helper)."""
    factory = await get_session_factory()
    async with factory() as session:
        yield session
