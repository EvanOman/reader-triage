"""Shared test fixtures for reader_triage tests."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Override DATABASE_URL before any app imports
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"
os.environ.setdefault("READWISE_TOKEN", "test-token")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app.models.article import (  # noqa: E402
    Article,
    ArticleScore,
    ArticleTag,
    Base,
    Summary,
)
from tests.factories import FakeReadwiseService  # noqa: E402


@pytest.fixture
async def engine():
    """Create an in-memory SQLite engine for testing."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Create FTS5 table
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts "
                "USING fts5(article_id UNINDEXED, title, author, content)"
            )
        )
    yield eng
    await eng.dispose()


@pytest.fixture
async def session_factory(engine):
    """Create a session factory bound to the test engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def fake_readwise():
    """Create a FakeReadwiseService for testing."""
    return FakeReadwiseService()


@pytest.fixture
async def session(session_factory):
    """Create a database session for testing."""
    async with session_factory() as sess:
        yield sess


@pytest.fixture
def sample_article() -> Article:
    """Create a sample article for testing."""
    return Article(
        id="test-article-001",
        title="The Future of AI Agents",
        url="https://example.com/ai-agents",
        author="Jane Smith",
        word_count=2500,
        content_preview="AI agents are transforming how we build software...",
        content="AI agents are transforming how we build software. In this article we explore the key architectural patterns that make agents effective.",
        location="new",
        category="article",
        site_name="example.com",
    )


@pytest.fixture
def sample_score() -> ArticleScore:
    """Create a sample article score for testing."""
    return ArticleScore(
        article_id="test-article-001",
        info_score=85,
        specificity_score=22,
        novelty_score=20,
        depth_score=23,
        actionability_score=20,
        score_reasons='["Quotable passages", "Novel framing", "Strong argument", "Applicable insight"]',
        overall_assessment="High capture value piece on AI agent architectures.",
        priority_score=85,
        author_boost=0.0,
        scoring_version="v2-categorical",
        skip_recommended=False,
    )


@pytest.fixture
async def populated_db(session, sample_article, sample_score):
    """Set up a DB with a sample article and score."""
    session.add(sample_article)
    session.add(sample_score)

    # Add a second article
    article2 = Article(
        id="test-article-002",
        title="Building with Obsidian and Claude",
        url="https://example.com/obsidian-claude",
        author="John Doe",
        word_count=1500,
        content_preview="A guide to using Obsidian with Claude Code for knowledge management.",
        content="A guide to using Obsidian with Claude Code for knowledge management. We explore plugins, workflows, and automation tips.",
        location="new",
        category="article",
        site_name="example.com",
    )
    score2 = ArticleScore(
        article_id="test-article-002",
        info_score=72,
        specificity_score=18,
        novelty_score=17,
        depth_score=20,
        actionability_score=17,
        score_reasons='["Practical guide", "Unique combo", "Step-by-step", "Useful tips"]',
        overall_assessment="Practical guide combining AI tools with knowledge management.",
        priority_score=72,
        author_boost=0.0,
        scoring_version="v2-categorical",
        skip_recommended=False,
    )
    session.add(article2)
    session.add(score2)

    # Add a third article with tags
    article3 = Article(
        id="test-article-003",
        title="Software Abundance in the Age of AI",
        url="https://example.com/software-abundance",
        author="Sarah Guo",
        word_count=3000,
        content_preview="When software becomes abundant, taste becomes the differentiator.",
        content=None,  # No content stored yet
        location="later",
        category="article",
        site_name="example.com",
    )
    score3 = ArticleScore(
        article_id="test-article-003",
        info_score=100,
        specificity_score=25,
        novelty_score=25,
        depth_score=25,
        actionability_score=25,
        score_reasons='["Perfect score", "Novel insight", "Strong argument", "Broadly applicable"]',
        overall_assessment="Exceptional piece on software abundance.",
        priority_score=100,
        author_boost=0.0,
        scoring_version="v2-categorical",
        skip_recommended=False,
    )
    tag1 = ArticleTag(article_id="test-article-003", tag_slug="ai-dev-tools")
    tag2 = ArticleTag(article_id="test-article-003", tag_slug="software-eng")
    session.add(article3)
    session.add(score3)
    session.add(tag1)
    session.add(tag2)

    # Add a low-score article with summary
    article4 = Article(
        id="test-article-004",
        title="Weekly Newsletter #42",
        url="https://example.com/newsletter-42",
        author="Newsletter Bot",
        word_count=800,
        location="new",
        category="email",
    )
    score4 = ArticleScore(
        article_id="test-article-004",
        info_score=15,
        specificity_score=3,
        novelty_score=5,
        depth_score=4,
        actionability_score=3,
        score_reasons='["Generic roundup", "Nothing novel", "Shallow", "Not applicable"]',
        overall_assessment="Standard newsletter roundup with no standout content.",
        priority_score=15,
        author_boost=0.0,
        scoring_version="v2-categorical",
        skip_recommended=True,
        skip_reason="Low information content",
    )
    summary4 = Summary(
        article_id="test-article-004",
        summary_text="A weekly roundup of AI news with no standout items.",
        key_points='["AI news roundup", "No notable insights"]',
    )
    session.add(article4)
    session.add(score4)
    session.add(summary4)

    await session.commit()
    return session
