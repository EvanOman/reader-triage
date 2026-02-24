"""Integration tests for scorer orchestration methods.

Tests _process_documents, rescore_failed_articles, and recompute_priorities
using fake Readwise, mocked DSPy strategies, and real in-memory SQLite.
"""

import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from readwise_sdk.exceptions import RateLimitError
from sqlalchemy import select

from app.models.article import Article, ArticleScore, Author, BinaryArticleScore, V4ArticleScore
from app.services.scorer import CURRENT_SCORING_VERSION, ArticleScorer
from app.services.scoring_models import V2CategoricalOutput, V3BinaryOutput, V4TieredOutput
from tests.factories import (
    make_claude_response,
    make_document,
    make_dspy_prediction,
    mock_lm_history,
)

# Default Claude response produces: specificity=17, novelty=25, depth=25, actionability=25 → total=92
_DEFAULT_TOTAL = 92


# ---------------------------------------------------------------------------
# DSPy mock helpers
# ---------------------------------------------------------------------------


def _make_v2_prediction(data: dict) -> MagicMock:
    """Create a mock DSPy Prediction for v2-categorical output."""
    output = V2CategoricalOutput(**data)
    return make_dspy_prediction(output)


def _make_v3_prediction(data: dict) -> MagicMock:
    """Create a mock DSPy Prediction for v3-binary output."""
    output = V3BinaryOutput(**data)
    return make_dspy_prediction(output)


def _make_v4_prediction(data: dict) -> MagicMock:
    """Create a mock DSPy Prediction for v4-binary output."""
    output = V4TieredOutput(**data)
    return make_dspy_prediction(output)


def _setup_all_strategy_mocks(scorer: ArticleScorer) -> None:
    """Set up default DSPy mocks on all three strategies of an ArticleScorer."""
    # v2 strategy
    v2_data = make_claude_response()
    scorer._strategy._predict_article = MagicMock(return_value=_make_v2_prediction(v2_data))
    scorer._strategy._predict_podcast = MagicMock(return_value=_make_v2_prediction(v2_data))
    scorer._strategy._lm = mock_lm_history()

    # v3 strategy
    v3_data = _make_v3_response()
    scorer._v3_strategy._predict_article = MagicMock(return_value=_make_v3_prediction(v3_data))
    scorer._v3_strategy._predict_podcast = MagicMock(return_value=_make_v3_prediction(v3_data))
    scorer._v3_strategy._lm = mock_lm_history()

    # v4 strategy
    v4_data = _make_v4_response()
    scorer._v4_strategy._predict_article = MagicMock(return_value=_make_v4_prediction(v4_data))
    scorer._v4_strategy._predict_podcast = MagicMock(return_value=_make_v4_prediction(v4_data))
    scorer._v4_strategy._lm = mock_lm_history()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer(fake_readwise):
    """ArticleScorer wired to fake Readwise with mocked DSPy strategies."""
    with patch("app.services.scorer.get_readwise_service"):
        s = ArticleScorer()
        s._readwise = fake_readwise

    _setup_all_strategy_mocks(s)
    return s


def _make_v3_response(overrides: dict | None = None) -> dict:
    """Build a v3-binary response with all 20 questions."""
    base: dict[str, object] = {}
    for i in range(1, 21):
        base[f"q{i}"] = True
        base[f"q{i}_reason"] = f"Reason for q{i}"
    for q in ["q8", "q12", "q16", "q20"]:
        base[q] = False
        base[f"{q}_reason"] = "Not a penalty case"
    base["overall_assessment"] = "Strong article."
    if overrides:
        base.update(overrides)
    return base


def _make_v4_response(overrides: dict | None = None) -> dict:
    """Build a v4-binary response with all 24 questions.

    Default: all positive questions yes, all penalty questions no → score 100.
    """
    base: dict[str, object] = {}
    for i in range(1, 25):
        base[f"q{i}_evidence"] = f"Evidence for q{i}"
        base[f"q{i}"] = True
    # Penalty questions → no (not penalized)
    for q in ["q6", "q12", "q18", "q24"]:
        base[q] = False
        base[f"{q}_evidence"] = ""
    base["overall_assessment"] = "Strong article."
    if overrides:
        base.update(overrides)
    return base


@pytest.fixture
def deps(session_factory):
    """Patch scorer's DB, FTS, usage, and sleep dependencies."""
    mock_fts = AsyncMock()
    mock_log = AsyncMock()
    mock_sleep = AsyncMock()
    with (
        patch(
            "app.services.scorer.get_session_factory",
            new=AsyncMock(return_value=session_factory),
        ),
        patch("app.services.scorer.upsert_fts_entry", new=mock_fts),
        patch("app.services.scoring_strategy.log_usage", new=mock_log),
        patch("asyncio.sleep", new=mock_sleep),
    ):
        yield {
            "fts": mock_fts,
            "log_usage": mock_log,
            "sleep": mock_sleep,
            "session_factory": session_factory,
        }


# ---------------------------------------------------------------------------
# _process_documents tests
# ---------------------------------------------------------------------------


class TestProcessDocuments:
    """Tests for the _process_documents orchestration method."""

    async def test_new_article_created_in_db(self, scorer, fake_readwise, deps):
        doc = make_document(id="new-1", title="Brand New Article", author="Jane Doe")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with deps["session_factory"]() as session:
            article = await session.get(Article, "new-1")
            assert article is not None
            assert article.title == "Brand New Article"
            assert article.author == "Jane Doe"
            assert article.location == "new"
            assert article.category == "article"

    async def test_existing_article_metadata_updated(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        # Pre-insert article with a current-version score (so no re-scoring)
        async with sf() as session:
            article = Article(
                id="exist-1",
                title="Existing Article",
                url="https://example.com/exist",
                author="Author",
                location="new",
                reading_progress=0.0,
                category="article",
            )
            score = ArticleScore(
                article_id="exist-1",
                info_score=50,
                specificity_score=12,
                novelty_score=13,
                depth_score=12,
                actionability_score=13,
                scoring_version=CURRENT_SCORING_VERSION,
            )
            session.add(article)
            session.add(score)
            await session.commit()

        # Pass doc with updated metadata
        doc = make_document(id="exist-1", location="later", reading_progress=0.75)
        meta_doc = replace(doc, content=None)
        await scorer._process_documents([meta_doc])

        async with sf() as session:
            article = await session.get(Article, "exist-1")
            assert article is not None
            assert article.location == "later"
            assert article.reading_progress == 0.75
            assert article.last_synced_at is not None

    async def test_unscored_article_fetches_content_and_scores(self, scorer, fake_readwise, deps):
        doc = make_document(id="score-1", content="<p>Full article content here.</p>")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        # Verify get_document was called to fetch full content
        assert "score-1" in fake_readwise.get_document_calls

        # Verify score was created
        async with deps["session_factory"]() as session:
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "score-1")
                )
            ).scalar_one()
            assert score.info_score == _DEFAULT_TOTAL
            assert score.scoring_version == CURRENT_SCORING_VERSION
            assert score.scored_at is not None

    async def test_already_scored_with_current_version_skipped(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        # Pre-insert with current version v2, v3, and v4 scores
        async with sf() as session:
            article = Article(
                id="skip-1",
                title="Already Scored",
                url="https://example.com/skip",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="skip-1",
                info_score=80,
                specificity_score=20,
                novelty_score=20,
                depth_score=20,
                actionability_score=20,
                scoring_version=CURRENT_SCORING_VERSION,
            )
            v3_score = BinaryArticleScore(
                article_id="skip-1",
                info_score=75,
                specificity_score=20,
                novelty_score=18,
                depth_score=20,
                actionability_score=17,
                scoring_version="v3-binary",
            )
            v4_score = V4ArticleScore(
                article_id="skip-1",
                info_score=70,
                specificity_score=18,
                novelty_score=18,
                depth_score=17,
                actionability_score=17,
                scoring_version="v4-binary",
            )
            session.add(article)
            session.add(score)
            session.add(v3_score)
            session.add(v4_score)
            await session.commit()

        doc = make_document(id="skip-1")
        meta_doc = replace(doc, content=None)
        result = await scorer._process_documents([meta_doc])

        # get_document should NOT be called (no scoring needed for any version)
        assert "skip-1" not in fake_readwise.get_document_calls
        assert result.newly_scored == 0

    async def test_old_scoring_version_triggers_rescore(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        # Pre-insert with OLD version score
        async with sf() as session:
            article = Article(
                id="old-ver-1",
                title="Old Version",
                url="https://example.com/old",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="old-ver-1",
                info_score=40,
                specificity_score=10,
                novelty_score=10,
                depth_score=10,
                actionability_score=10,
                scoring_version="v1-old",
            )
            session.add(article)
            session.add(score)
            await session.commit()

        doc = make_document(id="old-ver-1")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        result = await scorer._process_documents([meta_doc])

        assert result.newly_scored == 1
        assert "old-ver-1" in fake_readwise.get_document_calls

        # Verify score was updated (not a new row)
        async with sf() as session:
            scores = (
                (
                    await session.execute(
                        select(ArticleScore).where(ArticleScore.article_id == "old-ver-1")
                    )
                )
                .scalars()
                .all()
            )
            assert len(scores) == 1
            assert scores[0].info_score == _DEFAULT_TOTAL
            assert scores[0].scoring_version == CURRENT_SCORING_VERSION

    async def test_content_html_stripped_and_preview_backfilled(self, scorer, fake_readwise, deps):
        html = "<p>Hello <b>world</b></p> <div>Article body.</div>"
        doc = make_document(id="html-1", content=html, word_count=100)
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with deps["session_factory"]() as session:
            article = await session.get(Article, "html-1")
            assert article is not None
            # HTML tags stripped
            assert "<p>" not in (article.content or "")
            assert "<b>" not in (article.content or "")
            assert "Hello world" in (article.content or "")
            # content_preview backfilled (was None on the metadata doc)
            assert article.content_preview is not None
            assert article.content_preview.startswith("Hello world")

    async def test_fts_entry_created_for_new_articles(self, scorer, fake_readwise, deps):
        doc = make_document(id="fts-1", title="FTS Test Article", author="FTS Author")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        # Verify upsert_fts_entry was called
        deps["fts"].assert_called_once()
        call_args = deps["fts"].call_args
        assert call_args[0][0] == "fts-1"  # article_id
        assert call_args[0][1] == "FTS Test Article"  # title
        assert call_args[0][2] == "FTS Author"  # author

    async def test_author_boost_applied_to_priority(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        # Pre-insert author with 25 highlights → boost of 10
        async with sf() as session:
            author = Author(
                name="Boosted Author",
                normalized_name="boosted author",
                total_highlights=25,
            )
            session.add(author)
            await session.commit()

        doc = make_document(id="boost-1", author="Boosted Author")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with sf() as session:
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "boost-1")
                )
            ).scalar_one()
            assert score.author_boost == 10.0
            assert score.priority_score == score.info_score + 10.0

    async def test_skip_recommended_when_score_below_30(self, scorer, fake_readwise, deps):
        # Override v2 strategy to return all-zero scores
        low_data = make_claude_response(
            {
                "standalone_passages": "none",
                "novel_framing": False,
                "content_type": "news_or_roundup",
                "author_conviction": False,
                "practitioner_voice": False,
                "content_completeness": "summary_or_excerpt",
                "named_framework": False,
                "applicable_ideas": "not_really",
            }
        )
        scorer._strategy._predict_article = MagicMock(return_value=_make_v2_prediction(low_data))
        scorer._strategy._predict_podcast = MagicMock(return_value=_make_v2_prediction(low_data))

        doc = make_document(id="low-1")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with deps["session_factory"]() as session:
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "low-1")
                )
            ).scalar_one()
            assert score.info_score == 0
            assert score.skip_recommended is True
            assert score.skip_reason == "Low information content"

    async def test_returns_correct_scan_result(self, scorer, fake_readwise, deps):
        docs = [make_document(id=f"scan-{i}") for i in range(3)]
        for d in docs:
            fake_readwise.add_document(d)
        meta_docs = [replace(d, content=None) for d in docs]

        result = await scorer._process_documents(meta_docs)

        assert result.total_scanned == 3
        assert result.newly_scored == 3
        assert len(result.top_5) == 3

    async def test_get_document_returning_none_falls_back(self, scorer, fake_readwise, deps):
        """When get_document returns None, scorer falls back to the metadata doc."""
        # Create doc with summary but DON'T add to fake → get_document returns None
        doc = make_document(
            id="fallback-1",
            content=None,
            summary="Article summary for scoring.",
            word_count=400,  # Below 500 to avoid stub detection
        )

        result = await scorer._process_documents([doc])

        # get_document was called but returned None
        assert "fallback-1" in fake_readwise.get_document_calls
        # Scoring still proceeded using the summary
        assert result.newly_scored == 1

    async def test_claude_returning_none_skips_article(self, scorer, fake_readwise, deps):
        """When _score_document returns None, article is skipped."""
        # Make all strategies' predict raise to simulate API failure
        scorer._strategy._predict_article = MagicMock(side_effect=Exception("API error"))
        scorer._v3_strategy._predict_article = MagicMock(side_effect=Exception("API error"))
        scorer._v4_strategy._predict_article = MagicMock(side_effect=Exception("API error"))

        doc = make_document(id="fail-1")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        result = await scorer._process_documents([meta_doc])

        assert result.newly_scored == 0
        # Article still created in DB (just not scored)
        async with deps["session_factory"]() as session:
            article = await session.get(Article, "fail-1")
            assert article is not None

    async def test_multiple_documents_processed_in_sequence(self, scorer, fake_readwise, deps):
        for i in range(5):
            doc = make_document(id=f"multi-{i}", title=f"Article {i}")
            fake_readwise.add_document(doc)

        meta_docs = [replace(make_document(id=f"multi-{i}"), content=None) for i in range(5)]

        result = await scorer._process_documents(meta_docs)

        assert result.total_scanned == 5
        assert result.newly_scored == 5

        # All articles exist in DB
        async with deps["session_factory"]() as session:
            for i in range(5):
                article = await session.get(Article, f"multi-{i}")
                assert article is not None


# ---------------------------------------------------------------------------
# rescore_failed_articles tests
# ---------------------------------------------------------------------------


class TestRescoreFailedArticles:
    """Tests for the rescore_failed_articles orchestration method."""

    async def _insert_article_with_score(self, sf, article_id, **score_overrides):
        """Helper to insert an article and score for rescore tests."""
        defaults = {
            "info_score": 20,
            "specificity_score": 5,
            "novelty_score": 5,
            "depth_score": 5,
            "actionability_score": 5,
            "content_fetch_failed": True,
            "scoring_version": CURRENT_SCORING_VERSION,
            "overall_assessment": "Content not available from Readwise.",
        }
        defaults.update(score_overrides)

        async with sf() as session:
            article = Article(
                id=article_id,
                title=f"Article {article_id}",
                url=f"https://example.com/{article_id}",
                author="Test Author",
                word_count=1000,
                location="new",
                category="article",
            )
            score = ArticleScore(article_id=article_id, **defaults)
            session.add(article)
            session.add(score)
            await session.commit()

    async def test_rescores_content_fetch_failed_articles(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]
        await self._insert_article_with_score(
            sf,
            "rescore-1",
            content_fetch_failed=True,
            info_score=0,
            specificity_score=0,
            novelty_score=0,
            depth_score=0,
            actionability_score=0,
        )

        doc = make_document(
            id="rescore-1", content="Full content now available. " * 60, word_count=400
        )
        fake_readwise.add_document(doc)

        count = await scorer.rescore_failed_articles()

        assert count == 1
        async with sf() as session:
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "rescore-1")
                )
            ).scalar_one()
            assert score.info_score == _DEFAULT_TOTAL
            assert score.specificity_score == 17
            assert score.novelty_score == 25
            assert score.depth_score == 25
            assert score.actionability_score == 25
            assert score.content_fetch_failed is False
            assert score.scoring_version == CURRENT_SCORING_VERSION
            assert score.scored_at is not None
            reasons = json.loads(score.score_reasons)
            assert len(reasons) == 4

    async def test_rescores_bad_assessment_pattern_articles(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]
        await self._insert_article_with_score(
            sf,
            "bad-assess-1",
            content_fetch_failed=False,
            info_score=30,
            overall_assessment="The truncated content made scoring unreliable.",
        )

        doc = make_document(id="bad-assess-1", content="Full content now available.")
        fake_readwise.add_document(doc)

        count = await scorer.rescore_failed_articles()

        assert count == 1

    async def test_skips_high_score_articles_with_bad_patterns(self, scorer, fake_readwise, deps):
        """Articles with info_score >= 60 are not rescored even if assessment has bad patterns."""
        sf = deps["session_factory"]
        await self._insert_article_with_score(
            sf,
            "high-score-1",
            content_fetch_failed=False,
            info_score=65,
            overall_assessment="Despite truncated content, this is excellent.",
        )

        doc = make_document(id="high-score-1", content="Content here.")
        fake_readwise.add_document(doc)

        count = await scorer.rescore_failed_articles()

        assert count == 0

    async def test_skips_when_readwise_has_no_content(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]
        await self._insert_article_with_score(
            sf, "no-content-1", content_fetch_failed=True, info_score=0
        )

        # Add doc WITHOUT content
        doc = make_document(id="no-content-1", content=None, summary=None)
        fake_readwise.add_document(doc)

        count = await scorer.rescore_failed_articles()

        assert count == 0

    async def test_skips_content_fetch_failed_zero_score(self, scorer, fake_readwise, deps):
        """When rescored content is still a stub with zero score, skip the update."""
        sf = deps["session_factory"]
        await self._insert_article_with_score(
            sf, "still-stub-1", content_fetch_failed=True, info_score=0
        )

        # Add doc with stub content (short relative to word_count)
        doc = make_document(id="still-stub-1", content="Very short.", word_count=2000)
        fake_readwise.add_document(doc)

        # Configure v2 strategy to return all-zero scores
        zero_data = make_claude_response(
            {
                "standalone_passages": "none",
                "novel_framing": False,
                "content_type": "news_or_roundup",
                "author_conviction": False,
                "practitioner_voice": False,
                "content_completeness": "summary_or_excerpt",
                "named_framework": False,
                "applicable_ideas": "not_really",
            }
        )
        scorer._strategy._predict_article = MagicMock(return_value=_make_v2_prediction(zero_data))
        scorer._strategy._predict_podcast = MagicMock(return_value=_make_v2_prediction(zero_data))

        count = await scorer.rescore_failed_articles()

        assert count == 0

    async def test_handles_rate_limit_with_retry(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]
        await self._insert_article_with_score(sf, "rate-1", content_fetch_failed=True, info_score=0)

        doc = make_document(id="rate-1", content="Full content after retry.")
        fake_readwise.add_document(doc)

        # Schedule a rate limit error (consumed on first get_document call)
        exc = RateLimitError("rate limited")
        exc.retry_after = 0  # type: ignore[attr-defined]
        fake_readwise.schedule_failure("rate-1", exc)

        count = await scorer.rescore_failed_articles()

        # First call raised RateLimitError, retry succeeded
        assert fake_readwise.get_document_calls.count("rate-1") == 2
        assert count == 1
        # sleep was called for the rate limit wait
        deps["sleep"].assert_awaited()


# ---------------------------------------------------------------------------
# recompute_priorities tests
# ---------------------------------------------------------------------------


class TestRecomputePriorities:
    """Tests for the recompute_priorities method."""

    async def test_updates_priority_when_author_boost_changes(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        async with sf() as session:
            article = Article(
                id="recomp-1",
                title="Recompute Test",
                url="https://example.com/recomp",
                author="Popular Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="recomp-1",
                info_score=50,
                specificity_score=12,
                novelty_score=13,
                depth_score=12,
                actionability_score=13,
                priority_score=50,
                author_boost=0.0,
                scoring_version=CURRENT_SCORING_VERSION,
            )
            # Author with 25 highlights → boost=10
            author = Author(
                name="Popular Author",
                normalized_name="popular author",
                total_highlights=25,
            )
            session.add(article)
            session.add(score)
            session.add(author)
            await session.commit()

        count = await scorer.recompute_priorities()

        assert count == 1
        async with sf() as session:
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "recomp-1")
                )
            ).scalar_one()
            assert score.priority_score == 60.0  # 50 + 10
            assert score.author_boost == 10.0
            assert score.priority_computed_at is not None

    async def test_returns_count_of_updated_articles(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        async with sf() as session:
            # Two articles by the same boosted author
            author = Author(
                name="Multi Author",
                normalized_name="multi author",
                total_highlights=50,  # → boost=15
            )
            session.add(author)
            for i in range(3):
                art = Article(
                    id=f"count-{i}",
                    title=f"Count {i}",
                    url=f"https://example.com/count-{i}",
                    author="Multi Author",
                    location="new",
                    category="article",
                )
                sc = ArticleScore(
                    article_id=f"count-{i}",
                    info_score=40,
                    specificity_score=10,
                    novelty_score=10,
                    depth_score=10,
                    actionability_score=10,
                    priority_score=40,  # Will change to 55 (40+15)
                    author_boost=0.0,
                    scoring_version=CURRENT_SCORING_VERSION,
                )
                session.add(art)
                session.add(sc)
            await session.commit()

        count = await scorer.recompute_priorities()

        assert count == 3

    async def test_noop_when_priorities_already_match(self, scorer, fake_readwise, deps):
        sf = deps["session_factory"]

        async with sf() as session:
            article = Article(
                id="noop-1",
                title="No-op Test",
                url="https://example.com/noop",
                author="Unknown Person",  # Not in authors table → boost=0
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="noop-1",
                info_score=60,
                specificity_score=15,
                novelty_score=15,
                depth_score=15,
                actionability_score=15,
                priority_score=60,  # Already correct (60 + 0 boost)
                author_boost=0.0,
                scoring_version=CURRENT_SCORING_VERSION,
            )
            session.add(article)
            session.add(score)
            await session.commit()

        count = await scorer.recompute_priorities()

        assert count == 0


# ---------------------------------------------------------------------------
# Triple scoring (v2 + v3 + v4) tests
# ---------------------------------------------------------------------------


class TestTripleScoring:
    """Tests for v2 + v3 + v4 scoring running independently."""

    async def test_new_article_gets_all_three_scores(self, scorer, fake_readwise, deps):
        """A new article should get v2, v3, and v4 scores."""
        # All three strategies already have default mocks from the scorer fixture

        doc = make_document(id="triple-1", title="Triple Score Article")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with deps["session_factory"]() as session:
            v2_score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "triple-1")
                )
            ).scalar_one()
            assert v2_score.info_score == _DEFAULT_TOTAL
            assert v2_score.scoring_version == CURRENT_SCORING_VERSION

            v3_score = (
                await session.execute(
                    select(BinaryArticleScore).where(BinaryArticleScore.article_id == "triple-1")
                )
            ).scalar_one()
            assert v3_score.scoring_version == "v3-binary"
            assert v3_score.raw_responses is not None

            v4_score = (
                await session.execute(
                    select(V4ArticleScore).where(V4ArticleScore.article_id == "triple-1")
                )
            ).scalar_one()
            assert v4_score.info_score == 100  # All positive yes, all penalties no
            assert v4_score.scoring_version == "v4-binary"
            assert v4_score.raw_responses is not None

    async def test_v3_failure_does_not_block_v2_or_v4(self, scorer, fake_readwise, deps):
        """If v3 scoring fails, v2 and v4 should still succeed."""
        # Make v3 strategy's predict raise
        scorer._v3_strategy._predict_article = MagicMock(side_effect=Exception("v3 API fail"))

        doc = make_document(id="v3fail-1")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        result = await scorer._process_documents([meta_doc])

        assert result.newly_scored == 1  # v2 succeeded

        async with deps["session_factory"]() as session:
            v2 = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "v3fail-1")
                )
            ).scalar_one()
            assert v2.info_score == _DEFAULT_TOTAL

            v3 = (
                await session.execute(
                    select(BinaryArticleScore).where(BinaryArticleScore.article_id == "v3fail-1")
                )
            ).scalar_one_or_none()
            assert v3 is None  # v3 failed, no record

            v4 = (
                await session.execute(
                    select(V4ArticleScore).where(V4ArticleScore.article_id == "v3fail-1")
                )
            ).scalar_one()
            assert v4.info_score == 100

    async def test_already_v2_scored_article_still_gets_v3_and_v4(
        self, scorer, fake_readwise, deps
    ):
        """An article with a current v2 score but no v3/v4 should get both scored."""
        sf = deps["session_factory"]

        # Pre-insert article with current v2 score
        async with sf() as session:
            article = Article(
                id="v2only-1",
                title="V2 Only Article",
                url="https://example.com/v2only",
                author="Author",
                location="new",
                category="article",
            )
            score = ArticleScore(
                article_id="v2only-1",
                info_score=80,
                specificity_score=20,
                novelty_score=20,
                depth_score=20,
                actionability_score=20,
                scoring_version=CURRENT_SCORING_VERSION,
            )
            session.add(article)
            session.add(score)
            await session.commit()

        # v3 and v4 strategies already have default mocks from the scorer fixture

        doc = make_document(id="v2only-1")
        fake_readwise.add_document(doc)
        meta_doc = replace(doc, content=None)

        await scorer._process_documents([meta_doc])

        async with sf() as session:
            # v2 should be unchanged
            v2 = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == "v2only-1")
                )
            ).scalar_one()
            assert v2.info_score == 80  # Unchanged

            # v3 should now exist
            v3 = (
                await session.execute(
                    select(BinaryArticleScore).where(BinaryArticleScore.article_id == "v2only-1")
                )
            ).scalar_one()
            assert v3.scoring_version == "v3-binary"

            # v4 should now exist
            v4 = (
                await session.execute(
                    select(V4ArticleScore).where(V4ArticleScore.article_id == "v2only-1")
                )
            ).scalar_one()
            assert v4.info_score == 100
            assert v4.scoring_version == "v4-binary"
