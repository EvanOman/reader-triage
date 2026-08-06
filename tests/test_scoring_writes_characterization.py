"""Characterization tests (INTEGRATION tier) for the scoring WRITE paths.

These tests pin the *current* behavior of every place that writes an
``ArticleScore`` row, so a later extraction of the shared priority/skip rule
is provably behavior-preserving. They are a safety net, not a specification:
where the current code is inconsistent or wrong, these tests assert the
inconsistent/wrong value on purpose and say so.

The priority/skip rule is re-derived independently at five write sites:

  1. app/services/scorer.py:463-465  -- new-score path in ``_process_documents``
     (insert AND update branches). Rule: ``priority = total + author_boost``,
     ``skip = total < 30``, ``skip_reason = "Low information content"`` when skipping.
  2. app/services/scorer.py:730-731, :751 -- ``rescore_failed_articles``.
     Same rule, re-derived; does NOT rewrite ``priority_signals``.
  3. app/services/scorer.py:794      -- ``recompute_priorities``.
     ``priority = info_score + author_boost``; writes only on change;
     does NOT re-derive the skip flags.
  4. tools/backfill_v5.py:57-59      -- arithmetic v2 -> v5 conversion.
     Re-derives priority from the *stored* ``author_boost`` column (it never
     consults the authors table) and re-derives the skip flags.
  5. tools/backfill_highlighted.py -- looks the author's boost up and derives
     the outcome from it, like every other path. It used to hardcode
     ``author_boost=0.0``; ``test_backfill_highlighted_applies_author_boost``
     now asserts the two paths produce the same priority for the same article.

Everything runs against real in-memory SQLite through the real ORM write
paths and real transactions. The only stand-ins are the collaborators that
would otherwise cost money or touch the network: the scoring strategy
(``StubScoringStrategy``) and Readwise (``FakeReadwiseService``).
"""

import json
from dataclasses import replace
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.article import Article, ArticleScore, Author
from app.services.scorer import ArticleScorer, InfoScore
from app.services.scoring_strategy import reweight_total
from tests.factories import make_document
from tools import backfill_highlighted, backfill_v5

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Constants pinned from production behavior
# ---------------------------------------------------------------------------

# ArticleScorer.AUTHOR_BOOST_THRESHOLDS: 20+ highlights -> +10 priority.
BOOSTED_AUTHOR = "Boosted Author"
BOOSTED_AUTHOR_HIGHLIGHTS = 25
BOOST = 10.0

# Below the lowest threshold (2 highlights) -> boost 0.0 even though the
# author row exists.
QUIET_AUTHOR = "Quiet Author"
QUIET_AUTHOR_HIGHLIGHTS = 1

# Never inserted into the authors table.
UNKNOWN_AUTHOR = "Nobody In Particular"

DEFAULT_TOTAL = 84
SKIP_REASON = "Low information content"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _split_total(total: int) -> tuple[int, int]:
    """Pick (quotability, insight) subscores whose v5 reweight equals `total`.

    v5-reweighted totals are ``(quotability + insight) * 2`` and therefore
    always even; the 30-boundary tests below use 28/30 for that reason.
    """
    if total % 2 != 0:
        raise ValueError(f"v5 totals are always even, got {total}")
    pairs = total // 2
    quotability = min(25, pairs)
    insight = pairs - quotability
    if insight > 25:
        raise ValueError(f"total {total} is above the v5 maximum of 100")
    return quotability, insight


class StubScoringStrategy:
    """Deterministic stand-in for ReweightedCategoricalScoringStrategy.

    Implements the ScoringStrategy protocol surface the write paths touch
    (``version``, ``accepted_versions``, ``model_id``, ``score``) and mirrors
    how the real v5 strategy builds its total: ``total_override =
    reweight_total(quotability, insight)``. Injected via
    ``ArticleScorer(strategy=...)`` — no module-global patching, no LLM call.
    """

    version = "v5-reweighted"
    accepted_versions = frozenset({"v5-reweighted", "v2-categorical"})
    model_id = "stub-scoring-model"

    def __init__(self, total: int = DEFAULT_TOTAL) -> None:
        self.set_total(total)
        self.returns_none = False
        self.calls: list[str] = []

    def set_total(self, total: int) -> None:
        self.quotability, self.insight = _split_total(total)
        self.expected_total = total

    async def score(
        self,
        *,
        title: str,
        author: str | None,
        content: str,
        word_count: int | None,
        content_type_hint: str,
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        self.calls.append(entity_id)
        if self.returns_none:
            return None
        # A fresh instance per call: _score_document mutates
        # `content_fetch_failed` on the returned object.
        return InfoScore(
            specificity=self.quotability,
            specificity_reason="Quotable passage reason",
            novelty=13,
            novelty_reason="Surprise reason",
            depth=11,
            depth_reason="Argument reason",
            actionability=self.insight,
            actionability_reason="Applicable insight reason",
            overall_assessment="Stub assessment.",
            total_override=reweight_total(self.quotability, self.insight),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy():
    return StubScoringStrategy()


@pytest.fixture
def scorer(fake_readwise, strategy):
    """ArticleScorer with an injected strategy and the fake Readwise service.

    ``__init__`` reaches for the module-global ``get_readwise_service()``;
    tests/test_scorer_integration.py already established patching it during
    construction and then assigning ``_readwise``. Same pattern here.
    """
    with patch("app.services.scorer.get_readwise_service"):
        s = ArticleScorer(strategy=strategy)
    s._readwise = fake_readwise
    return s


@pytest.fixture
def scorer_deps(session_factory):
    """Bind the scorer's DB/FTS/sleep dependencies to the test engine."""
    with (
        patch(
            "app.services.scorer.get_session_factory",
            new=AsyncMock(return_value=session_factory),
        ),
        patch("app.services.scorer.upsert_fts_entry", new=AsyncMock()),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        yield session_factory


@pytest.fixture
def tool_db(session_factory):
    """Bind the backfill tools' DB access to the test engine.

    Both tools import ``get_session_factory``/``init_db`` from
    ``app.models.article`` *inside* their coroutine, so patching the module
    attribute is the only available seam — the tools take no collaborators.
    This also guarantees they can never reach the real reader_triage.db.
    """
    with (
        patch(
            "app.models.article.get_session_factory",
            new=AsyncMock(return_value=session_factory),
        ),
        patch("app.models.article.init_db", new=AsyncMock()),
    ):
        yield session_factory


# ---------------------------------------------------------------------------
# Seeding / sensing helpers (local to this file by design)
# ---------------------------------------------------------------------------


async def _seed_author(sf, name: str, total_highlights: int) -> None:
    async with sf() as session:
        session.add(
            Author(
                name=name,
                normalized_name=name.lower().strip(),
                total_highlights=total_highlights,
            )
        )
        await session.commit()


async def _seed_article(sf, article_id: str, author: str | None = None, **kwargs) -> None:
    defaults = {
        "title": f"Article {article_id}",
        "url": f"https://example.com/{article_id}",
        "author": author,
        "word_count": 400,
        "location": "new",
        "category": "article",
    }
    defaults.update(kwargs)
    async with sf() as session:
        session.add(Article(id=article_id, **defaults))
        await session.commit()


async def _seed_score(sf, article_id: str, **kwargs) -> int:
    """Insert an ArticleScore row and return its primary key."""
    defaults = {
        "info_score": 40.0,
        "specificity_score": 10,
        "novelty_score": 10,
        "depth_score": 10,
        "actionability_score": 10,
        "priority_score": 40.0,
        "author_boost": 0.0,
        "scoring_version": "v2-categorical",
        "skip_recommended": False,
    }
    defaults.update(kwargs)
    async with sf() as session:
        score = ArticleScore(article_id=article_id, **defaults)
        session.add(score)
        await session.commit()
        return score.id


async def _all_scores(sf, article_id: str) -> list[ArticleScore]:
    async with sf() as session:
        result = await session.execute(
            select(ArticleScore).where(ArticleScore.article_id == article_id)
        )
        return list(result.scalars().all())


async def _one_score(sf, article_id: str) -> ArticleScore:
    rows = await _all_scores(sf, article_id)
    assert len(rows) == 1, f"expected exactly one score row for {article_id}, got {len(rows)}"
    return rows[0]


async def _score_document_via_scorer(scorer, fake_readwise, doc) -> None:
    """Run the real _process_documents path for one document."""
    fake_readwise.add_document(doc)
    await scorer._process_documents([replace(doc, content=None)])


async def _run_backfill_highlighted(
    article_ids: list[str],
    strategy: StubScoringStrategy,
    fake_readwise,
    *,
    dry_run: bool = False,
) -> None:
    """Drive tools/backfill_highlighted.backfill() against the test DB.

    ``get_unscored_highlighted_ids()`` is patched out: it opens the real
    reader_triage.db with raw sqlite3 and calls the Readwise-backed
    ``tools.cal_data.fetch_highlights()``. Neither is reachable (or
    permissible) from a hermetic test, and the tool exposes no seam for it.
    The write path under test — everything after ID discovery — is real.
    """
    with (
        patch.object(
            backfill_highlighted, "get_unscored_highlighted_ids", return_value=article_ids
        ),
        patch("app.services.readwise.get_readwise_service", return_value=fake_readwise),
        patch("app.services.scorer._get_default_strategy", return_value=strategy),
    ):
        await backfill_highlighted.backfill(limit=0, dry_run=dry_run)


# ===========================================================================
# THE FIXED BUG
# ===========================================================================


async def test_backfill_highlighted_applies_author_boost(
    session_factory, fake_readwise, strategy, scorer, scorer_deps, tool_db
):
    """The two write paths agree. This test used to assert that they did not.

    Until the fix, tools/backfill_highlighted.py hardcoded ``author_boost=0.0``
    and ``priority_score=float(result.total)``, so an article it backfilled
    silently lost the boost its author had earned and ranked below an identical
    article that happened to go through the scorer. This test was written to
    pin that, named ``..._drops_author_boost_CHARACTERIZATION``, and it asserted
    the wrong values on purpose so the extraction refactor could be proved
    behavior-preserving before the behavior was changed.

    The extraction is done and the boost is now looked up, so the assertions
    below are inverted to the correct values in the same commit that fixed the
    tool. Both rows are still built here, because the property worth protecting
    is not "the backfill applies a boost" but "the two paths cannot disagree".
    """
    sf = session_factory
    await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
    await _seed_article(sf, "bug-backfill-1", author=BOOSTED_AUTHOR, word_count=2000)

    backfill_doc = make_document(id="bug-backfill-1", author=BOOSTED_AUTHOR)
    fake_readwise.add_document(backfill_doc)
    await _run_backfill_highlighted(["bug-backfill-1"], strategy, fake_readwise)

    backfilled = await _one_score(sf, "bug-backfill-1")

    # --- the corrected values ---
    assert backfilled.author_boost == BOOST
    assert backfilled.priority_score == float(strategy.expected_total) + BOOST
    assert backfilled.priority_score > backfilled.info_score

    # --- same author, same score, scorer path: identical outcome ---
    scorer_doc = make_document(id="bug-scorer-1", author=BOOSTED_AUTHOR)
    await _score_document_via_scorer(scorer, fake_readwise, scorer_doc)

    scored = await _one_score(sf, "bug-scorer-1")
    assert scored.author_boost == BOOST
    assert scored.priority_score == float(strategy.expected_total) + BOOST

    # The point of the whole exercise: which path wrote the row no longer
    # changes where the article ranks.
    assert scored.info_score == backfilled.info_score
    assert scored.priority_score == backfilled.priority_score
    assert scored.author_boost == backfilled.author_boost


# ===========================================================================
# SITE 1 -- app/services/scorer.py:463-465, _process_documents
# ===========================================================================


class TestSite1NewScorePath:
    """The new-score path in the main scoring loop (insert and update branches)."""

    async def test_insert_branch_applies_boost_and_no_skip(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id="s1-insert", author=BOOSTED_AUTHOR)
        )

        score = await _one_score(sf, "s1-insert")
        assert score.info_score == DEFAULT_TOTAL
        assert score.author_boost == BOOST
        assert score.priority_score == DEFAULT_TOTAL + BOOST
        assert score.skip_recommended is False
        assert score.skip_reason is None
        # Only the insert branch writes priority_signals.
        assert json.loads(score.priority_signals) == {"author_highlights": True}
        assert score.scoring_version == "v5-reweighted"
        assert score.model_used == "stub-scoring-model"
        assert score.scored_at is not None
        assert score.priority_computed_at is not None

    async def test_update_branch_overwrites_existing_row_in_place(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        await _seed_article(sf, "s1-update", author=BOOSTED_AUTHOR, word_count=2000)
        # An unaccepted version forces a re-score of the existing row.
        row_id = await _seed_score(
            sf,
            "s1-update",
            scoring_version="v1-old",
            info_score=12.0,
            priority_score=12.0,
            author_boost=0.0,
            skip_recommended=True,
            skip_reason=SKIP_REASON,
        )

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id="s1-update", author=BOOSTED_AUTHOR)
        )

        score = await _one_score(sf, "s1-update")
        assert score.id == row_id  # updated in place, not re-inserted
        assert score.info_score == DEFAULT_TOTAL
        assert score.author_boost == BOOST
        assert score.priority_score == DEFAULT_TOTAL + BOOST
        assert score.skip_recommended is False
        assert score.skip_reason is None
        assert json.loads(score.priority_signals) == {"author_highlights": True}
        assert score.scoring_version == "v5-reweighted"

    @pytest.mark.parametrize(
        ("total", "expect_skip"),
        [
            (28, True),  # below 30 -> skip
            (30, False),  # exactly 30 -> the boundary is NOT inclusive of skip
            (60, False),  # the "high value" cutoff is a read-side concept only
        ],
    )
    async def test_skip_rule_at_the_30_boundary(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps, total, expect_skip
    ):
        sf = session_factory
        strategy.set_total(total)
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id=f"s1-bnd-{total}", author=BOOSTED_AUTHOR)
        )

        score = await _one_score(sf, f"s1-bnd-{total}")
        assert score.info_score == total
        assert score.skip_recommended is expect_skip
        assert score.skip_reason == (SKIP_REASON if expect_skip else None)
        # The boost is applied regardless of the skip decision.
        assert score.priority_score == total + BOOST

    async def test_zero_boost_author_row_present_but_under_threshold(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await _seed_author(sf, QUIET_AUTHOR, QUIET_AUTHOR_HIGHLIGHTS)

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id="s1-quiet", author=QUIET_AUTHOR)
        )

        score = await _one_score(sf, "s1-quiet")
        assert score.author_boost == 0.0
        assert score.priority_score == float(DEFAULT_TOTAL)
        assert json.loads(score.priority_signals) == {"author_highlights": False}

    async def test_unknown_author_gets_zero_boost(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id="s1-unknown", author=UNKNOWN_AUTHOR)
        )

        score = await _one_score(sf, "s1-unknown")
        assert score.author_boost == 0.0
        assert score.priority_score == float(DEFAULT_TOTAL)

    async def test_missing_author_gets_zero_boost(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory

        await _score_document_via_scorer(
            scorer, fake_readwise, make_document(id="s1-noauthor", author=None)
        )

        score = await _one_score(sf, "s1-noauthor")
        assert score.author_boost == 0.0
        assert score.priority_score == float(DEFAULT_TOTAL)


# ===========================================================================
# SITE 2 -- app/services/scorer.py:730-731 and :751, rescore_failed_articles
# ===========================================================================


class TestSite2RescorePath:
    """The rescore path re-derives the same rule independently."""

    async def _seed_rescore_candidate(self, sf, article_id: str, author: str | None, **overrides):
        """A row the rescore query selects: fetch failed and info_score < 60."""
        await _seed_article(sf, article_id, author=author, word_count=400, content="x" * 500)
        score_kwargs = {
            "info_score": 0.0,
            "specificity_score": 0,
            "novelty_score": 0,
            "depth_score": 0,
            "actionability_score": 0,
            "priority_score": 0.0,
            "author_boost": 0.0,
            "content_fetch_failed": True,
            "scoring_version": "v5-reweighted",
            "overall_assessment": "Content not available from Readwise.",
        }
        score_kwargs.update(overrides)
        return await _seed_score(sf, article_id, **score_kwargs)

    async def test_rescore_applies_boost_and_clears_skip(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        row_id = await self._seed_rescore_candidate(
            sf, "s2-boost", BOOSTED_AUTHOR, skip_recommended=True, skip_reason=SKIP_REASON
        )
        fake_readwise.add_document(
            make_document(id="s2-boost", author=BOOSTED_AUTHOR, word_count=400)
        )

        count = await scorer.rescore_failed_articles()

        assert count == 1
        score = await _one_score(sf, "s2-boost")
        assert score.id == row_id
        assert score.info_score == DEFAULT_TOTAL
        assert score.author_boost == BOOST
        assert score.priority_score == DEFAULT_TOTAL + BOOST
        assert score.skip_recommended is False
        assert score.skip_reason is None
        assert score.content_fetch_failed is False
        assert score.scoring_version == "v5-reweighted"
        assert score.model_used == "stub-scoring-model"
        assert score.priority_computed_at is not None

    @pytest.mark.parametrize(("total", "expect_skip"), [(28, True), (30, False)])
    async def test_rescore_skip_rule_at_the_30_boundary(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps, total, expect_skip
    ):
        sf = session_factory
        strategy.set_total(total)
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        article_id = f"s2-bnd-{total}"
        await self._seed_rescore_candidate(sf, article_id, BOOSTED_AUTHOR)
        fake_readwise.add_document(
            make_document(id=article_id, author=BOOSTED_AUTHOR, word_count=400)
        )

        count = await scorer.rescore_failed_articles()

        assert count == 1
        score = await _one_score(sf, article_id)
        assert score.info_score == total
        assert score.skip_recommended is expect_skip
        assert score.skip_reason == (SKIP_REASON if expect_skip else None)
        assert score.priority_score == total + BOOST

    async def test_rescore_leaves_priority_signals_stale(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        """DIVERGENCE, pinned: unlike the insert branch at site 1, the rescore
        path never rewrites ``priority_signals``. A row whose author has since
        earned a boost keeps its stale ``{"author_highlights": false}``."""
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        await self._seed_rescore_candidate(
            sf,
            "s2-signals",
            BOOSTED_AUTHOR,
            priority_signals=json.dumps({"author_highlights": False}),
        )
        fake_readwise.add_document(
            make_document(id="s2-signals", author=BOOSTED_AUTHOR, word_count=400)
        )

        await scorer.rescore_failed_articles()

        score = await _one_score(sf, "s2-signals")
        assert score.author_boost == BOOST
        assert json.loads(score.priority_signals) == {"author_highlights": False}

    async def test_rescore_of_unknown_author_gets_zero_boost(
        self, session_factory, fake_readwise, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await self._seed_rescore_candidate(sf, "s2-unknown", UNKNOWN_AUTHOR)
        fake_readwise.add_document(
            make_document(id="s2-unknown", author=UNKNOWN_AUTHOR, word_count=400)
        )

        count = await scorer.rescore_failed_articles()

        assert count == 1
        score = await _one_score(sf, "s2-unknown")
        assert score.author_boost == 0.0
        assert score.priority_score == float(DEFAULT_TOTAL)


# ===========================================================================
# SITE 3 -- app/services/scorer.py:794, recompute_priorities
# ===========================================================================


class TestSite3RecomputePriorities:
    """recompute_priorities re-derives priority only -- never the skip flags."""

    async def test_writes_info_score_plus_boost(
        self, session_factory, strategy, scorer, scorer_deps
    ):
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        await _seed_article(sf, "s3-boost", author=BOOSTED_AUTHOR)
        await _seed_score(sf, "s3-boost", info_score=50.0, priority_score=50.0, author_boost=0.0)

        count = await scorer.recompute_priorities()

        assert count == 1
        score = await _one_score(sf, "s3-boost")
        assert score.info_score == 50.0
        assert score.author_boost == BOOST
        assert score.priority_score == 60.0
        assert score.priority_computed_at is not None

    async def test_no_write_when_value_is_already_correct(
        self, session_factory, strategy, scorer, scorer_deps
    ):
        """The guard is ``!=`` on both priority and boost; when neither moves,
        the row is untouched -- including priority_computed_at."""
        sf = session_factory
        stamp = datetime(2020, 1, 1, 12, 0, 0)
        await _seed_article(sf, "s3-noop", author=UNKNOWN_AUTHOR)
        await _seed_score(
            sf,
            "s3-noop",
            info_score=60.0,
            priority_score=60.0,
            author_boost=0.0,
            priority_computed_at=stamp,
        )

        count = await scorer.recompute_priorities()

        assert count == 0
        score = await _one_score(sf, "s3-noop")
        assert score.priority_score == 60.0
        assert score.priority_computed_at == stamp

    async def test_boost_change_alone_triggers_a_write(
        self, session_factory, strategy, scorer, scorer_deps
    ):
        """Pins the second half of the guard: the priority value is unchanged
        (both are 40.0) but the stored boost is wrong, so the row is rewritten."""
        sf = session_factory
        await _seed_article(sf, "s3-boostonly", author=UNKNOWN_AUTHOR)
        await _seed_score(
            sf, "s3-boostonly", info_score=40.0, priority_score=40.0, author_boost=7.0
        )

        count = await scorer.recompute_priorities()

        assert count == 1
        score = await _one_score(sf, "s3-boostonly")
        assert score.priority_score == 40.0
        assert score.author_boost == 0.0

    async def test_does_not_rederive_the_skip_flags(
        self, session_factory, strategy, scorer, scorer_deps
    ):
        """DIVERGENCE, pinned: an info_score of 10 is well below the skip
        threshold, but recompute_priorities only touches priority/boost. The
        stale ``skip_recommended=False`` survives."""
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, BOOSTED_AUTHOR_HIGHLIGHTS)
        await _seed_article(sf, "s3-skip", author=BOOSTED_AUTHOR)
        await _seed_score(
            sf,
            "s3-skip",
            info_score=10.0,
            priority_score=10.0,
            skip_recommended=False,
            skip_reason=None,
        )

        count = await scorer.recompute_priorities()

        assert count == 1
        score = await _one_score(sf, "s3-skip")
        assert score.priority_score == 20.0  # 10 + boost
        assert score.skip_recommended is False
        assert score.skip_reason is None

    async def test_score_without_an_article_row_is_skipped(
        self, session_factory, strategy, scorer, scorer_deps
    ):
        """The query joins Article, so an orphan score is never even visited."""
        sf = session_factory
        await _seed_score(sf, "s3-orphan", info_score=50.0, priority_score=1.0)

        count = await scorer.recompute_priorities()

        assert count == 0
        score = await _one_score(sf, "s3-orphan")
        assert score.priority_score == 1.0


# ===========================================================================
# SITE 4 -- tools/backfill_v5.py:57-59
# ===========================================================================


class TestSite4BackfillV5:
    """Arithmetic v2 -> v5 conversion, including the dry-run rollback."""

    async def test_rewrites_total_priority_and_skip_flags(self, session_factory, tool_db):
        sf = session_factory
        await _seed_article(sf, "s4-convert", author=BOOSTED_AUTHOR)
        row_id = await _seed_score(
            sf,
            "s4-convert",
            info_score=70.0,
            specificity_score=20,
            novelty_score=15,
            depth_score=15,
            actionability_score=20,
            priority_score=70.0,
            author_boost=5.0,
            scoring_version="v2-categorical",
            skip_recommended=False,
        )

        rc = await backfill_v5.run(dry_run=False)

        assert rc == 0
        score = await _one_score(sf, "s4-convert")
        assert score.id == row_id
        assert score.info_score == 80.0  # (20 + 20) * 2
        assert score.priority_score == 85.0  # new total + STORED author_boost
        assert score.author_boost == 5.0  # never recomputed
        assert score.skip_recommended is False
        assert score.skip_reason is None
        assert score.scoring_version == "v5-reweighted"
        assert score.priority_computed_at is not None
        # Subscores are preserved verbatim -- only the total is reweighted.
        assert score.specificity_score == 20
        assert score.novelty_score == 15
        assert score.depth_score == 15
        assert score.actionability_score == 20

    async def test_uses_stored_boost_not_the_authors_table(self, session_factory, tool_db):
        """DIVERGENCE, pinned: the tool selects ArticleScore alone. It never
        joins Article or consults the authors table, so an author whose boost
        has since changed is ignored -- the stored column wins."""
        sf = session_factory
        await _seed_author(sf, BOOSTED_AUTHOR, 50)  # would be a +15 boost today
        await _seed_article(sf, "s4-stale-boost", author=BOOSTED_AUTHOR)
        await _seed_score(
            sf,
            "s4-stale-boost",
            info_score=70.0,
            specificity_score=20,
            actionability_score=20,
            author_boost=3.0,
            scoring_version="v2-categorical",
        )

        await backfill_v5.run(dry_run=False)

        score = await _one_score(sf, "s4-stale-boost")
        assert score.author_boost == 3.0
        assert score.priority_score == 83.0  # 80 + the stale stored 3.0

    @pytest.mark.parametrize(
        ("quotability", "insight", "expected_total", "expect_skip"),
        [
            (7, 7, 28.0, True),  # below 30 -> skip
            (8, 7, 30.0, False),  # exactly 30 -> no skip
            (15, 15, 60.0, False),
        ],
    )
    async def test_skip_rule_at_the_30_boundary(
        self, session_factory, tool_db, quotability, insight, expected_total, expect_skip
    ):
        sf = session_factory
        article_id = f"s4-bnd-{expected_total:.0f}"
        await _seed_article(sf, article_id, author=UNKNOWN_AUTHOR)
        await _seed_score(
            sf,
            article_id,
            info_score=99.0,
            specificity_score=quotability,
            actionability_score=insight,
            author_boost=0.0,
            scoring_version="v2-categorical",
            skip_recommended=not expect_skip,
            skip_reason=None if expect_skip else SKIP_REASON,
        )

        await backfill_v5.run(dry_run=False)

        score = await _one_score(sf, article_id)
        assert score.info_score == expected_total
        assert score.priority_score == expected_total
        assert score.skip_recommended is expect_skip
        assert score.skip_reason == (SKIP_REASON if expect_skip else None)

    async def test_dry_run_rolls_everything_back(self, session_factory, tool_db):
        sf = session_factory
        await _seed_article(sf, "s4-dry", author=UNKNOWN_AUTHOR)
        await _seed_score(
            sf,
            "s4-dry",
            info_score=70.0,
            specificity_score=20,
            novelty_score=15,
            depth_score=15,
            actionability_score=20,
            priority_score=70.0,
            author_boost=5.0,
            scoring_version="v2-categorical",
            skip_recommended=False,
            priority_computed_at=None,
        )

        rc = await backfill_v5.run(dry_run=True)

        assert rc == 0
        score = await _one_score(sf, "s4-dry")
        assert score.info_score == 70.0
        assert score.priority_score == 70.0
        assert score.scoring_version == "v2-categorical"
        assert score.priority_computed_at is None

    async def test_only_v2_categorical_rows_are_touched(self, session_factory, tool_db):
        sf = session_factory
        await _seed_article(sf, "s4-already-v5", author=UNKNOWN_AUTHOR)
        await _seed_score(
            sf,
            "s4-already-v5",
            info_score=99.0,
            specificity_score=20,
            actionability_score=20,
            priority_score=99.0,
            scoring_version="v5-reweighted",
        )

        await backfill_v5.run(dry_run=False)

        score = await _one_score(sf, "s4-already-v5")
        assert score.info_score == 99.0  # NOT rewritten to 80.0
        assert score.priority_score == 99.0


# ===========================================================================
# SITE 5 -- tools/backfill_highlighted.py
# ===========================================================================


class TestSite5BackfillHighlighted:
    """Scores articles the reader has highlighted but that were never scored."""

    async def test_inserts_a_new_score_row(self, session_factory, fake_readwise, strategy, tool_db):
        sf = session_factory
        await _seed_article(sf, "s5-insert", author=UNKNOWN_AUTHOR, word_count=2000)
        fake_readwise.add_document(make_document(id="s5-insert", author=UNKNOWN_AUTHOR))

        await _run_backfill_highlighted(["s5-insert"], strategy, fake_readwise)

        score = await _one_score(sf, "s5-insert")
        assert score.info_score == DEFAULT_TOTAL
        assert score.specificity_score == strategy.quotability
        assert score.actionability_score == strategy.insight
        assert score.priority_score == float(DEFAULT_TOTAL)
        assert score.author_boost == 0.0
        assert score.skip_recommended is False
        assert score.skip_reason is None
        assert score.scoring_version == "v5-reweighted"
        # Hardcoded in the tool, independent of the strategy actually used.
        assert score.model_used == "claude-sonnet-4-5-20250929"
        assert score.priority_signals is None
        assert score.scored_at is not None
        assert score.priority_computed_at is not None
        assert json.loads(score.score_reasons) == [
            "Quotable passage reason",
            "Surprise reason",
            "Argument reason",
            "Applicable insight reason",
        ]

    @pytest.mark.parametrize(("total", "expect_skip"), [(28, True), (30, False), (60, False)])
    async def test_skip_rule_at_the_30_boundary(
        self, session_factory, fake_readwise, strategy, tool_db, total, expect_skip
    ):
        sf = session_factory
        strategy.set_total(total)
        article_id = f"s5-bnd-{total}"
        await _seed_article(sf, article_id, author=UNKNOWN_AUTHOR, word_count=2000)
        fake_readwise.add_document(make_document(id=article_id, author=UNKNOWN_AUTHOR))

        await _run_backfill_highlighted([article_id], strategy, fake_readwise)

        score = await _one_score(sf, article_id)
        assert score.info_score == total
        assert score.priority_score == float(total)
        assert score.skip_recommended is expect_skip
        assert score.skip_reason == (SKIP_REASON if expect_skip else None)

    async def test_dry_run_writes_nothing(self, session_factory, fake_readwise, strategy, tool_db):
        sf = session_factory
        await _seed_article(sf, "s5-dry", author=BOOSTED_AUTHOR, word_count=2000)
        fake_readwise.add_document(make_document(id="s5-dry", author=BOOSTED_AUTHOR))

        await _run_backfill_highlighted(["s5-dry"], strategy, fake_readwise, dry_run=True)

        assert await _all_scores(sf, "s5-dry") == []
        assert strategy.calls == []  # dry run returns before scoring

    async def test_missing_content_writes_no_row(
        self, session_factory, fake_readwise, strategy, tool_db
    ):
        sf = session_factory
        await _seed_article(sf, "s5-nocontent", author=BOOSTED_AUTHOR, word_count=2000)
        fake_readwise.add_document(
            make_document(id="s5-nocontent", author=BOOSTED_AUTHOR, content=None)
        )

        await _run_backfill_highlighted(["s5-nocontent"], strategy, fake_readwise)

        assert await _all_scores(sf, "s5-nocontent") == []

    async def test_strategy_returning_none_writes_no_row(
        self, session_factory, fake_readwise, strategy, tool_db
    ):
        sf = session_factory
        strategy.returns_none = True
        await _seed_article(sf, "s5-none", author=BOOSTED_AUTHOR, word_count=2000)
        fake_readwise.add_document(make_document(id="s5-none", author=BOOSTED_AUTHOR))

        await _run_backfill_highlighted(["s5-none"], strategy, fake_readwise)

        assert await _all_scores(sf, "s5-none") == []


# ===========================================================================
# Cross-site comparison -- the reason the rule is worth extracting
# ===========================================================================


async def test_all_five_sites_agree_on_the_skip_rule_at_the_boundary(
    session_factory, fake_readwise, strategy, scorer, scorer_deps, tool_db
):
    """Sites 1, 2, 4 and 5 each re-derive ``skip = total < 30`` independently.

    Site 3 (recompute_priorities) deliberately derives no skip flag at all, so
    it is absent here -- that asymmetry is pinned in
    TestSite3RecomputePriorities.test_does_not_rederive_the_skip_flags.
    """
    sf = session_factory
    strategy.set_total(28)

    # Site 1: new-score path.
    await _score_document_via_scorer(
        scorer, fake_readwise, make_document(id="agree-s1", author=UNKNOWN_AUTHOR)
    )

    # Site 2: rescore path.
    await _seed_article(sf, "agree-s2", author=UNKNOWN_AUTHOR, word_count=400, content="x" * 500)
    await _seed_score(
        sf,
        "agree-s2",
        info_score=0.0,
        priority_score=0.0,
        content_fetch_failed=True,
        scoring_version="v5-reweighted",
    )
    fake_readwise.add_document(make_document(id="agree-s2", author=UNKNOWN_AUTHOR, word_count=400))
    await scorer.rescore_failed_articles()

    # Site 4: v5 backfill (7 + 7) * 2 == 28.
    await _seed_article(sf, "agree-s4", author=UNKNOWN_AUTHOR)
    await _seed_score(
        sf,
        "agree-s4",
        info_score=99.0,
        specificity_score=7,
        actionability_score=7,
        scoring_version="v2-categorical",
    )
    await backfill_v5.run(dry_run=False)

    # Site 5: highlighted backfill.
    await _seed_article(sf, "agree-s5", author=UNKNOWN_AUTHOR, word_count=2000)
    fake_readwise.add_document(make_document(id="agree-s5", author=UNKNOWN_AUTHOR))
    await _run_backfill_highlighted(["agree-s5"], strategy, fake_readwise)

    for article_id in ("agree-s1", "agree-s2", "agree-s4", "agree-s5"):
        score = await _one_score(sf, article_id)
        assert score.info_score == 28.0, article_id
        assert score.skip_recommended is True, article_id
        assert score.skip_reason == SKIP_REASON, article_id
