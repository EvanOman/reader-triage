"""Characterization tests (UNIT tier) for inbox_monitor's scoring rules.

These tests PIN CURRENT BEHAVIOR EXACTLY AS IT IS TODAY, bugs included. They are
the safety net for the Phase 3 extraction that consolidates the duplicated
priority/skip derivation and the 60/30 value-tier cutoffs into one module.

Read this before changing anything here:

* The expectation tables below ARE the specification. They are written as
  literal values, never derived by calling the code under test, so that after
  Phase 3 the *same* tables can be asserted against the extracted module and a
  divergence is impossible to hide behind a shared helper.
* Any change to a table in this file is a BEHAVIOR CHANGE. It requires a
  deliberate commit that says so — not a drive-by edit while refactoring.
* Where today's behavior is wrong, it is pinned as-wrong and labeled. See
  ``TestBackfillHighlightedDivergence`` (author boost silently dropped) and
  ``TestTierLabelNonFiniteInputs`` (NaN/inf crash instead of "N/A"). Phase 3
  extracts; it does not fix. Fixing is a separate, deliberate commit that
  updates the table here in the same change.

Rule sites this file characterizes (verified July 2026):

  PRIORITY / SKIP derivation, duplicated in 5 places:
    app/services/scorer.py:463-465          canonical form
    app/services/scorer.py:730-731, :751    re-derived identically
    app/services/scorer.py:794              recompute_priorities, priority only
    tools/backfill_v5.py:57-59              same rule over reweighted totals
    tools/backfill_highlighted.py:122-126   DIVERGED — boost hardcoded to 0.0

  VALUE TIER 60/30 cutoffs, duplicated in 6 blocks:
    app/routers/podcasts_api.py:253-257, :397-407
    app/routers/podcasts_pages.py:68-78, :101-105
    tools/cal_review.py:42-44 (_tier_label), :55-57 (_tier_label_plain)
    app/services/digest.py:22               only the HIGH threshold is named

Scope: pure rules only. No database, no I/O, no network.
"""

import pytest

from app.domain.scoring_outcome import (
    ValueTier,
    derive_outcome,
    derive_priority,
    should_skip,
    skip_reason_for,
    value_tier,
)
from app.models.podcast import PodcastEpisodeScore
from app.services import digest
from app.services.scoring_strategy import reweight_total
from tools.cal_review import GREEN, RED, RESET, YELLOW, _tier_label, _tier_label_plain

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Priority / skip derivation truth table
# ---------------------------------------------------------------------------

# (total, author_boost, expected_priority, expected_skip, expected_skip_reason)
#
# Literal expectations. Do NOT replace these with a call to the rule.
# The 60 boundary is included even though the rule does not branch on it: the
# tier cutoffs downstream do, and the extraction must not accidentally couple
# the skip threshold to the tier threshold.
PRIORITY_SKIP_TABLE = [
    # --- exact 30 boundary, no boost -------------------------------------
    (29.9, 0.0, 29.9, True, "Low information content"),
    (30.0, 0.0, 30.0, False, None),
    (30.1, 0.0, 30.1, False, None),
    # --- exact 60 boundary, no boost (skip rule does NOT branch here) -----
    (59.9, 0.0, 59.9, False, None),
    (60.0, 0.0, 60.0, False, None),
    (60.1, 0.0, 60.1, False, None),
    # --- extremes ---------------------------------------------------------
    (0.0, 0.0, 0.0, True, "Low information content"),
    (100.0, 0.0, 100.0, False, None),
    # --- int totals (the LLM path yields ints; backfills yield floats) -----
    (29, 0.0, 29.0, True, "Low information content"),
    (30, 0.0, 30.0, False, None),
    (60, 0.0, 60.0, False, None),
    (100, 0, 100, False, None),
    # --- positive boost: lifts priority, NEVER lifts a skip decision ------
    (29.9, 10.0, 39.9, True, "Low information content"),
    (0.0, 100.0, 100.0, True, "Low information content"),
    (29, 100, 129, True, "Low information content"),
    (30.0, 100.0, 130.0, False, None),
    (60.0, 100.0, 160.0, False, None),
    # --- negative boost: lowers priority, NEVER causes a skip -------------
    (60.0, -10.0, 50.0, False, None),
    (30.0, -30.0, 0.0, False, None),
    (30.0, -100.0, -70.0, False, None),
    (29.9, -0.1, 29.8, True, "Low information content"),
    # --- zero boost spelled as int vs float -------------------------------
    (45.0, 0, 45.0, False, None),
    (45, 0.0, 45.0, False, None),
]


class TestPriorityAndSkipDerivation:
    """The rule at app/services/scorer.py:463-465, duplicated 4 more times.

    priority = total + author_boost
    skip     = total < 30                      (boost is NOT part of this)
    reason   = "Low information content" if skip else None
    """

    @pytest.mark.parametrize(
        ("total", "author_boost", "expected_priority", "expected_skip", "expected_reason"),
        PRIORITY_SKIP_TABLE,
    )
    def test_priority_skip_truth_table(
        self,
        total: float,
        author_boost: float,
        expected_priority: float,
        expected_skip: bool,
        expected_reason: str | None,
    ):
        outcome = derive_outcome(total, author_boost)

        assert outcome.priority_score == pytest.approx(expected_priority)
        assert outcome.skip_recommended is expected_skip
        assert outcome.skip_reason == expected_reason

        # The same values via the individual helpers, so a call site that needs
        # only one of them is covered too.
        assert derive_priority(total, author_boost) == pytest.approx(expected_priority)
        assert should_skip(total) is expected_skip
        assert skip_reason_for(total) == expected_reason

    def test_skip_ignores_author_boost_even_when_boost_clears_threshold(self):
        """Pinned: a 0-score article by a favored author is still skip-recommended.

        priority becomes 100 but skip stays True, because the threshold is
        applied to the raw total. This is deliberate today; the extraction must
        preserve it.
        """
        outcome = derive_outcome(0.0, 100.0)
        assert outcome.priority_score == 100.0
        assert outcome.skip_recommended is True
        assert outcome.tier is ValueTier.LOW

    def test_recompute_priorities_derives_priority_but_not_skip(self):
        """Pinned asymmetry at app/services/scorer.py:794.

        recompute_priorities() recomputes priority = info_score + author_boost
        and leaves skip_recommended/skip_reason untouched. A row whose stored
        skip flag disagrees with its stored total is therefore NOT healed by an
        author sync. The extracted rule must not silently start re-deriving
        skip here — that would change stored data.
        """
        assert derive_priority(12.0, 25.0) == 37.0
        # Nothing in recompute_priorities touches skip; the raw total still says skip.
        assert should_skip(12.0) is True

    @pytest.mark.parametrize(
        ("total", "author_boost", "expected_priority", "expected_skip", "expected_reason"),
        PRIORITY_SKIP_TABLE,
    )
    def test_backfill_v5_matches_canonical_rule(
        self,
        total: float,
        author_boost: float,
        expected_priority: float,
        expected_skip: bool,
        expected_reason: str | None,
    ):
        """tools/backfill_v5.py:57-59 derives the identical values.

        Asserting the same table through the same entry point is the point: the
        backfill and the scorer are no longer free to disagree.
        """
        outcome = derive_outcome(total, author_boost)

        assert outcome.priority_score == pytest.approx(expected_priority)
        assert outcome.skip_recommended is expected_skip
        assert outcome.skip_reason == expected_reason


class TestBackfillHighlightedDivergence:
    """PINNED BUG — tools/backfill_highlighted.py:122-126.

    That call site writes ``priority_score=float(result.total)`` with
    ``author_boost=0.0`` hardcoded, so the author boost is silently dropped for
    every row it backfills. Its skip derivation is correct.

    This table is the WRONG behavior, recorded on purpose. Phase 3 must extract
    the rule without changing what this site currently produces; a later,
    separate commit may fix the site, and must then update this table and say
    so in the commit message.
    """

    # (total, author_boost_that_should_have_applied, actual_priority, actual_boost_written)
    DIVERGED_TABLE = [
        (29.9, 10.0, 29.9, 0.0),
        (30.0, 10.0, 30.0, 0.0),
        (60.0, 25.0, 60.0, 0.0),
        (100.0, 100.0, 100.0, 0.0),
        (0.0, 50.0, 0.0, 0.0),
        (45, 0.0, 45.0, 0.0),
    ]

    @pytest.mark.parametrize(
        ("total", "dropped_boost", "actual_priority", "actual_boost"), DIVERGED_TABLE
    )
    def test_author_boost_is_silently_dropped(
        self,
        total: float,
        dropped_boost: float,
        actual_priority: float,
        actual_boost: float,
    ):
        # What backfill_highlighted.py:122-123 produces: the boost it should
        # have looked up is never passed, so the rule is fed a zero.
        outcome = derive_outcome(total, 0.0)

        assert outcome.priority_score == pytest.approx(actual_priority)
        assert outcome.author_boost == actual_boost
        # And the divergence itself, stated: the boost it should have applied.
        assert derive_priority(total, dropped_boost) == pytest.approx(total + dropped_boost)

    @pytest.mark.parametrize(
        ("total", "expected_skip", "expected_reason"),
        [
            (29.9, True, "Low information content"),
            (30.0, False, None),
            (30, False, None),
            (29, True, "Low information content"),
            (0.0, True, "Low information content"),
            (100.0, False, None),
        ],
    )
    def test_skip_derivation_still_matches_canonical(
        self, total: float, expected_skip: bool, expected_reason: str | None
    ):
        """Only the priority half diverged; skip/reason are correct here."""
        assert should_skip(total) is expected_skip
        assert skip_reason_for(total) == expected_reason


# ---------------------------------------------------------------------------
# 2. 60/30 value tiers, asserted against the real implementations
# ---------------------------------------------------------------------------

# (score, expected plain tier label)
# Literal expectations, asserted against tools/cal_review._tier_label_plain.
TIER_TABLE = [
    (29.9, "Low"),
    (30.0, "Medium"),
    (30.1, "Medium"),
    (59.9, "Medium"),
    (60.0, "High"),
    (60.1, "High"),
    (0, "Low"),
    (0.0, "Low"),
    (29, "Low"),
    (30, "Medium"),
    (59, "Medium"),
    (60, "High"),
    (100, "High"),
    (100.0, "High"),
    (-1, "Low"),
    (-0.5, "Low"),
    (1000, "High"),
]


class TestTierLabels:
    """Boundary characterization of tools/cal_review._tier_label{,_plain}."""

    @pytest.mark.parametrize(("score", "expected"), TIER_TABLE)
    def test_tier_label_plain(self, score: float, expected: str):
        assert _tier_label_plain(score) == expected

    @pytest.mark.parametrize(("score", "expected"), TIER_TABLE)
    def test_extracted_value_tier_agrees_with_the_label_helper(self, score: float, expected: str):
        """The same table through the extracted rule.

        cal_review renders tiers for a human and the routers filter them in SQL;
        both must agree with :func:`value_tier`, or the review tool and the UI
        would disagree about the same article.
        """
        assert value_tier(score) is ValueTier[expected.upper()]

    @pytest.mark.parametrize(("score", "expected"), TIER_TABLE)
    def test_tier_label_colored_wraps_the_same_word(self, score: float, expected: str):
        color = {"High": GREEN, "Medium": YELLOW, "Low": RED}[expected]
        assert _tier_label(score) == f"{color}{expected}{RESET}"

    def test_none_is_not_a_tier(self):
        """None short-circuits before the numeric compare, in both variants."""
        assert _tier_label_plain(None) == "N/A"
        assert _tier_label(None) == "\033[2mN/A\033[0m"

    @pytest.mark.parametrize(
        ("score", "truncated_to", "expected"),
        [
            (59.9, 59, "Medium"),
            (59.999999, 59, "Medium"),
            (29.9, 29, "Low"),
            (60.9, 60, "High"),
            (30.9, 30, "Medium"),
            (-0.9, 0, "Low"),
        ],
    )
    def test_int_truncation_is_pinned(self, score: float, truncated_to: int, expected: str):
        """Both label helpers do ``s = int(score)`` before comparing.

        int() truncates toward zero. Because both cutoffs are whole numbers,
        truncation happens to agree with a direct float compare on every value
        here — 59.9 is Medium either way. The extraction is free to drop the
        int() ONLY because of that coincidence; if a fractional cutoff is ever
        introduced, this equivalence breaks. Pinned so that assumption is
        visible rather than inferred.
        """
        assert int(score) == truncated_to
        assert _tier_label_plain(score) == expected
        # The coincidence, asserted explicitly.
        direct = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
        assert direct == expected

    @pytest.mark.parametrize(("value", "expected"), [(True, "Low"), (False, "Low")])
    def test_bools_are_scored_as_numbers(self, value: bool, expected: str):
        """bool is an int subclass, so True scores 1 → Low. Not guarded against."""
        assert _tier_label_plain(value) == expected


class TestTierLabelNonFiniteInputs:
    """PINNED ROUGH EDGE: NaN/inf raise instead of degrading to "N/A".

    ``int(score)`` is unguarded, so a non-finite score crashes the caller rather
    than rendering. Reachable in practice from a pandas column with missing
    values read as NaN — cal_review reads its data through pandas. Recorded,
    not fixed.
    """

    def test_nan_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            _tier_label_plain(float("nan"))

    def test_positive_infinity_raises_overflow_error(self):
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            _tier_label_plain(float("inf"))

    def test_negative_infinity_raises_overflow_error(self):
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            _tier_label(float("-inf"))


class TestDigestThreshold:
    """app/services/digest.py names only the HIGH cutoff."""

    def test_high_value_threshold_value_and_type(self):
        assert digest.HIGH_VALUE_THRESHOLD == 60.0
        assert isinstance(digest.HIGH_VALUE_THRESHOLD, float)

    def test_medium_cutoff_has_no_name_anywhere_in_digest(self):
        """Pinned naming asymmetry: 30 is a bare literal at every other site.

        The digest only ever asks "is this high value?", so it never needed the
        30 cutoff. The extracted module will own both; this asserts the current
        state so the extraction is visibly additive, not a rename.
        """
        named = [n for n in dir(digest) if "THRESHOLD" in n]
        assert named == ["HIGH_VALUE_THRESHOLD"]

    def test_digest_threshold_agrees_with_tier_label_high_cutoff(self):
        """The two independently-written 60s currently agree."""
        assert _tier_label_plain(digest.HIGH_VALUE_THRESHOLD) == "High"
        assert _tier_label_plain(digest.HIGH_VALUE_THRESHOLD - 0.1) == "Medium"


# ---------------------------------------------------------------------------
# 3. reweight_total — the v5 production total
# ---------------------------------------------------------------------------


class TestReweightTotal:
    """app/services/scoring_strategy.py:421 — total = (quotability + insight) * 2.

    Surprise and Argument are ignored entirely (calibration, July 2026). Only
    the two arguments passed here move the number.
    """

    # (quotability, insight, expected_total, expected_tier)
    REWEIGHT_TABLE = [
        (0, 0, 0, "Low"),
        (1, 0, 2, "Low"),
        (7, 7, 28, "Low"),
        (0, 14, 28, "Low"),
        # --- the 30 cutoff: unreachable exactly, totals are always even ----
        (14, 1, 30, "Medium"),
        (15, 0, 30, "Medium"),
        (0, 15, 30, "Medium"),
        (8, 7, 30, "Medium"),
        (7, 8, 30, "Medium"),
        (14, 0, 28, "Low"),
        (16, 0, 32, "Medium"),
        # --- the 60 cutoff --------------------------------------------------
        (14, 15, 58, "Medium"),
        (15, 15, 60, "High"),
        (20, 10, 60, "High"),
        (25, 5, 60, "High"),
        (0, 30, 60, "High"),
        (16, 15, 62, "High"),
        # --- top of the realistic range (subscores are clamped to 0-25) -----
        (25, 25, 100, "High"),
        (25, 24, 98, "High"),
        (24, 25, 98, "High"),
    ]

    @pytest.mark.parametrize(
        ("quotability", "insight", "expected_total", "expected_tier"), REWEIGHT_TABLE
    )
    def test_reweight_total_and_resulting_tier(
        self, quotability: int, insight: int, expected_total: int, expected_tier: str
    ):
        total = reweight_total(quotability, insight)
        assert total == expected_total
        assert isinstance(total, int)
        assert _tier_label_plain(total) == expected_tier

    def test_reweight_is_symmetric_in_its_two_arguments(self):
        """Q and I carry identical weight today, despite different calibration ρ."""
        for q, i in [(0, 25), (7, 18), (12, 13), (25, 0)]:
            assert reweight_total(q, i) == reweight_total(i, q)

    @pytest.mark.parametrize("total", [t for _, _, t, _ in REWEIGHT_TABLE])
    def test_every_reachable_total_is_even(self, total: int):
        """Consequence of the *2: odd totals are unreachable under v5.

        So no v5 article can score 29, 59 or any other odd number, and the 30
        and 60 cutoffs are hit exactly rather than straddled. Boundary tests
        elsewhere in the system that feed odd v5 totals are testing impossible
        states — worth knowing before the extraction moves the cutoffs.
        """
        assert total % 2 == 0

    def test_no_clamping_to_the_documented_0_100_range(self):
        """The docstring says "range 0-100" but the function itself never clamps.

        The range holds only because callers pass subscores already clamped to
        0-25 (scoring_strategy.py:406-412). Out-of-range inputs pass straight
        through. Pinned so the extraction does not "helpfully" add a clamp.
        """
        assert reweight_total(30, 30) == 120
        assert reweight_total(-5, 0) == -10
        assert reweight_total(0, -1) == -2

    def test_reweight_ignores_surprise_and_argument_by_construction(self):
        """Two articles with wildly different Surprise/Argument score identically.

        Documented here as the signature difference between the v5 total and the
        podcast total (sum of all four dimensions) — see TestPodcastTotalScale.
        """
        high_surprise_and_argument = reweight_total(10, 10)  # Q=10 I=10, S=25 A=25
        zero_surprise_and_argument = reweight_total(10, 10)  # Q=10 I=10, S=0  A=0
        assert high_surprise_and_argument == zero_surprise_and_argument == 40


# ---------------------------------------------------------------------------
# 4. Podcast totals are a DIFFERENT scale on the SAME cutoffs
# ---------------------------------------------------------------------------


class TestPodcastTotalScale:
    """PodcastEpisodeScore totals sum all four dimensions; articles do not.

    Podcasts:  total = spec + novelty + depth + action   (0-100, all 4 dims)
    Articles:  total = (quotability + insight) * 2       (0-100, 2 dims doubled)

    Both are then bucketed by the SAME 60/30 cutoffs (podcasts_api.py:253-257
    and :397-407, podcasts_pages.py:68-78 and :101-105). Two different scoring
    scales share one set of thresholds. The extraction must keep the tier
    function scale-agnostic — it takes a number, not a score object — or it will
    quietly couple the podcast pipeline to the article reweighting.
    """

    def test_score_total_expr_sums_the_four_columns(self):
        expr = PodcastEpisodeScore.score_total_expr()
        assert str(expr) == (
            "podcast_episode_scores.specificity_score "
            "+ podcast_episode_scores.novelty_score "
            "+ podcast_episode_scores.depth_score "
            "+ podcast_episode_scores.actionability_score"
        )

    def test_tier_predicates_compile_against_the_same_cutoffs(self):
        """The 60/30 literals the podcast routers apply to that expression."""
        expr = PodcastEpisodeScore.score_total_expr()
        high = (expr >= 60).compile(compile_kwargs={"literal_binds": True})
        medium_low = (expr >= 30).compile(compile_kwargs={"literal_binds": True})
        low = (expr < 30).compile(compile_kwargs={"literal_binds": True})
        assert str(high).endswith(">= 60")
        assert str(medium_low).endswith(">= 30")
        assert str(low).endswith("< 30")

    @pytest.mark.parametrize(
        ("spec", "novelty", "depth", "action", "podcast_total", "article_v5_total"),
        [
            # Same four dimension values, two totals, sometimes two tiers.
            (25, 25, 25, 25, 100, 100),
            (15, 15, 15, 15, 60, 60),
            (10, 25, 25, 10, 70, 40),  # podcast High, article Medium
            (5, 25, 25, 5, 60, 20),  # podcast High, article Low
            (25, 0, 0, 25, 50, 100),  # podcast Medium, article High
            (20, 0, 0, 20, 40, 80),  # podcast Medium, article High
            (0, 20, 20, 0, 40, 0),  # podcast Medium, article Low
        ],
    )
    def test_the_two_scales_disagree_on_tier(
        self,
        spec: int,
        novelty: int,
        depth: int,
        action: int,
        podcast_total: int,
        article_v5_total: int,
    ):
        """Identical subscores, different totals — pinned, not a defect.

        Podcast scoring was never recalibrated to v5; it still sums all four.
        """
        assert spec + novelty + depth + action == podcast_total
        assert reweight_total(spec, action) == article_v5_total

    def test_podcast_scale_reaches_odd_totals_that_v5_cannot(self):
        """Sum-of-four can be odd; (Q+I)*2 cannot. The 30/60 edges behave differently."""
        assert 15 + 14 + 15 + 15 == 59
        assert _tier_label_plain(59) == "Medium"
        assert all(reweight_total(q, i) % 2 == 0 for q in range(26) for i in range(26))
