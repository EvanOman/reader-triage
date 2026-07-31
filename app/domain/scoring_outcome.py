"""The one place inbox_monitor decides what a score *means*.

A scoring strategy answers "how good is this article" and produces a total.
Everything downstream of that total — the priority it gets ranked by, whether
it is skipped, and which value tier it lands in — is decided here and nowhere
else.

Before this module the rule was re-derived at five write sites and six tier
sites, and one of them had already drifted: ``tools/backfill_highlighted.py``
hardcoded ``author_boost=0.0``, so rows it wrote silently lost the boost every
other path applied. Duplication is how that happened; a single home is how it
stops happening again.

Two rules, deliberately independent:

  priority = total + author_boost
      The boost reorders the queue. It is *ranking*, not *value*.

  skip / tier = f(total), boost excluded
      Value judgements read the raw total. A favoured author does not make a
      contentless article worth reading, so a boost can lift an item's rank
      while it stays skip-recommended and in the low tier.

Keeping those two apart is the invariant most likely to be broken by a
well-meaning simplification, so it is stated here and pinned by
``tests/test_scoring_golden_master.py`` (see fixtures gm-06, gm-07, gm-16).

Scope note: this is a rules module, not a domain layer. inbox_monitor's
``app/models/`` is a SQLAlchemy schema and stays one; nothing here knows about
sessions, rows, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import ColumnElement, SQLColumnExpression

__all__ = [
    "HIGH_VALUE_THRESHOLD",
    "MEDIUM_VALUE_THRESHOLD",
    "SKIP_REASON",
    "SKIP_THRESHOLD",
    "ScoringOutcome",
    "ValueTier",
    "derive_outcome",
    "derive_priority",
    "should_skip",
    "skip_reason_for",
    "tier_criteria",
    "value_tier",
]

# A total below this is not worth the reader's attention.
SKIP_THRESHOLD = 30.0

# Value tiers. Note these are *inclusive lower bounds*: exactly 60.0 is High and
# exactly 30.0 is Medium.
HIGH_VALUE_THRESHOLD = 60.0
MEDIUM_VALUE_THRESHOLD = 30.0

# The only reason a low total is skipped for. Stored verbatim in
# ``ArticleScore.skip_reason``, so changing this string rewrites user-visible
# data and is a migration, not an edit.
SKIP_REASON = "Low information content"


class ValueTier(StrEnum):
    """Which band a total falls in. Values match the API/query tier strings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def derive_priority(total: float, author_boost: float) -> float:
    """Rank order: the total, nudged by how much the reader likes this author.

    Deliberately not clamped. A boost can carry priority above 100 or, if a
    boost is ever negative, below the total. Priority is a sort key, not a
    score, so it has no meaningful ceiling.
    """
    return total + author_boost


def should_skip(total: float) -> bool:
    """Whether the reader should skip this. Reads the raw total, not priority.

    An author boost must never rescue a contentless article.
    """
    return total < SKIP_THRESHOLD


def skip_reason_for(total: float) -> str | None:
    """The stored reason string, or None when the item is not skipped."""
    return SKIP_REASON if should_skip(total) else None


def value_tier(total: float) -> ValueTier:
    """Which value band a raw total falls in. Excludes the author boost."""
    if total >= HIGH_VALUE_THRESHOLD:
        return ValueTier.HIGH
    if total >= MEDIUM_VALUE_THRESHOLD:
        return ValueTier.MEDIUM
    return ValueTier.LOW


@dataclass(frozen=True, slots=True)
class ScoringOutcome:
    """Everything derived from a total, computed once and passed around whole.

    Call sites that persist a score should build one of these rather than
    deriving the fields individually — deriving them separately is exactly how
    the sites drifted apart in the first place.
    """

    total: float
    author_boost: float
    priority_score: float
    skip_recommended: bool
    skip_reason: str | None

    @property
    def tier(self) -> ValueTier:
        return value_tier(self.total)


def derive_outcome(total: float, author_boost: float = 0.0) -> ScoringOutcome:
    """Derive every score-dependent field at once from a total and a boost."""
    return ScoringOutcome(
        total=total,
        author_boost=author_boost,
        priority_score=derive_priority(total, author_boost),
        skip_recommended=should_skip(total),
        skip_reason=skip_reason_for(total),
    )


def tier_criteria(
    score_total: SQLColumnExpression[int | float], tier: str
) -> tuple[ColumnElement[bool], ...]:
    """SQL predicates selecting one value tier, for ``.where(*criteria)``.

    The same cutoffs as :func:`value_tier`, expressed against a column so
    queries filter in the database instead of re-implementing the bands. An
    unrecognised tier selects nothing, matching the previous behaviour of the
    routers' if/elif chains, which simply left the query unfiltered.

    Takes ``SQLColumnExpression`` rather than ``ColumnElement`` so that both
    kinds of caller fit: a mapped column such as ``ArticleScore.info_score``
    (an ``InstrumentedAttribute``) and a computed one such as
    ``PodcastEpisodeScore.score_total_expr()``.
    """
    if tier == ValueTier.HIGH:
        return (score_total >= HIGH_VALUE_THRESHOLD,)
    if tier == ValueTier.MEDIUM:
        return (score_total >= MEDIUM_VALUE_THRESHOLD, score_total < HIGH_VALUE_THRESHOLD)
    if tier == ValueTier.LOW:
        return (score_total < MEDIUM_VALUE_THRESHOLD,)
    return ()
