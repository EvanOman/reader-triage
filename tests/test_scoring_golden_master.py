"""Golden-master characterization test for the article scoring pipeline.

WHAT THIS PINS
--------------
This is a characterization ("golden master") test in the Working-Effectively-
With-Legacy-Code sense: it does not assert that the scoring pipeline is *right*,
it asserts that it is *unchanged*. A hand-authored corpus of 16 fixed articles
(``tests/golden/scoring_corpus.json``) is driven through the real
``ArticleScorer`` orchestration against in-memory SQLite, and every persisted
``ArticleScore`` column plus the derived value tier is serialized as canonical
JSON and compared byte-for-byte against ``tests/golden/scoring_pipeline_golden.json``.

Three pipeline write sites are exercised in sequence, each snapshotted:

  1. ``_process_documents``      -- first scan: create articles, score, derive
                                    priority_score / author_boost / priority_signals
                                    / skip_recommended / skip_reason.
  2. ``recompute_priorities``    -- author highlight counts change; priority and
                                    boost are re-derived *without* touching
                                    skip_recommended or priority_signals.
  3. ``rescore_failed_articles`` -- content-fetch failures are re-fetched and
                                    re-scored, re-deriving priority and skip.

The corpus deliberately covers every case that a careless extraction of the
scoring-outcome logic would break:

  * a high scorer (100.0), a mid scorer (44.0) and a low scorer (18.0);
  * info_score landing EXACTLY on 60.0 (``gm-02``) and EXACTLY on 30.0
    (``gm-04``) -- the two tier/skip cutoffs;
  * 26.0, the highest reachable score that still skips (``gm-05``);
  * every rung of ``ArticleScorer.AUTHOR_BOOST_THRESHOLDS`` (+15/+10/+7/+5/+3),
    an author row that exists but is below every threshold (boost 0.0, ``gm-15``),
    an author with no row at all (``gm-01``), and a null author (``gm-09``);
  * author-name normalisation -- ``"  PROLIFIC highlighter "`` matching
    ``"prolific highlighter"`` (``gm-16``);
  * TWO articles whose author boost pushes priority_score across the 60 line
    while info_score stays below it (``gm-07``: 56.0 -> 61.0, ``gm-16``: 48.0 ->
    63.0). This is exactly the case that breaks when a naive extraction derives
    the value tier from priority_score instead of info_score;
  * a low scorer whose boost lifts priority into the medium band while
    skip_recommended stays True (``gm-06``);
  * both content-fetch-failure shapes -- summary-only (the model is never
    called, ``gm-10``) and paywall stub (the model IS called with a
    content_warning, ``gm-11``);
  * a ``highlight``-category document that is excluded from scoring entirely and
    therefore has no ArticleScore row at all (``gm-12``).

DETERMINISM
-----------
Nothing is generated at test time. Every model answer is a recorded response in
the corpus file, replayed through the real ``ReweightedCategoricalScoringStrategy``
so that the production categorical-to-numeric mapping and the v5 reweight are
themselves under test; only the DSPy predict call is substituted. The strategy
raises if it is asked for an article it has no recording for, so a live model
call is impossible even by accident. Two fields are normalised because they are
wall-clock values: ``scored_at`` and ``priority_computed_at`` become
``"<timestamp>"`` when set and stay ``null`` when unset (the set/unset
distinction is load-bearing -- ``recompute_priorities`` only stamps
``priority_computed_at`` when a value actually changed). The ``ArticleScore.id``
autoincrement column is omitted entirely. Nothing else is normalised, elided or
compared loosely.

REGENERATING THE GOLDEN FILE
----------------------------
Run with ``UPDATE_GOLDEN=1`` to rewrite ``tests/golden/scoring_pipeline_golden.json``
and its ``.sha256`` sidecar. The regeneration run always FAILS on purpose, so a
regenerate can never be mistaken for a green run.

    UPDATE_GOLDEN=1 uv run pytest tests/test_scoring_golden_master.py

Regenerating is a deliberate act. The whole point of this file is that a change
to priority_score, skip_recommended, skip_reason, author_boost or the derived
value tier shows up as a diff, so the new golden MUST be reviewed line by line
in ``git diff`` before it is committed. If you cannot explain every changed line,
the change is a regression, not a new baseline. Do not "fix" a mismatch by
loosening the comparison.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.article import Article, ArticleScore, Author, Base
from app.services.readwise import ReaderDocument
from app.services.scorer import ArticleScorer, InfoScore
from app.services.scoring_models import V2CategoricalOutput
from app.services.scoring_strategy import ReweightedCategoricalScoringStrategy
from tests.factories import FakeReadwiseService, make_dspy_prediction, mock_lm_history

pytestmark = pytest.mark.golden


GOLDEN_DIR = Path(__file__).parent / "golden"
CORPUS_PATH = GOLDEN_DIR / "scoring_corpus.json"
GOLDEN_PATH = GOLDEN_DIR / "scoring_pipeline_golden.json"
GOLDEN_SHA_PATH = GOLDEN_DIR / "scoring_pipeline_golden.sha256"

# Pinned so the persisted `model_used` column never depends on SCORING_MODEL in
# the ambient environment. The `openai/` prefix is stripped by the strategy's
# `model_id` property exactly as it is in production.
FIXTURE_MODEL_ID = "openai/golden-fixture-model"

TIMESTAMP_PLACEHOLDER = "<timestamp>"

# Production tier cutoffs (app/routers/api.py, app/templates/*.html).
HIGH_TIER_CUTOFF = 60
MEDIUM_TIER_CUTOFF = 30


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def as_dict(value: object) -> dict[str, object]:
    """Narrow a decoded JSON value to a string-keyed dict."""
    assert isinstance(value, dict), f"expected a JSON object, got {type(value).__name__}"
    return {str(key): item for key, item in value.items()}


def as_dict_list(value: object) -> list[dict[str, object]]:
    """Narrow a decoded JSON value to a list of string-keyed dicts."""
    assert isinstance(value, list), f"expected a JSON array, got {type(value).__name__}"
    return [as_dict(item) for item in value]


def _opt_str(value: object) -> str | None:
    assert value is None or isinstance(value, str)
    return value


def _opt_int(value: object) -> int | None:
    assert value is None or isinstance(value, int)
    return value


def _opt_float(value: object) -> float | None:
    assert value is None or isinstance(value, int | float)
    return None if value is None else float(value)


def load_corpus() -> dict[str, object]:
    """Load the hand-authored fixture corpus."""
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return as_dict(json.load(fh))


def corpus_articles(corpus: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the corpus article entries."""
    return as_dict_list(corpus["articles"])


def _entry_field(entry: Mapping[str, object], key: str) -> dict[str, object] | None:
    value = entry.get(key)
    return None if value is None else as_dict(value)


def make_reader_document(doc_id: str, spec: Mapping[str, object]) -> ReaderDocument:
    """Build a ReaderDocument from a corpus `document` block.

    Every field is type-checked on the way through, so a malformed fixture fails
    here rather than surfacing as a mysterious golden diff.

    Timestamps are deliberately None: the pipeline copies them straight onto the
    Article row and they play no part in scoring, so leaving them out keeps the
    fixtures free of dates.
    """
    return ReaderDocument(
        id=doc_id,
        title=str(spec["title"]),
        url=str(spec["url"]),
        author=_opt_str(spec["author"]),
        word_count=_opt_int(spec["word_count"]),
        content=_opt_str(spec["content"]),
        summary=_opt_str(spec["summary"]),
        location=_opt_str(spec["location"]),
        category=_opt_str(spec["category"]),
        site_name=_opt_str(spec["site_name"]),
        source_url=_opt_str(spec["source_url"]),
        reading_progress=_opt_float(spec["reading_progress"]),
        created_at=None,
        updated_at=None,
        published_date=None,
    )


# ---------------------------------------------------------------------------
# Recorded scoring strategy
# ---------------------------------------------------------------------------


class RecordedScoringStrategy:
    """Replays recorded model responses through the REAL v5 scoring strategy.

    Only the DSPy predict call is substituted. Prompt construction, content
    truncation, the categorical-to-numeric mapping and the v5 reweight all run
    for real, so a change to any of them shows up in the golden file.

    Calls are counted per article id: the first call replays `model_response`
    and any later call (i.e. `rescore_failed_articles`) replays
    `rescore_model_response` when the entry defines one. An article with no
    recording raises -- nothing may be generated at test time.
    """

    def __init__(self, corpus: Mapping[str, object]) -> None:
        with patch(
            "app.services.scoring_strategy._make_lm",
            return_value=mock_lm_history(input_tokens=1234, output_tokens=567),
        ):
            self._inner = ReweightedCategoricalScoringStrategy(model_id=FIXTURE_MODEL_ID)
        self._entries: dict[str, dict[str, object]] = {
            str(entry["id"]): entry for entry in corpus_articles(corpus)
        }
        self.call_counts: dict[str, int] = {}
        self.prompts: dict[str, list[str]] = {}

    @property
    def version(self) -> str:
        return self._inner.version

    @property
    def accepted_versions(self) -> frozenset[str]:
        return self._inner.accepted_versions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

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
        entry = self._entries.get(entity_id)
        if entry is None:
            raise AssertionError(
                f"No recorded model response for {entity_id!r}. The golden master "
                "must never generate a score at test time."
            )

        call_index = self.call_counts.get(entity_id, 0)
        self.call_counts[entity_id] = call_index + 1

        recordings: list[dict[str, object] | None] = [_entry_field(entry, "model_response")]
        rescore_recording = _entry_field(entry, "rescore_model_response")
        if rescore_recording is not None:
            recordings.append(rescore_recording)
        recorded = recordings[min(call_index, len(recordings) - 1)]
        if recorded is None:
            raise AssertionError(
                f"{entity_id!r} declares model_response: null -- the pipeline is not "
                "supposed to invoke the scoring model for it, but it just did."
            )

        predict = MagicMock(
            return_value=make_dspy_prediction(V2CategoricalOutput(**recorded))  # type: ignore[missing-argument]
        )
        self._inner._predict_article = predict

        result = await self._inner.score(
            title=title,
            author=author,
            content=content,
            word_count=word_count,
            content_type_hint=content_type_hint,
            entity_id=entity_id,
            content_warning=content_warning,
        )

        assert predict.call_count == 1, f"strategy did not reach the model for {entity_id!r}"
        self.prompts.setdefault(entity_id, []).append(
            str(predict.call_args.kwargs["evaluation_prompt"])
        )

        override = entry.get("total_override")
        if result is not None and override is not None:
            assert isinstance(override, int)
            result.total_override = override
        return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def value_tier(score: float | None) -> str | None:
    """Derive the high/medium/low tier the way production does (60/30 cutoffs)."""
    if score is None:
        return None
    if score >= HIGH_TIER_CUTOFF:
        return "high"
    if score >= MEDIUM_TIER_CUTOFF:
        return "medium"
    return "low"


def _normalized_timestamp(value: object) -> str | None:
    """Collapse a wall-clock timestamp to a placeholder, preserving set/unset."""
    return None if value is None else TIMESTAMP_PLACEHOLDER


def _serialize_score(score: ArticleScore) -> dict[str, object]:
    """Serialize every ArticleScore column except the autoincrement id.

    `value_tier` is the production tier (derived from info_score).
    `value_tier_from_priority_score` is the tier you would get if the cutoffs
    were applied to priority_score instead -- it exists so that the two
    boost-crosses-60 fixtures make the divergence explicit in the diff.
    """
    return {
        "actionability_score": score.actionability_score,
        "author_boost": score.author_boost,
        "content_fetch_failed": score.content_fetch_failed,
        "depth_score": score.depth_score,
        "info_score": score.info_score,
        "model_used": score.model_used,
        "novelty_score": score.novelty_score,
        "overall_assessment": score.overall_assessment,
        "priority_computed_at": _normalized_timestamp(score.priority_computed_at),
        "priority_score": score.priority_score,
        "priority_signals": score.priority_signals,
        "score_reasons": score.score_reasons,
        "scored_at": _normalized_timestamp(score.scored_at),
        "scoring_version": score.scoring_version,
        "skip_reason": score.skip_reason,
        "skip_recommended": score.skip_recommended,
        "specificity_score": score.specificity_score,
        "value_tier": value_tier(score.info_score),
        "value_tier_from_priority_score": value_tier(score.priority_score),
    }


def _serialize_article(article: Article) -> dict[str, object]:
    """Serialize the article-side fields the scoring pipeline writes."""
    return {
        "author": article.author,
        "category": article.category,
        "content_length": None if article.content is None else len(article.content),
        "content_preview_length": (
            None if article.content_preview is None else len(article.content_preview)
        ),
        "location": article.location,
        "reading_progress": article.reading_progress,
        "site_name": article.site_name,
        "title": article.title,
        "url": article.url,
        "word_count": article.word_count,
    }


async def snapshot_state(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, object]:
    """Snapshot every article, its ArticleScore row and the authors table."""
    async with session_factory() as session:
        articles = list((await session.execute(select(Article))).scalars().all())
        authors = list((await session.execute(select(Author))).scalars().all())

        rows: list[dict[str, object]] = []
        for article in sorted(articles, key=lambda a: a.id):
            score = (
                await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == article.id)
                )
            ).scalar_one_or_none()
            rows.append(
                {
                    "article": _serialize_article(article),
                    "article_id": article.id,
                    "score": None if score is None else _serialize_score(score),
                }
            )

    return {
        "articles": rows,
        "authors": [
            {
                "name": author.name,
                "normalized_name": author.normalized_name,
                "total_highlights": author.total_highlights,
            }
            for author in sorted(authors, key=lambda a: a.normalized_name)
        ],
    }


def canonical_json(document: Mapping[str, object]) -> str:
    """Canonical serialization: sorted keys, 2-space indent, ASCII, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------------------
# Pipeline harness
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _fresh_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A private in-memory SQLite database, torn down at the end of the block.

    Deliberately self-contained rather than reusing the shared `session_factory`
    fixture: `test_output_is_stable_across_repeated_runs` needs two independent
    databases inside a single test.
    """
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@asynccontextmanager
async def _patched_pipeline(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[None]:
    """Point the scorer at the test database and stub its out-of-scope collaborators.

    FTS indexing, usage logging and rate-limit sleeps are not part of the scored
    output; they are the only things stubbed besides Readwise and the model.
    """
    with (
        patch(
            "app.services.scorer.get_session_factory",
            new=AsyncMock(return_value=session_factory),
        ),
        patch("app.services.scorer.upsert_fts_entry", new=AsyncMock()),
        patch("app.services.scoring_strategy.log_usage", new=AsyncMock()),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        yield


async def _seed_authors(
    session_factory: async_sessionmaker[AsyncSession],
    authors: Sequence[Mapping[str, object]],
) -> None:
    async with session_factory() as session:
        for spec in authors:
            session.add(
                Author(
                    name=str(spec["name"]),
                    normalized_name=str(spec["normalized_name"]),
                    total_highlights=int(str(spec["total_highlights"])),
                )
            )
        await session.commit()


async def _apply_author_updates(
    session_factory: async_sessionmaker[AsyncSession],
    updates: Sequence[Mapping[str, object]],
) -> None:
    async with session_factory() as session:
        for spec in updates:
            name = str(spec["name"])
            highlights = int(str(spec["total_highlights"]))
            if spec.get("insert"):
                session.add(
                    Author(
                        name=name,
                        normalized_name=str(spec["normalized_name"]),
                        total_highlights=highlights,
                    )
                )
                continue
            author = (await session.execute(select(Author).where(Author.name == name))).scalar_one()
            author.total_highlights = highlights
        await session.commit()


async def run_pipeline() -> tuple[dict[str, object], RecordedScoringStrategy]:
    """Drive the real scoring pipeline over the corpus and build the golden document."""
    corpus = load_corpus()
    entries = corpus_articles(corpus)
    authors = as_dict_list(corpus["authors"])
    author_updates = as_dict_list(corpus["author_updates_before_recompute"])

    strategy = RecordedScoringStrategy(corpus)
    readwise = FakeReadwiseService()
    documents: list[ReaderDocument] = []
    for entry in entries:
        spec = _entry_field(entry, "document")
        assert spec is not None
        doc = make_reader_document(str(entry["id"]), spec)
        readwise.add_document(doc)
        documents.append(doc)

    async with _fresh_session_factory() as session_factory:
        await _seed_authors(session_factory, authors)

        async with _patched_pipeline(session_factory):
            with patch("app.services.scorer.get_readwise_service"):
                scorer = ArticleScorer(strategy=strategy)
            scorer._readwise = readwise

            # --- Stage 1: first scan. Readwise hands over metadata only; the
            # pipeline re-fetches each document with content when it decides to
            # score, exactly as it does in production.
            metadata_docs = [replace(doc, content=None) for doc in documents]
            scan = await scorer._process_documents(metadata_docs)
            stage_1 = await snapshot_state(session_factory)
            stage_1["name"] = "1_process_documents"
            stage_1["description"] = (
                "First scan of the corpus: articles created, scored, and "
                "priority/boost/skip derived."
            )
            stage_1["result"] = {
                "newly_scored": scan.newly_scored,
                "total_scanned": scan.total_scanned,
            }

            # --- Stage 2: author highlight counts move, priorities recomputed.
            await _apply_author_updates(session_factory, author_updates)
            updated = await scorer.recompute_priorities()
            stage_2 = await snapshot_state(session_factory)
            stage_2["name"] = "2_recompute_priorities"
            stage_2["description"] = (
                "Author highlight counts changed; priority_score and author_boost "
                "are re-derived. skip_recommended, skip_reason and priority_signals "
                "are deliberately NOT revisited by this write site."
            )
            stage_2["result"] = {"updated": updated}

            # --- Stage 3: Readwise finally returns the full text for the stub
            # article, and the fetch-failure sweep re-scores it.
            for entry in entries:
                rescore_spec = _entry_field(entry, "rescore_document")
                if rescore_spec is not None:
                    readwise.add_document(make_reader_document(str(entry["id"]), rescore_spec))
            rescored = await scorer.rescore_failed_articles()
            stage_3 = await snapshot_state(session_factory)
            stage_3["name"] = "3_rescore_failed_articles"
            stage_3["description"] = (
                "Content-fetch failures re-fetched. gm-11 now has full text and is "
                "re-scored; gm-10 still has none and is only re-flagged."
            )
            stage_3["result"] = {"rescored": rescored}

    document: dict[str, object] = {
        "_about": (
            "Golden master for the article scoring pipeline. Regenerate with "
            "UPDATE_GOLDEN=1 and review every changed line in git diff -- see the "
            "module docstring of tests/test_scoring_golden_master.py."
        ),
        "corpus_article_count": len(entries),
        "normalized_fields": {
            "article_scores.id": "omitted (autoincrement rowid)",
            "article_scores.priority_computed_at": (
                f"{TIMESTAMP_PLACEHOLDER!r} when set, null when unset (wall clock)"
            ),
            "article_scores.scored_at": (
                f"{TIMESTAMP_PLACEHOLDER!r} when set, null when unset (wall clock)"
            ),
        },
        "stages": [stage_1, stage_2, stage_3],
    }
    return document, strategy


async def build_golden_text() -> str:
    """Run the pipeline and render the canonical golden text."""
    document, _ = await run_pipeline()
    return canonical_json(document)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unified_diff(expected: str, actual: str) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="tests/golden/scoring_pipeline_golden.json (committed)",
            tofile="pipeline output (current code)",
            n=4,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_scored_output_matches_golden_master() -> None:
    """The full scored output of the corpus is byte-for-byte the committed golden.

    A diff here means the derived scoring outcome changed. Read the diff before
    doing anything else: `priority_score`, `author_boost`, `skip_recommended`,
    `skip_reason` and `value_tier` are the fields this test exists to protect.
    """
    actual = await build_golden_text()

    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.write_text(actual, encoding="utf-8")
        GOLDEN_SHA_PATH.write_text(sha256_of(actual) + "\n", encoding="utf-8")
        pytest.fail(
            "UPDATE_GOLDEN=1: the golden file was regenerated and this run is "
            "failed on purpose so it cannot be mistaken for a passing run.\n"
            f"New SHA-256: {sha256_of(actual)}\n"
            "Review every changed line in `git diff tests/golden/` before "
            "committing, then re-run without UPDATE_GOLDEN."
        )

    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. Regenerate it deliberately with "
        "UPDATE_GOLDEN=1 and review the result."
    )
    expected = GOLDEN_PATH.read_text(encoding="utf-8")

    if actual != expected:
        pytest.fail(
            "Scoring output no longer matches the golden master.\n\n"
            f"{_unified_diff(expected, actual)}\n"
            f"golden SHA-256: {sha256_of(expected)}\n"
            f"actual SHA-256: {sha256_of(actual)}\n\n"
            "If this change is intended, regenerate with UPDATE_GOLDEN=1 and "
            "review every line of the diff."
        )


async def test_golden_file_sha256_matches_sidecar() -> None:
    """The committed .sha256 sidecar matches the committed golden file.

    The hash is printed so it can be recorded in the refactor baseline.
    """
    golden_text = GOLDEN_PATH.read_text(encoding="utf-8")
    digest = sha256_of(golden_text)
    print(f"\ntests/golden/scoring_pipeline_golden.json SHA-256: {digest}")

    recorded = GOLDEN_SHA_PATH.read_text(encoding="utf-8").strip()
    assert recorded == digest, (
        f"Sidecar hash {recorded} does not match the golden file hash {digest}. "
        "Regenerate both together with UPDATE_GOLDEN=1."
    )


async def test_output_is_stable_across_repeated_runs() -> None:
    """Two independent runs produce identical bytes.

    Guards the golden against wall-clock values, autoincrement ids, set/dict
    iteration order and any other source of churn. A golden master that rewrites
    itself every run is worthless.
    """
    first = await build_golden_text()
    second = await build_golden_text()
    assert first == second, "The pipeline snapshot is not deterministic:\n" + _unified_diff(
        first, second
    )


async def test_stage_one_matches_hand_authored_expectations() -> None:
    """Each fixture's `expected` block holds, independently of the golden file.

    This is the second safety net: it survives a careless regeneration of the
    golden file, because the expectations are hand-authored intent rather than
    recorded output.
    """
    corpus = load_corpus()
    document, _ = await run_pipeline()
    stage_1 = as_dict_list(document["stages"])[0]
    rows = as_dict_list(stage_1["articles"])
    by_id = {str(row["article_id"]): row for row in rows}

    for entry in corpus_articles(corpus):
        article_id = str(entry["id"])
        assert article_id in by_id, f"{article_id} was never persisted"
        actual_score = by_id[article_id]["score"]
        expected = _entry_field(entry, "expected")

        if expected is None:
            assert actual_score is None, (
                f"{article_id} is excluded from scoring and must have no ArticleScore row"
            )
            continue

        assert actual_score is not None, f"{article_id} has no ArticleScore row"
        persisted = as_dict(actual_score)
        for field, want in expected.items():
            assert persisted[field] == want, (
                f"{article_id}.{field}: expected {want!r}, got {persisted[field]!r}"
            )


async def test_corpus_covers_the_boundary_and_boost_cases() -> None:
    """The corpus still contains the cases this golden master exists to protect.

    Without this, someone could quietly delete the awkward fixtures and the
    golden would happily go green over a much weaker corpus.
    """
    corpus = load_corpus()
    expectations = [
        exp
        for entry in corpus_articles(corpus)
        if (exp := _entry_field(entry, "expected")) is not None
    ]
    info_scores = {exp["info_score"] for exp in expectations}

    assert 60.0 in info_scores, "no fixture lands exactly on the 60.0 high cutoff"
    assert 30.0 in info_scores, "no fixture lands exactly on the 30.0 skip/medium cutoff"
    assert any(s >= 60 for s in info_scores if isinstance(s, float)), "no high scorer"
    assert any(30 <= s < 60 for s in info_scores if isinstance(s, float)), "no mid scorer"
    assert any(s < 30 for s in info_scores if isinstance(s, float)), "no low scorer"

    assert any(exp["skip_recommended"] is True for exp in expectations), "nothing skips"
    assert any(exp["content_fetch_failed"] is True for exp in expectations), (
        "no content-fetch-failure fixture"
    )

    boosts = {exp["author_boost"] for exp in expectations}
    assert boosts >= {0.0, 3.0, 5.0, 7.0, 10.0, 15.0}, (
        f"author boost thresholds are not fully covered: {sorted(str(b) for b in boosts)}"
    )

    crossings = [
        exp
        for exp in expectations
        if isinstance(exp["info_score"], float)
        and isinstance(exp["priority_score"], float)
        and exp["info_score"] < 60 <= exp["priority_score"]
    ]
    assert len(crossings) >= 2, (
        "the corpus must keep at least two articles whose author boost pushes "
        "priority_score across 60 while info_score stays below it -- that is the "
        "case a naive extraction of the tier rule breaks"
    )


async def test_recorded_responses_cover_every_model_call() -> None:
    """No model call happens without a recording, and the two null-recording
    fixtures really are never sent to the model."""
    _, strategy = await run_pipeline()

    assert strategy.call_counts.get("gm-10-fetch-failure-summary-only", 0) == 0
    assert strategy.call_counts.get("gm-12-excluded-highlight-category", 0) == 0

    # gm-11 is scored once on the stub and once again on the rescore sweep.
    assert strategy.call_counts["gm-11-fetch-failure-stub-content"] == 2

    # The stub path must splice the truncated-content warning into the prompt,
    # immediately ahead of the article metadata block.
    warning = "NOTE: The content below appears incomplete"
    stub_prompt = strategy.prompts["gm-11-fetch-failure-stub-content"][0]
    assert warning in stub_prompt
    assert stub_prompt.index(warning) < stub_prompt.index("Article Title:")
    # ...and the rescore, on full text, must not carry it.
    rescore_prompt = strategy.prompts["gm-11-fetch-failure-stub-content"][1]
    assert warning not in rescore_prompt
