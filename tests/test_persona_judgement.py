"""PERSONA tier: end-to-end judgement over a small hand-authored article corpus.

This tier answers a question the other tiers cannot: are the scoring *outcomes*
sane? Unit and golden tests pin arithmetic and byte-for-byte output; neither
notices if the whole rubric inverts and starts ranking listicles above
practitioner essays. Here we run six deliberately different documents through
the real `ArticleScorer` pipeline and check the shape of what comes out.

Two layers, and they are independent:

  LAYER 1 (deterministic, always runs) -- ordinary assertions on the persisted
  rows: skip decisions, tier assignment against the documented 60/30 cutoffs,
  and the relative priority ordering. No agent, no subprocess, no network.

  LAYER 2 (the judge) -- shells out to `claude -p` and asks an external LLM
  whether the outcomes are defensible, then asserts on its structured verdict.
  If the CLI is missing, times out, or answers unparseably, the test SKIPS.
  A judgement tier that turns red because a CLI was unavailable is worse than
  no judgement tier at all.

Two LLM roles are in play here and they are kept strictly apart:

  (a) THE JUDGE reads the outcomes and renders a verdict. It runs as a
      subscription-backed `claude -p` subprocess. Subscription CLIs do not
      route through the LiteLLM gateway, so it costs no metered budget. It is
      out-of-process, which is why the autouse socket guard in conftest.py --
      which patches sockets inside the pytest process -- neither blocks it nor
      needs to be relaxed for it. The guard stays fully armed throughout, and
      `test_socket_guard_stays_armed` proves it. The subprocess environment is
      scrubbed of every provider credential before launch, so the judge cannot
      fall back to a metered API key even if one is present in the parent.

  (b) inbox_monitor's OWN scoring calls are the system under test. They are
      served entirely from the recorded responses in tests/persona/*.json,
      replayed through the real ReweightedCategoricalScoringStrategy so that
      the production mapping tables, the v5 reweighting, stub detection and the
      skip threshold are all genuinely exercised. Only the DSPy predictor is
      substituted. Nothing here can reach a provider.

The corpus lives in tests/persona/ and is hand-authored -- never sampled from
the production database.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.models.article import Article, ArticleScore
from app.services.readwise import ReaderDocument
from app.services.scorer import CURRENT_SCORING_VERSION, ArticleScorer
from app.services.scoring_models import V2CategoricalOutput
from app.services.scoring_strategy import ReweightedCategoricalScoringStrategy
from tests.conftest import NetworkAccessAttempted
from tests.factories import make_document, make_dspy_prediction, mock_lm_history

pytestmark = pytest.mark.persona

CORPUS_DIR = Path(__file__).parent / "persona"

# Tier cutoffs, per AGENTS.md: >= 60 high, 30-59 medium, < 30 low.
HIGH_CUTOFF = 60
MEDIUM_CUTOFF = 30


def tier_for(info_score: int) -> str:
    """Bucket a 0-100 info score into the documented value tiers."""
    if info_score >= HIGH_CUTOFF:
        return "high"
    if info_score >= MEDIUM_CUTOFF:
        return "medium"
    return "low"


# What sane behaviour looks like. Stated here rather than in the corpus files so
# that the fixtures describe only what the LLM said, and the test describes what
# the system is expected to do about it.
#
# `surfaced_rank` is the 1-based position in the ranking the app surfaces
# (ArticleScorer top-N, which excludes archived items); None means "correctly
# not surfaced".
EXPECTED: dict[str, dict[str, object]] = {
    "persona-01-technical-deep-dive": {
        "info_score": 100,
        "tier": "high",
        "skip_recommended": False,
        "content_fetch_failed": False,
        "surfaced_rank": 1,
    },
    "persona-02-listicle": {
        "info_score": 18,
        "tier": "low",
        "skip_recommended": True,
        "content_fetch_failed": False,
        "surfaced_rank": 4,
    },
    "persona-03-paywalled-stub": {
        "info_score": 0,
        "tier": "low",
        "skip_recommended": True,
        "content_fetch_failed": True,
        "surfaced_rank": 5,
    },
    "persona-04-newsletter": {
        "info_score": 48,
        "tier": "medium",
        "skip_recommended": False,
        "content_fetch_failed": False,
        "surfaced_rank": 3,
    },
    "persona-05-non-english": {
        "info_score": 84,
        "tier": "high",
        "skip_recommended": False,
        "content_fetch_failed": False,
        "surfaced_rank": 2,
    },
    "persona-06-already-read": {
        "info_score": 60,
        "tier": "high",
        "skip_recommended": False,
        "content_fetch_failed": False,
        "surfaced_rank": None,
    },
}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusEntry:
    """One hand-authored article plus the scoring response recorded for it."""

    id: str
    kind: str
    expectation: str
    judge_summary: str
    document: ReaderDocument
    recorded_response: dict[str, object] | None

    @property
    def title(self) -> str:
        return self.document.title


def _load_corpus() -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc = raw["document"]
        word_count = doc["word_count"]
        content = doc["content"]

        # Stub detection keys off word_count > 500. Corpus bodies are short and
        # declare their real length, so only the paywalled stub (which declares
        # 1800 words and returns none of them) can trip it. That keeps the stub
        # path unambiguous instead of accidentally catching a normal article.
        if content is not None:
            assert word_count <= 500, f"{path.name}: authored body must stay under the stub gate"

        entries.append(
            CorpusEntry(
                id=raw["id"],
                kind=raw["kind"],
                expectation=raw["expectation"],
                judge_summary=raw["judge_summary"],
                document=make_document(
                    id=raw["id"],
                    title=doc["title"],
                    url=doc["url"],
                    author=doc["author"],
                    word_count=word_count,
                    content=content,
                    summary=doc["summary"],
                    location=doc["location"],
                    category=doc["category"],
                    site_name=doc["site_name"],
                    reading_progress=doc["reading_progress"],
                ),
                recorded_response=raw["recorded_scoring_response"],
            )
        )

    assert len(entries) == len(EXPECTED), "corpus and expectations disagree on size"
    assert {e.id for e in entries} == set(EXPECTED), "corpus and expectations disagree on ids"
    return entries


@pytest.fixture(scope="module")
def corpus() -> list[CorpusEntry]:
    return _load_corpus()


# ---------------------------------------------------------------------------
# Recorded scoring (role (b): the system under test, never a live call)
# ---------------------------------------------------------------------------


class RecordedPredict:
    """Replays a recorded scoring response in place of `dspy.Predict`.

    Substituted for the strategy's predictor only. Everything downstream --
    the categorical-to-numeric mapping tables, the v5 reweighting, the author
    boost, the skip threshold -- is the real production code path.

    The prompt the strategy builds carries the article title, which is how a
    response is matched back to its corpus entry.
    """

    _TITLE_RE = re.compile(r"^Article Title: (?P<title>.+)$", re.MULTILINE)

    def __init__(self, corpus: list[CorpusEntry]) -> None:
        self._by_title = {e.title: e for e in corpus if e.recorded_response is not None}
        self.titles_scored: list[str] = []
        self.misses: list[str] = []

    def __call__(self, *, evaluation_prompt: str) -> MagicMock:
        match = self._TITLE_RE.search(evaluation_prompt)
        title = match.group("title").strip() if match else "<no title in prompt>"
        entry = self._by_title.get(title)
        if entry is None or entry.recorded_response is None:
            # Recorded outside the fixture set: never fall through to a live
            # call, and make the miss visible to the assertions below.
            self.misses.append(title)
            raise AssertionError(f"No recorded scoring response for {title!r}")
        self.titles_scored.append(title)
        return make_dspy_prediction(V2CategoricalOutput(**entry.recorded_response))


@pytest.fixture
def recorded_predict(corpus: list[CorpusEntry]) -> RecordedPredict:
    return RecordedPredict(corpus)


@pytest.fixture
def persona_deps(session_factory):
    """Patch the scorer's DB, FTS and usage-logging collaborators."""
    with (
        patch(
            "app.services.scorer.get_session_factory",
            new=AsyncMock(return_value=session_factory),
        ),
        patch("app.services.scorer.upsert_fts_entry", new=AsyncMock()),
        patch("app.services.scoring_strategy.log_usage", new=AsyncMock()),
    ):
        yield session_factory


@pytest.fixture
def persona_scorer(fake_readwise, corpus, recorded_predict):
    """ArticleScorer running the real v5 strategy over recorded responses."""
    strategy = ReweightedCategoricalScoringStrategy()
    strategy._predict_article = recorded_predict
    strategy._lm = mock_lm_history()

    with patch("app.services.scorer.get_readwise_service"):
        scorer = ArticleScorer(strategy=strategy)
    scorer._readwise = fake_readwise

    for entry in corpus:
        fake_readwise.add_document(entry.document)
    return scorer


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What the pipeline actually decided about one corpus article."""

    id: str
    kind: str
    title: str
    judge_summary: str
    expectation: str
    location: str | None
    reading_progress: float | None
    info_score: int
    priority_score: float | None
    quotability: int
    surprise: int
    argument: int
    applicable_insight: int
    skip_recommended: bool
    skip_reason: str | None
    content_fetch_failed: bool
    scoring_version: str | None
    overall_assessment: str | None
    surfaced_rank: int | None

    @property
    def tier(self) -> str:
        return tier_for(self.info_score)


@pytest.fixture
async def outcomes(persona_scorer, persona_deps, corpus) -> list[Outcome]:
    """Score the whole corpus once and read the persisted decisions back."""
    # Readwise hands out metadata-only documents on the list call; the scorer
    # fetches full content per document. Mirror that.
    metadata_docs = [
        make_document(
            id=e.document.id,
            title=e.document.title,
            url=e.document.url,
            author=e.document.author,
            word_count=e.document.word_count,
            content=None,
            summary=None,
            location=e.document.location,
            category=e.document.category,
            site_name=e.document.site_name,
            reading_progress=e.document.reading_progress,
        )
        for e in corpus
    ]

    result = await persona_scorer._process_documents(metadata_docs)
    surfaced = {article.id: i + 1 for i, article in enumerate(result.top_5)}

    by_id = {e.id: e for e in corpus}
    collected: list[Outcome] = []
    async with persona_deps() as session:
        rows = (
            (
                await session.execute(
                    select(Article, ArticleScore).join(
                        ArticleScore, ArticleScore.article_id == Article.id
                    )
                )
            )
            .tuples()
            .all()
        )
        for article, score in rows:
            entry = by_id[article.id]
            collected.append(
                Outcome(
                    id=article.id,
                    kind=entry.kind,
                    title=article.title,
                    judge_summary=entry.judge_summary,
                    expectation=entry.expectation,
                    location=article.location,
                    reading_progress=article.reading_progress,
                    info_score=score.info_score,
                    priority_score=score.priority_score,
                    quotability=score.specificity_score,
                    surprise=score.novelty_score,
                    argument=score.depth_score,
                    applicable_insight=score.actionability_score,
                    skip_recommended=bool(score.skip_recommended),
                    skip_reason=score.skip_reason,
                    content_fetch_failed=bool(score.content_fetch_failed),
                    scoring_version=score.scoring_version,
                    overall_assessment=score.overall_assessment,
                    surfaced_rank=surfaced.get(article.id),
                )
            )

    collected.sort(key=lambda o: o.id)
    return collected


def _by_id(outcomes: list[Outcome]) -> dict[str, Outcome]:
    return {o.id: o for o in outcomes}


# ---------------------------------------------------------------------------
# LAYER 1 -- deterministic outcome assertions. No judge involved.
# ---------------------------------------------------------------------------


class TestScoringOutcomes:
    """Concrete assertions on the persisted decisions."""

    async def test_socket_guard_stays_armed(self):
        """The persona tier does not relax hermeticity to make the judge work.

        The judge is a subprocess; in-process network access is still fatal.
        """
        with pytest.raises(NetworkAccessAttempted):
            socket.getaddrinfo("api.anthropic.com", 443)
        with pytest.raises(NetworkAccessAttempted):
            socket.create_connection(("localhost", 18400))

    async def test_whole_corpus_is_scored_and_persisted(self, outcomes, recorded_predict):
        assert len(outcomes) == len(EXPECTED)
        assert all(o.scoring_version == CURRENT_SCORING_VERSION for o in outcomes)
        assert recorded_predict.misses == [], "a document was scored without a recorded response"

    async def test_paywalled_stub_is_skipped_without_an_llm_call(
        self, outcomes, recorded_predict, corpus
    ):
        """Almost no content must not be scored as if it were an article."""
        stub = _by_id(outcomes)["persona-03-paywalled-stub"]
        assert stub.skip_recommended is True
        assert stub.content_fetch_failed is True
        assert stub.info_score == 0
        assert stub.skip_reason == "Low information content"

        stub_title = next(e.title for e in corpus if e.id == "persona-03-paywalled-stub")
        assert stub_title not in recorded_predict.titles_scored
        # Every other article did reach the scorer.
        assert len(recorded_predict.titles_scored) == len(EXPECTED) - 1

    async def test_only_the_stub_is_flagged_as_a_content_failure(self, outcomes):
        flagged = {o.id for o in outcomes if o.content_fetch_failed}
        assert flagged == {"persona-03-paywalled-stub"}

    @pytest.mark.parametrize("article_id", sorted(EXPECTED))
    async def test_tier_assignment_matches_cutoffs(self, outcomes, article_id):
        outcome = _by_id(outcomes)[article_id]
        expected = EXPECTED[article_id]
        assert outcome.info_score == expected["info_score"]
        assert outcome.tier == expected["tier"]

    @pytest.mark.parametrize("article_id", sorted(EXPECTED))
    async def test_skip_decisions(self, outcomes, article_id):
        outcome = _by_id(outcomes)[article_id]
        assert outcome.skip_recommended is EXPECTED[article_id]["skip_recommended"]
        assert outcome.content_fetch_failed is EXPECTED[article_id]["content_fetch_failed"]

    async def test_technical_piece_tops_the_ordering(self, outcomes):
        by_id = _by_id(outcomes)
        technical = by_id["persona-01-technical-deep-dive"]
        assert technical.surfaced_rank == 1
        assert technical.tier == "high"
        assert technical.skip_recommended is False
        assert technical.info_score > by_id["persona-02-listicle"].info_score
        assert technical.info_score > by_id["persona-04-newsletter"].info_score

    async def test_non_english_article_is_not_penalised_for_language(self, outcomes):
        """A substantive Spanish essay must rank on its merits, not its language."""
        by_id = _by_id(outcomes)
        spanish = by_id["persona-05-non-english"]
        assert spanish.tier == "high"
        assert spanish.skip_recommended is False
        assert spanish.info_score > by_id["persona-04-newsletter"].info_score
        assert spanish.info_score > by_id["persona-02-listicle"].info_score

    async def test_listicle_is_low_value(self, outcomes):
        listicle = _by_id(outcomes)["persona-02-listicle"]
        assert listicle.tier == "low"
        assert listicle.skip_recommended is True

    async def test_newsletter_lands_below_the_substantive_articles(self, outcomes):
        by_id = _by_id(outcomes)
        newsletter = by_id["persona-04-newsletter"]
        assert newsletter.tier == "medium"
        assert newsletter.skip_recommended is False
        assert newsletter.info_score < by_id["persona-01-technical-deep-dive"].info_score
        assert newsletter.info_score > by_id["persona-02-listicle"].info_score

    async def test_already_read_item_is_scored_but_not_surfaced(self, outcomes):
        """Scoring a finished item is fine. Surfacing it again is not."""
        read = _by_id(outcomes)["persona-06-already-read"]
        assert read.location == "archive"
        assert read.reading_progress == pytest.approx(0.95)
        assert read.tier == "high", "archived items are still scored on their merits"
        assert read.surfaced_rank is None

    async def test_relative_priority_ordering(self, outcomes):
        """The full surfaced ranking, worst-to-best inversions included."""
        surfaced = sorted(
            (o for o in outcomes if o.surfaced_rank is not None),
            key=lambda o: o.surfaced_rank or 0,
        )
        assert [o.id for o in surfaced] == [
            "persona-01-technical-deep-dive",
            "persona-05-non-english",
            "persona-04-newsletter",
            "persona-02-listicle",
            "persona-03-paywalled-stub",
        ]
        scores = [o.info_score for o in surfaced]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("article_id", sorted(EXPECTED))
    async def test_surfaced_rank(self, outcomes, article_id):
        assert _by_id(outcomes)[article_id].surfaced_rank == EXPECTED[article_id]["surfaced_rank"]

    async def test_priority_tracks_info_score_without_author_boosts(self, outcomes):
        """No corpus author is in the authors table, so priority == info score."""
        for outcome in outcomes:
            assert outcome.priority_score == pytest.approx(float(outcome.info_score))


# ---------------------------------------------------------------------------
# LAYER 2 -- the judge (role (a): out-of-process, subscription-backed)
# ---------------------------------------------------------------------------

# Scrubbed from the judge's environment. Without this the CLI could authenticate
# with a provider key inherited from the pytest process (conftest sets a dummy
# ANTHROPIC_API_KEY, and the shell may export real OpenAI/gateway credentials)
# and bill a metered account. Removing them forces subscription auth.
_CREDENTIAL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "LLM_GATEWAY_API_KEY",
    "LLM_GATEWAY_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)

# Generous: the judge is a full CLI cold start plus one inference turn.
_JUDGE_TIMEOUT_SECONDS = 300

# Opus, not the inherited session model: delegated work must not land on Fable.
_JUDGE_MODEL = "opus"

_JUDGE_PROMPT = """You are auditing the output of an article triage system. It scores saved
articles 0-100 for "capture value" -- how likely a reader is to want to save and highlight
passages -- and uses the score to decide what to surface, what to rank first, and what to skip.

Tiers: score >= 60 is high value, 30-59 is medium, below 30 is low. A score below 30 sets
skip_recommended. Archived or already-finished items are scored but deliberately left out of
the surfaced ranking.

Below are six articles and the decisions the system made about each. Judge whether each decision
is defensible given the description of the article. You are judging the OUTCOME, not the prose.

{articles}

Answer with ONLY a JSON object, no markdown fences and no other text, in exactly this shape:

{{"articles": [{{"id": "<article id>", "reasonable": <true or false>, "reason": "<one line>"}}],
 "systematic_inversion": <true or false>,
 "overall": "<pass or fail>",
 "notes": "<one line>"}}

Include one entry in "articles" for every article id listed above, in the order given.
Set "systematic_inversion" to true only if the ranking is broadly backwards -- for example if
low-value or contentless items outrank substantive ones, or if the system is systematically
rewarding what it should penalise. An individual debatable score is not a systematic inversion.
Set "overall" to "fail" if the outcomes as a whole would mislead a reader deciding what to read.
"""


def _render_article(outcome: Outcome) -> str:
    rank = "not surfaced" if outcome.surfaced_rank is None else f"#{outcome.surfaced_rank}"
    return "\n".join(
        [
            f"--- {outcome.id} ---",
            f"Title: {outcome.title}",
            f"What it is: {outcome.judge_summary}",
            f"Readwise location: {outcome.location}; read progress: {outcome.reading_progress}",
            "System decisions:",
            f"  total score: {outcome.info_score}/100 ({outcome.tier} tier)",
            f"  subscores -- quotability {outcome.quotability}/25, "
            f"surprise {outcome.surprise}/25, argument {outcome.argument}/25, "
            f"applicable insight {outcome.applicable_insight}/25",
            f"  skip_recommended: {outcome.skip_recommended}"
            + (f" ({outcome.skip_reason})" if outcome.skip_reason else ""),
            f"  content_fetch_failed: {outcome.content_fetch_failed}",
            f"  position in surfaced ranking: {rank}",
            f"  system's own assessment: {outcome.overall_assessment}",
        ]
    )


def build_judge_prompt(outcomes: list[Outcome]) -> str:
    ordered = sorted(outcomes, key=lambda o: o.id)
    return _JUDGE_PROMPT.format(articles="\n\n".join(_render_article(o) for o in ordered))


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Pull the first balanced JSON object out of a CLI response."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def run_judge(prompt: str, cwd: Path) -> dict[str, object]:
    """Ask the subscription-backed CLI for a verdict, or skip the test.

    Any unavailability -- CLI missing, timeout, non-zero exit, unparseable
    answer -- is a skip, never a failure. Layer 1 already covers the outcomes
    deterministically; this layer is additive.
    """
    executable = shutil.which("claude")
    if executable is None:
        pytest.skip("persona judge unavailable: `claude` CLI not on PATH")

    env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_ENV_VARS}

    try:
        completed = subprocess.run(
            [
                executable,
                "-p",
                prompt,
                "--model",
                _JUDGE_MODEL,
                "--output-format",
                "json",
                "--strict-mcp-config",
            ],
            capture_output=True,
            text=True,
            timeout=_JUDGE_TIMEOUT_SECONDS,
            env=env,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("persona judge unavailable: `claude` CLI could not be executed")
    except subprocess.TimeoutExpired:
        pytest.skip(f"persona judge timed out after {_JUDGE_TIMEOUT_SECONDS}s")

    if completed.returncode != 0:
        pytest.skip(
            f"persona judge exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )

    envelope = _extract_json_object(completed.stdout)
    if envelope is None:
        pytest.skip("persona judge returned no parseable envelope")
    if envelope.get("is_error"):
        pytest.skip(f"persona judge reported an error: {str(envelope.get('result'))[:300]}")

    result = envelope.get("result")
    if not isinstance(result, str):
        pytest.skip("persona judge envelope carried no textual result")

    verdict = _extract_json_object(result)
    if verdict is None:
        pytest.skip(f"persona judge answer was not JSON: {result.strip()[:300]}")
    return verdict


class TestJudgeVerdict:
    """The judge reads the recorded, deterministic outcomes and rules on them."""

    async def test_judge_finds_no_systematic_inversion(self, outcomes, tmp_path, capsys):
        verdict = run_judge(build_judge_prompt(outcomes), cwd=tmp_path)

        judged = verdict.get("articles")
        if not isinstance(judged, list) or not judged:
            pytest.skip(f"persona judge returned no per-article verdicts: {verdict}")

        per_article = {
            str(item.get("id")): item
            for item in judged
            if isinstance(item, dict) and item.get("id")
        }
        missing = set(EXPECTED) - set(per_article)
        if missing:
            pytest.skip(f"persona judge skipped articles: {sorted(missing)}")

        with capsys.disabled():
            print("\n--- persona judge verdict ---")
            print(f"overall: {verdict.get('overall')}")
            print(f"systematic_inversion: {verdict.get('systematic_inversion')}")
            print(f"notes: {verdict.get('notes')}")
            for article_id in sorted(per_article):
                item = per_article[article_id]
                print(f"  {article_id}: {item.get('reasonable')} -- {item.get('reason')}")

        unreasonable = {
            article_id: item.get("reason")
            for article_id, item in per_article.items()
            if item.get("reasonable") is False
        }

        # The load-bearing assertion: the rubric has not inverted.
        assert verdict.get("systematic_inversion") is not True, (
            "judge flagged a systematic scoring inversion: "
            f"{verdict.get('notes')!r}; objections: {unreasonable}"
        )

        # Robustness floor rather than an all-or-nothing check -- an individual
        # score the judge would have set differently is not a regression.
        reasonable_count = sum(1 for item in per_article.values() if item.get("reasonable") is True)
        assert reasonable_count >= 4, (
            f"judge called only {reasonable_count}/{len(per_article)} outcomes reasonable "
            f"(overall={verdict.get('overall')!r}); objections: {unreasonable}"
        )
