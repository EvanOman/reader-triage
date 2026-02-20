"""Tests for v3-binary scoring: computation, strategy, and gatekeeper logic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import TextBlock

from app.services.scoring_strategy import (
    BinaryScoringStrategy,
    compute_binary_dimension,
    compute_binary_total,
)

# ---------------------------------------------------------------------------
# Pure score computation tests
# ---------------------------------------------------------------------------


class TestComputeBinaryTotal:
    """Tests for compute_binary_total()."""

    def test_all_positive_yes_all_negative_no_gives_max(self):
        """All positive questions yes, all negative questions no → 100."""
        responses = {f"q{i}": True for i in range(1, 21)}
        # Set penalty questions to False (best case)
        for q in ["q8", "q12", "q16", "q20"]:
            responses[q] = False
        assert compute_binary_total(responses) == 100

    def test_all_no_gives_baseline(self):
        """All questions no → maps -0 from 0 to scaled score."""
        responses = {f"q{i}": False for i in range(1, 21)}
        # raw = 0, min = -24, max = 95
        # scaled = 100 * (0 - (-24)) / (95 - (-24)) = 100 * 24/119 ≈ 20
        score = compute_binary_total(responses)
        assert score == 20  # 100 * 24 / 119 = 20.16... → int → 20

    def test_all_yes_including_penalties(self):
        """All questions yes including penalties."""
        responses = {f"q{i}": True for i in range(1, 21)}
        # raw = sum of all weights = 95 + (-24) = 71
        # scaled = 100 * (71 - (-24)) / (95 - (-24)) = 100 * 95/119 ≈ 79
        score = compute_binary_total(responses)
        assert score == 79  # 100 * 95 / 119 = 79.83... → int → 79

    def test_only_penalties_yes_gives_minimum(self):
        """Only penalty questions yes, everything else no → 0."""
        responses = {f"q{i}": False for i in range(1, 21)}
        for q in ["q8", "q12", "q16", "q20"]:
            responses[q] = True
        # raw = -6 + -6 + -4 + -8 = -24
        # scaled = 100 * (-24 - (-24)) / (95 - (-24)) = 0
        assert compute_binary_total(responses) == 0

    def test_missing_keys_treated_as_false(self):
        """Missing keys default to False."""
        score = compute_binary_total({})
        assert score == 20  # Same as all-no

    def test_single_essential_question(self):
        """A single essential question (q3, weight=8) yields expected score."""
        responses = {f"q{i}": False for i in range(1, 21)}
        responses["q3"] = True
        # raw = 8, scaled = 100 * (8 + 24) / 119 = 100 * 32/119 ≈ 26
        score = compute_binary_total(responses)
        assert score == 26


class TestComputeBinaryDimension:
    """Tests for compute_binary_dimension()."""

    def test_quotability_all_yes(self):
        """All quotability questions yes → max 25."""
        responses = {"q1": True, "q2": True, "q3": True, "q4": True}
        assert compute_binary_dimension("quotability", responses) == 25

    def test_quotability_all_no(self):
        """All quotability questions no → 0 (no penalties in this dimension)."""
        responses = {"q1": False, "q2": False, "q3": False, "q4": False}
        assert compute_binary_dimension("quotability", responses) == 0

    def test_surprise_best_case(self):
        """Surprise: q5-q7 yes, q8 (penalty) no → max 25."""
        responses = {"q5": True, "q6": True, "q7": True, "q8": False}
        assert compute_binary_dimension("surprise", responses) == 25

    def test_surprise_worst_case(self):
        """Surprise: q5-q7 no, q8 (penalty) yes → 0."""
        responses = {"q5": False, "q6": False, "q7": False, "q8": True}
        assert compute_binary_dimension("surprise", responses) == 0

    def test_argument_mixed(self):
        """Argument: some yes, some no → intermediate score."""
        responses = {"q9": True, "q10": False, "q11": True, "q12": False}
        # dim_max = 6+6+8 = 20, dim_min = -6
        # raw = 6+0+8+0 = 14
        # scaled = 25 * (14 - (-6)) / (20 - (-6)) = 25 * 20/26 ≈ 19
        score = compute_binary_dimension("argument", responses)
        assert score == 19

    def test_insight_with_penalty(self):
        """Insight: all positive yes + penalty yes → intermediate."""
        responses = {"q13": True, "q14": True, "q15": True, "q16": True}
        # dim_max = 8+6+5 = 19, dim_min = -4
        # raw = 8+6+5+(-4) = 15
        # scaled = 25 * (15 - (-4)) / (19 - (-4)) = 25 * 19/23 ≈ 20
        score = compute_binary_dimension("insight", responses)
        assert score == 20


# ---------------------------------------------------------------------------
# BinaryScoringStrategy tests (mocked Claude)
# ---------------------------------------------------------------------------


def _make_binary_response(overrides: dict | None = None) -> dict:
    """Build a full v3-binary Claude response with all 20 questions."""
    base: dict[str, object] = {}
    for i in range(1, 21):
        base[f"q{i}"] = True
        base[f"q{i}_reason"] = f"Reason for q{i}"
    # Set penalty questions to false by default (best case)
    for q in ["q8", "q12", "q16", "q20"]:
        base[q] = False
        base[f"{q}_reason"] = f"Not applicable for {q}"
    base["overall_assessment"] = "Strong article with good insights."
    if overrides:
        base.update(overrides)
    return base


def _mock_claude_response(
    data: dict, input_tokens: int = 500, output_tokens: int = 200
) -> MagicMock:
    """Create a mock Anthropic messages.create() return value."""
    content_block = TextBlock(type="text", text=json.dumps(data))
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


class TestBinaryScoringStrategy:
    """Tests for BinaryScoringStrategy.score()."""

    @pytest.fixture
    def strategy(self):
        return BinaryScoringStrategy()

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        data = _make_binary_response()
        client.messages.create.return_value = _mock_claude_response(data)
        return client

    async def test_version(self, strategy):
        assert strategy.version == "v3-binary"

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_score_returns_info_score(self, mock_log, strategy, mock_client):
        result = await strategy.score(
            title="Test Article",
            author="Test Author",
            content="Some content here.",
            word_count=1000,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="test-1",
        )
        assert result is not None
        assert result.total == 100  # All positive yes, all negative no
        assert result.specificity == 25
        assert result.novelty == 25
        assert result.depth == 25
        assert result.actionability == 25
        assert result.raw_responses is not None
        assert result.content_fetch_failed is False

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_gatekeeper_q17_false_returns_content_fetch_failed(
        self, mock_log, strategy, mock_client
    ):
        data = _make_binary_response({"q17": False, "q17_reason": "Content is truncated"})
        mock_client.messages.create.return_value = _mock_claude_response(data)

        result = await strategy.score(
            title="Truncated Article",
            author="Author",
            content="Short stub.",
            word_count=5000,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="gate-1",
        )
        assert result is not None
        assert result.content_fetch_failed is True
        assert result.total == 0
        assert result.raw_responses is not None

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_empty_content_returns_none(self, mock_log, strategy, mock_client):
        result = await strategy.score(
            title="Empty",
            author="Author",
            content="",
            word_count=0,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="empty-1",
        )
        assert result is None

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_api_error_returns_none(self, mock_log, strategy, mock_client):
        mock_client.messages.create.side_effect = Exception("API Error")
        result = await strategy.score(
            title="Error Article",
            author="Author",
            content="Some content.",
            word_count=500,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="err-1",
        )
        assert result is None

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_podcast_uses_podcast_prompt(self, mock_log, strategy, mock_client):
        await strategy.score(
            title="Test Podcast",
            author="Host Name",
            content="Podcast transcript here.",
            word_count=10000,
            content_type_hint="podcast",
            anthropic_client=mock_client,
            entity_id="pod-1",
        )
        # Verify it was called and used the podcast service name for logging
        mock_log.assert_called_once()
        assert mock_log.call_args.kwargs["service"] == "podcast_scorer_v3"

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_dimension_reasons_populated(self, mock_log, strategy, mock_client):
        result = await strategy.score(
            title="Test",
            author="Author",
            content="Content.",
            word_count=500,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="reasons-1",
        )
        assert result is not None
        # Reasons should be populated from positive-yes questions
        assert result.specificity_reason != ""
        assert result.novelty_reason != ""
        assert result.depth_reason != ""
        assert result.actionability_reason != ""

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_low_score_with_penalties(self, mock_log, strategy, mock_client):
        # All positive no, all penalties yes
        data = _make_binary_response()
        for i in range(1, 21):
            data[f"q{i}"] = False
        for q in ["q8", "q12", "q16", "q20"]:
            data[q] = True
        data["q17"] = True  # Keep gatekeeper passing
        mock_client.messages.create.return_value = _mock_claude_response(data)

        result = await strategy.score(
            title="Bad Article",
            author="Author",
            content="Low quality content.",
            word_count=500,
            content_type_hint="article",
            anthropic_client=mock_client,
            entity_id="low-1",
        )
        assert result is not None
        assert result.total == 0
