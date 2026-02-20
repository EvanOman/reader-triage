"""Tests for the article scorer service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import TextBlock

from app.models.article import Author
from app.services.scorer import (
    _BAD_ASSESSMENT_PATTERNS,
    APPLICABLE_SCORES,
    AUTHOR_CONVICTION_POINTS,
    COMPLETENESS_SCORES,
    CONTENT_TYPE_SCORES,
    NAMED_FRAMEWORK_POINTS,
    NOVEL_FRAMING_POINTS,
    PRACTITIONER_VOICE_POINTS,
    STANDALONE_SCORES,
    ArticleScorer,
    InfoScore,
    _assessment_indicates_bad_content,
    normalize_author_name,
)
from tests.factories import make_claude_response, make_document, mock_anthropic_response


def _make_scorer_instance() -> ArticleScorer:
    """Build a scorer instance with mocked dependencies (no Claude response configured)."""
    with (
        patch("app.services.scorer.Anthropic"),
        patch("app.services.scorer.get_readwise_service"),
    ):
        return ArticleScorer()


# ---------------------------------------------------------------------------
# 1. Score calculation from categorical responses
# ---------------------------------------------------------------------------


class TestScoreCalculation:
    """Test point mappings from categorical responses to numeric scores."""

    async def test_standalone_scores_mapping(self):
        assert STANDALONE_SCORES == {"none": 0, "a_few": 9, "several": 17, "many": 25}

    async def test_novel_framing_points(self):
        assert NOVEL_FRAMING_POINTS == 15

    async def test_content_type_scores_mapping(self):
        assert CONTENT_TYPE_SCORES == {
            "original_analysis": 10,
            "opinion_with_evidence": 8,
            "informational_summary": 3,
            "product_review": 2,
            "news_or_roundup": 0,
        }

    async def test_author_conviction_points(self):
        assert AUTHOR_CONVICTION_POINTS == 12

    async def test_practitioner_voice_points(self):
        assert PRACTITIONER_VOICE_POINTS == 8

    async def test_completeness_scores_mapping(self):
        assert COMPLETENESS_SCORES == {
            "complete": 5,
            "appears_truncated": 2,
            "summary_or_excerpt": 0,
        }

    async def test_named_framework_points(self):
        assert NAMED_FRAMEWORK_POINTS == 12

    async def test_applicable_scores_mapping(self):
        assert APPLICABLE_SCORES == {"broadly": 13, "narrowly": 7, "not_really": 0}

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_max_score_all_categories(self, mock_log_usage):
        """All-max categorical answers produce (25+25+25+25) = 100."""
        data = make_claude_response(
            {
                "standalone_passages": "many",  # 25
                "novel_framing": True,  # 15
                "content_type": "original_analysis",  # 10  -> 25
                "author_conviction": True,  # 12
                "practitioner_voice": True,  # 8
                "content_completeness": "complete",  # 5  -> 25
                "named_framework": True,  # 12
                "applicable_ideas": "broadly",  # 13 -> 25
            }
        )
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        assert result.specificity == 25
        assert result.novelty == 25
        assert result.depth == 25
        assert result.actionability == 25
        assert result.total == 100

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_min_score_all_categories(self, mock_log_usage):
        """All-min categorical answers produce 0."""
        data = make_claude_response(
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
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        assert result.specificity == 0
        assert result.novelty == 0
        assert result.depth == 0
        assert result.actionability == 0
        assert result.total == 0

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_mixed_score_calculation(self, mock_log_usage):
        """Mixed categoricals produce expected intermediate scores."""
        data = make_claude_response(
            {
                "standalone_passages": "a_few",  # 9
                "novel_framing": False,  # 0
                "content_type": "opinion_with_evidence",  # 8 -> 8
                "author_conviction": True,  # 12
                "practitioner_voice": False,  # 0
                "content_completeness": "appears_truncated",  # 2 -> 14
                "named_framework": False,  # 0
                "applicable_ideas": "narrowly",  # 7 -> 7
            }
        )
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        assert result.specificity == 9
        assert result.novelty == 8
        assert result.depth == 14
        assert result.actionability == 7
        assert result.total == 38


# ---------------------------------------------------------------------------
# 2. InfoScore dataclass
# ---------------------------------------------------------------------------


class TestInfoScore:
    """Test InfoScore total property and score clamping."""

    async def test_total_sums_all_dimensions(self):
        score = InfoScore(
            specificity=10,
            specificity_reason="reason1",
            novelty=20,
            novelty_reason="reason2",
            depth=15,
            depth_reason="reason3",
            actionability=5,
            actionability_reason="reason4",
            overall_assessment="assessment",
        )
        assert score.total == 50

    async def test_total_zero_when_all_zero(self):
        score = InfoScore(
            specificity=0,
            specificity_reason="",
            novelty=0,
            novelty_reason="",
            depth=0,
            depth_reason="",
            actionability=0,
            actionability_reason="",
            overall_assessment="",
        )
        assert score.total == 0

    async def test_total_max_when_all_max(self):
        score = InfoScore(
            specificity=25,
            specificity_reason="",
            novelty=25,
            novelty_reason="",
            depth=25,
            depth_reason="",
            actionability=25,
            actionability_reason="",
            overall_assessment="",
        )
        assert score.total == 100

    async def test_content_fetch_failed_default_false(self):
        score = InfoScore(
            specificity=0,
            specificity_reason="",
            novelty=0,
            novelty_reason="",
            depth=0,
            depth_reason="",
            actionability=0,
            actionability_reason="",
            overall_assessment="",
        )
        assert score.content_fetch_failed is False

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_score_clamped_to_0_25(self, mock_log_usage):
        """Scores from _score_document are clamped to [0, 25] per dimension."""
        # novel_framing(15) + original_analysis(10) = 25 -- at the max boundary
        data = make_claude_response(
            {
                "standalone_passages": "many",  # 25 -- clamped to 25
                "novel_framing": True,  # 15
                "content_type": "original_analysis",  # 10 -> 25 (exact boundary)
                "author_conviction": True,  # 12
                "practitioner_voice": True,  # 8
                "content_completeness": "complete",  # 5 -> 25 (exact boundary)
                "named_framework": True,  # 12
                "applicable_ideas": "broadly",  # 13 -> 25 (exact boundary)
            }
        )
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        # All dimensions clamped to max 25
        assert result.specificity <= 25
        assert result.novelty <= 25
        assert result.depth <= 25
        assert result.actionability <= 25
        # All dimensions non-negative
        assert result.specificity >= 0
        assert result.novelty >= 0
        assert result.depth >= 0
        assert result.actionability >= 0


# ---------------------------------------------------------------------------
# 3. Content stub detection
# ---------------------------------------------------------------------------


class TestContentIsStub:
    """Test _content_is_stub() method."""

    async def test_stub_when_actual_words_far_below_reported(self):
        """Content with < 15% of reported word count is a stub (for articles > 500 words)."""
        scorer = _make_scorer_instance()
        # 1000 reported words, but content has only ~10 words -> 1% -> stub
        doc = make_document(word_count=1000, content="This is a very short stub of content.")
        assert scorer._content_is_stub(doc) is True

    async def test_not_stub_for_short_articles(self):
        """Articles with word_count <= 500 are never stubs (they are short by nature)."""
        scorer = _make_scorer_instance()
        doc = make_document(word_count=400, content="Short article.")
        assert scorer._content_is_stub(doc) is False

    async def test_not_stub_when_content_proportional(self):
        """Content with >= 15% of reported word count is not a stub."""
        scorer = _make_scorer_instance()
        # 1000 reported words, content has ~200 words -> 20% -> not a stub
        words = " ".join(["word"] * 200)
        doc = make_document(word_count=1000, content=words)
        assert scorer._content_is_stub(doc) is False

    async def test_not_stub_when_no_word_count(self):
        """No reported word_count means we cannot determine stub."""
        scorer = _make_scorer_instance()
        doc = make_document(word_count=None, content="Some content here.")
        assert scorer._content_is_stub(doc) is False

    async def test_not_stub_when_no_content(self):
        """Empty/None content still evaluates (0 words vs reported)."""
        scorer = _make_scorer_instance()
        # word_count=1000, content="" -> 0 actual words -> 0% -> stub
        doc = make_document(word_count=1000, content="")
        assert scorer._content_is_stub(doc) is True

    async def test_stub_boundary_exactly_15_percent(self):
        """Exactly 15% is NOT a stub (only strictly below triggers it)."""
        scorer = _make_scorer_instance()
        # word_count=1000, exactly 150 words -> 15% -> not a stub (>= not <)
        words = " ".join(["word"] * 150)
        doc = make_document(word_count=1000, content=words)
        assert scorer._content_is_stub(doc) is False

    async def test_stub_just_under_15_percent(self):
        """Just under 15% IS a stub."""
        scorer = _make_scorer_instance()
        # word_count=1000, 149 words -> 14.9% -> stub
        words = " ".join(["word"] * 149)
        doc = make_document(word_count=1000, content=words)
        assert scorer._content_is_stub(doc) is True

    async def test_word_count_at_boundary_500(self):
        """word_count=500 does not trigger stub detection (must be > 500)."""
        scorer = _make_scorer_instance()
        doc = make_document(word_count=500, content="short")
        assert scorer._content_is_stub(doc) is False

    async def test_word_count_at_501(self):
        """word_count=501 does trigger stub detection when content is short."""
        scorer = _make_scorer_instance()
        doc = make_document(word_count=501, content="short")
        assert scorer._content_is_stub(doc) is True


# ---------------------------------------------------------------------------
# 4. Assessment bad content detection
# ---------------------------------------------------------------------------


class TestAssessmentBadContentDetection:
    """Test _assessment_indicates_bad_content() against each known pattern."""

    @pytest.mark.parametrize("pattern", _BAD_ASSESSMENT_PATTERNS)
    async def test_each_bad_pattern_detected(self, pattern):
        """Each pattern in _BAD_ASSESSMENT_PATTERNS should trigger True."""
        assessment = f"The article's {pattern} made it hard to evaluate."
        assert _assessment_indicates_bad_content(assessment) is True

    @pytest.mark.parametrize("pattern", _BAD_ASSESSMENT_PATTERNS)
    async def test_patterns_case_insensitive(self, pattern):
        """Patterns should match regardless of casing."""
        assessment = f"The {pattern.upper()} was an issue."
        assert _assessment_indicates_bad_content(assessment) is True

    async def test_normal_assessment_not_flagged(self):
        """A normal positive assessment does not trigger."""
        assessment = "High value article with novel insights and practical frameworks."
        assert _assessment_indicates_bad_content(assessment) is False

    async def test_empty_assessment_not_flagged(self):
        assert _assessment_indicates_bad_content("") is False

    async def test_assessment_about_article_quality_not_flagged(self):
        """Words like 'incomplete argument' should NOT trigger (they describe article quality, not fetch failure)."""
        assessment = "The article presents an incomplete argument about software design."
        assert _assessment_indicates_bad_content(assessment) is False

    async def test_all_patterns_present(self):
        """Verify the expected patterns are all present."""
        expected = [
            "truncated content",
            "truncated article",
            "severely limited by",
            "prevents full eval",
            "cannot assess",
            "cannot properly",
            "only a summary was returned",
            "only a trivial fragment",
            "content not available",
        ]
        assert _BAD_ASSESSMENT_PATTERNS == expected


# ---------------------------------------------------------------------------
# 5. Author boost calculation (requires DB)
# ---------------------------------------------------------------------------


class TestAuthorBoost:
    """Test _get_author_boost() threshold logic."""

    async def _insert_author(self, session, name: str, highlights: int) -> Author:
        author = Author(
            name=name,
            normalized_name=normalize_author_name(name),
            total_highlights=highlights,
        )
        session.add(author)
        await session.commit()
        return author

    async def test_50_plus_highlights_gives_15(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Prolific Author", 50)
        boost = await scorer._get_author_boost(session, "Prolific Author")
        assert boost == 15.0

    async def test_100_highlights_gives_15(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Super Author", 100)
        boost = await scorer._get_author_boost(session, "Super Author")
        assert boost == 15.0

    async def test_20_plus_highlights_gives_10(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Good Author", 20)
        boost = await scorer._get_author_boost(session, "Good Author")
        assert boost == 10.0

    async def test_10_plus_highlights_gives_7(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Decent Author", 10)
        boost = await scorer._get_author_boost(session, "Decent Author")
        assert boost == 7.0

    async def test_5_plus_highlights_gives_5(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Rising Author", 5)
        boost = await scorer._get_author_boost(session, "Rising Author")
        assert boost == 5.0

    async def test_2_plus_highlights_gives_3(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "New Author", 2)
        boost = await scorer._get_author_boost(session, "New Author")
        assert boost == 3.0

    async def test_1_highlight_gives_0(self, session):
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Rare Author", 1)
        boost = await scorer._get_author_boost(session, "Rare Author")
        assert boost == 0.0

    async def test_no_author_gives_0(self, session):
        scorer = _make_scorer_instance()
        boost = await scorer._get_author_boost(session, None)
        assert boost == 0.0

    async def test_unknown_author_gives_0(self, session):
        """Author name not in the DB returns 0."""
        scorer = _make_scorer_instance()
        boost = await scorer._get_author_boost(session, "Unknown Person")
        assert boost == 0.0

    async def test_author_name_case_insensitive(self, session):
        """Author lookup uses normalized (lowercase) name."""
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Jane Smith", 25)
        boost = await scorer._get_author_boost(session, "  Jane Smith  ")
        assert boost == 10.0

    async def test_threshold_boundary_49_highlights(self, session):
        """49 highlights should get +10, not +15."""
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Almost Prolific", 49)
        boost = await scorer._get_author_boost(session, "Almost Prolific")
        assert boost == 10.0

    async def test_threshold_boundary_19_highlights(self, session):
        """19 highlights should get +7, not +10."""
        scorer = _make_scorer_instance()
        await self._insert_author(session, "Almost Good", 19)
        boost = await scorer._get_author_boost(session, "Almost Good")
        assert boost == 7.0


# ---------------------------------------------------------------------------
# 6. _score_document integration with mocked Claude API
# ---------------------------------------------------------------------------


def _build_scorer(claude_response_data: dict) -> ArticleScorer:
    """Build an ArticleScorer with mocked Anthropic client."""
    with (
        patch("app.services.scorer.Anthropic") as mock_anthropic_cls,
        patch("app.services.scorer.get_readwise_service"),
    ):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_anthropic_response(claude_response_data)
        mock_anthropic_cls.return_value = mock_client
        scorer = ArticleScorer()
    return scorer


class TestScoreDocumentIntegration:
    """Test _score_document with mocked Claude responses."""

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_returns_none_for_empty_content(self, mock_log_usage):
        """Documents with no content and no summary return None."""
        scorer = _build_scorer({})
        doc = make_document(content=None, summary=None)
        result = await scorer._score_document(doc)
        assert result is None

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_returns_none_for_blank_content(self, mock_log_usage):
        """Documents with blank content and no summary return None."""
        scorer = _build_scorer({})
        doc = make_document(content="", summary=None)
        result = await scorer._score_document(doc)
        assert result is None

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_falls_back_to_summary_when_no_content(self, mock_log_usage):
        """When content is empty but summary exists, summary is used for scoring.

        Use word_count <= 500 to avoid triggering stub detection, which would
        short-circuit before the Claude API call.
        """
        data = make_claude_response({"standalone_passages": "a_few"})
        scorer = _build_scorer(data)
        doc = make_document(
            content=None, summary="This is a summary of the article.", word_count=400
        )
        result = await scorer._score_document(doc)
        assert result is not None
        assert result.specificity == 9  # a_few -> 9

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_stub_without_content_returns_fetch_failed(self, mock_log_usage):
        """Stub document with only summary returns content_fetch_failed InfoScore."""
        scorer = _build_scorer({})
        # word_count=2000 but only a short summary, no actual content
        doc = make_document(word_count=2000, content=None, summary="Brief summary.")
        result = await scorer._score_document(doc)
        assert result is not None
        assert result.content_fetch_failed is True
        assert result.total == 0
        assert "Content not available" in result.overall_assessment

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_stub_with_content_marks_content_failed(self, mock_log_usage):
        """Stub that has content (but way too short) marks content_fetch_failed."""
        data = make_claude_response({"standalone_passages": "none"})
        scorer = _build_scorer(data)
        # word_count=2000 but content only has ~10 words
        doc = make_document(
            word_count=2000, content="This is only ten words of content in total here."
        )
        result = await scorer._score_document(doc)
        assert result is not None
        assert result.content_fetch_failed is True

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_reasons_passed_through(self, mock_log_usage):
        """Reason strings from Claude response are passed through to InfoScore."""
        data = make_claude_response(
            {
                "quotability_reason": "Great quotes",
                "surprise_reason": "Very novel",
                "argument_reason": "Strong argument",
                "insight_reason": "Broadly useful",
                "overall_assessment": "Excellent article overall.",
            }
        )
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        assert result.specificity_reason == "Great quotes"
        assert result.novelty_reason == "Very novel"
        assert result.depth_reason == "Strong argument"
        assert result.actionability_reason == "Broadly useful"
        assert result.overall_assessment == "Excellent article overall."

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_handles_unknown_categorical_values_gracefully(self, mock_log_usage):
        """Unknown enum values should map to 0 via .get() defaults."""
        data = make_claude_response(
            {
                "standalone_passages": "unknown_value",
                "content_type": "unknown_type",
                "content_completeness": "unknown_completeness",
                "applicable_ideas": "unknown_applicability",
            }
        )
        scorer = _build_scorer(data)
        doc = make_document()
        result = await scorer._score_document(doc)

        assert result is not None
        assert result.specificity == 0  # unknown maps to 0
        # novelty: novel_framing(True)=15 + unknown_type=0 -> 15
        assert result.novelty == 15

    @patch("app.services.scoring_strategy.log_usage", new_callable=AsyncMock)
    async def test_handles_markdown_wrapped_json(self, mock_log_usage):
        """Claude sometimes wraps JSON in markdown code fences."""
        data = make_claude_response()
        with (
            patch("app.services.scorer.Anthropic") as mock_anthropic_cls,
            patch("app.services.scorer.get_readwise_service"),
        ):
            mock_client = MagicMock()
            content_block = TextBlock(type="text", text=f"```json\n{json.dumps(data)}\n```")
            usage = MagicMock()
            usage.input_tokens = 500
            usage.output_tokens = 100
            response = MagicMock()
            response.content = [content_block]
            response.usage = usage
            mock_client.messages.create.return_value = response
            mock_anthropic_cls.return_value = mock_client
            scorer = ArticleScorer()

        doc = make_document()
        result = await scorer._score_document(doc)
        assert result is not None
        assert result.total > 0


# ---------------------------------------------------------------------------
# Normalize author name
# ---------------------------------------------------------------------------


class TestNormalizeAuthorName:
    async def test_lowercases(self):
        assert normalize_author_name("Jane Smith") == "jane smith"

    async def test_strips_whitespace(self):
        assert normalize_author_name("  Jane Smith  ") == "jane smith"

    async def test_empty_string(self):
        assert normalize_author_name("") == ""
