"""Scoring strategy protocol and implementations.

Defines the ScoringStrategy interface, the default CategoricalScoringStrategy
(v2-categorical), and the BinaryScoringStrategy (v3-binary).
"""

import json
import logging
import re
from typing import Protocol

from anthropic import Anthropic
from anthropic.types import TextBlock

from app.services.scorer import (
    _ARTICLE_SCORING_PROMPT,
    _MAX_CONTENT_LENGTH,
    _PODCAST_SCORING_PROMPT,
    APPLICABLE_SCORES,
    AUTHOR_CONVICTION_POINTS,
    COMPLETENESS_SCORES,
    CONTENT_TYPE_SCORES,
    NAMED_FRAMEWORK_POINTS,
    NOVEL_FRAMING_POINTS,
    PODCAST_COMPLETENESS_SCORES,
    PODCAST_CONTENT_TYPE_SCORES,
    PRACTITIONER_VOICE_POINTS,
    STANDALONE_SCORES,
    InfoScore,
)
from app.services.usage import log_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v3-binary scoring: weights, computation, and prompt templates
# ---------------------------------------------------------------------------

# Weights for each binary question (q17 is gatekeeper, not scored)
BINARY_WEIGHTS: dict[str, int] = {
    "q1": 5,
    "q2": 5,
    "q3": 8,
    "q4": 4,  # Quotability
    "q5": 8,
    "q6": 6,
    "q7": 6,
    "q8": -6,  # Surprise
    "q9": 6,
    "q10": 6,
    "q11": 8,
    "q12": -6,  # Argument
    "q13": 8,
    "q14": 6,
    "q15": 5,
    "q16": -4,  # Insight
    # q17 is gatekeeper, not in weights
    "q18": 6,
    "q19": 8,
    "q20": -8,  # Cross-cutting
}

_BINARY_MAX_POSITIVE = sum(w for w in BINARY_WEIGHTS.values() if w > 0)  # 95
_BINARY_MIN_RAW = sum(w for w in BINARY_WEIGHTS.values() if w < 0)  # -24

BINARY_DIMENSION_QUESTIONS: dict[str, list[str]] = {
    "quotability": ["q1", "q2", "q3", "q4"],
    "surprise": ["q5", "q6", "q7", "q8"],
    "argument": ["q9", "q10", "q11", "q12"],
    "insight": ["q13", "q14", "q15", "q16"],
}


def compute_binary_total(responses: dict[str, bool]) -> int:
    """Compute total score (0-100) from binary question responses.

    Raw score range is [-24, 95], linearly scaled to [0, 100].
    """
    raw = sum(BINARY_WEIGHTS[q] * (1 if responses.get(q, False) else 0) for q in BINARY_WEIGHTS)
    return max(
        0, min(100, int(100 * (raw - _BINARY_MIN_RAW) / (_BINARY_MAX_POSITIVE - _BINARY_MIN_RAW)))
    )


def compute_binary_dimension(dim: str, responses: dict[str, bool]) -> int:
    """Compute a single dimension sub-score (0-25) from binary responses."""
    qs = BINARY_DIMENSION_QUESTIONS[dim]
    dim_weights = {q: BINARY_WEIGHTS[q] for q in qs}
    dim_max = sum(w for w in dim_weights.values() if w > 0)
    dim_min = sum(w for w in dim_weights.values() if w < 0)
    dim_raw = sum(dim_weights[q] * (1 if responses.get(q, False) else 0) for q in qs)
    return max(0, min(25, int(25 * (dim_raw - dim_min) / (dim_max - dim_min))))


# --- Binary prompt templates ---

_BINARY_ARTICLE_PROMPT = """Evaluate this article for capture value — how likely a reader is to want to save and highlight passages.

For each question, answer ONLY "yes" or "no". Be critical and honest — most articles should NOT pass the harder questions. A "yes" means the content clearly and unambiguously meets the criterion.

QUOTABILITY — Would a reader want to highlight passages?
Q1: Does it contain a memorable phrase, vivid metaphor, or striking sentence that encapsulates a key idea?
Q2: Does it include a specific data point, statistic, or quantified claim worth remembering?
Q3: Could you extract at least one passage of 2-3 sentences that would work as a standalone saved note worth revisiting?
Q4: Does it contain a direct quote from a practitioner or expert that adds credibility or color?

SURPRISE — Does it challenge or expand the reader's understanding?
Q5: Does it present a perspective or conclusion that contradicts common assumptions or conventional wisdom in its domain?
Q6: Does it reframe a familiar topic through an unexpected lens, analogy, or cross-domain connection?
Q7: Does it contain a finding, case study, or example that an informed reader in this domain likely hasn't encountered before?
Q8: Is this primarily a summary, restatement, or roundup of widely-known ideas rather than original analysis?

ARGUMENT — Is the reasoning strong and grounded?
Q9: Does the author argue for a clear, specific position rather than presenting multiple viewpoints neutrally?
Q10: Does the author support their central claims with concrete evidence such as data, case studies, or specific real-world examples?
Q11: Does the author write from first-person professional experience, sharing hard-won lessons or opinions from practice?
Q12: Could this article's central argument or message be adequately captured in a single tweet-length summary?

INSIGHT — Can the reader use what they've read?
Q13: Does it introduce, name, or build upon a specific framework, mental model, or structured methodology?
Q14: Could a reader apply a specific technique or idea from this in their own work within the next month?
Q15: Does it provide enough context and concrete detail for a reader to act on its key recommendations?
Q16: Is the content so narrowly domain-specific that it would only be useful to readers in a single niche?

OVERALL QUALITY
Q17: Is the available text complete and substantial enough to properly evaluate? (Not truncated, paywalled, or a stub.)
Q18: Does this offer substantive value beyond what a reader could easily find in the first page of search results on this topic?
Q19: Would a knowledgeable reader in this domain likely encounter at least one idea they haven't seen before?
Q20: Is this primarily a news article, product announcement, press release, or event recap?

{content_warning}Title: {title}
Author: {author}
Word Count: {word_count}

Content:
{content}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"q1": <true/false>, "q1_reason": "<brief reason>", "q2": <true/false>, "q2_reason": "<brief reason>", "q3": <true/false>, "q3_reason": "<brief reason>", "q4": <true/false>, "q4_reason": "<brief reason>", "q5": <true/false>, "q5_reason": "<brief reason>", "q6": <true/false>, "q6_reason": "<brief reason>", "q7": <true/false>, "q7_reason": "<brief reason>", "q8": <true/false>, "q8_reason": "<brief reason>", "q9": <true/false>, "q9_reason": "<brief reason>", "q10": <true/false>, "q10_reason": "<brief reason>", "q11": <true/false>, "q11_reason": "<brief reason>", "q12": <true/false>, "q12_reason": "<brief reason>", "q13": <true/false>, "q13_reason": "<brief reason>", "q14": <true/false>, "q14_reason": "<brief reason>", "q15": <true/false>, "q15_reason": "<brief reason>", "q16": <true/false>, "q16_reason": "<brief reason>", "q17": <true/false>, "q17_reason": "<brief reason>", "q18": <true/false>, "q18_reason": "<brief reason>", "q19": <true/false>, "q19_reason": "<brief reason>", "q20": <true/false>, "q20_reason": "<brief reason>", "overall_assessment": "<1-2 sentence summary>"}}"""

_BINARY_PODCAST_PROMPT = """Evaluate this podcast episode transcript for capture value — how likely a listener is to want to save and highlight passages.

For each question, answer ONLY "yes" or "no". Be critical and honest — most episodes should NOT pass the harder questions. A "yes" means the content clearly and unambiguously meets the criterion.

QUOTABILITY — Would a listener want to highlight passages?
Q1: Does the transcript contain a memorable statement, analogy, or phrasing from a host or guest?
Q2: Does it include a specific data point, statistic, or quantified claim worth remembering?
Q3: Could you extract at least one passage of 2-3 sentences that would work as a standalone saved note worth revisiting?
Q4: Does a guest share a specific professional insight or hard-won lesson from their experience?

SURPRISE — Does it challenge or expand the listener's understanding?
Q5: Does it present a perspective or conclusion that contradicts common assumptions or conventional wisdom in its domain?
Q6: Does it reframe a familiar topic through an unexpected lens, analogy, or cross-domain connection?
Q7: Does it contain a finding, case study, or example that an informed listener in this domain likely hasn't encountered before?
Q8: Is this primarily a summary, restatement, or roundup of widely-known ideas rather than original analysis?

ARGUMENT — Is the reasoning strong and grounded?
Q9: Does a speaker argue for a clear, specific position rather than presenting multiple viewpoints neutrally?
Q10: Does a speaker support their central claims with concrete evidence such as data, case studies, or specific real-world examples?
Q11: Does a host or guest share in-depth first-person professional experience?
Q12: Could this episode's central topic be adequately covered in a blog post introduction?

INSIGHT — Can the listener use what they've heard?
Q13: Does it introduce, name, or build upon a specific framework, mental model, or structured methodology?
Q14: Could a listener apply a specific technique or idea from this in their own work within the next month?
Q15: Does it provide enough context and concrete detail for a listener to act on its key recommendations?
Q16: Is the content so narrowly domain-specific that it would only be useful to listeners in a single niche?

OVERALL QUALITY
Q17: Is the transcript quality sufficient to evaluate the episode's content? (Not garbled, heavily truncated, or mostly filler.)
Q18: Does this offer substantive value beyond what a listener could easily find in the first page of search results on this topic?
Q19: Would a knowledgeable listener in this domain likely encounter at least one idea they haven't seen before?
Q20: Is this primarily a news commentary episode, product launch discussion, or event recap?

{content_warning}Title: {title}
Host/Show: {author}
Word Count: {word_count}

Transcript:
{content}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"q1": <true/false>, "q1_reason": "<brief reason>", "q2": <true/false>, "q2_reason": "<brief reason>", "q3": <true/false>, "q3_reason": "<brief reason>", "q4": <true/false>, "q4_reason": "<brief reason>", "q5": <true/false>, "q5_reason": "<brief reason>", "q6": <true/false>, "q6_reason": "<brief reason>", "q7": <true/false>, "q7_reason": "<brief reason>", "q8": <true/false>, "q8_reason": "<brief reason>", "q9": <true/false>, "q9_reason": "<brief reason>", "q10": <true/false>, "q10_reason": "<brief reason>", "q11": <true/false>, "q11_reason": "<brief reason>", "q12": <true/false>, "q12_reason": "<brief reason>", "q13": <true/false>, "q13_reason": "<brief reason>", "q14": <true/false>, "q14_reason": "<brief reason>", "q15": <true/false>, "q15_reason": "<brief reason>", "q16": <true/false>, "q16_reason": "<brief reason>", "q17": <true/false>, "q17_reason": "<brief reason>", "q18": <true/false>, "q18_reason": "<brief reason>", "q19": <true/false>, "q19_reason": "<brief reason>", "q20": <true/false>, "q20_reason": "<brief reason>", "overall_assessment": "<1-2 sentence summary>"}}"""


class ScoringStrategy(Protocol):
    """Interface for content scoring strategies."""

    @property
    def version(self) -> str:
        """Scoring version identifier (e.g., 'v2-categorical', 'v3-binary-weighted')."""
        ...

    async def score(
        self,
        *,
        title: str,
        author: str | None,
        content: str,
        word_count: int | None,
        content_type_hint: str,
        anthropic_client: Anthropic,
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        """Score content and return an InfoScore result.

        Args:
            title: Title of the content.
            author: Author or host name (may be None).
            content: The text content to score (already extracted by caller).
            word_count: Reported word count (may be None).
            content_type_hint: "article" or "podcast" -- selects prompt variant and mappings.
            anthropic_client: Anthropic client instance.
            entity_id: ID string for usage logging (article ID or episode ID).
            content_warning: Optional warning text prepended to the article metadata in the
                prompt (used for truncated article content). Empty string by default.

        Returns:
            InfoScore or None if scoring failed.
        """
        ...


class CategoricalScoringStrategy:
    """v2-categorical scoring strategy.

    Uses 8 categorical questions mapped to point values via lookup tables.
    This is the original scoring approach extracted from score_content().
    """

    @property
    def version(self) -> str:
        return "v2-categorical"

    async def score(
        self,
        *,
        title: str,
        author: str | None,
        content: str,
        word_count: int | None,
        content_type_hint: str,
        anthropic_client: Anthropic,
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        """Score content using the v2-categorical rubric.

        Selects prompt variant based on content_type_hint, truncates content,
        calls Claude, parses JSON, and maps categorical answers to numeric scores.
        """
        if not content:
            return None

        # Select prompt template and score mappings based on content type
        if content_type_hint == "podcast":
            scoring_prompt = _PODCAST_SCORING_PROMPT
            content_type_scores = PODCAST_CONTENT_TYPE_SCORES
            completeness_scores = PODCAST_COMPLETENESS_SCORES
        else:
            scoring_prompt = _ARTICLE_SCORING_PROMPT
            content_type_scores = CONTENT_TYPE_SCORES
            completeness_scores = COMPLETENESS_SCORES

        # Truncate content if too long
        max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"

        prompt = scoring_prompt.format(
            title=title,
            author=author or "Unknown",
            word_count=word_count or "Unknown",
            content=content,
            content_warning=content_warning,
        )

        try:
            model = "claude-sonnet-4-20250514"
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )

            # Log usage
            service = "podcast_scorer" if content_type_hint == "podcast" else "scorer"
            await log_usage(
                service=service,
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                article_id=entity_id,
            )

            first_block = response.content[0]
            assert isinstance(first_block, TextBlock)
            text = first_block.text.strip()
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\n?", "", text)
                text = text.strip()

            data = json.loads(text)

            # Map categorical responses to numeric scores
            quotability = STANDALONE_SCORES.get(data.get("standalone_passages", "none"), 0)
            surprise = (NOVEL_FRAMING_POINTS if data.get("novel_framing") else 0) + (
                content_type_scores.get(data.get("content_type", ""), 0)
            )
            argument = (
                (AUTHOR_CONVICTION_POINTS if data.get("author_conviction") else 0)
                + (PRACTITIONER_VOICE_POINTS if data.get("practitioner_voice") else 0)
                + completeness_scores.get(data.get("content_completeness", ""), 0)
            )
            insight = (NAMED_FRAMEWORK_POINTS if data.get("named_framework") else 0) + (
                APPLICABLE_SCORES.get(data.get("applicable_ideas", ""), 0)
            )

            return InfoScore(
                specificity=min(25, max(0, quotability)),
                specificity_reason=data.get("quotability_reason", ""),
                novelty=min(25, max(0, surprise)),
                novelty_reason=data.get("surprise_reason", ""),
                depth=min(25, max(0, argument)),
                depth_reason=data.get("argument_reason", ""),
                actionability=min(25, max(0, insight)),
                actionability_reason=data.get("insight_reason", ""),
                overall_assessment=data.get("overall_assessment", ""),
            )
        except Exception as e:
            logger.error("Error scoring content %s: %s", entity_id, e)
            return None


class BinaryScoringStrategy:
    """v3-binary scoring strategy.

    Uses 20 weighted binary (yes/no) questions with penalty questions
    for better score discrimination. See docs/research/binary-scoring/.
    """

    @property
    def version(self) -> str:
        return "v3-binary"

    async def score(
        self,
        *,
        title: str,
        author: str | None,
        content: str,
        word_count: int | None,
        content_type_hint: str,
        anthropic_client: Anthropic,
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        """Score content using the v3-binary rubric."""
        if not content:
            return None

        if content_type_hint == "podcast":
            scoring_prompt = _BINARY_PODCAST_PROMPT
        else:
            scoring_prompt = _BINARY_ARTICLE_PROMPT

        # Truncate content if too long
        max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"

        prompt = scoring_prompt.format(
            title=title,
            author=author or "Unknown",
            word_count=word_count or "Unknown",
            content=content,
            content_warning=content_warning,
        )

        try:
            model = "claude-sonnet-4-20250514"
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )

            service = "scorer_v3" if content_type_hint != "podcast" else "podcast_scorer_v3"
            await log_usage(
                service=service,
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                article_id=entity_id,
            )

            first_block = response.content[0]
            assert isinstance(first_block, TextBlock)
            text = first_block.text.strip()
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\n?", "", text)
                text = text.strip()

            data: dict[str, object] = json.loads(text)

            # Extract boolean responses for q1-q20
            responses: dict[str, bool] = {}
            for i in range(1, 21):
                key = f"q{i}"
                responses[key] = bool(data.get(key, False))

            # Gatekeeper: q17=false means content is incomplete
            if not responses.get("q17", False):
                return InfoScore(
                    specificity=0,
                    specificity_reason="Content incomplete",
                    novelty=0,
                    novelty_reason="Content incomplete",
                    depth=0,
                    depth_reason="Content incomplete",
                    actionability=0,
                    actionability_reason="Content incomplete",
                    overall_assessment=str(
                        data.get("overall_assessment", "Content incomplete — cannot evaluate.")
                    ),
                    content_fetch_failed=True,
                    raw_responses=data,
                )

            # Compute scores
            total = compute_binary_total(responses)
            quotability = compute_binary_dimension("quotability", responses)
            surprise = compute_binary_dimension("surprise", responses)
            argument = compute_binary_dimension("argument", responses)
            insight = compute_binary_dimension("insight", responses)

            # Extract per-dimension reasons (combine individual question reasons)
            def _dim_reasons(qs: list[str]) -> str:
                parts = []
                for q in qs:
                    reason = data.get(f"{q}_reason", "")
                    if reason and responses.get(q, False):
                        parts.append(str(reason))
                return "; ".join(parts) if parts else "No positive signals"

            return InfoScore(
                specificity=quotability,
                specificity_reason=_dim_reasons(["q1", "q2", "q3", "q4"]),
                novelty=surprise,
                novelty_reason=_dim_reasons(["q5", "q6", "q7"]),
                depth=argument,
                depth_reason=_dim_reasons(["q9", "q10", "q11"]),
                actionability=insight,
                actionability_reason=_dim_reasons(["q13", "q14", "q15"]),
                overall_assessment=str(data.get("overall_assessment", "")),
                raw_responses=data,
                total_override=total,
            )
        except Exception as e:
            logger.error("Error in v3-binary scoring for %s: %s", entity_id, e)
            return None
