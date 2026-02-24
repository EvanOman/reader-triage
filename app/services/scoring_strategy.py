"""Scoring strategy protocol and implementations.

Defines the ScoringStrategy interface, the default CategoricalScoringStrategy
(v2-categorical), the BinaryScoringStrategy (v3-binary), and the
TieredBinaryScoringStrategy (v4-binary).
"""

import logging
from typing import Protocol

import dspy

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
from app.services.scoring_models import V2CategoricalOutput, V3BinaryOutput, V4TieredOutput
from app.services.scoring_signatures import (
    V2ArticleScoring,
    V2PodcastScoring,
    V3ArticleScoring,
    V3PodcastScoring,
    V4ArticleScoring,
    V4PodcastScoring,
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


def _make_lm(model_id: str, max_tokens: int, temperature: float = 0.0) -> dspy.LM:
    """Create a dspy.LM, adjusting params for OpenAI reasoning models."""
    try:
        return dspy.LM(model_id, max_tokens=max_tokens, temperature=temperature)
    except ValueError:
        # OpenAI reasoning models require temperature=1.0 and max_tokens >= 16000
        return dspy.LM(model_id, max_tokens=max(max_tokens, 16000), temperature=1.0)


def _strip_json_instruction(prompt: str) -> str:
    """Strip JSON format instruction from prompt -- DSPy handles output formatting."""
    if "Respond with ONLY" in prompt:
        return prompt[: prompt.index("Respond with ONLY")].rstrip()
    return prompt


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
            entity_id: ID string for usage logging (article ID or episode ID).
            content_warning: Optional warning text prepended to the article metadata in the
                prompt (used for truncated article content). Empty string by default.

        Returns:
            InfoScore or None if scoring failed.
        """
        ...


def _extract_usage(lm: dspy.LM) -> tuple[int, int]:
    """Extract input/output token counts from the most recent DSPy LM call."""
    history = lm.history
    if history:
        last_entry = history[-1]
        usage = last_entry.get("response", {}).get("usage", {})
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)
        return input_tokens, output_tokens
    return 0, 0


class CategoricalScoringStrategy:
    """v2-categorical scoring strategy.

    Uses 8 categorical questions mapped to point values via lookup tables.
    This is the original scoring approach extracted from score_content().
    """

    def __init__(self, model_id: str = "anthropic/claude-sonnet-4-5-20250929") -> None:
        self._model_id = model_id
        self._lm = _make_lm(model_id, max_tokens=700)
        self._predict_article = dspy.Predict(V2ArticleScoring)
        self._predict_podcast = dspy.Predict(V2PodcastScoring)

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
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        """Score content using the v2-categorical rubric.

        Selects prompt variant based on content_type_hint, truncates content,
        calls Claude via DSPy, and maps categorical answers to numeric scores.
        """
        if not content:
            return None

        # Select prompt template and score mappings based on content type
        if content_type_hint == "podcast":
            scoring_prompt = _PODCAST_SCORING_PROMPT
            content_type_scores = PODCAST_CONTENT_TYPE_SCORES
            completeness_scores = PODCAST_COMPLETENESS_SCORES
            predict = self._predict_podcast
        else:
            scoring_prompt = _ARTICLE_SCORING_PROMPT
            content_type_scores = CONTENT_TYPE_SCORES
            completeness_scores = COMPLETENESS_SCORES
            predict = self._predict_article

        # Truncate content if too long
        max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [content trimmed for evaluation]"

        prompt = scoring_prompt.format(
            title=title,
            author=author or "Unknown",
            word_count=word_count or "Unknown",
            content=content,
            content_warning=content_warning,
        )

        # Strip JSON format instruction -- DSPy handles output formatting
        prompt = _strip_json_instruction(prompt)

        try:
            with dspy.context(lm=self._lm):
                prediction = predict(evaluation_prompt=prompt)

            output: V2CategoricalOutput = prediction.result

            # Log usage
            input_tokens, output_tokens = _extract_usage(self._lm)
            model = self._model_id.split("/", 1)[-1]
            service = "podcast_scorer" if content_type_hint == "podcast" else "scorer"
            await log_usage(
                service=service,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                article_id=entity_id,
            )

            # Map categorical responses to numeric scores
            quotability = STANDALONE_SCORES.get(output.standalone_passages, 0)
            surprise = (NOVEL_FRAMING_POINTS if output.novel_framing else 0) + (
                content_type_scores.get(output.content_type, 0)
            )
            argument = (
                (AUTHOR_CONVICTION_POINTS if output.author_conviction else 0)
                + (PRACTITIONER_VOICE_POINTS if output.practitioner_voice else 0)
                + completeness_scores.get(output.content_completeness, 0)
            )
            insight = (NAMED_FRAMEWORK_POINTS if output.named_framework else 0) + (
                APPLICABLE_SCORES.get(output.applicable_ideas, 0)
            )

            return InfoScore(
                specificity=min(25, max(0, quotability)),
                specificity_reason=output.quotability_reason,
                novelty=min(25, max(0, surprise)),
                novelty_reason=output.surprise_reason,
                depth=min(25, max(0, argument)),
                depth_reason=output.argument_reason,
                actionability=min(25, max(0, insight)),
                actionability_reason=output.insight_reason,
                overall_assessment=output.overall_assessment,
            )
        except Exception as e:
            logger.error("Error scoring content %s: %s", entity_id, e)
            return None


class BinaryScoringStrategy:
    """v3-binary scoring strategy.

    Uses 20 weighted binary (yes/no) questions with penalty questions
    for better score discrimination. See docs/research/binary-scoring/.
    """

    def __init__(self, model_id: str = "anthropic/claude-sonnet-4-5-20250929") -> None:
        self._model_id = model_id
        self._lm = _make_lm(model_id, max_tokens=1200)
        self._predict_article = dspy.Predict(V3ArticleScoring)
        self._predict_podcast = dspy.Predict(V3PodcastScoring)

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
        entity_id: str,
        content_warning: str = "",
    ) -> InfoScore | None:
        """Score content using the v3-binary rubric."""
        if not content:
            return None

        if content_type_hint == "podcast":
            scoring_prompt = _BINARY_PODCAST_PROMPT
            predict = self._predict_podcast
        else:
            scoring_prompt = _BINARY_ARTICLE_PROMPT
            predict = self._predict_article

        # Truncate content if too long -- track whether we did so
        max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
        was_truncated = len(content) > max_content_length
        if was_truncated:
            content = content[:max_content_length] + "... [content trimmed for evaluation]"

        prompt = scoring_prompt.format(
            title=title,
            author=author or "Unknown",
            word_count=word_count or "Unknown",
            content=content,
            content_warning=content_warning,
        )

        # Strip JSON format instruction -- DSPy handles output formatting
        prompt = _strip_json_instruction(prompt)

        try:
            with dspy.context(lm=self._lm):
                prediction = predict(evaluation_prompt=prompt)

            output: V3BinaryOutput = prediction.result
            data = output.model_dump()

            # Log usage
            input_tokens, output_tokens = _extract_usage(self._lm)
            model = self._model_id.split("/", 1)[-1]
            service = "scorer_v3" if content_type_hint != "podcast" else "podcast_scorer_v3"
            await log_usage(
                service=service,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                article_id=entity_id,
            )

            # Extract boolean responses for q1-q20
            responses: dict[str, bool] = {}
            for i in range(1, 21):
                key = f"q{i}"
                responses[key] = bool(getattr(output, key, False))

            # Gatekeeper: q17=false means content may be incomplete.
            # Only hard-gate when content is truly minimal (< 500 chars).
            # For longer content -- whether naturally short posts or our own
            # truncation of long articles -- let Q17 contribute as a soft
            # signal through normal scoring instead of zeroing everything.
            content_len = len(content)
            if not responses.get("q17", False) and not was_truncated and content_len < 500:
                return InfoScore(
                    specificity=0,
                    specificity_reason="Content incomplete",
                    novelty=0,
                    novelty_reason="Content incomplete",
                    depth=0,
                    depth_reason="Content incomplete",
                    actionability=0,
                    actionability_reason="Content incomplete",
                    overall_assessment=output.overall_assessment
                    or "Content incomplete — cannot evaluate.",
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
                    reason = getattr(output, f"{q}_reason", "")
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
                overall_assessment=output.overall_assessment,
                raw_responses=data,
                total_override=total,
            )
        except Exception as e:
            logger.error("Error in v3-binary scoring for %s: %s", entity_id, e)
            return None


# ---------------------------------------------------------------------------
# v4-binary scoring: tiered-difficulty questions with evidence grounding
# ---------------------------------------------------------------------------

# Weights for each of 24 binary questions, tiered by difficulty.
# Positive questions: yes = add points. Negative questions: yes = subtract.
V4_WEIGHTS: dict[str, int] = {
    # Quotability
    "q1": 4,  # Easy: specific claim/data/example
    "q2": 6,  # Medium: memorable phrasing
    "q3": 6,  # Medium: quantified finding
    "q4": 9,  # Hard: extractable standalone passage
    "q5": 5,  # Medium: attributed statement
    "q6": -6,  # Negative: primarily linking/summarizing
    # Surprise
    "q7": 4,  # Easy: substantive depth
    "q8": 6,  # Medium: contradicts assumption
    "q9": 10,  # Hard: unexpected reframing
    "q10": 6,  # Medium: unfamiliar evidence
    "q11": 5,  # Medium: multi-stage argument
    "q12": -7,  # Negative: conveyable in one sentence
    # Argument
    "q13": 3,  # Easy: clear debatable position
    "q14": 6,  # Medium: 2+ evidence types
    "q15": 6,  # Medium: first-person experience
    "q16": 9,  # Hard: engages counterargument
    "q17": 10,  # Hard: intellectual risk
    "q18": -6,  # Negative: primarily reporting
    # Insight
    "q19": 4,  # Easy: applicable idea
    "q20": 10,  # Hard: named framework/model
    "q21": 6,  # Medium: enough detail to apply
    "q22": 5,  # Medium: technique for next month
    "q23": 9,  # Hard: perspective shift
    "q24": -5,  # Negative: narrowly domain-specific
}

_V4_MAX_POSITIVE = sum(w for w in V4_WEIGHTS.values() if w > 0)  # 129
_V4_MIN_RAW = sum(w for w in V4_WEIGHTS.values() if w < 0)  # -24

V4_DIMENSION_QUESTIONS: dict[str, list[str]] = {
    "quotability": ["q1", "q2", "q3", "q4", "q5", "q6"],
    "surprise": ["q7", "q8", "q9", "q10", "q11", "q12"],
    "argument": ["q13", "q14", "q15", "q16", "q17", "q18"],
    "insight": ["q19", "q20", "q21", "q22", "q23", "q24"],
}

# Questions where "yes" is a positive signal (used for reason extraction)
_V4_POSITIVE_QUESTIONS: dict[str, list[str]] = {
    "quotability": ["q1", "q2", "q3", "q4", "q5"],
    "surprise": ["q7", "q8", "q9", "q10", "q11"],
    "argument": ["q13", "q14", "q15", "q16", "q17"],
    "insight": ["q19", "q20", "q21", "q22", "q23"],
}


def compute_v4_total(responses: dict[str, bool]) -> int:
    """Compute total score (0-100) from v4 binary question responses.

    Raw score range is [_V4_MIN_RAW, _V4_MAX_POSITIVE], linearly scaled to [0, 100].
    """
    raw = sum(V4_WEIGHTS[q] * (1 if responses.get(q, False) else 0) for q in V4_WEIGHTS)
    return max(0, min(100, int(100 * (raw - _V4_MIN_RAW) / (_V4_MAX_POSITIVE - _V4_MIN_RAW))))


def compute_v4_dimension(dim: str, responses: dict[str, bool]) -> int:
    """Compute a single dimension sub-score (0-25) from v4 binary responses."""
    qs = V4_DIMENSION_QUESTIONS[dim]
    dim_weights = {q: V4_WEIGHTS[q] for q in qs}
    dim_max = sum(w for w in dim_weights.values() if w > 0)
    dim_min = sum(w for w in dim_weights.values() if w < 0)
    dim_raw = sum(dim_weights[q] * (1 if responses.get(q, False) else 0) for q in qs)
    span = dim_max - dim_min
    if span == 0:
        return 0
    return max(0, min(25, int(25 * (dim_raw - dim_min) / span)))


# --- V4 system prompt (shared rubric + calibration instruction) ---

_V4_SYSTEM_PROMPT = """You are a content evaluation assistant. Your task is to assess articles for "capture value" — how likely a thoughtful reader is to want to save and highlight passages.

CALIBRATION: Be critical and discriminating. Most articles should pass 10-14 of 24 questions. An article passing more than 20 is exceptional and should be very rare. Easy questions test minimum quality — most articles pass them. Hard questions test for excellence — most articles fail them. If you find yourself answering "yes" to nearly every question, you are not applying the criteria strictly enough.

RESPONSE FORMAT: For each question, first provide brief evidence (a specific quote or observation from the text), then your yes/no judgment. For "yes" answers, the evidence must cite a concrete passage or feature. For "no" answers, evidence can be empty.

Respond with ONLY a JSON object. No markdown fences, no extra text."""

# --- V4 article prompt ---

_V4_ARTICLE_PROMPT = """Evaluate this article across 24 questions in four dimensions.

QUOTABILITY — Would a reader want to highlight passages?
Q1 [Easy]: Does the article contain at least one specific claim, data point, or concrete example that goes beyond abstract generalization?
Q2 [Medium]: Does the article include a sentence or short passage that is memorable for its *phrasing* — a striking metaphor, crisp formulation, or vivid example?
Q3 [Medium]: Does the article cite a specific quantified finding (a number, percentage, measurement, or research result) that would be worth remembering on its own?
Q4 [Hard]: Could you extract a self-contained passage of 2-4 sentences from this article that would be valuable as a standalone note — making sense without the surrounding context?
Q5 [Medium]: Does the article contain a direct, attributed statement from a named individual (the author speaking from experience, or a quoted expert/practitioner) that adds credibility or insight beyond what unsourced claims would provide?
Q6 [Penalty]: Is the article's value primarily in linking to or summarizing other sources, rather than in its own prose?

SURPRISE — Does it challenge or expand the reader's understanding?
Q7 [Easy]: Does the article address a real topic with enough depth that a reader could discuss it substantively with a colleague?
Q8 [Medium]: Does the article present a specific finding, conclusion, or perspective that directly contradicts or complicates a widely-held assumption in its domain?
Q9 [Hard]: Does the article reframe a familiar topic by connecting it to an unexpected domain, analogy, or historical parallel that changes how the reader might think about it?
Q10 [Medium]: Does the article contain a case study, example, or piece of evidence that the reader is unlikely to have encountered in prior reading on this topic?
Q11 [Medium]: Does the article develop an idea through multiple stages or layers, reaching a conclusion that is not obvious from its opening premise?
Q12 [Penalty]: Could the article's central point be accurately conveyed in a single sentence without significant loss of meaning?

ARGUMENT — Is the reasoning strong and grounded?
Q13 [Easy]: Does the author take a clear position on a debatable topic — one where a reasonable, informed person could argue the opposite?
Q14 [Medium]: Does the author support their central claim with at least two distinct types of evidence (e.g., personal experience AND external data; case study AND logical argument; historical example AND current research)?
Q15 [Medium]: Does the author write from direct, first-person professional experience — describing specific situations they personally faced, decisions they made, or results they observed?
Q16 [Hard]: Does the author engage with and respond to the strongest counterargument to their position, rather than ignoring or strawmanning opposing views?
Q17 [Hard]: Does the argument contain an element of genuine intellectual risk — where the author commits to a specific, falsifiable prediction, recommends against a popular approach, or admits to a significant failure?
Q18 [Penalty]: Is the article primarily reporting or explaining what others have said or done, rather than advancing the author's own analysis or argument?

INSIGHT — Can the reader use what they've read?
Q19 [Easy]: Does the article contain at least one idea, recommendation, or observation that could influence how a reader thinks about or approaches a problem in their own work?
Q20 [Hard]: Does the article introduce, name, or clearly articulate a framework, mental model, or structured approach that organizes thinking about a class of problems?
Q21 [Medium]: Does the article provide enough concrete detail (specific steps, criteria, examples, or conditions) that a reader could attempt to apply its core idea without needing to seek additional sources?
Q22 [Medium]: Does the article describe a specific technique, practice, or decision process that the reader could try in a concrete situation within the next month?
Q23 [Hard]: Does the article offer a perspective shift — a way of seeing a familiar situation that, once understood, would persistently change the reader's default interpretation or approach?
Q24 [Penalty]: Is the article's content so narrowly domain-specific that it would be useful only to practitioners of a single specialized profession or technology?

{content_warning}Title: {title}
Author: {author}
Word Count: {word_count}

Content:
{content}

Respond with ONLY a JSON object:
{{"q1_evidence": "<quote or observation>", "q1": true, "q2_evidence": "", "q2": false, "q3_evidence": "<quote or observation>", "q3": true, "q4_evidence": "", "q4": false, "q5_evidence": "<quote or observation>", "q5": true, "q6_evidence": "", "q6": false, "q7_evidence": "<quote or observation>", "q7": true, "q8_evidence": "", "q8": false, "q9_evidence": "", "q9": false, "q10_evidence": "<quote or observation>", "q10": true, "q11_evidence": "", "q11": false, "q12_evidence": "", "q12": false, "q13_evidence": "<quote or observation>", "q13": true, "q14_evidence": "", "q14": false, "q15_evidence": "<quote or observation>", "q15": true, "q16_evidence": "", "q16": false, "q17_evidence": "", "q17": false, "q18_evidence": "", "q18": false, "q19_evidence": "<quote or observation>", "q19": true, "q20_evidence": "", "q20": false, "q21_evidence": "", "q21": false, "q22_evidence": "", "q22": false, "q23_evidence": "", "q23": false, "q24_evidence": "", "q24": false, "overall_assessment": "<1-2 sentence summary>"}}"""

# --- V4 podcast prompt ---

_V4_PODCAST_PROMPT = """Evaluate this podcast episode transcript across 24 questions in four dimensions.

QUOTABILITY — Would a listener want to highlight passages?
Q1 [Easy]: Does the transcript contain at least one specific claim, data point, or concrete example that goes beyond abstract generalization?
Q2 [Medium]: Does the transcript include a statement or exchange that is memorable for its phrasing — a striking analogy, crisp formulation, or vivid example?
Q3 [Medium]: Does a speaker cite a specific quantified finding (a number, percentage, measurement, or research result) that would be worth remembering on its own?
Q4 [Hard]: Could you extract a self-contained passage of 2-4 sentences from this transcript that would be valuable as a standalone note — making sense without the surrounding context?
Q5 [Medium]: Does a host or guest share a direct, attributed statement from their own professional experience that adds credibility or insight beyond what unsourced claims would provide?
Q6 [Penalty]: Is the episode's value primarily in referencing or summarizing other people's work, rather than offering original insight from the speakers?

SURPRISE — Does it challenge or expand the listener's understanding?
Q7 [Easy]: Does the episode address a real topic with enough depth that a listener could discuss it substantively with a colleague?
Q8 [Medium]: Does a speaker present a specific finding, conclusion, or perspective that directly contradicts or complicates a widely-held assumption in the episode's domain?
Q9 [Hard]: Does the episode reframe a familiar topic by connecting it to an unexpected domain, analogy, or historical parallel that changes how the listener might think about it?
Q10 [Medium]: Does the episode contain a case study, example, or piece of evidence that the listener is unlikely to have encountered in prior reading on this topic?
Q11 [Medium]: Does the conversation develop an idea through multiple stages or layers, reaching a conclusion that is not obvious from the episode's opening premise?
Q12 [Penalty]: Could the episode's central point be accurately conveyed in a single sentence without significant loss of meaning?

ARGUMENT — Is the reasoning strong and grounded?
Q13 [Easy]: Does a speaker take a clear position on a debatable topic — one where a reasonable, informed person could argue the opposite?
Q14 [Medium]: Does a speaker support their central claim with at least two distinct types of evidence (e.g., personal experience AND external data; case study AND logical argument)?
Q15 [Medium]: Does a host or guest speak from direct, first-person professional experience — describing specific situations they personally faced, decisions they made, or results they observed?
Q16 [Hard]: Does a speaker engage with and respond to the strongest counterargument to their position, rather than ignoring or dismissing opposing views?
Q17 [Hard]: Does the argument contain an element of genuine intellectual risk — where a speaker commits to a specific prediction, recommends against a popular approach, or admits to a significant failure?
Q18 [Penalty]: Is the episode primarily reporting or explaining what others have said or done, rather than advancing the speakers' own analysis or argument?

INSIGHT — Can the listener use what they've heard?
Q19 [Easy]: Does the episode contain at least one idea, recommendation, or observation that could influence how a listener thinks about or approaches a problem in their own work?
Q20 [Hard]: Does the episode introduce, name, or clearly articulate a framework, mental model, or structured approach that organizes thinking about a class of problems?
Q21 [Medium]: Does the episode provide enough concrete detail (specific steps, criteria, examples, or conditions) that a listener could attempt to apply its core idea without needing to seek additional sources?
Q22 [Medium]: Does the episode describe a specific technique, practice, or decision process that the listener could try in a concrete situation within the next month?
Q23 [Hard]: Does the episode offer a perspective shift — a way of seeing a familiar situation that, once understood, would persistently change the listener's default interpretation or approach?
Q24 [Penalty]: Is the episode's content so narrowly domain-specific that it would be useful only to practitioners of a single specialized profession or technology?

{content_warning}Title: {title}
Host/Show: {author}
Word Count: {word_count}

Transcript:
{content}

Respond with ONLY a JSON object:
{{"q1_evidence": "<quote or observation>", "q1": true, "q2_evidence": "", "q2": false, "q3_evidence": "<quote or observation>", "q3": true, "q4_evidence": "", "q4": false, "q5_evidence": "<quote or observation>", "q5": true, "q6_evidence": "", "q6": false, "q7_evidence": "<quote or observation>", "q7": true, "q8_evidence": "", "q8": false, "q9_evidence": "", "q9": false, "q10_evidence": "<quote or observation>", "q10": true, "q11_evidence": "", "q11": false, "q12_evidence": "", "q12": false, "q13_evidence": "<quote or observation>", "q13": true, "q14_evidence": "", "q14": false, "q15_evidence": "<quote or observation>", "q15": true, "q16_evidence": "", "q16": false, "q17_evidence": "", "q17": false, "q18_evidence": "", "q18": false, "q19_evidence": "<quote or observation>", "q19": true, "q20_evidence": "", "q20": false, "q21_evidence": "", "q21": false, "q22_evidence": "", "q22": false, "q23_evidence": "", "q23": false, "q24_evidence": "", "q24": false, "overall_assessment": "<1-2 sentence summary>"}}"""


class TieredBinaryScoringStrategy:
    """v4-binary scoring strategy.

    Uses 24 difficulty-tiered binary questions with evidence grounding,
    penalty questions, and calibration instructions for better score
    discrimination. See docs/v4-implementation-plan.md.
    """

    def __init__(self, model_id: str = "anthropic/claude-sonnet-4-5-20250929") -> None:
        self._model_id = model_id
        self._lm = _make_lm(model_id, max_tokens=2000, temperature=0.0)
        self._predict_article = dspy.Predict(V4ArticleScoring)
        self._predict_podcast = dspy.Predict(V4PodcastScoring)

    @property
    def version(self) -> str:
        return "v4-binary"

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
        """Score content using the v4-binary tiered rubric."""
        if not content:
            return None

        if content_type_hint == "podcast":
            scoring_prompt = _V4_PODCAST_PROMPT
            predict = self._predict_podcast
        else:
            scoring_prompt = _V4_ARTICLE_PROMPT
            predict = self._predict_article

        # Truncate content if too long
        max_content_length = _MAX_CONTENT_LENGTH.get(content_type_hint, 15000)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [content trimmed for evaluation]"

        prompt = scoring_prompt.format(
            title=title,
            author=author or "Unknown",
            word_count=word_count or "Unknown",
            content=content,
            content_warning=content_warning,
        )

        # Strip JSON format instruction -- DSPy handles output formatting
        prompt = _strip_json_instruction(prompt)

        try:
            with dspy.context(lm=self._lm):
                prediction = predict(evaluation_prompt=prompt)

            output: V4TieredOutput = prediction.result
            data = output.model_dump()

            # Log usage
            input_tokens, output_tokens = _extract_usage(self._lm)
            model = self._model_id.split("/", 1)[-1]
            service = "scorer_v4" if content_type_hint != "podcast" else "podcast_scorer_v4"
            await log_usage(
                service=service,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                article_id=entity_id,
            )

            # Extract boolean responses for q1-q24
            responses: dict[str, bool] = {}
            for i in range(1, 25):
                key = f"q{i}"
                responses[key] = bool(getattr(output, key, False))

            # Compute scores
            total = compute_v4_total(responses)
            quotability = compute_v4_dimension("quotability", responses)
            surprise = compute_v4_dimension("surprise", responses)
            argument = compute_v4_dimension("argument", responses)
            insight = compute_v4_dimension("insight", responses)

            # Extract per-dimension reasons from evidence fields
            def _dim_reasons(qs: list[str]) -> str:
                parts = []
                for q in qs:
                    evidence = getattr(output, f"{q}_evidence", "")
                    if evidence and responses.get(q, False):
                        parts.append(str(evidence))
                return "; ".join(parts) if parts else "No positive signals"

            return InfoScore(
                specificity=quotability,
                specificity_reason=_dim_reasons(_V4_POSITIVE_QUESTIONS["quotability"]),
                novelty=surprise,
                novelty_reason=_dim_reasons(_V4_POSITIVE_QUESTIONS["surprise"]),
                depth=argument,
                depth_reason=_dim_reasons(_V4_POSITIVE_QUESTIONS["argument"]),
                actionability=insight,
                actionability_reason=_dim_reasons(_V4_POSITIVE_QUESTIONS["insight"]),
                overall_assessment=output.overall_assessment,
                raw_responses=data,
                total_override=total,
            )
        except Exception as e:
            logger.error("Error in v4-binary scoring for %s: %s", entity_id, e)
            return None
