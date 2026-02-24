"""DSPy Signatures for scoring strategies.

Each Signature defines input/output fields and a system prompt (via docstring).
The existing prompt templates are passed as the InputField value, preserving
the exact evaluation rubric text. DSPy handles output formatting via the
Pydantic output models.
"""

import dspy

from app.services.scoring_models import V2CategoricalOutput, V3BinaryOutput, V4TieredOutput

# ---------------------------------------------------------------------------
# V2 Categorical Signatures
# ---------------------------------------------------------------------------


class V2ArticleScoring(dspy.Signature):
    """You are a content evaluation assistant. Evaluate the article for capture value — how likely a reader is to want to save and highlight passages. Answer each question by selecting from the provided options."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with article content and scoring questions"
    )
    result: V2CategoricalOutput = dspy.OutputField(
        desc="Structured scoring result with categorical answers and reasons"
    )


class V2PodcastScoring(dspy.Signature):
    """You are a content evaluation assistant. Evaluate the podcast episode for capture value — how likely a listener is to want to save and highlight passages. Answer each question by selecting from the provided options."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with podcast transcript and scoring questions"
    )
    result: V2CategoricalOutput = dspy.OutputField(
        desc="Structured scoring result with categorical answers and reasons"
    )


# ---------------------------------------------------------------------------
# V3 Binary Signatures
# ---------------------------------------------------------------------------


class V3ArticleScoring(dspy.Signature):
    """You are a content evaluation assistant. Evaluate the article for capture value — how likely a reader is to want to save and highlight passages. For each question, answer ONLY "yes" or "no". Be critical and honest — most articles should NOT pass the harder questions."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with article content and 20 binary questions"
    )
    result: V3BinaryOutput = dspy.OutputField(
        desc="Structured scoring result with boolean answers and reasons"
    )


class V3PodcastScoring(dspy.Signature):
    """You are a content evaluation assistant. Evaluate the podcast episode for capture value — how likely a listener is to want to save and highlight passages. For each question, answer ONLY "yes" or "no". Be critical and honest — most episodes should NOT pass the harder questions."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with podcast transcript and 20 binary questions"
    )
    result: V3BinaryOutput = dspy.OutputField(
        desc="Structured scoring result with boolean answers and reasons"
    )


# ---------------------------------------------------------------------------
# V4 Tiered Binary Signatures
# ---------------------------------------------------------------------------


class V4ArticleScoring(dspy.Signature):
    """You are a content evaluation assistant. Your task is to assess articles for "capture value" — how likely a thoughtful reader is to want to save and highlight passages.

    CALIBRATION: Be critical and discriminating. Most articles should pass 10-14 of 24 questions. An article passing more than 20 is exceptional and should be very rare. Easy questions test minimum quality — most articles pass them. Hard questions test for excellence — most articles fail them. If you find yourself answering "yes" to nearly every question, you are not applying the criteria strictly enough.

    RESPONSE FORMAT: For each question, first provide brief evidence (a specific quote or observation from the text), then your yes/no judgment. For "yes" answers, the evidence must cite a concrete passage or feature. For "no" answers, evidence can be empty."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with article content and 24 tiered questions"
    )
    result: V4TieredOutput = dspy.OutputField(
        desc="Structured scoring result with boolean answers, evidence, and assessment"
    )


class V4PodcastScoring(dspy.Signature):
    """You are a content evaluation assistant. Your task is to assess podcast episodes for "capture value" — how likely a thoughtful listener is to want to save and highlight passages.

    CALIBRATION: Be critical and discriminating. Most episodes should pass 10-14 of 24 questions. An episode passing more than 20 is exceptional and should be very rare. Easy questions test minimum quality — most episodes pass them. Hard questions test for excellence — most episodes fail them. If you find yourself answering "yes" to nearly every question, you are not applying the criteria strictly enough.

    RESPONSE FORMAT: For each question, first provide brief evidence (a specific quote or observation from the transcript), then your yes/no judgment. For "yes" answers, the evidence must cite a concrete passage or feature. For "no" answers, evidence can be empty."""

    evaluation_prompt: str = dspy.InputField(
        desc="Full evaluation prompt with podcast transcript and 24 tiered questions"
    )
    result: V4TieredOutput = dspy.OutputField(
        desc="Structured scoring result with boolean answers, evidence, and assessment"
    )
