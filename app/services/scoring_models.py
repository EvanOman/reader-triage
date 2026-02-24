"""Pydantic output models for DSPy scoring strategies."""

from pydantic import BaseModel, Field


class V2CategoricalOutput(BaseModel):
    """Output model for v2-categorical scoring."""

    standalone_passages: str = Field(default="none", description="none / a_few / several / many")
    quotability_reason: str = Field(default="", description="Brief reason for quotability score")
    novel_framing: bool = Field(default=False, description="Does it reframe a familiar topic?")
    content_type: str = Field(
        default="informational_summary",
        description="original_analysis / opinion_with_evidence / informational_summary / product_review / news_or_roundup",
    )
    surprise_reason: str = Field(default="", description="Brief reason for surprise score")
    author_conviction: bool = Field(
        default=False, description="Does the author argue with conviction?"
    )
    practitioner_voice: bool = Field(
        default=False, description="Written from first-person practitioner experience?"
    )
    content_completeness: str = Field(
        default="complete", description="complete / appears_truncated / summary_or_excerpt"
    )
    argument_reason: str = Field(default="", description="Brief reason for argument score")
    named_framework: bool = Field(
        default=False, description="Does it introduce a named concept or framework?"
    )
    applicable_ideas: str = Field(
        default="not_really", description="broadly / narrowly / not_really"
    )
    insight_reason: str = Field(default="", description="Brief reason for insight score")
    overall_assessment: str = Field(default="", description="1-2 sentence summary")


class V3BinaryOutput(BaseModel):
    """Output model for v3-binary scoring (20 yes/no questions with reasons)."""

    q1: bool = Field(
        default=False, description="Memorable phrase, vivid metaphor, or striking sentence?"
    )
    q1_reason: str = Field(default="", description="Brief reason for q1")
    q2: bool = Field(
        default=False, description="Specific data point, statistic, or quantified claim?"
    )
    q2_reason: str = Field(default="", description="Brief reason for q2")
    q3: bool = Field(default=False, description="Extractable 2-3 sentence standalone note?")
    q3_reason: str = Field(default="", description="Brief reason for q3")
    q4: bool = Field(default=False, description="Direct quote from practitioner or expert?")
    q4_reason: str = Field(default="", description="Brief reason for q4")
    q5: bool = Field(default=False, description="Contradicts common assumptions?")
    q5_reason: str = Field(default="", description="Brief reason for q5")
    q6: bool = Field(default=False, description="Reframes through unexpected lens?")
    q6_reason: str = Field(default="", description="Brief reason for q6")
    q7: bool = Field(default=False, description="Unfamiliar finding or case study?")
    q7_reason: str = Field(default="", description="Brief reason for q7")
    q8: bool = Field(default=False, description="Primarily a summary/restatement of known ideas?")
    q8_reason: str = Field(default="", description="Brief reason for q8")
    q9: bool = Field(default=False, description="Clear specific position argued?")
    q9_reason: str = Field(default="", description="Brief reason for q9")
    q10: bool = Field(default=False, description="Supported with concrete evidence?")
    q10_reason: str = Field(default="", description="Brief reason for q10")
    q11: bool = Field(default=False, description="First-person professional experience?")
    q11_reason: str = Field(default="", description="Brief reason for q11")
    q12: bool = Field(default=False, description="Could be captured in a tweet-length summary?")
    q12_reason: str = Field(default="", description="Brief reason for q12")
    q13: bool = Field(default=False, description="Introduces framework or mental model?")
    q13_reason: str = Field(default="", description="Brief reason for q13")
    q14: bool = Field(default=False, description="Applicable technique within next month?")
    q14_reason: str = Field(default="", description="Brief reason for q14")
    q15: bool = Field(default=False, description="Enough detail to act on recommendations?")
    q15_reason: str = Field(default="", description="Brief reason for q15")
    q16: bool = Field(default=False, description="Too narrowly domain-specific?")
    q16_reason: str = Field(default="", description="Brief reason for q16")
    q17: bool = Field(
        default=False, description="Content complete and substantial enough to evaluate?"
    )
    q17_reason: str = Field(default="", description="Brief reason for q17")
    q18: bool = Field(default=False, description="Substantive value beyond search results?")
    q18_reason: str = Field(default="", description="Brief reason for q18")
    q19: bool = Field(default=False, description="At least one idea reader hasn't seen before?")
    q19_reason: str = Field(default="", description="Brief reason for q19")
    q20: bool = Field(
        default=False, description="Primarily news/product announcement/press release?"
    )
    q20_reason: str = Field(default="", description="Brief reason for q20")
    overall_assessment: str = Field(default="", description="1-2 sentence summary")


class V4TieredOutput(BaseModel):
    """Output model for v4-binary scoring (24 tiered questions with evidence)."""

    q1_evidence: str = Field(default="", description="Quote or observation for q1")
    q1: bool = Field(
        default=False, description="[Easy] Specific claim, data point, or concrete example?"
    )
    q2_evidence: str = Field(default="", description="Quote or observation for q2")
    q2: bool = Field(default=False, description="[Medium] Memorable phrasing?")
    q3_evidence: str = Field(default="", description="Quote or observation for q3")
    q3: bool = Field(default=False, description="[Medium] Specific quantified finding?")
    q4_evidence: str = Field(default="", description="Quote or observation for q4")
    q4: bool = Field(default=False, description="[Hard] Self-contained 2-4 sentence passage?")
    q5_evidence: str = Field(default="", description="Quote or observation for q5")
    q5: bool = Field(
        default=False, description="[Medium] Attributed statement from named individual?"
    )
    q6_evidence: str = Field(default="", description="Quote or observation for q6")
    q6: bool = Field(default=False, description="[Penalty] Primarily linking/summarizing?")
    q7_evidence: str = Field(default="", description="Quote or observation for q7")
    q7: bool = Field(default=False, description="[Easy] Substantive depth?")
    q8_evidence: str = Field(default="", description="Quote or observation for q8")
    q8: bool = Field(default=False, description="[Medium] Contradicts widely-held assumption?")
    q9_evidence: str = Field(default="", description="Quote or observation for q9")
    q9: bool = Field(default=False, description="[Hard] Unexpected reframing?")
    q10_evidence: str = Field(default="", description="Quote or observation for q10")
    q10: bool = Field(default=False, description="[Medium] Unfamiliar evidence?")
    q11_evidence: str = Field(default="", description="Quote or observation for q11")
    q11: bool = Field(default=False, description="[Medium] Multi-stage argument?")
    q12_evidence: str = Field(default="", description="Quote or observation for q12")
    q12: bool = Field(default=False, description="[Penalty] Conveyable in one sentence?")
    q13_evidence: str = Field(default="", description="Quote or observation for q13")
    q13: bool = Field(default=False, description="[Easy] Clear debatable position?")
    q14_evidence: str = Field(default="", description="Quote or observation for q14")
    q14: bool = Field(default=False, description="[Medium] 2+ evidence types?")
    q15_evidence: str = Field(default="", description="Quote or observation for q15")
    q15: bool = Field(default=False, description="[Medium] First-person experience?")
    q16_evidence: str = Field(default="", description="Quote or observation for q16")
    q16: bool = Field(default=False, description="[Hard] Engages counterargument?")
    q17_evidence: str = Field(default="", description="Quote or observation for q17")
    q17: bool = Field(default=False, description="[Hard] Intellectual risk?")
    q18_evidence: str = Field(default="", description="Quote or observation for q18")
    q18: bool = Field(default=False, description="[Penalty] Primarily reporting?")
    q19_evidence: str = Field(default="", description="Quote or observation for q19")
    q19: bool = Field(default=False, description="[Easy] Applicable idea?")
    q20_evidence: str = Field(default="", description="Quote or observation for q20")
    q20: bool = Field(default=False, description="[Hard] Named framework/model?")
    q21_evidence: str = Field(default="", description="Quote or observation for q21")
    q21: bool = Field(default=False, description="[Medium] Enough detail to apply?")
    q22_evidence: str = Field(default="", description="Quote or observation for q22")
    q22: bool = Field(default=False, description="[Medium] Technique for next month?")
    q23_evidence: str = Field(default="", description="Quote or observation for q23")
    q23: bool = Field(default=False, description="[Hard] Perspective shift?")
    q24_evidence: str = Field(default="", description="Quote or observation for q24")
    q24: bool = Field(default=False, description="[Penalty] Narrowly domain-specific?")
    overall_assessment: str = Field(default="", description="1-2 sentence summary")
