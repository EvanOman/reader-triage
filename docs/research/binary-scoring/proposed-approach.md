# Proposed Approach: v3-binary Scoring Strategy

## Summary

Replace the current v2-categorical scoring (8 multi-class/boolean questions mapped to point buckets) with a v3-binary strategy using 20 weighted binary questions. The new approach addresses three problems with the current system:

1. **Score ceiling clustering**: Most articles score 80-100, providing little differentiation
2. **Categorical ambiguity**: Questions like "a_few vs several vs many" invite sycophantic inflation
3. **Coarse resolution**: 8 questions with varying answer types produce a lumpy score distribution

---

## Architecture: Pluggable ScoringStrategy

The v3-binary scorer should be implemented as a new `ScoringStrategy` class, enabling A/B comparison with v2-categorical.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScoringResult:
    """Result from any scoring strategy."""
    total: int                          # 0-100
    specificity: int                    # 0-25 (quotability)
    specificity_reason: str
    novelty: int                        # 0-25 (surprise)
    novelty_reason: str
    depth: int                          # 0-25 (argument quality)
    depth_reason: str
    actionability: int                  # 0-25 (applicable insight)
    actionability_reason: str
    overall_assessment: str
    content_fetch_failed: bool = False
    raw_responses: dict | None = None   # Full LLM response for debugging


class ScoringStrategy(ABC):
    """Base class for pluggable scoring strategies."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Scoring version string, e.g. 'v3-binary'."""
        ...

    @abstractmethod
    async def score(
        self,
        *,
        title: str,
        author: str | None,
        content: str,
        word_count: int | None,
        content_type: str,  # "article" or "podcast"
    ) -> ScoringResult | None:
        ...
```

---

## The 20 Binary Questions

### Quotability Dimension (4 questions)

| ID | Question | Weight | Tier | Polarity |
|----|----------|--------|------|----------|
| q1 | Does the article contain a memorable phrase, vivid metaphor, or striking sentence that encapsulates a key idea? | 5 | Supplementary | Positive |
| q2 | Does the article include a specific data point, statistic, or quantified claim worth remembering? | 5 | Supplementary | Positive |
| q3 | Could you extract at least one passage of 2-3 sentences that would work as a standalone saved note worth revisiting? | 8 | Essential | Positive |
| q4 | Does the article contain a direct quote from a practitioner or expert that adds credibility or color? | 4 | Supplementary | Positive |

**Dimension max (all positive "yes")**: 22 points raw

### Surprise Dimension (4 questions)

| ID | Question | Weight | Tier | Polarity |
|----|----------|--------|------|----------|
| q5 | Does the article present a perspective or conclusion that contradicts common assumptions or conventional wisdom in its domain? | 8 | Essential | Positive |
| q6 | Does the article reframe a familiar topic through an unexpected lens, analogy, or cross-domain connection? | 6 | Important | Positive |
| q7 | Does the article contain a finding, case study, or example that an informed reader in this domain likely hasn't encountered before? | 6 | Important | Positive |
| q8 | Is this primarily a summary, restatement, or roundup of widely-known ideas rather than original analysis? | -6 | Penalty | Negative |

**Dimension max (best case: q5-q7 yes, q8 no)**: 20 points raw
**Dimension min (worst case: q5-q7 no, q8 yes)**: -6 points raw

### Argument Quality Dimension (4 questions)

| ID | Question | Weight | Tier | Polarity |
|----|----------|--------|------|----------|
| q9 | Does the author argue for a clear, specific position rather than presenting multiple viewpoints neutrally? | 6 | Important | Positive |
| q10 | Does the author support their central claims with concrete evidence such as data, case studies, or specific real-world examples? | 6 | Important | Positive |
| q11 | Does the author write from first-person professional experience, sharing hard-won lessons or opinions from practice? | 8 | Essential | Positive |
| q12 | Could this article's central argument or message be adequately captured in a single tweet-length summary? | -6 | Penalty | Negative |

**Dimension max**: 20 points raw
**Dimension min**: -6 points raw

### Applicable Insight Dimension (4 questions)

| ID | Question | Weight | Tier | Polarity |
|----|----------|--------|------|----------|
| q13 | Does the article introduce, name, or build upon a specific framework, mental model, or structured methodology? | 8 | Essential | Positive |
| q14 | Could a reader apply a specific technique or idea from this article in their own work within the next month? | 6 | Important | Positive |
| q15 | Does the article provide enough context and concrete detail for a reader to act on its key recommendations? | 5 | Supplementary | Positive |
| q16 | Is the content so narrowly domain-specific that it would only be useful to readers in a single niche? | -4 | Penalty (moderate) | Negative |

**Dimension max**: 19 points raw
**Dimension min**: -4 points raw

### Cross-Cutting Quality (4 questions)

| ID | Question | Weight | Tier | Polarity |
|----|----------|--------|------|----------|
| q17 | Is the available text complete and substantial enough to properly evaluate? (Not truncated, paywalled, or a stub.) | N/A | Gatekeeper | Special |
| q18 | Does this article offer substantive value beyond what a reader could easily find in the first page of search results on this topic? | 6 | Important | Positive |
| q19 | Would a knowledgeable reader in this domain likely encounter at least one idea they haven't seen before? | 8 | Essential | Positive |
| q20 | Is this primarily a news article, product announcement, press release, or event recap? | -8 | Penalty (strong) | Negative |

**Cross-cutting max (q18+q19 yes, q20 no)**: 14 points raw
**Cross-cutting min (q18+q19 no, q20 yes)**: -8 points raw

---

## Score Computation

### Step 1: Gatekeeper Check

If `q17 = false` (content incomplete), return `content_fetch_failed = True` and flag for rescore. Do not compute a score.

### Step 2: Raw Score Calculation

```python
WEIGHTS = {
    "q1": 5,  "q2": 5,  "q3": 8,   "q4": 4,
    "q5": 8,  "q6": 6,  "q7": 6,   "q8": -6,
    "q9": 6,  "q10": 6, "q11": 8,  "q12": -6,
    "q13": 8, "q14": 6, "q15": 5,  "q16": -4,
    # q17 is gatekeeper, not scored
    "q18": 6, "q19": 8, "q20": -8,
}

# Positive weights only (for scaling)
MAX_POSITIVE = sum(w for w in WEIGHTS.values() if w > 0)
# = 5+5+8+4 + 8+6+6 + 6+6+8 + 8+6+5 + 6+8 = 95

raw_score = sum(
    WEIGHTS[q] * (1 if responses[q] else 0)
    for q in WEIGHTS
)
# raw_score range: -24 (all negative yes, all positive no) to 95 (all positive yes, all negative no)
```

### Step 3: Scale to 0-100

```python
# Linear scaling: map [-24, 95] to [0, 100]
MIN_RAW = sum(w for w in WEIGHTS.values() if w < 0)  # -24
total = max(0, min(100, int(100 * (raw_score - MIN_RAW) / (MAX_POSITIVE - MIN_RAW))))
```

### Step 4: Dimension Sub-Scores (0-25 each)

```python
DIMENSION_QUESTIONS = {
    "quotability": ["q1", "q2", "q3", "q4"],
    "surprise": ["q5", "q6", "q7", "q8"],
    "argument": ["q9", "q10", "q11", "q12"],
    "insight": ["q13", "q14", "q15", "q16"],
}

def dimension_score(dim: str, responses: dict[str, bool]) -> int:
    qs = DIMENSION_QUESTIONS[dim]
    dim_weights = {q: WEIGHTS[q] for q in qs}
    dim_max = sum(w for w in dim_weights.values() if w > 0)
    dim_min = sum(w for w in dim_weights.values() if w < 0)
    dim_raw = sum(dim_weights[q] * (1 if responses[q] else 0) for q in qs)
    return max(0, min(25, int(25 * (dim_raw - dim_min) / (dim_max - dim_min))))
```

Note: Cross-cutting questions (q17-q20) contribute to the total score but not to any single dimension sub-score. This is intentional -- they represent holistic quality signals.

---

## Expected Score Distribution

Based on the question difficulty tiers and weights, here's the expected score distribution for a typical article mix:

| Article Type | Expected Score Range | Explanation |
|-------------|---------------------|-------------|
| News roundup / press release | 0-15 | Fails most quality questions, hits q8 and q20 penalties |
| Generic blog post / summary | 15-35 | Passes some baseline questions, fails quality and exceptional |
| Solid informational article | 35-55 | Passes most baseline + some quality questions |
| Strong opinion/analysis piece | 55-75 | Passes quality questions, some exceptional |
| Exceptional practitioner essay | 75-90 | Passes most questions including exceptional tier |
| Truly outstanding (rare) | 90-100 | Near-perfect across all dimensions |

This contrasts with the current v2-categorical distribution where most articles cluster at 80-100. The v3-binary system achieves better discrimination through:

1. **Penalty questions**: Active pull-down for commodity content (current system only has "none/0" as the minimum, never subtracts)
2. **Hard questions**: High-tier questions that most articles won't pass
3. **Binary clarity**: No ambiguous "a_few vs several" where LLMs default to the generous interpretation

---

## Prompt Template

```python
BINARY_SCORING_PROMPT = """Evaluate this {content_label} for capture value — how likely a reader is to want to save and highlight passages.

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
```

### Key Prompt Design Decisions

1. **"Be critical and honest"**: Direct anti-sycophancy instruction, shown to be the most effective single mitigation technique (see literature review)
2. **"most articles should NOT pass the harder questions"**: Sets expectations that "no" is the normal answer for quality/exceptional tier questions
3. **Grouped by dimension**: Helps the LLM reason about related questions together
4. **Reasons required**: Chain-of-thought reasoning improves judgment quality (Cameron Wolfe's recommendation) and provides debugging transparency
5. **Single call**: All 20 questions in one prompt rather than 20 separate calls (lower cost, similar reliability)

---

## Podcast Variant

For podcasts, questions need transcript-specific adjustments:

| Article Question | Podcast Adaptation |
|-----------------|-------------------|
| Q1: memorable phrase | Q1: Does the transcript contain a memorable statement, analogy, or phrasing from a host or guest? |
| Q4: direct quote from expert | Q4: Does a guest share a specific professional insight or hard-won lesson from their experience? |
| Q11: first-person experience | Q11: Does a host or guest share in-depth first-person professional experience? |
| Q12: fits in a tweet | Q12: Could this episode's central topic be adequately covered in a blog post introduction? |
| Q17: text complete | Q17: Is the transcript quality sufficient to evaluate the episode's content? (Not garbled, heavily truncated, or mostly filler.) |
| Q20: news/announcement | Q20: Is this primarily a news commentary episode, product launch discussion, or event recap? |

---

## Comparison with Current v2-categorical

| Aspect | v2-categorical | v3-binary (proposed) |
|--------|---------------|---------------------|
| Questions | 8 (mixed types) | 20 (all binary) |
| Answer types | 4-5 categories, boolean, 3-option | Binary (yes/no) only |
| Score ceiling | 100 (easily reached) | ~90-95 (hard to reach) |
| Penalty signals | None (minimum is 0 per question) | 4 negative questions actively pull down |
| Discrimination | Poor (clusters at 80-100) | Good (expected spread across 15-85) |
| Interpretability | Moderate (category mappings non-obvious) | High (each yes/no directly explained) |
| Calibratable | Limited (categorical weights are fixed) | Yes (weights can be learned from data) |
| LLM reliability | Moderate (categorical choices invite inflation) | High (binary decisions more consistent) |
| Token cost | ~600-700 output tokens | ~800-1000 output tokens (20 reasons) |
| Dimension sub-scores | Yes (4 x 0-25) | Yes (4 x 0-25, backward compatible) |

---

## Calibrating Weights with Existing Highlight Data

The existing calibration toolkit (`just cal-*`) provides the foundation for weight calibration.

### Step 1: Parallel Scoring

Score existing articles with both v2-categorical and v3-binary (batch process, e.g., 200 articles). Store v3-binary raw responses alongside existing scores.

### Step 2: Correlation Analysis

For each binary question, compute:
- **Point-biserial correlation** with highlight count
- **Pass rate** (what percentage of articles get "yes")
- **Conditional engagement**: mean highlights when q=yes vs q=no

Questions with low correlation or extreme pass rates (>90% or <5%) are candidates for revision or removal.

### Step 3: Logistic Regression

```bash
# Extend the calibration toolkit
just cal-binary-weights
```

Fit logistic regression with binary question responses predicting engagement (highlight_count >= 2). Compare learned weights to expert-assigned weights.

### Step 4: Iterate

Based on calibration results:
- Adjust weights for questions that over/under-predict
- Revise or replace questions with poor discrimination
- Add new questions if a dimension consistently under-performs

### Step 5: A/B Comparison

Run `just cal-report` on both v2-categorical and v3-binary scores against the same engagement data. Compare Spearman correlation, tier accuracy, and miss rates. The v3-binary system should show:
- Better spread (lower score clustering)
- Fewer false positives (high score, no highlights)
- Similar or better Spearman correlation with highlight counts

---

## Implementation Roadmap

### Phase 1: Implement Strategy Class (1-2 days)

- Create `BinaryScoringStrategy` implementing `ScoringStrategy` interface
- Implement prompt template with all 20 questions
- Implement weight-based score computation
- Add JSON response parsing with validation
- Unit tests for score computation logic

### Phase 2: Shadow Scoring (1 week)

- Run v3-binary alongside v2-categorical on new articles
- Store both scores (dual-write to DB or separate table)
- Do NOT use v3-binary for UI display or skip recommendations yet

### Phase 3: Calibration (After ~100 shadow-scored articles)

- Analyze v3-binary question pass rates
- Correlate with engagement data
- Adjust weights and question wording as needed
- Compare discrimination with v2-categorical

### Phase 4: Switch Over (After calibration validates improvement)

- Update `CURRENT_SCORING_VERSION` to "v3-binary"
- Enable v3-binary as the primary scorer
- Keep v2-categorical available for comparison
- Update calibration tools to understand v3-binary responses

### Phase 5: Data-Calibrated Weights (After ~200 articles)

- Fit logistic regression on v3-binary responses vs. engagement
- Replace expert weights with learned weights
- Set up periodic recalibration (monthly)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LLM answers "yes" to everything | Anti-sycophancy prompt instruction; monitor pass rates; if avg >70%, questions are too easy |
| Score too low (overcorrection) | Monitor score distribution; if median < 30, reduce penalty weights |
| Inconsistent across runs | Test-retest reliability on sample of articles; binary judgments are inherently more stable |
| Higher token cost | 20 questions with reasons ~1000 tokens vs ~700 current; marginal cost increase of ~$0.001/article |
| Breaking changes to DB schema | Reuse existing columns; store raw binary responses in a JSON field |
| Questions become stale | Periodic review against calibration data; replace low-discrimination questions |

---

## Appendix: Full Weight Table

| Question | Weight | Max Contribution | Category |
|----------|--------|-----------------|----------|
| q1: Memorable phrase | +5 | 5 | Quotability |
| q2: Specific data point | +5 | 5 | Quotability |
| q3: Standalone passage | +8 | 8 | Quotability |
| q4: Expert quote | +4 | 4 | Quotability |
| q5: Contradicts assumptions | +8 | 8 | Surprise |
| q6: Unexpected lens | +6 | 6 | Surprise |
| q7: Novel example | +6 | 6 | Surprise |
| q8: Summary of known (NEG) | -6 | 0 (or -6) | Surprise |
| q9: Clear position | +6 | 6 | Argument |
| q10: Concrete evidence | +6 | 6 | Argument |
| q11: Practitioner experience | +8 | 8 | Argument |
| q12: Fits in a tweet (NEG) | -6 | 0 (or -6) | Argument |
| q13: Named framework | +8 | 8 | Insight |
| q14: Applicable technique | +6 | 6 | Insight |
| q15: Actionable detail | +5 | 5 | Insight |
| q16: Narrow audience (NEG) | -4 | 0 (or -4) | Insight |
| q17: Content complete | N/A | Gatekeeper | Quality |
| q18: Beyond search results | +6 | 6 | Quality |
| q19: Expert learns new thing | +8 | 8 | Quality |
| q20: News/announcement (NEG) | -8 | 0 (or -8) | Quality |
| **Totals** | | **Max positive: 95** | **Max penalty: -24** |

Scaled 0-100 score range with realistic article mix: most articles 25-65, good articles 55-80, exceptional 80-95.
