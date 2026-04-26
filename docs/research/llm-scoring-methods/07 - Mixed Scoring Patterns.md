# Mixed Scoring Patterns: Hybrid Approaches for LLM Evaluation

## Context

Our current v2-categorical scoring system uses 8 questions in a mixed-format design:

| # | Question | Format | Current Mapping |
|---|----------|--------|-----------------|
| 1 | STANDALONE PASSAGES | 4-level ordinal (none/a_few/several/many) | 0/9/17/25 pts |
| 2 | NOVEL FRAMING | Binary (true/false) | 0/15 pts |
| 3 | CONTENT TYPE | 5-option categorical | 0-10 pts |
| 4 | AUTHOR CONVICTION | Binary (true/false) | 0/12 pts |
| 5 | PRACTITIONER VOICE | Binary (true/false) | 0/8 pts |
| 6 | CONTENT COMPLETENESS | 3-option categorical | 0/2/5 pts |
| 7 | NAMED FRAMEWORK | Binary (true/false) | 0/12 pts |
| 8 | APPLICABLE IDEAS | 3-level ordinal (broadly/narrowly/not_really) | 0/7/13 pts |

This is a mixed-format questionnaire: some true/false, some multi-option categorical, some ordinal. This document investigates whether mixed formats are defensible, what the research says about hybrid approaches, and what specific changes (if any) we should make.

---

## 1. Pure Binary vs. Mixed Binary + Ordinal: When to Use Each

### The Case for Pure Binary

The strongest practitioner voices in the LLM evaluation space advocate for pure binary scoring.

**Hamel Husain** states it bluntly: "A common mistake is straying from binary pass/fail judgments. If your evaluations consist of metrics that LLMs score on a 1-5 scale or any other scale, you're doing it wrong." His reasoning: the distinction between adjacent scale points is subjective and inconsistent across reviewers, creating noisy data. Binary forces clarity.

Source: [Using LLM-as-a-Judge For Evaluation](https://hamel.dev/blog/posts/llm-judge/)

**Eugene Yan** concurs, preferring binary outputs because they enable straightforward classification metrics (precision, recall) rather than unreliable correlation metrics on ordinal scales.

Source: [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)

**CheckEval** (Lee et al., EMNLP 2025) demonstrated that decomposing evaluation into binary checklist questions improved inter-evaluator agreement by 0.45 over G-Eval's Likert-scale approach, achieving Krippendorff's alpha of 0.67 -- comparable to human rater agreement.

Source: [CheckEval paper](https://arxiv.org/abs/2403.18771)

**Arize AI's 2025 empirical testing** across GPT-5-nano, Claude Opus, and Qwen3 found that numeric scores "plateaued quickly after only a small amount of corruption, with scores saturating and collapsing into narrow bands." Binary labels "consistently separate clean from corrupted passages, with low variance across runs." Their follow-up 2025 study confirmed that "categorical or checklist-style judgments preserve distinctions more reliably, with discrete rubrics aligning more closely with human annotations than open numeric scoring."

Source: [Testing Binary vs Score Evals on the Latest Models](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/)

### The Case for Mixed Formats

Mixed formats are not always wrong. There are specific situations where non-binary questions earn their place:

1. **When the dimension is genuinely ordinal and the categories are well-separated.** Our Q1 (STANDALONE PASSAGES: none/a_few/several/many) is a quantity estimation. "None" and "many" are far apart; the LLM is making a count judgment, not a quality judgment. This is fundamentally different from asking "how good is the writing on a scale of 1-5."

2. **When the question is taxonomic, not evaluative.** Q3 (CONTENT TYPE) classifies the genre of the content. It is not asking "how good is this?" but "what kind of thing is this?" Categorical classification is a natural task for LLMs and does not suffer from the same calibration problems as quality rating.

3. **When the scale captures genuinely distinct behaviors.** Q8 (APPLICABLE IDEAS: broadly/narrowly/not_really) maps to meaningfully different real-world outcomes: broadly applicable ideas are used frequently, narrowly applicable ideas are used rarely, and not-really-applicable ideas are not used at all. These are not arbitrary gradations of a continuous quality; they describe different states.

4. **"Likert or Not" (Godfrey et al., 2025)** found that the gap between pointwise scoring and listwise ranking shrinks with sufficiently large ordinal label spaces. The paper tested 11-point scales and found they performed competitively with listwise ranking on most model-dataset combinations. Small ordinal scales (3-4 points) with well-defined anchors fall into a sweet spot of reliability.

   Source: [Likert or Not: LLM Absolute Relevance Judgments on Fine-Grained Ordinal Scales](https://arxiv.org/abs/2505.19334)

### Verdict: Mixed Is Defensible, But With Constraints

The literature's binary advocacy is primarily aimed at replacing numeric scales (1-10, 1-100) and Likert scales where adjacent points lack clear semantic boundaries. It does not condemn well-anchored ordinal scales with 3-4 clearly distinguished levels.

**Rule of thumb**: Use binary when the question is about the *presence* of a quality. Use ordinal (3-4 levels max) when the question is about *quantity* or *degree* and each level maps to a meaningfully different real-world state. Use categorical when the question is about *classification* rather than evaluation.

---

## 2. Hierarchical Scoring: Binary Gating Questions

### Concept

A hierarchical or gated scoring system uses binary questions as gates that control whether subsequent, more detailed questions are even asked. The idea is to short-circuit evaluation for content that fails basic thresholds, saving tokens and avoiding meaningless fine-grained scoring of low-quality material.

### Multi-Stage LLM Classification Pipelines

Research on multi-stage LLM pipelines demonstrates that decomposing evaluation into a series of sequential stages with early-exit gates delivers significant improvements in both quality and cost:

- Multi-stage pipelines deliver up to 18.4% increase in agreement (Krippendorff's alpha) over single-model baselines, with up to 90-fold cost savings in large-scale annotation.
- Cascade pipelines with a cheaper model as first-pass filter and a stronger model for detail can reduce cost by ~60% with comparable performance.
- "I don't know" (IDK) gates allow uncertain cases to be escalated while confident cases exit early.

Source: [Multi-Stage LLM Classification Pipeline](https://www.emergentmind.com/topics/multi-stage-llm-based-classification-pipeline)

### How It Applies to Our System

Our Q6 (CONTENT COMPLETENESS) already functions as an implicit gate: if content is truncated or a stub, the entire score is suspect. We formalize this in `_content_is_stub()` and `content_fetch_failed`, but the gating happens in Python code, not in the prompt.

A hierarchical design for our 8 questions might look like:

```
GATE 1: Is the available text complete enough to evaluate?
  NO  -> Flag for rescore, return null score
  YES -> Continue

GATE 2: Is this primarily news, a product announcement, or event recap?
  YES -> Cap score at 20, skip detailed evaluation
  NO  -> Continue

GATE 3: Does the content have substantive depth (not a link roundup or brief)?
  NO  -> Cap score at 30, skip detailed evaluation
  YES -> Evaluate all remaining questions
```

### Trade-offs

**Advantages:**
- Saves tokens on low-value content (potentially 40-60% of articles).
- Prevents the nonsensical situation where a news roundup scores high on "author conviction."
- Makes score capping explicit in the evaluation logic rather than post-hoc.

**Disadvantages:**
- Requires multiple LLM calls or conditional prompting (which most structured output APIs don't support well).
- Introduces discontinuities in the score distribution at gate boundaries.
- A false negative at an early gate permanently blocks a good article from full evaluation.

### Practical Recommendation

Within a single prompt, gating is impractical -- the LLM answers all questions regardless. The benefit comes from a *two-pass architecture* where a cheap first pass gates whether a full evaluation happens. See Section 3.

Within the current single-pass design, we can approximate gating by using multiplicative penalties in the scoring formula (see Section 8 for specific recommendations).

---

## 3. Two-Pass Approaches: Quick Screen, Then Detailed Evaluation

### The Pattern

A two-pass scoring system uses a fast, cheap initial screen to classify content into tiers, then applies detailed (and expensive) evaluation only to content that passes the screen.

```
Pass 1: Quick binary screen (cheap model, few tokens)
  -> Classify as: SKIP / EVALUATE / PRIORITY

Pass 2: Detailed scoring (full model, full prompt)
  -> Only runs on EVALUATE and PRIORITY items
  -> Uses complete 8-question rubric
```

### Evidence from Practice

**Evidently AI** recommends combining rule-based systems for filtering obvious issues with LLM judges for nuanced tasks: "Start with rule-based systems to filter out obvious issues, then use LLM judges or human reviews for more complex, nuanced tasks."

Source: [LLM-as-a-judge: a complete guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)

**Datadog's LLM evaluation framework** advocates for a tiered approach: "use cheaper methods first, and reserve expensive LLM eval for cases that pass some initial filter."

Source: [Building an LLM evaluation framework: best practices](https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/)

### Cost Model

The cost savings depend on the filter rate at Pass 1:

```
Total Cost = cost_pass1 + cost_pass2 * (1 - filter_rate)
```

If Pass 1 uses a smaller/cheaper model and filters 50% of content:
- Pass 1 cost: ~$0.001 per article (short prompt, small model)
- Pass 2 cost: ~$0.01 per article (full prompt, claude-sonnet)
- Total: $0.001 + $0.01 * 0.5 = $0.006 per article
- Savings: ~40% vs. scoring everything with full prompt

### Design for Our System

**Pass 1 prompt** (targeting ~200 tokens output):
```
Is this content worth a detailed evaluation for a reader who values
original analysis, practitioner insights, and actionable frameworks?
Consider: Is it substantive (not a brief, roundup, or stub)? Does it
appear to contain original thinking rather than just reporting?

Respond: SKIP (news, roundups, stubs, announcements) / EVALUATE (substantive content) / PRIORITY (appears exceptional)
Also provide a one-line reason.
```

**Pass 2**: Full 8-question evaluation, only for EVALUATE and PRIORITY items.

**Pass 1 model options**:
- `claude-haiku` or equivalent fast/cheap model
- Same `claude-sonnet` but with a much shorter prompt (still saves on output tokens and evaluation prompt tokens)

### Trade-offs

**Advantages:**
- Significant cost reduction for high-volume scoring
- Fast feedback loop: articles can be immediately categorized as SKIP
- Prevents wasting detailed evaluation on content that will obviously score low

**Disadvantages:**
- Adds latency for good content (two serial calls instead of one)
- False negatives at Pass 1 mean some good articles are never fully evaluated
- More complex orchestration code
- For our current volume (~50-100 articles/day), the cost savings may not justify the complexity

### Practical Recommendation

A two-pass architecture makes sense at scale (hundreds or thousands of items per day). At our current volume, the added complexity is not justified. However, a lightweight version is feasible: use the existing `_content_is_stub()` check as a Pass 0 (already implemented), and consider adding a metadata-only pre-filter (word count < 200, category == "tweet") to skip obviously low-value items before even calling the LLM.

---

## 4. Ensemble Methods: Multiple LLM Calls with Different Formats

### The Concept

Instead of a single evaluation call, make multiple calls with different question formats and aggregate the results. The hypothesis is that different question formats capture different aspects of quality, and aggregation reduces the noise inherent in any single format.

### LLM Juries

**LLM Juries** involve multiple models independently evaluating outputs, with results aggregated through voting:

- **Max pooling** for binary classification
- **Average or median pooling** for rating scales
- **Soft or hard voting** for multi-class classification
- **Minority-veto ensemble**: a few vetoes force an "invalid" label, increasing true negative rate

Research shows that "a diverse panel of smaller models outperforms a single large judge, reduces bias, and does so at over 7x lower cost." LLM Jury-on-Demand systems achieve consistent improvements in detection rates over naive majority voting.

Source: [LLM Juries for Evaluation](https://www.comet.com/site/blog/llm-juries-for-evaluation/)
Source: [Who Judges the Judge? LLM Jury-on-Demand](https://arxiv.org/abs/2512.01786)

### Ensemble Across Formats

A more novel approach: run the *same* model with *different* question formats and aggregate:

| Call | Format | Purpose |
|------|--------|---------|
| Call 1 | 8 binary yes/no questions | High-reliability signal detection |
| Call 2 | 4 ordinal questions (none/few/several/many) | Quantity estimation for graded dimensions |
| Call 3 | 1 holistic Likert score + reasoning | Captures gestalt quality that decomposed questions may miss |

Aggregate: weighted average of the three calls' normalized scores.

### Cost-Benefit Analysis

For our use case, this is almost certainly not worth it:

- 3x the API cost per article
- 3x the latency (if serial) or orchestration complexity (if parallel)
- Research indicates that high pairwise correlation among LLM judgments limits gains from ensembling with the same model
- The diversity benefit comes from using *different* models, not the same model with different prompts

### When Ensembling Makes Sense

- High-stakes evaluation where a single misjudgment is costly
- When you have access to multiple models with different strengths
- When cost is not a primary constraint
- For calibration: occasionally run an ensemble to validate that your primary scorer is not drifting

### Practical Recommendation

Do not implement routine ensembling. Instead, consider occasional ensemble runs as a calibration tool: score a sample of 50-100 articles with both the current v2-categorical format and a proposed v3-binary format, then compare. This gives you confidence that a format change is not degrading quality without committing to ongoing ensemble cost.

---

## 5. Pairwise Comparison + Binary: Calibration via Ranking

### Pairwise Comparison Overview

Pairwise comparison asks the LLM to compare two items and pick the better one, rather than assigning absolute scores. This is fundamentally different from pointwise scoring.

**Recent research (2025)** found conflicting results on pairwise vs. pointwise reliability:
- Pairwise preferences flip in ~35% of cases vs. ~9% for absolute scores
- Yet other studies find pairwise comparisons lead to more stable results with smaller differences from human annotations
- Pairwise protocols are more vulnerable to "distracted evaluation" where spurious features influence judgment

Source: [Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation](https://arxiv.org/abs/2504.14716)

The key limitation: pairwise comparison is not scalable. Comparing N items requires O(N^2) comparisons. For our ~50-100 articles per day, full pairwise ranking would require 1,250-5,000 comparison calls.

### Hybrid: Pairwise for Calibration, Binary for Scoring

A more practical hybrid:

1. **Primary scoring**: Use binary/categorical questions for routine scoring (cheap, scalable).
2. **Periodic calibration**: Take a sample of 20-30 recently scored articles spanning the score range. Run pairwise comparisons to produce a ranking. Compare this ranking with the score-based ranking to detect systematic biases.

If pairwise ranking consistently puts articles that scored 45 above articles that scored 65, the scoring rubric has a bias that needs correction.

### Implementation Sketch

```python
# Calibration prompt
"""Compare these two articles for overall reading value.
Which would a reader who values original analysis and actionable insights
prefer to read?

Article A: {title_a} (score: hidden)
[first 2000 chars of content_a]

Article B: {title_b} (score: hidden)
[first 2000 chars of content_b]

Respond: A_BETTER / B_BETTER / TIE
"""

# Run on 20 random pairs, compute Kendall's tau between
# pairwise ranking and score-based ranking
```

### Practical Recommendation

Pairwise calibration is a useful diagnostic tool but not a production scoring method. Consider running it quarterly on a sample of 20-30 articles to validate that scores correspond to actual quality ordering. Add it to the calibration toolkit alongside the existing `just cal-*` commands.

---

## 6. Confidence-Weighted Binary: Binary + Confidence Level

### The Concept

Ask the LLM to answer binary (yes/no) plus provide a confidence level (e.g., high/medium/low, or a 1-5 scale). The confidence level modulates the weight of the binary answer.

```
Q: Does this article contain a memorable phrase worth highlighting?
A: yes (confidence: high)     -> 8 points (full weight)
A: yes (confidence: medium)   -> 5 points (partial weight)
A: yes (confidence: low)      -> 3 points (reduced weight)
A: no  (confidence: high)     -> 0 points
A: no  (confidence: low)      -> 2 points (hedged "no")
```

### Research on LLM Confidence

Recent research reveals significant challenges with LLM confidence self-assessment:

**Overconfidence is pervasive.** LLM-as-a-Judge systems "exhibit an overconfidence phenomenon where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment."

Source: [Overconfidence in LLM-as-a-Judge](https://arxiv.org/abs/2508.06225)

**Verbalized confidence is poorly calibrated.** The "verbalized method is generally overconfident," meaning when you ask an LLM to express confidence, it systematically overestimates certainty. However, "prompting models to reason through uncertainty with hedging phrases yields more calibrated confidence."

Source: [Confidence-Diversity Calibration](https://arxiv.org/pdf/2508.02029)

**Model-specific behavior varies.** Multi-generation confidence methods (running the same prompt multiple times and measuring consistency) may be underconfident or overconfident depending on the model family.

### Design Considerations

If implementing confidence-weighted binary:

1. **Don't ask for numeric confidence.** LLMs are poorly calibrated for numeric confidence scores. Instead, use categorical: "certain / probable / uncertain."
2. **Treat "uncertain" as signal, not noise.** An uncertain "yes" is still valuable information -- it suggests the quality is borderline, which is exactly where score granularity matters.
3. **Map confidence to weight multipliers, not additive scores:**

```python
CONFIDENCE_MULTIPLIERS = {
    "certain": 1.0,
    "probable": 0.7,
    "uncertain": 0.4,
}

score_contribution = base_weight * CONFIDENCE_MULTIPLIERS[confidence]
```

### Trade-offs

**Advantages:**
- Adds granularity without the full complexity of ordinal scales
- Preserves the reliability of binary judgments as the primary signal
- "Uncertain" responses highlight borderline content for potential human review

**Disadvantages:**
- LLM confidence is systematically overconfident, potentially nullifying the benefit
- Adds prompt complexity and output parsing complexity
- The confidence signal may not carry additional information beyond the binary answer itself
- Testing indicates that binary without confidence achieves comparable or better inter-evaluator agreement than binary-with-confidence

### Practical Recommendation

**Do not implement confidence-weighted binary for production scoring.** The research on LLM overconfidence is damning: the confidence signal is too noisy to reliably modulate scores. Instead, achieve granularity through *more binary questions at different difficulty tiers* (see Section 7).

The one place confidence is useful: **flagging uncertain evaluations for review.** If the LLM expresses low confidence on multiple questions for a single article, flag that article for manual spot-check. This uses confidence as a meta-signal about evaluation quality rather than as a score component.

---

## 7. Graduated Binary: Multiple Binary Questions at Different Thresholds

### The Concept

Instead of a single ordinal question (none/a_few/several/many), decompose it into multiple binary questions at ascending thresholds:

```
Q1a: Does this article contain ANY notable passage worth saving?     (easy)
Q1b: Does this article contain MULTIPLE notable passages?            (medium)
Q1c: Does this article contain MANY notable passages (5+)?           (hard)
```

The sum of "yes" answers naturally produces a 0-3 ordinal score, but each individual judgment is binary. This is the core insight from CheckEval and Item Response Theory applied to our specific question format.

### Why This Works Better Than Direct Ordinal

1. **Each threshold is independently calibratable.** You can measure reliability of "any vs. none" separately from "multiple vs. a few." Binary agreement at each threshold is higher than ordinal agreement across the full scale.

2. **Difficulty tiers emerge naturally.** The "any" question has a high pass rate (~70-80% of substantive articles), while "many" has a low pass rate (~10-20%). This creates the score spread that fights ceiling effects.

3. **The LLM never has to map a subjective impression to an arbitrary scale point.** It only has to answer: "Is this above or below this specific threshold?" This is the same cognitive task as binary, just asked multiple times.

4. **Monotonicity is guaranteed.** If Q1c is "yes," Q1a must also be "yes" (assuming the LLM is consistent). Any violations are detectable and can be corrected.

### Mapping Our Current Ordinal Questions to Graduated Binary

**Q1: STANDALONE PASSAGES (none/a_few/several/many)**

Current: Single 4-level ordinal question.
Graduated binary equivalent:

| Question | Threshold | Expected Pass Rate | Points |
|----------|-----------|--------------------|--------|
| Does this contain ANY passage worth saving as a standalone note? | Low | ~65% | 7 |
| Does this contain at least 2-3 passages worth saving? | Medium | ~35% | 9 |
| Does this contain 5+ passages worth saving? | High | ~12% | 9 |

Total possible: 25 (matches current dimension max).

**Q8: APPLICABLE IDEAS (broadly/narrowly/not_really)**

Current: Single 3-level ordinal question.
Graduated binary equivalent:

| Question | Threshold | Expected Pass Rate | Points |
|----------|-----------|--------------------|--------|
| Could a reader apply ANY idea from this in their work? | Low | ~60% | 5 |
| Could ideas from this apply across multiple domains or roles? | High | ~25% | 8 |

Total possible: 13 (matches current max for `broadly`).

### Consistency Checking

Graduated binary enables automatic consistency validation:

```python
def check_monotonicity(responses: dict) -> list[str]:
    """Flag logically inconsistent graduated binary responses."""
    violations = []
    if responses["many_passages"] and not responses["any_passages"]:
        violations.append("many_passages=yes but any_passages=no")
    if responses["broadly_applicable"] and not responses["any_applicable"]:
        violations.append("broadly_applicable=yes but any_applicable=no")
    return violations
```

If violations are detected, the safer (lower) response can be used, or the article can be flagged for re-evaluation.

### Practical Recommendation

**This is the highest-value change for our current ordinal questions.** Converting Q1 and Q8 from direct ordinal to graduated binary:
- Preserves the score range and point mappings
- Improves reliability by making each judgment binary
- Enables consistency checking
- Fits naturally into the existing prompt format
- Does not require changing the database schema or UI

---

## 8. Analysis of Current Questions: What to Keep, What to Change

### Questions That Should Stay Binary

**Q2: NOVEL FRAMING (true/false)** -- Already binary, well-designed. The concept of "novel framing" is inherently present-or-absent. Keep as-is.

**Q4: AUTHOR CONVICTION (true/false)** -- Already binary, well-designed. An author either argues for a clear position or does not. Keep as-is.

**Q5: PRACTITIONER VOICE (true/false)** -- Already binary, well-designed. First-person professional experience is either present or absent. Keep as-is.

**Q7: NAMED FRAMEWORK (true/false)** -- Already binary, well-designed. A named concept/framework either exists in the text or does not. Keep as-is.

### Questions That Should Convert to Graduated Binary

**Q1: STANDALONE PASSAGES (none/a_few/several/many)** -- Convert to 3 graduated binary questions:
1. "Does this contain ANY passage that works as a standalone saved note?"
2. "Does this contain MULTIPLE (2-3+) such passages?"
3. "Does this contain MANY (5+) such passages?"

Rationale: This is a quantity estimation where the thresholds are clear and testable. Graduated binary gets the same information with higher reliability at each threshold.

**Q8: APPLICABLE IDEAS (broadly/narrowly/not_really)** -- Convert to 2 graduated binary questions:
1. "Could a reader apply ANY specific idea from this in their own work?"
2. "Could ideas from this apply across multiple domains or roles (not just one narrow specialty)?"

Rationale: "Broadly" vs. "narrowly" is ambiguous as a direct question, but decomposed into "any applicability?" and "cross-domain applicability?" each threshold is clear.

### Questions That Should Stay Categorical

**Q3: CONTENT TYPE (5-option categorical)** -- Keep as categorical. This is a classification task, not an evaluation task. The LLM is identifying what kind of content this is, not judging its quality. LLMs are highly reliable at classification tasks, and the 5 options are well-separated taxonomic categories (original_analysis / opinion_with_evidence / informational_summary / product_review / news_or_roundup).

However, consider whether Q3 should be a *scoring* input at all, or just a *metadata* field. The content type scoring (original_analysis = 10, news_or_roundup = 0) encodes a strong prior that original analysis is always more valuable than news. This may be defensible but is a separate question from format choice.

**Q6: CONTENT COMPLETENESS (3-option categorical)** -- Keep as categorical, but reframe as a gatekeeper rather than a scoring input. Completeness is not a quality of the content but a quality of our *access* to the content. A truncated masterpiece is still a masterpiece; we just can't score it. Currently completeness contributes 0-5 points to the argument quality bucket, but it should instead function as a binary gate: complete -> continue scoring, truncated -> flag for rescore.

### Summary of Recommended Changes

| Question | Current Format | Recommended Format | Change |
|----------|---------------|-------------------|--------|
| Q1: Standalone Passages | 4-level ordinal | 3 graduated binary | Convert |
| Q2: Novel Framing | Binary | Binary | Keep |
| Q3: Content Type | 5-option categorical | 5-option categorical | Keep (but review scoring role) |
| Q4: Author Conviction | Binary | Binary | Keep |
| Q5: Practitioner Voice | Binary | Binary | Keep |
| Q6: Content Completeness | 3-option categorical | Binary gatekeeper | Convert to gate |
| Q7: Named Framework | Binary | Binary | Keep |
| Q8: Applicable Ideas | 3-level ordinal | 2 graduated binary | Convert |

Net result: 4 existing binary questions stay, 2 ordinal questions become 5 graduated binary questions, 1 categorical stays, 1 categorical becomes a gate. Total question count goes from 8 to 12 (or 11 if the gate is handled in code rather than the prompt).

---

## 9. Practitioner Guidance on Mixed Approaches

### What the Literature Actually Recommends

The strong advocacy for "all binary" comes from a specific context: replacing 1-5 or 1-10 Likert scales for *subjective quality assessment*. The literature is not as prescriptive about well-designed mixed formats.

**Databricks** recommends "an integer scale of 0-3 or 0-4, with binary grading working well for simple metrics" -- implicitly endorsing mixed approaches where some questions use low-precision ordinal scales.

Source: [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)

**GoDaddy's Rubrics as Rewards** framework organizes binary checklist questions into importance tiers (essential, important, optional, pitfall) and uses both "explicit aggregation" (fixed formulas) and "implicit aggregation" (LLM weighs holistically). This is effectively a mixed approach: binary questions with tiered weighting.

Source: [Calibrating Scores of LLM-as-a-Judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge)

**Evidently AI** summarizes the practical consensus: "You can use a three-option approach, like 'relevant,' 'irrelevant,' and 'partially relevant.'" Three options with clear semantic anchors are still "low precision" enough to maintain reliability.

Source: [LLM-as-a-judge: a complete guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)

### The Real Rule

The actual principle behind "use binary" is: **minimize the cognitive load of each individual judgment.** Binary is the minimum, but the important constraint is not "exactly 2 options" -- it is "each option must be clearly distinguishable with no ambiguous middle ground."

A 3-level scale where each level has a distinct, concrete definition (complete / appears_truncated / summary_or_excerpt) is fine. A 5-level scale where levels 2, 3, and 4 are fuzzy gradations of "medium" is not.

Our current system mostly follows this principle. The main weakness is Q1 (STANDALONE PASSAGES), where the boundary between "a_few" and "several" is genuinely ambiguous. The graduated binary conversion resolves this.

---

## 10. Implementation Recommendation: Evolutionary, Not Revolutionary

### What Not to Do

Do not tear out the entire scoring system and replace it with a pure binary approach. The existing system is functional, the score distributions are understood, and the calibration toolkit is built around the current format. A complete replacement would:
- Invalidate all existing scores and require re-scoring the entire database
- Break the score trend analysis in the calibration tools
- Introduce new question-design bugs while fixing format bugs

### What to Do: Targeted Conversions

Implement changes as a new scoring version (v3-mixed-optimized or similar) that:

1. **Converts Q1 to graduated binary** (3 threshold questions replacing 1 ordinal question)
2. **Converts Q8 to graduated binary** (2 threshold questions replacing 1 ordinal question)
3. **Converts Q6 to a binary gatekeeper** (handled in code, not in the scoring formula)
4. **Keeps Q2, Q4, Q5, Q7 as binary** (unchanged)
5. **Keeps Q3 as categorical** (unchanged)

The point mappings can be adjusted to produce a similar score distribution to v2-categorical, making A/B comparison straightforward.

### Prompt Structure

The modified prompt would look like:

```
Evaluate this article for capture value. Answer each question.

1a. STANDALONE PASSAGES (any): Does this contain ANY passage that works as a
    standalone saved note? Options: true / false
1b. STANDALONE PASSAGES (multiple): Does this contain MULTIPLE (2-3+) such
    passages? Options: true / false
1c. STANDALONE PASSAGES (many): Does this contain MANY (5+) such passages?
    Options: true / false

2. NOVEL FRAMING: Does it reframe a familiar topic or present a surprising lens?
   Options: true / false

3. CONTENT TYPE: What best describes this content?
   Options: original_analysis / opinion_with_evidence / informational_summary
   / product_review / news_or_roundup

4. AUTHOR CONVICTION: Does the author argue for a clear position with conviction?
   Options: true / false

5. PRACTITIONER VOICE: Is this written from first-person practitioner experience?
   Options: true / false

6. CONTENT COMPLETENESS: Does the available text appear to be a complete piece?
   Options: true / false

7. NAMED FRAMEWORK: Does it introduce a named concept, framework, or mental model?
   Options: true / false

8a. APPLICABLE IDEAS (any): Could a reader apply ANY specific idea from this
    in their own work? Options: true / false
8b. APPLICABLE IDEAS (broad): Could ideas from this apply across multiple
    domains or roles? Options: true / false
```

Total questions: 12 (9 binary, 2 graduated-binary pairs sharing a base question, 1 categorical).

### Scoring Formula

```python
# Quotability (max 25)
quotability = (
    (7 if any_passages else 0) +
    (9 if multiple_passages else 0) +
    (9 if many_passages else 0)
)

# Surprise (max 25) -- unchanged
surprise = (
    (NOVEL_FRAMING_POINTS if novel_framing else 0) +
    content_type_scores.get(content_type, 0)
)

# Argument (max 25) -- completeness removed from scoring
argument = (
    (AUTHOR_CONVICTION_POINTS if author_conviction else 0) +
    (PRACTITIONER_VOICE_POINTS if practitioner_voice else 0) +
    (5 if content_complete else 0)  # or remove entirely
)

# Insight (max 25)
insight = (
    (NAMED_FRAMEWORK_POINTS if named_framework else 0) +
    (5 if any_applicable else 0) +
    (8 if broadly_applicable else 0)
)
```

### Migration Path

1. Implement v3 scoring strategy alongside v2 (the `ScoringStrategy` protocol already supports this).
2. Score a sample of 50-100 articles with both v2 and v3.
3. Compare distributions, correlations, and alignment with calibration data.
4. If v3 is an improvement, bump `CURRENT_SCORING_VERSION` and let the rescore mechanism handle migration.

---

## References

- [Testing Binary vs Score Evals on the Latest Models (Arize AI, 2025)](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/)
- [Evaluating the Effectiveness of LLM-Evaluators (Eugene Yan)](https://eugeneyan.com/writing/llm-evaluators/)
- [Using LLM-as-a-Judge For Evaluation (Hamel Husain)](https://hamel.dev/blog/posts/llm-judge/)
- [LLM-as-a-judge: a complete guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [CheckEval: Reliable LLM-as-a-Judge framework (Lee et al.)](https://arxiv.org/abs/2403.18771)
- [Likert or Not: LLM Absolute Relevance Judgments (Godfrey et al., 2025)](https://arxiv.org/abs/2505.19334)
- [Pairwise or Pointwise? Evaluating Feedback Protocols (2025)](https://arxiv.org/abs/2504.14716)
- [Multi-Stage LLM Classification Pipeline](https://www.emergentmind.com/topics/multi-stage-llm-based-classification-pipeline)
- [LLM Juries for Evaluation (Comet)](https://www.comet.com/site/blog/llm-juries-for-evaluation/)
- [Who Judges the Judge? LLM Jury-on-Demand](https://arxiv.org/abs/2512.01786)
- [Overconfidence in LLM-as-a-Judge (2025)](https://arxiv.org/abs/2508.06225)
- [Confidence-Diversity Calibration](https://arxiv.org/pdf/2508.02029)
- [Calibrating Scores of LLM-as-a-Judge (GoDaddy RaR)](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge)
- [Best Practices for LLM Evaluation of RAG Applications (Databricks)](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
- [Building an LLM evaluation framework (Datadog)](https://www.datadoghq.com/blog/llm-evaluation-framework-best-practices/)
- [Using LLMs for Evaluation (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- [Harnessing Multiple Large Language Models: A Survey on LLM Ensemble](https://arxiv.org/html/2502.18036)
