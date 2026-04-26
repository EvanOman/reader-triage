# Weighting Strategies for Binary Scoring

## Overview

With binary questions, weighting determines how much each "yes" or "no" contributes to the final score. The simplest approach (equal weights, final score = percentage of "yes" answers) is a reasonable baseline, but calibrated weights can significantly improve discrimination and correlation with actual engagement.

---

## Strategy 1: Equal Weights (Baseline)

**How it works**: Each question contributes equally. Final score = (number of "yes" on positive questions - number of "yes" on negative questions) / total questions, scaled to 0-100.

**Advantages**:
- Simplest to implement and explain
- No historical data needed
- Used successfully by CheckEval and Criteria-Eval
- Good starting point before calibration data exists

**Disadvantages**:
- Assumes all questions are equally predictive of engagement
- No way to emphasize what matters most
- Negative questions need special handling

**When to use**: As the initial deployment strategy before enough engagement data exists to calibrate.

---

## Strategy 2: Expert-Assigned Weights

**How it works**: A domain expert assigns weights to each question based on their judgment of importance. This is the approach used in traditional weighted rubrics and in GoDaddy's "Rubrics as Rewards" framework.

### Tiered Importance Model (from GoDaddy RaR)

Organize questions into importance tiers:

| Tier | Weight | Purpose | Example Questions |
|------|--------|---------|-------------------|
| **Essential** | 8-10 pts | Core value signals; failing these is a strong negative | Standalone passages exist, Expert would learn something new |
| **Important** | 5-7 pts | Strong quality indicators | Author conviction, Practitioner voice, Named framework |
| **Supplementary** | 3-4 pts | Nice-to-have quality markers | Direct expert quote, Specific data point |
| **Gatekeeper** | Special | Binary pass/fail that gates the entire score | Content completeness (if "no," flag for rescore instead of scoring 0) |
| **Penalty** | -3 to -8 pts | "Yes" reduces score | Summary of known ideas, Fits in a tweet, News/announcement |

### Example Weight Assignment

For a 20-question system targeting 0-100 scale:

**Positive questions (16 questions, max ~108 points before penalties)**:
- 4 Essential questions at 8 points each = 32 pts
- 6 Important questions at 6 points each = 36 pts
- 5 Supplementary questions at 4 points each = 20 pts
- 1 Gatekeeper (special handling, not scored)

**Negative questions (4 questions, max penalty ~-24 points)**:
- 2 Strong penalties at -8 points each = -16 pts
- 2 Moderate penalties at -4 points each = -8 pts

**Score range**: Theoretical 0 to 108, clamped to 0-100. A "typical decent article" that passes most baseline and some quality questions but few exceptional ones might score 40-55. An exceptional article passing most questions might score 70-85. Only truly outstanding articles would break 85.

**Advantages**:
- Encodes domain knowledge about what matters
- Can be tuned quickly based on inspection of results
- No historical data required

**Disadvantages**:
- Subjective -- different experts would assign different weights
- May not correlate well with actual user engagement
- Requires iteration and manual tuning

---

## Strategy 3: Data-Calibrated Weights (Logistic Regression)

**How it works**: Use historical engagement data (highlights in Readwise) to learn which binary question responses best predict actual user engagement. This is the most rigorous approach.

### Process

1. **Collect training data**: Score a set of articles (ideally 200+) with the binary questions. For each article, record the yes/no vector and the actual highlight count.

2. **Define the target variable**: Binary engagement outcome, e.g.:
   - `engaged = 1` if highlight_count >= 2 (or some threshold)
   - `engaged = 0` otherwise

3. **Fit logistic regression**: Use the binary question responses as features, engagement as target.
   ```
   P(engaged) = sigmoid(w1*q1 + w2*q2 + ... + w20*q20 + bias)
   ```
   The learned weights `w1...w20` become the question weights. Negative weights naturally emerge for questions that anti-correlate with engagement.

4. **Convert to scoring**: Transform the logistic regression coefficients into a point-based scoring system. Scale so that the maximum achievable score maps to 100.

### Practical Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# X = matrix of binary question responses (n_articles x n_questions)
# y = binary engagement labels (n_articles,)

model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y)

# The coefficients are the learned weights
weights = model.coef_[0]  # shape: (n_questions,)

# Scale to a 0-100 scoring system
# For a new article with binary responses `q`:
raw_score = np.dot(weights, q) + model.intercept_[0]
# Map to 0-100 using sigmoid and scaling
score = int(100 * sigmoid(raw_score))
```

### Regularization Considerations

- **L2 regularization** (Ridge): Shrinks all weights toward zero, good when questions are correlated
- **L1 regularization** (Lasso): Can zero out unimportant questions, useful for feature selection
- **ElasticNet**: Combination, often the best default

### Sample Size Requirements

- Minimum viable: ~100 articles with engagement data (50+ with highlights, 50+ without)
- Recommended: 200+ articles for stable weight estimates
- Eugene Yan recommends: "at least 50-100 failures out of 200+ total samples"

**Advantages**:
- Weights are empirically grounded in actual user behavior
- Automatically discovers which questions are most predictive
- Can reveal that some questions are noise (weight near zero)
- Can discover negative-signal questions automatically

**Disadvantages**:
- Requires substantial historical data
- Risk of overfitting to a specific user's preferences
- Weights may need periodic recalibration as reading habits change
- Cold-start problem: can't use until enough articles are scored with the new system

---

## Strategy 4: Hierarchical / Gatekeeper Weights

**How it works**: Some questions serve as "gates" that must be passed before other questions are even considered. This creates a hierarchical scoring structure.

### Implementation Pattern

```
IF NOT q17_complete_content:
    flag_for_rescore()
    return None  # Don't score incomplete content

IF q20_is_news_recap AND q8_is_summary_of_known:
    return max(20, regular_score)  # Cap at 20 for commodity content

# Otherwise, compute regular weighted score
score = weighted_sum(all_questions)
```

### Gatekeeper Examples

| Gate | Condition | Action |
|------|-----------|--------|
| Content completeness | Q17 = "no" | Skip scoring, flag for rescore |
| Commodity content | Q20 = "yes" AND Q8 = "yes" | Cap score at 20 |
| Zero quotability | Q1-Q4 all "no" | Cap score at 50 (no memorable passages = limited value) |

**Advantages**:
- Prevents pathological scores (e.g., news roundup scoring 80)
- Encodes hard business rules about what content is worth reading
- Simple to reason about and debug

**Disadvantages**:
- Can create score discontinuities
- Harder to calibrate with regression
- Risk of over-engineering edge cases

---

## Strategy 5: Bayesian Updating / Online Learning

**How it works**: Start with expert-assigned prior weights, then update them incrementally as engagement data arrives.

### Process

1. Start with expert weights as priors (Strategy 2)
2. After each article is scored and engagement is observed, update weights using Bayesian online learning
3. Weights gradually shift toward the data-calibrated optimum

### Simplified Implementation

```python
# Initialize with expert priors
weights = expert_weights.copy()
learning_rate = 0.01

for article in scored_articles_with_engagement:
    predicted = sigmoid(np.dot(weights, article.binary_responses))
    actual = 1.0 if article.highlight_count >= 2 else 0.0
    error = actual - predicted

    # Gradient update
    for i, q_response in enumerate(article.binary_responses):
        weights[i] += learning_rate * error * q_response
```

**Advantages**:
- Smooth transition from expert to data-driven weights
- Adapts over time as reading preferences evolve
- No need to wait for large dataset

**Disadvantages**:
- More complex to implement correctly
- Risk of instability with small data
- Harder to debug and explain

---

## Recommended Approach: Phased Weighting

Given the codebase's existing calibration toolkit, a phased approach is most practical:

### Phase 1: Expert-Assigned Weights (Immediate)

Deploy with hand-tuned weights based on the tiered model:

```python
QUESTION_WEIGHTS = {
    # Quotability (max ~24 pts)
    "q1_memorable_phrase": 5,
    "q2_specific_data": 5,
    "q3_standalone_passage": 8,     # Essential
    "q4_expert_quote": 4,

    # Surprise (max ~24 pts)
    "q5_contradicts_assumptions": 8, # Essential
    "q6_unexpected_lens": 6,
    "q7_novel_example": 6,
    "q8_summary_of_known": -6,       # Penalty

    # Argument (max ~24 pts)
    "q9_clear_position": 6,
    "q10_concrete_evidence": 6,
    "q11_practitioner_experience": 8, # Essential
    "q12_fits_in_tweet": -6,          # Penalty

    # Insight (max ~24 pts)
    "q13_named_framework": 8,         # Essential
    "q14_applicable_technique": 6,
    "q15_actionable_detail": 5,
    "q16_narrow_audience": -4,        # Moderate penalty

    # Cross-cutting (max ~16 pts)
    "q17_complete_content": None,     # Gatekeeper, not scored
    "q18_beyond_search": 6,
    "q19_expert_learns": 8,           # Essential
    "q20_news_announcement": -8,      # Strong penalty
}
```

Maximum possible score: ~108 (from positive questions). With typical penalties, realistic max ~85-95. Scale/clamp to 0-100.

### Phase 2: Calibration Analysis (After ~100 articles scored)

Use the existing `just cal-*` tools to:
1. Correlate each binary question's responses with highlight counts
2. Identify questions with zero or negative correlation (candidates for removal or weight adjustment)
3. Run logistic regression to get data-suggested weights
4. Compare data-suggested vs. expert weights

### Phase 3: Data-Calibrated Weights (After ~200+ articles)

Switch to logistic-regression-derived weights, with manual overrides for business logic (gatekeepers, hard caps).

### Phase 4: Ongoing Recalibration

Run calibration analysis monthly using `just cal-dimensions` and `just cal-report`. Adjust weights when correlation with engagement degrades.

---

## Handling Negative Questions

Three approaches to negative (penalty) questions:

### Approach A: Subtract from Total
```python
score = sum(weight * response for q, weight, response in positive_qs)
score -= sum(abs(weight) * response for q, weight, response in negative_qs)
score = max(0, min(100, score))
```

### Approach B: Multiplicative Penalty
```python
base_score = sum(weight * response for q, weight, response in positive_qs)
penalty_factor = 1.0
for q, weight, response in negative_qs:
    if response:  # "yes" on a negative question
        penalty_factor *= (1.0 - abs(weight) / 100)
score = base_score * penalty_factor
```

### Approach C: Integrated Weights (Recommended)
Negative weights are simply negative numbers in the weight vector. The final score naturally decreases when negative questions are answered "yes." This is what logistic regression would produce and is simplest to implement.

```python
raw = sum(weights[q] * (1 if responses[q] else 0) for q in all_questions)
score = max(0, min(100, raw))
```

---

## Dimension Sub-Scores

To maintain backward compatibility with the 4-dimension structure (quotability, surprise, argument, insight), compute sub-scores per dimension:

```python
dimension_scores = {}
for dim, questions in dimension_question_mapping.items():
    dim_weights = {q: weights[q] for q in questions}
    dim_max = sum(w for w in dim_weights.values() if w > 0)
    dim_raw = sum(dim_weights[q] * responses[q] for q in questions)
    dimension_scores[dim] = max(0, min(25, int(25 * dim_raw / dim_max)))
```

This maps each dimension's weighted binary responses to a 0-25 sub-score, preserving the existing DB schema and UI display.
