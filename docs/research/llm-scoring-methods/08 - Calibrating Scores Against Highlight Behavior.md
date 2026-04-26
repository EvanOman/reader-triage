# Calibrating Scores Against Highlight Behavior

Using reader highlight behavior as ground truth for an LLM-based content scoring system.

## Context

We have an LLM-based article scoring system that predicts content value across four dimensions (Quotability, Surprise Factor, Argument Quality, Applicable Insight), each scored 0-25 for a total of 0-100. The system uses Claude to evaluate articles and assign categorical rubric responses that map to numeric scores.

The question: **How well do these predicted scores correspond to actual reader engagement, measured by highlighting behavior?**

Our existing calibration toolkit (`tools/calibrate.py`) computes Spearman correlation between `info_score` and `num_highlights`, performs dimension analysis, and identifies false positives/negatives. This document examines whether that approach is sound and how to strengthen it.

---

## 1. Choosing the Dependent Variable: Highlight Count vs. Highlighted Word Count vs. Highlight Density

### 1.1 Highlight Count (number of passages saved)

**Pros:**
- Simple, intuitive, and directly available from the Readwise API.
- Each highlight represents a discrete decision by the reader -- "this passage is worth saving." That deliberate act of selection is closer to what our scoring system tries to predict (capture value).
- Less sensitive to highlight style differences. Whether someone highlights a sentence or a paragraph, it counts as one engagement event.
- Already implemented in our calibration toolkit.

**Cons:**
- Conflates articles where a reader made 3 short highlights vs. 3 very long highlights. The latter may indicate deeper engagement.
- Longer articles naturally provide more opportunities for highlights, creating a length bias. A 10,000-word article with 5 highlights might represent less impressive density than a 2,000-word article with 5 highlights.
- Does not capture the "weight" of each highlight. A single highlighted paragraph may contain more value than three highlighted sentences.

### 1.2 Highlighted Word Count (total words across all highlights)

**Pros:**
- Captures the volume of content the reader found worth preserving.
- Partially reflects depth of engagement -- highlighting long passages suggests sustained interest rather than just noting a quick quote.

**Cons:**
- Highly sensitive to individual highlighting style. Some readers highlight single sentences; others highlight entire paragraphs or sections. This introduces noise that has nothing to do with content quality.
- Research on reading comprehension shows that when students highlight too much text, it actually indicates less selective engagement -- they are not discriminating between key and peripheral content. Over-highlighting is a known marker of shallow processing.
- More complex to obtain from the API (requires fetching and processing individual highlight texts rather than just counts).
- The relationship to content quality is less clear: is an article where someone highlighted 500 words "better" than one where they highlighted 200 carefully chosen words?

### 1.3 Highlight Density (highlights per 1,000 words)

**Pros:**
- Controls for article length, which is the biggest confounder when using raw highlight count. A 1,500-word article with 4 highlights (2.67/1k words) clearly engaged the reader more densely than a 15,000-word article with 4 highlights (0.27/1k words).
- Creates a more comparable metric across articles of varying length.
- Aligns well with the scoring system's design, which already evaluates content quality independent of length.

**Cons:**
- Very short articles (under 500 words) can produce extreme density values from a single highlight (e.g., 2.0 per 1k words), creating noise at the low end.
- Requires `word_count` to be available and accurate. Our data shows some articles have null or unreliable word counts.
- May undervalue long-form pieces where the valuable content is concentrated in certain sections. A 10,000-word article where someone highlighted 3 passages in one brilliant section might score low on density despite being highly valuable.
- Introduces a denominator problem: articles with missing or very low word counts need to be handled carefully.

### 1.4 Recommendation

**Use highlight count as the primary metric, with highlight density as a secondary/confirmatory metric.**

Rationale:
- Highlight count is the most robust single signal. Each highlight is a deliberate "this is worth saving" decision, which directly maps to what our scoring rubric evaluates (quotability, surprise, argument quality, insight).
- Run correlations against both highlight count and highlight density. If they agree in direction and significance, confidence in findings is higher. If they diverge, investigate whether article length is driving the discrepancy.
- For articles where `word_count` is available, compute density as `num_highlights / (word_count / 1000)` and include it as a secondary dependent variable.
- Filter out articles with `word_count < 300` when using density, to avoid extreme values from very short content.

**Implementation note for our toolkit:** The current approach of using raw `num_highlights` is reasonable. Consider adding a `highlight_density` column to `load_dataset()` in `tools/cal_data.py` and reporting correlation against both metrics in `cal_report.py`.

---

## 2. The Zero-Inflation Problem

### 2.1 Nature of the Problem

In a typical reading workflow, the majority of articles will have 0 highlights. This happens for multiple distinct reasons:

1. **Genuinely low-quality content** -- the article was not worth highlighting (true negative).
2. **Unread articles** -- the reader never opened or finished the article, so had no opportunity to highlight (missing data, not a signal of quality).
3. **Read but not highlighted** -- the reader consumed the content but does not highlight habitually, or the content was good but not "highlightable" (e.g., a narrative essay that was enjoyable but did not produce quotable passages).
4. **Platform effects** -- the reader consumed the content outside of Readwise Reader (e.g., on the original website) and never had the opportunity to highlight.

These are fundamentally different processes generating the same observed value of 0, which is exactly the scenario described by zero-inflated models in the statistics literature.

### 2.2 Why This Matters for Calibration

If we treat all zeros as "article was not worth highlighting," we systematically penalize our scoring model for articles that were simply never read. A high-scoring article that the reader never opened will appear as a false positive, dragging down the correlation coefficient even though the scoring model may have been correct.

Our existing toolkit already partially addresses this with the `--min-progress` flag, but the default is 0.0 (include all), and the recommended value in the docs is only 0.1. This is a good start but may not be sufficient.

### 2.3 Approaches to Handle Zero-Inflation

#### Approach A: Filter on Reading Progress (Current Approach)

Filter out articles below a reading progress threshold before computing correlations.

- **Threshold of 0.0 (current default):** Includes all articles. Problematic because unread articles inflate the zero count.
- **Threshold of 0.1:** Filters out articles that were barely opened. Removes the most obvious "never read" cases but still includes articles where the reader bounced early.
- **Threshold of 0.5:** Ensures the reader engaged with at least half the article. More conservative but loses sample size.
- **Threshold of 0.9 or 1.0:** Restricts to "fully read" articles. Most accurate for calibration but may have a small sample size.

**Recommendation:** Use `--min-progress 0.5` as the default for calibration reports. This balances signal quality against sample size. Run sensitivity analysis at 0.1, 0.5, and 0.9 thresholds to see if the correlation changes meaningfully.

#### Approach B: Hurdle Model (Two-Part Model)

A hurdle model separates the analysis into two stages:

**Part 1 -- Binary model:** Did the reader highlight at all? (0 vs. 1+)
This can be modeled with logistic regression: `P(highlights > 0) ~ info_score + reading_progress + word_count`.

**Part 2 -- Count model (conditional on highlighting):** Given that the reader highlighted, how many highlights did they make?
This can be modeled with a truncated Poisson or negative binomial regression: `num_highlights | highlights > 0 ~ info_score + dimensions`.

The hurdle model is conceptually the right fit for our data because the process that determines whether someone highlights at all (did they read it? are they a highlighter?) is different from the process that determines how many highlights they make once they start (how much quotable content was there?).

**Implementation recommendation:** This is more complex than simple correlation but would provide much richer insights. Use the `statsmodels` library which is already a dependency in our toolkit:

```python
# Part 1: Binary -- did they highlight?
import statsmodels.api as sm
df['highlighted'] = (df['num_highlights'] > 0).astype(int)
logit = sm.Logit(df['highlighted'], sm.add_constant(df[['info_score', 'reading_progress', 'word_count']]))
logit_result = logit.fit()

# Part 2: Count -- how many, among those who highlighted?
highlighted_df = df[df['num_highlights'] > 0]
# Use negative binomial for overdispersion
from statsmodels.discrete.count_model import NegativeBinomial
nb = NegativeBinomial(highlighted_df['num_highlights'],
                      sm.add_constant(highlighted_df[['info_score', 'reading_progress']]))
nb_result = nb.fit()
```

#### Approach C: Binarize the Dependent Variable

Convert highlights to a binary variable (0 = no highlights, 1 = any highlights) and use point-biserial correlation or logistic regression. This sidesteps the zero-inflation problem entirely.

**Pros:** Simple, avoids the distributional issues of count data.
**Cons:** Loses information about the degree of engagement. An article with 1 highlight is treated the same as one with 15 highlights.

This is useful as a quick sanity check (does our scoring model predict whether an article gets highlighted at all?) but should not replace count-based analysis.

### 2.4 Recommendation

Use a layered approach:
1. **Primary analysis:** Spearman correlation with `min_progress >= 0.5` (addresses the most common source of structural zeros).
2. **Secondary analysis:** Hurdle model that formally separates the "did they read/highlight" decision from the "how many highlights" count.
3. **Sensitivity check:** Binarized analysis (highlighted vs. not) to verify the scoring model at least predicts the binary outcome.

---

## 3. Reading Progress as a Confounder

### 3.1 The Confounding Mechanism

Reading progress affects both the likelihood of highlighting (you can't highlight what you haven't read) and is potentially correlated with content quality (readers may finish high-quality articles more often). This makes it a classic confounder: it is associated with both the predictor (article quality/score) and the outcome (highlight count), and it is not on the causal pathway between score and highlights (the score is assigned before the reader sees the article).

However, reading progress has a nuanced relationship:
- It is partly a **mediator**: high-quality content causes more reading, which causes more highlighting. If we control for reading progress too aggressively, we may remove the very signal we are trying to measure.
- It is partly a **confounder**: articles in the "later" queue may have low reading progress regardless of quality, simply because the reader has not gotten to them yet.

### 3.2 Should We Only Calibrate Against Fully-Read Articles?

**Arguments for restricting to fully-read (progress >= 0.9):**
- Eliminates the "never read" problem completely.
- Ensures the reader had full opportunity to highlight, so 0 highlights truly means "nothing worth saving."
- Produces the cleanest signal for calibration.

**Arguments against:**
- Dramatically reduces sample size. If only 30% of articles are fully read, we lose 70% of our data.
- Selection bias: fully-read articles may not be representative. Readers may finish articles for reasons unrelated to quality (short articles, obligation, topic interest). The articles that get finished may already be biased toward higher scores, compressing the score range and reducing correlation power.
- In production, the scoring system needs to work for all articles, not just fully-read ones.

### 3.3 Recommended Strategy

**Stratified analysis by reading progress bands:**

| Band | Progress Range | Purpose |
|------|---------------|---------|
| Unread | 0.0 - 0.05 | Exclude from calibration entirely |
| Started | 0.05 - 0.30 | Include with caution; note that zero highlights may mean "stopped reading" |
| Substantial | 0.30 - 0.80 | Include; reasonable expectation that reader had opportunity to highlight |
| Completed | 0.80 - 1.00 | Primary calibration set; cleanest signal |

**Implementation:** Add reading progress as a covariate in regression models rather than just using it as a filter. This lets the model account for its effect without discarding data:

```python
# Include reading_progress as a covariate, not just a filter
X = sm.add_constant(df[['info_score', 'reading_progress']])
y = df['num_highlights']
model = sm.OLS(y, X).fit()
# Check: does info_score remain significant after controlling for reading_progress?
```

Run the primary calibration report at two thresholds:
1. `--min-progress 0.5` for the main correlation (good balance).
2. `--min-progress 0.8` for a confirmatory analysis on completed articles.

If the correlation is notably different between these two groups, reading progress is acting as a significant confounder and needs to be modeled explicitly.

---

## 4. Correlation Metrics: Spearman vs. Pearson vs. Kendall Tau

### 4.1 Pearson Correlation (r)

**What it measures:** Linear association between two continuous variables.

**Assumptions:** Both variables are continuous, normally distributed (for inference), and the relationship is linear.

**Appropriateness for our data:** Poor as a primary metric.
- Highlight counts are discrete, zero-inflated, and right-skewed -- violating normality assumptions.
- The relationship between scores and highlights is unlikely to be strictly linear. We expect a monotonic trend (higher scores correspond to more highlights) but not necessarily a proportional one.
- Sensitive to outliers. A few articles with extreme highlight counts (20+) can dominate the correlation.

**When to use it:** As a secondary metric alongside Spearman for comparison. If Pearson and Spearman diverge significantly, it suggests the relationship is monotonic but non-linear, which is useful diagnostic information. Also appropriate for the dimension redundancy analysis (correlating dimension scores with each other), since dimension scores are more nearly continuous.

### 4.2 Spearman Rank Correlation (rho)

**What it measures:** Monotonic association between two variables, based on ranks rather than raw values.

**Assumptions:** Both variables are at least ordinal. No normality or linearity assumptions.

**Appropriateness for our data:** Good as the primary metric. This is the right default choice.
- Works well with ordinal/ranked data, which is essentially what our scoring system produces (articles ordered by predicted quality).
- Robust to outliers since it operates on ranks.
- Handles non-linear monotonic relationships (e.g., a logarithmic relationship between score and highlights).
- Well-understood, easy to interpret, and already implemented in our toolkit.

**Limitations:**
- With many tied values (common in our data: many articles have 0 highlights, many have scores clustered at certain rubric breakpoints), Spearman can be affected. However, the correction for ties is applied by default in `scipy.stats.spearmanr`.
- Does not distinguish between different types of monotonic relationships (e.g., whether the relationship is stronger at the top or bottom of the scale).

### 4.3 Kendall Tau (tau-b)

**What it measures:** Proportion of concordant vs. discordant pairs, adjusted for ties.

**Assumptions:** Same as Spearman -- ordinal data, no distributional assumptions.

**Appropriateness for our data:** Viable alternative to Spearman, with specific advantages for tied data.

**Advantages over Spearman:**
- More robust with small sample sizes.
- Handles tied ranks more gracefully (Kendall tau-b explicitly adjusts for ties in the denominator).
- Has a more natural probabilistic interpretation: tau = P(concordant) - P(discordant). A tau of 0.30 means that if you randomly pick two articles, the one with the higher score has a 65% chance of also having more highlights (since (1 + 0.30) / 2 = 0.65).
- Confidence intervals tend to be narrower and more reliable than Spearman's.

**Disadvantages:**
- Produces lower absolute values than Spearman for the same data (typically tau ~= 2/3 * rho), which can be confusing if comparing to literature that reports Spearman correlations.
- More computationally expensive for large datasets (O(n^2) vs. O(n log n) for Spearman), though this is irrelevant for datasets of 500-1000 articles.
- Less widely recognized in applied research, making it harder to benchmark against other systems.

**Relevant research finding:** Studies on the effective use of rank correlations show that when there are ties in the data, Spearman's measure returns values closer to the desired confidence interval coverage rates, while Kendall's results can differ more from the desired level as the number of ties increases. However, other research shows Kendall tau-b is more robust specifically because it adjusts for ties -- the findings are somewhat contradictory and depend on the nature of the ties.

### 4.4 Recommendation for Our System

**Report Spearman as the primary metric, Kendall tau-b as a secondary metric, and Pearson as a diagnostic.**

Rationale:
- Spearman is the most appropriate primary metric: it handles our ordinal, non-normal data without assumptions, is widely understood, and is already implemented.
- Kendall tau-b provides a useful cross-check and has a cleaner probabilistic interpretation. Its concordance-based framing ("65% of article pairs are correctly ranked by our scoring model") is particularly intuitive for communicating calibration quality.
- Pearson is useful for diagnosing non-linearity (if Spearman >> Pearson, the relationship is monotonic but not linear).

**Interpretation thresholds for our context:**

| Spearman rho | Kendall tau-b (approx.) | Interpretation | Action |
|-------------|------------------------|----------------|--------|
| >= 0.50 | >= 0.35 | Strong | Model is well-calibrated; maintain |
| 0.30 - 0.49 | 0.20 - 0.34 | Moderate | Acceptable; look for dimension-specific improvements |
| 0.15 - 0.29 | 0.10 - 0.19 | Weak | Prompt revision needed; examine false positives/negatives |
| < 0.15 | < 0.10 | Negligible | Model is not predictive; major rubric redesign needed |

**Implementation addition for `cal_report.py`:**

```python
from scipy.stats import kendalltau

def _safe_kendall(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < MIN_CORR_N:
        return None
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return None
    tau, p = kendalltau(valid["x"], valid["y"])
    return float(tau), float(p)
```

---

## 5. Sample Size and Power Analysis

### 5.1 How Many Articles Are Needed for Stable Calibration?

The answer depends on the expected effect size (correlation strength) and the desired precision.

**Minimum sample size for detecting a significant Spearman correlation (alpha=0.05, power=0.80):**

| Expected rho | Required n |
|-------------|-----------|
| 0.50 (strong) | 29 |
| 0.40 (moderate-strong) | 46 |
| 0.30 (moderate) | 84 |
| 0.20 (weak-moderate) | 193 |
| 0.15 (weak) | 346 |
| 0.10 (very weak) | 782 |

These numbers come from the standard power analysis formula for correlation (which applies to Spearman via its equivalence to Pearson on ranked data): n = ((z_alpha + z_beta) / arctanh(rho))^2 + 3.

### 5.2 Stability vs. Detection

There is an important distinction between having enough data to **detect** a statistically significant correlation and having enough data for the estimate to be **stable** (i.e., not fluctuating wildly if you add or remove a few articles).

For stability (95% CI width of approximately 0.2 around the estimated rho):
- rho ~= 0.30: need approximately 100-150 articles
- rho ~= 0.20: need approximately 200-300 articles

For a narrower CI (width of 0.1):
- rho ~= 0.30: need approximately 400+ articles
- rho ~= 0.20: need approximately 1,000+ articles

### 5.3 Assessment of Our Dataset

With 500+ scored articles, our dataset is adequate for:
- Detecting moderate correlations (rho >= 0.20) with high power.
- Producing reasonably stable estimates at the overall level.
- Stratified analysis by broad categories (e.g., article vs. email) if each stratum has 100+ articles.

Our dataset is **marginal** for:
- Tag-level analysis. With the 10-article minimum in the current toolkit, we are underpowered for detecting anything short of very strong correlations. The current threshold of 10 articles per tag is too low for meaningful correlation; consider raising to 30+ for any correlation-based analysis.
- Monthly trend analysis. A 30-day rolling window may contain only 15-40 articles, which is underpowered for detecting weak-to-moderate correlations. The `run_trends()` function's current `MIN_CORR_N = 5` threshold is far too low for reliable correlation estimates -- those early windows will produce highly unstable rho values. Consider raising to 20-30 minimum.
- Detecting small effects (rho < 0.15), which may require 500+ articles after filtering for reading progress.

### 5.4 Bootstrapping for Confidence Intervals

Rather than relying on asymptotic formulas, use bootstrap resampling to get empirical confidence intervals for the correlation. This is more appropriate for our non-normal, zero-inflated data:

```python
import numpy as np
from scipy.stats import spearmanr

def bootstrap_spearman(x, y, n_bootstrap=1000, ci=0.95):
    """Compute Spearman rho with bootstrap confidence interval."""
    rhos = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        rho, _ = spearmanr(x.iloc[idx], y.iloc[idx])
        rhos.append(rho)
    rhos = sorted(rhos)
    lower = rhos[int((1 - ci) / 2 * n_bootstrap)]
    upper = rhos[int((1 + ci) / 2 * n_bootstrap)]
    return np.median(rhos), lower, upper
```

**Recommendation:** Add bootstrap CIs to the calibration report alongside p-values. A bootstrap CI is more informative than a p-value for ongoing monitoring because it tells you the range of plausible correlation values, not just whether it is distinguishable from zero.

---

## 6. Temporal Effects

### 6.1 Sources of Temporal Drift

Reading and highlighting habits change over time for several reasons:

1. **Reader behavior changes.** The reader may go through phases of heavy highlighting vs. light highlighting, independent of content quality. A vacation, a busy work period, or a change in reading tools all affect engagement.
2. **Content mix changes.** If the reader subscribes to new sources or drops old ones, the type of content entering the system changes.
3. **Scoring model changes.** Prompt updates, model version changes (e.g., switching Claude model versions), or rubric modifications all create discontinuities in the score distribution.
4. **Seasonal effects.** Holiday periods, year-end reviews, or conference seasons can affect both reading volume and highlighting intensity.
5. **Highlight backfill.** Highlights may be added to articles days or weeks after initial reading, meaning the `num_highlights` for recent articles may be artificially low (right-censoring).

### 6.2 Handling Temporal Effects

#### Time-Windowed Analysis (Already Implemented)

The `run_trends()` function already computes rolling Spearman correlation over time windows. This is the right approach. Improvements:

- **Increase the minimum window size.** The current `MIN_CORR_N = 5` is too small. Use at least 20-30 articles per window for any correlation to be meaningful.
- **Track the trend of trends.** The current code checks whether the rolling correlation is improving or degrading, which is good. Consider adding an alert threshold: if the most recent window's correlation drops below 0.15, flag the model for review.

#### Lag-Adjusted Analysis

Highlights are often added days or weeks after an article is read. Consider excluding articles scored in the last 30 days from calibration, to allow time for highlights to accumulate:

```python
cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
df_settled = df[df['scored_at'] < cutoff]
```

This "settlement period" prevents right-censoring from deflating engagement metrics for recently scored articles.

#### Scoring Version Tracking

The existing `scoring_version` column in the database is valuable. When computing calibration metrics, always stratify by scoring version. If the system transitions from v1 to v2, mixing articles scored under different rubrics will produce misleading aggregate correlations.

```python
for version, group in df.groupby('scoring_version'):
    rho = _safe_spearman(group['info_score'], group['num_highlights'])
    print(f"Version {version}: rho = {rho}")
```

#### Detrending

If both scores and highlights show temporal trends (e.g., both increasing over time), the correlation between them may be inflated by the shared trend rather than reflecting a genuine article-level relationship. To check for this:

1. Compute the month-level residuals for both scores and highlights (subtract each month's mean).
2. Correlate the residuals. If the residual correlation is much weaker than the raw correlation, the apparent calibration is partly driven by shared temporal trends.

### 6.3 Recommendation

1. Add a 30-day settlement period for recent articles in calibration reports.
2. Raise `MIN_CORR_N` to 25 in `cal_report.py` for rolling window analysis.
3. Stratify by `scoring_version` whenever computing aggregate metrics.
4. Add a detrending check: compute both raw and detrended correlations, and flag if they diverge by more than 0.10.

---

## 7. Evaluation Protocol: Data Splitting for Calibration vs. Validation

### 7.1 Why Not Simple Random Splits?

In traditional ML, you randomly split data into train/test sets. For calibration of a scoring system against temporal engagement data, this approach has problems:

- **Temporal leakage:** Randomly selecting articles means your "calibration" set may contain future articles whose engagement patterns inform the model applied to past articles.
- **Non-stationarity:** If highlighting behavior or content mix changes over time, a random split may mix different regimes.
- **Small dataset:** With only 500+ articles, aggressive splitting leaves small subsets.

### 7.2 Recommended Evaluation Protocol

#### Protocol A: Temporal Holdout (Primary)

Use a chronological split:

```
|--- Calibration Set (older 70%) ---|--- Validation Set (recent 30%) ---|
```

- Sort articles by `scored_at` (or `added_at`).
- Use the oldest 70% to develop calibration insights (identify which dimensions predict best, set thresholds, etc.).
- Validate on the most recent 30%. If the insights hold on recent data, they are likely genuine.
- **Critical:** Do not iterate on the validation set. If you use the validation results to modify the model and then re-validate on the same set, you have data leakage.

#### Protocol B: Expanding Window Cross-Validation

For ongoing monitoring, use an expanding window approach similar to time series cross-validation:

```
Fold 1: Train on months 1-3,   Validate on month 4
Fold 2: Train on months 1-4,   Validate on month 5
Fold 3: Train on months 1-5,   Validate on month 6
...
```

This mirrors how the system operates in practice: you calibrate on historical data and evaluate whether the model generalizes to new articles. Average the correlation across validation folds for an overall performance estimate.

#### Protocol C: Stratified Validation

If using Protocol A, ensure the validation set has reasonable representation across:
- Score tiers (High/Medium/Low).
- Categories (article, email, rss).
- Major tags.

If the validation set is heavily skewed toward one category, the validation results may not generalize.

### 7.3 What to Validate

The calibration/validation split should test specific hypotheses:

| Hypothesis | Calibration Set | Validation Set |
|-----------|----------------|----------------|
| Overall correlation is >= 0.20 | Estimate rho, identify dimensions | Confirm rho >= 0.20 on new data |
| Tier gradient is monotonic (High > Medium > Low) | Measure engagement rates per tier | Confirm gradient holds |
| Dimension X is the best predictor | Rank dimensions by correlation | Confirm same dimension ranks highest |
| False positives share common traits | Identify patterns (short articles, certain topics) | Confirm patterns repeat |

### 7.4 Recommendation

Use temporal holdout (Protocol A) for one-time calibration reviews. Implement expanding window (Protocol B) as an automated monthly check in the calibration toolkit. Add a `cal-validate` subcommand:

```bash
just cal-validate --holdout-pct 30  # temporal holdout validation
```

---

## 8. Category and Topic Confounders

### 8.1 The Problem

Different content categories and topics inherently attract different highlighting behavior:
- **Technical tutorials** may get many highlights (code snippets, key commands).
- **Opinion essays** may get few highlights (the value is in the argument arc, not extractable passages).
- **News summaries** may get zero highlights (ephemeral content, nothing to save).
- **Long-form investigations** may get moderate highlights (concentrated in key findings sections).

If our scoring model assigns high scores to opinion essays (which do have high argument quality and insight) but readers do not highlight them, the calibration will show a false positive pattern that reflects a mismatch between what the model values and how the reader engages, not a flaw in the model's quality assessment.

### 8.2 Stratified Correlation Analysis

Compute correlations within each category and tag separately, rather than pooling all articles. This is already partially implemented in `cal_report.py` (the Category Breakdown and Tag Performance sections).

**What to look for:**

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| High overall rho, but low rho within most categories | Simpson's paradox -- the correlation is driven by between-category differences, not within-category prediction | Model may not be discriminating within topics; review rubric |
| Some categories have high rho, others have low rho | Model works well for certain content types | Consider category-specific scoring weights or rubric variants |
| Consistent rho across categories | Model generalizes well | Good calibration; maintain current approach |

### 8.3 Partial Correlation

Compute the partial correlation between `info_score` and `num_highlights`, controlling for category:

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def partial_spearman(df, score_col, highlight_col, control_col):
    """Spearman partial correlation controlling for a categorical variable.

    Residualizes both variables by subtracting the group mean within
    each category, then computes Spearman on the residuals.
    """
    group_means_score = df.groupby(control_col)[score_col].transform('mean')
    group_means_hl = df.groupby(control_col)[highlight_col].transform('mean')

    residual_score = df[score_col] - group_means_score
    residual_hl = df[highlight_col] - group_means_hl

    rho, p = spearmanr(residual_score, residual_hl)
    return rho, p
```

If the partial correlation (controlling for category) is substantially lower than the raw correlation, category is a significant confounder.

### 8.4 Within-Category Normalization

An alternative to stratified analysis is to normalize highlights within each category. Convert `num_highlights` to a z-score within each category (or percentile rank within category), then correlate with scores:

```python
df['hl_zscore'] = df.groupby('category')['num_highlights'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
)
```

This answers the question: "Within this type of content, does a higher score predict relatively more highlights?" -- which is what we actually want the scoring model to do.

### 8.5 Recommendation

1. **Always report both overall and per-category correlations** (already done in the toolkit).
2. **Add partial correlation** controlling for category to the standard report.
3. **Flag Simpson's paradox:** If overall rho > 0.20 but the mean within-category rho < 0.10, add a warning to the report.
4. For categories with consistently low correlation, investigate whether the scoring rubric's dimensions are appropriate for that content type. Our system already has different rubric variants for articles vs. podcasts -- this approach could extend to other category-specific adjustments.
5. Consider **word count** as an additional confounder. Longer articles provide more opportunity for highlights. Including `log(word_count)` as a covariate in regression models helps isolate the effect of content quality from content length.

---

## 9. Practical Recommendations Summary

### 9.1 Which Metric to Use as Dependent Variable

| Context | Metric | Rationale |
|---------|--------|-----------|
| Primary calibration | `num_highlights` (raw count) | Most direct signal; each highlight = deliberate save decision |
| Length-controlled check | `highlight_density` (per 1k words) | Controls for article length bias |
| Binary check | `highlighted` (0/1) | Sidesteps zero-inflation; useful as sanity check |
| Advanced modeling | Hurdle model components | Separates "did they engage" from "how much" |

### 9.2 Minimum Dataset Size for Reliable Calibration

| Analysis Type | Minimum n | Ideal n |
|--------------|----------|---------|
| Overall correlation | 80 | 200+ |
| Per-category analysis | 30 per category | 100+ per category |
| Per-tag analysis | 30 per tag | 50+ per tag |
| Rolling temporal window | 25 per window | 50+ per window |
| Regression (4 predictors) | 50 | 100+ |
| Bootstrap CI (width ~0.15) | 150 | 300+ |

With our 500+ articles, we are adequately powered for overall and per-category analysis. Raise the minimum thresholds in the toolkit accordingly.

### 9.3 How to Set Up Ongoing Calibration Monitoring

**Monthly automated check (add to CI or cron):**

1. `just cal-sync` -- refresh highlight data.
2. Run calibration report with `--min-progress 0.5`, excluding articles scored in the last 30 days (settlement period).
3. Record the following metrics to a log/dashboard:
   - Overall Spearman rho and Kendall tau-b.
   - Per-category Spearman rho.
   - Tier gradient (is High > Medium > Low for engagement rate?).
   - Score drift (mean score this month vs. trailing 3-month mean).
   - Engagement drift (mean highlights this month vs. trailing 3-month mean).

**Alert thresholds:**

| Metric | Yellow Alert | Red Alert |
|--------|-------------|-----------|
| Overall Spearman rho | Drops below 0.20 | Drops below 0.10 or goes negative |
| Tier gradient | Non-monotonic | High tier has lowest engagement |
| Score drift | > 8 points shift in monthly mean | > 15 points shift |
| Rolling rho trend | Declining for 3+ consecutive windows | Below 0.10 for 2+ windows |

### 9.4 When Calibration Suggests the Scoring Model Needs Updating

**Immediate action triggers:**
- Overall Spearman rho is below 0.10 for 60+ days with sufficient data (100+ articles in the window).
- Tier gradient is inverted (Low-scored articles consistently get more highlights than High-scored ones).
- A major scoring dimension has negative correlation with highlights (the dimension is actively misleading).

**Investigation triggers (review, then decide):**
- One scoring dimension consistently has rho < 0.05 while others are > 0.20 -- the weak dimension may be adding noise without predictive value. Consider de-weighting or replacing it.
- A specific content category consistently shows rho < 0.10 -- the rubric may not suit that content type.
- False positive analysis reveals a systematic pattern (e.g., short opinion pieces always score high but never get highlighted) -- the rubric may be overweighting argument quality for content that is not naturally highlightable.

**How to update:**
1. Analyze false positives and false negatives to identify rubric gaps.
2. Adjust categorical rubric mappings (e.g., change point values for `STANDALONE_SCORES`, `CONTENT_TYPE_SCORES`, etc. in `scorer.py`).
3. Bump the `CURRENT_SCORING_VERSION` to trigger re-scoring.
4. After re-scoring, run calibration on the holdout/validation set to confirm improvement.
5. Track whether the correlation improves over the next 30-60 days of new articles.

---

## 10. Assessment of Our Current Calibration Approach

### 10.1 What We Are Doing Right

1. **Spearman correlation is appropriate** for this data type. Good primary metric choice.
2. **Dimension analysis** (per-dimension correlations, redundancy matrix, OLS regression) provides actionable insights about which scoring components predict engagement.
3. **False positive/negative analysis** using percentile-based gap detection is a sound methodology for identifying the worst misses.
4. **Temporal trend tracking** with rolling windows catches calibration degradation.
5. **Reading progress filtering** is available and addresses the zero-inflation problem at a basic level.
6. **Scoring version tracking** allows us to isolate the effect of rubric changes.
7. **Category and tag breakdowns** address topic-level confounders.

### 10.2 What to Improve

1. **Raise default `--min-progress` from 0.0 to 0.5** in `cal_report.py` for calibration contexts. The current default includes unread articles which dilute the signal.
2. **Add a 30-day settlement period** to exclude recently scored articles from calibration. Recent articles have not yet had time to accumulate highlights.
3. **Add Kendall tau-b** as a secondary correlation metric alongside Spearman.
4. **Add bootstrap confidence intervals** for the primary correlation estimate.
5. **Raise `MIN_CORR_N` from 5 to 25** for any sub-group analysis (rolling windows, per-tag, per-category).
6. **Add highlight density** (`num_highlights / (word_count / 1000)`) as a secondary dependent variable.
7. **Add partial correlation** controlling for category, to detect Simpson's paradox.
8. **Add a `cal-validate` command** that performs temporal holdout validation.
9. **Consider a hurdle model** for formal handling of zero-inflation in advanced analysis.
10. **Log calibration metrics over time** to a file or database for trend visualization, rather than only printing to stdout.

### 10.3 Priority Order for Implementation

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Raise `min_progress` default to 0.5 | Trivial | High -- cleaner signal |
| 2 | Add 30-day settlement period | Low | High -- removes right-censoring |
| 3 | Raise `MIN_CORR_N` to 25 | Trivial | Medium -- prevents unreliable sub-group stats |
| 4 | Add Kendall tau-b to report | Low | Medium -- richer picture |
| 5 | Add bootstrap CIs | Low | Medium -- better uncertainty quantification |
| 6 | Add highlight density metric | Low | Medium -- length-controlled confirmation |
| 7 | Add partial correlation for category | Medium | Medium -- detects Simpson's paradox |
| 8 | Implement `cal-validate` command | Medium | High -- proper validation protocol |
| 9 | Implement hurdle model | High | Medium -- rigorous zero-inflation handling |
| 10 | Add calibration logging/dashboard | Medium | High for long-term monitoring |

---

## References and Further Reading

- Lambert, D. (1992). Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing. *Technometrics*, 34(1), 1-14. Foundation for zero-inflated count models.
- Bonett, D.G. & Wright, T.A. (2000). Sample Size Requirements for Estimating Pearson, Kendall and Spearman Correlations. *Psychometrika*, 65(1), 23-28. Power analysis for rank correlations.
- Xu, W. et al. (2013). Effective Use of Spearman's and Kendall's Correlation Coefficients for Association Between Two Measured Traits. *Animal Behaviour*, 77(5), 1447-1451. Comparison of rank correlation methods with tied data.
- Atkins, D.C. & Gallop, R.J. (2007). Rethinking How Family Researchers Model Infrequent Outcomes: A Tutorial on Count Regression and Zero-Inflated Models. *Journal of Family Psychology*, 21(4), 726-735. Practical guide to hurdle models for behavioral count data.
- GoDaddy Engineering (2025). Calibrating Scores of LLM-as-a-Judge. GoDaddy Blog. Rubric-based calibration methodology for LLM scoring systems.
- Guo, Y. et al. (2021). Detection of Calibration Drift in Clinical Prediction Models to Inform Model Updating. *Journal of the American Medical Informatics Association*. Methods for monitoring calibration degradation over time.
