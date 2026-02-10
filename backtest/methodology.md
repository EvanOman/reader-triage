# Backtest Methodology: Scoring System vs. User Engagement

Statistical methodology for evaluating whether the AI article scoring system
(0-100, four sub-dimensions of 0-25 each) predicts user engagement as measured
by number of highlights and total highlighted words.

## Table of Contents

1. [Data Preparation](#1-data-preparation)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Correlation Measures](#3-correlation-measures)
4. [Regression Approaches](#4-regression-approaches)
5. [Handling Zero-Inflation](#5-handling-zero-inflation)
6. [Sub-Dimension Analysis](#6-sub-dimension-analysis)
7. [Calibration Metrics](#7-calibration-metrics)
8. [Visualization Approaches](#8-visualization-approaches)
9. [Sample Size Considerations](#9-sample-size-considerations)
10. [Controlling for Confounders](#10-controlling-for-confounders)
11. [Implementation Sequence](#11-implementation-sequence)

---

## 1. Data Preparation

### 1.1 Data Sources

The backtest requires joining two datasets:

- **Scored articles**: From `article_scores` table (info_score, specificity_score,
  novelty_score, depth_score, actionability_score)
- **Highlight data**: From `archived_articles.json` produced by `collect_highlights.py`
  (num_highlights, total_highlighted_words, word_count, reading_progress, category)

The join key is the article's Readwise document ID or URL matching.

### 1.2 Loading and Merging

```python
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# Load highlight data
with open("backtest/archived_articles.json") as f:
    highlights_data = json.load(f)
highlights_df = pd.DataFrame(highlights_data)

# Load scores from database
conn = sqlite3.connect("inbox_monitor.db")
scores_df = pd.read_sql_query("""
    SELECT
        a.id as doc_id,
        a.title,
        a.author,
        a.word_count,
        a.category,
        a.site_name,
        a.reading_progress,
        s.info_score,
        s.specificity_score,
        s.novelty_score,
        s.depth_score,
        s.actionability_score
    FROM articles a
    JOIN article_scores s ON a.id = s.article_id
    WHERE a.location = 'archive'
""", conn)
conn.close()

# Merge on document ID
df = scores_df.merge(highlights_df[['doc_id', 'num_highlights', 'total_highlighted_words']],
                     on='doc_id', how='inner')
print(f"Merged dataset: {len(df)} articles with both scores and highlight data")
```

### 1.3 Derived Variables

```python
# Highlight density: highlights per 1000 words (normalizes for article length)
df['highlight_density'] = np.where(
    df['word_count'] > 0,
    df['num_highlights'] / (df['word_count'] / 1000),
    0
)

# Highlighted word fraction: what fraction of the article was highlighted
df['highlight_word_fraction'] = np.where(
    df['word_count'] > 0,
    df['total_highlighted_words'] / df['word_count'],
    0
)

# Binary engagement indicator
df['has_highlights'] = (df['num_highlights'] > 0).astype(int)

# Log-transformed counts (for visualization; add 1 to handle zeros)
df['log_highlights'] = np.log1p(df['num_highlights'])
df['log_highlighted_words'] = np.log1p(df['total_highlighted_words'])

# Score quintiles (5 equal-frequency bins)
df['score_quintile'] = pd.qcut(df['info_score'], q=5, labels=False, duplicates='drop')

# Score tertiles for simpler analysis
df['score_tier'] = pd.cut(df['info_score'],
                          bins=[-1, 29, 59, 100],
                          labels=['Low (<30)', 'Medium (30-59)', 'High (>=60)'])
```

### 1.4 Filtering Decisions

Create multiple analysis subsets to handle noise:

```python
# Full dataset -- includes potential noise from unread articles
df_full = df.copy()

# "Likely read" subset -- articles with any reading progress
df_read = df[df['reading_progress'] > 0].copy()

# "Definitely read" subset -- articles substantially read
df_read_50 = df[df['reading_progress'] >= 0.5].copy()

# "Engaged" subset -- articles with at least 1 highlight (for
# conditional analysis: "given that someone highlighted, how many?")
df_engaged = df[df['num_highlights'] > 0].copy()

print(f"Full:           {len(df_full)} articles")
print(f"reading_progress > 0:  {len(df_read)} articles")
print(f"reading_progress >= 0.5: {len(df_read_50)} articles")
print(f"With highlights: {len(df_engaged)} articles")
```

---

## 2. Exploratory Data Analysis

Before any modeling, understand the marginal distributions and basic patterns.

### 2.1 Distribution Summaries

```python
print("=== Score Distribution ===")
print(df[['info_score', 'specificity_score', 'novelty_score',
          'depth_score', 'actionability_score']].describe())

print("\n=== Highlight Distribution ===")
print(df[['num_highlights', 'total_highlighted_words',
          'highlight_density', 'highlight_word_fraction']].describe())

# Zero-inflation assessment
zero_pct = (df['num_highlights'] == 0).mean() * 100
print(f"\nArticles with 0 highlights: {zero_pct:.1f}%")

# Overdispersion check: variance vs mean for count data
mean_hl = df['num_highlights'].mean()
var_hl = df['num_highlights'].var()
print(f"num_highlights -- mean: {mean_hl:.2f}, variance: {var_hl:.2f}, "
      f"ratio: {var_hl/mean_hl:.2f}")
# ratio >> 1 indicates overdispersion (need NegBin, not Poisson)
```

### 2.2 Quick Contingency Check

```python
# Cross-tabulate score tiers with engagement binary
ct = pd.crosstab(df['score_tier'], df['has_highlights'],
                 margins=True, normalize='index')
print("\nHighlight rate by score tier:")
print(ct.round(3))
```

---

## 3. Correlation Measures

### 3.1 Which Correlation Coefficient to Use

**Recommendation: Use all three, report Spearman as the primary measure.**

| Measure | When to Use | Rationale |
|---------|-------------|-----------|
| **Spearman rho** | Primary measure | Robust to non-normality, handles the skewed count data well, captures monotonic relationships without assuming linearity. Interpretable as "do higher scores tend to co-occur with more highlights?" |
| **Kendall tau-b** | Robustness check | More robust than Spearman with smaller gross error sensitivity and smaller asymptotic variance. Handles tied values well (important because many articles share the same highlight count, especially 0). Better for small samples. |
| **Pearson r** | Secondary / comparison | Only valid if the relationship is approximately linear and both variables are roughly normal. Will be biased by the zero-inflated highlight distribution. Report it for completeness but do not rely on it. |

**Why Spearman over Kendall as primary**: Spearman is more widely understood and
has a direct interpretation as the Pearson correlation of ranks. It is also more
statistically powerful when the sample size is moderate-to-large (N > 50).
Kendall tau-b is preferred for very small samples or when there are many ties.
Since we expect moderate samples (200+), Spearman is the better default.

**Why not Pearson alone**: The highlight counts are zero-inflated and right-skewed.
Pearson assumes bivariate normality and is sensitive to outliers. A single
article with 50 highlights could dominate the Pearson correlation. Spearman and
Kendall operate on ranks, making them robust to this issue.

### 3.2 Implementation

```python
from scipy import stats

def compute_correlations(score_col, engagement_col, data):
    """Compute all three correlation measures with p-values and CIs."""

    x = data[score_col].values
    y = data[engagement_col].values

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    results = {}

    # Pearson
    r_pearson, p_pearson = stats.pearsonr(x, y)
    results['pearson'] = {'r': r_pearson, 'p': p_pearson}

    # Spearman
    r_spearman, p_spearman = stats.spearmanr(x, y)
    results['spearman'] = {'r': r_spearman, 'p': p_spearman}

    # Kendall
    tau, p_kendall = stats.kendalltau(x, y)
    results['kendall'] = {'tau': tau, 'p': p_kendall}

    # Bootstrap confidence interval for Spearman (primary measure)
    from scipy.stats import bootstrap
    def spearman_stat(x_boot, y_boot, axis):
        # scipy.stats.bootstrap expects a function that operates along axis
        rs = np.array([stats.spearmanr(x_boot[:, i], y_boot[:, i])[0]
                       for i in range(x_boot.shape[1])])
        return rs

    # Manual bootstrap for Spearman CI
    rng = np.random.default_rng(42)
    n_boot = 5000
    boot_spearman = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_spearman[i] = stats.spearmanr(x[idx], y[idx])[0]

    ci_low, ci_high = np.percentile(boot_spearman, [2.5, 97.5])
    results['spearman']['ci_95'] = (ci_low, ci_high)

    return results, n


# Run correlations for all engagement measures
engagement_measures = ['num_highlights', 'total_highlighted_words',
                       'highlight_density', 'log_highlights']

print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

for eng_col in engagement_measures:
    print(f"\n--- info_score vs {eng_col} ---")
    for subset_name, subset_df in [('Full', df_full), ('Read', df_read),
                                     ('Engaged', df_engaged)]:
        if len(subset_df) < 10:
            continue
        corrs, n = compute_correlations('info_score', eng_col, subset_df)
        sp = corrs['spearman']
        print(f"  {subset_name:>8} (n={n:3d}): "
              f"Spearman={sp['r']:.3f} [{sp['ci_95'][0]:.3f}, {sp['ci_95'][1]:.3f}] "
              f"p={sp['p']:.4f}  |  "
              f"Kendall={corrs['kendall']['tau']:.3f} p={corrs['kendall']['p']:.4f}  |  "
              f"Pearson={corrs['pearson']['r']:.3f} p={corrs['pearson']['p']:.4f}")
```

### 3.3 Interpretation Guide

| Spearman rho | Interpretation |
|---|---|
| 0.00 - 0.10 | No meaningful association -- scoring does not predict engagement |
| 0.10 - 0.30 | Weak association -- score captures some signal, but noisy |
| 0.30 - 0.50 | Moderate association -- scoring is a useful (but imperfect) predictor |
| 0.50 - 0.70 | Strong association -- scoring is a solid engagement predictor |
| 0.70+ | Very strong -- would be surprising given the noise factors |

---

## 4. Regression Approaches

### 4.1 Model Selection Strategy

Highlights are **count data** (non-negative integers). The correct regression
framework is a Generalized Linear Model (GLM) with a count distribution, not
OLS linear regression.

**Recommended model progression** (fit in order, compare):

| Model | When Appropriate | Python |
|-------|-----------------|--------|
| 1. Poisson GLM | Baseline; assumes mean = variance | `statsmodels.GLM(family=Poisson())` |
| 2. Negative Binomial GLM | When overdispersed (variance >> mean) | `statsmodels.NegativeBinomial()` |
| 3. Zero-Inflated Poisson (ZIP) | Many excess zeros, no overdispersion | `statsmodels.ZeroInflatedPoisson()` |
| 4. Zero-Inflated Neg. Bin. (ZINB) | Many excess zeros + overdispersion | `statsmodels.ZeroInflatedNegativeBinomialP()` |
| 5. Hurdle Model | Zeros come from a single process (not reading) | Two-part: logistic + truncated NB |

**The decision tree:**

```
                     Is variance >> mean?
                      /              \
                   Yes                No
                    |                  |
            Use Neg. Binomial    Use Poisson
                    |                  |
              Are there          Are there
           excess zeros?       excess zeros?
            /        \          /        \
          Yes        No       Yes        No
           |          |        |          |
         ZINB        NB       ZIP      Poisson
```

### 4.2 Poisson Baseline

```python
import statsmodels.api as sm

# Poisson GLM: E[highlights] = exp(b0 + b1 * info_score)
X = sm.add_constant(df['info_score'])
y = df['num_highlights']

poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# Check for overdispersion: Pearson chi-squared / df_resid
pearson_chi2 = poisson_model.pearson_chi2
df_resid = poisson_model.df_resid
dispersion_ratio = pearson_chi2 / df_resid
print(f"\nOverdispersion test: Pearson chi2/df = {dispersion_ratio:.2f}")
print(f"  (>> 1 means overdispersed, use Negative Binomial instead)")
```

### 4.3 Negative Binomial (Expected Primary Model)

```python
# Negative Binomial allows variance > mean (overdispersion)
nb_model = sm.NegativeBinomial(y, X, loglike_method='nb2').fit(
    maxiter=100, disp=False
)
print(nb_model.summary())

# Extract and interpret coefficients
b1 = nb_model.params[1]  # coefficient for info_score
irr = np.exp(b1)          # Incidence Rate Ratio
print(f"\nIncidence Rate Ratio (IRR) for info_score: {irr:.4f}")
print(f"  Interpretation: A 1-point increase in info_score is associated with "
      f"a {(irr - 1) * 100:.2f}% change in expected highlight count")
print(f"  A 10-point increase: {(np.exp(b1 * 10) - 1) * 100:.1f}% change")
```

### 4.4 Cameron-Trivedi Overdispersion Test

A formal test for whether Negative Binomial is needed over Poisson:

```python
def cameron_trivedi_test(poisson_model, y):
    """
    Cameron & Trivedi (1990) regression-based test for overdispersion.

    Tests H0: Var(Y) = mu (Poisson) vs H1: Var(Y) = mu + alpha * mu^2

    A significant positive alpha indicates overdispersion.
    """
    mu = poisson_model.predict()
    # Auxiliary regression: (y - mu)^2 - y  on  mu^2
    z = ((y - mu) ** 2 - y) / mu
    aux_model = sm.OLS(z, mu).fit()

    alpha_hat = aux_model.params[0]
    t_stat = aux_model.tvalues[0]
    p_value = aux_model.pvalues[0]

    print(f"Cameron-Trivedi overdispersion test:")
    print(f"  alpha = {alpha_hat:.4f}")
    print(f"  t-stat = {t_stat:.3f}")
    print(f"  p-value = {p_value:.4f}")
    if p_value < 0.05 and alpha_hat > 0:
        print(f"  --> Significant overdispersion detected. Use Negative Binomial.")
    else:
        print(f"  --> No significant overdispersion. Poisson may be adequate.")

    return alpha_hat, t_stat, p_value

cameron_trivedi_test(poisson_model, y)
```

### 4.5 Model Comparison with AIC/BIC

```python
def compare_models(models_dict):
    """Compare fitted models using AIC, BIC, and log-likelihood."""
    comparison = []
    for name, model in models_dict.items():
        comparison.append({
            'Model': name,
            'LogLik': model.llf,
            'AIC': model.aic,
            'BIC': model.bic,
            'Params': model.df_model + 1,
        })
    comp_df = pd.DataFrame(comparison).sort_values('AIC')
    comp_df['Delta_AIC'] = comp_df['AIC'] - comp_df['AIC'].min()
    print(comp_df.to_string(index=False))
    return comp_df

# Example usage (after fitting all models):
# compare_models({'Poisson': poisson_model, 'NegBin': nb_model,
#                  'ZIP': zip_model, 'ZINB': zinb_model})
```

### 4.6 Logistic Regression for Binary Engagement

As a complementary analysis, predict whether an article gets any highlights at
all (a binary outcome). This sidesteps count-model complexity and is useful for
evaluating the score as a "quality filter."

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

# Binary outcome: any highlights at all
X_lr = df[['info_score']].values
y_lr = df['has_highlights'].values

# Logistic regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_lr, y_lr)

# Cross-validated AUC
cv_auc = cross_val_score(log_model, X_lr, y_lr, cv=5, scoring='roc_auc')
print(f"5-fold CV AUC for predicting any highlights: {cv_auc.mean():.3f} "
      f"+/- {cv_auc.std():.3f}")

# Full-sample AUC
y_prob = log_model.predict_proba(X_lr)[:, 1]
full_auc = roc_auc_score(y_lr, y_prob)
print(f"Full-sample AUC: {full_auc:.3f}")

# Interpretation
b = log_model.coef_[0][0]
or_per_10 = np.exp(b * 10)
print(f"Odds ratio per 10-point score increase: {or_per_10:.2f}")
```

---

## 5. Handling Zero-Inflation

### 5.1 Diagnosing the Zero Problem

The fundamental issue: an article with 0 highlights could mean:

- **Structural zero**: The article was never read (or barely opened). The user
  never had the opportunity to highlight. No amount of quality would change this.
- **Sampling zero**: The article was read but the user chose not to highlight
  anything. This is a genuine signal of low engagement.

This distinction maps directly to the choice between **zero-inflated** and
**hurdle** models.

### 5.2 Recommended Approach: Hurdle Model (Primary) + ZINB (Comparison)

**The hurdle model is the better conceptual fit for this problem.**

Rationale: The zeros in our data primarily come from a single, well-understood
process -- the user did not read the article. This is the "hurdle." Once the
hurdle is crossed (the article was read), the number of highlights follows a
count distribution. Hurdle models cleanly separate these two processes:

1. **Part 1 (Logistic)**: Did the user engage at all? (P(highlights > 0))
2. **Part 2 (Truncated NegBin)**: Given engagement, how many highlights?

The zero-inflated model, by contrast, assumes zeros come from two mixed
processes (structural zeros from a latent "always-zero" class AND sampling zeros
from the count process). This is harder to interpret here because we have an
observable variable (`reading_progress`) that directly explains most structural
zeros.

### 5.3 Hurdle Model Implementation

Statsmodels does not have a built-in hurdle model, but it is straightforward to
implement as two separate regressions:

```python
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def fit_hurdle_model(df, score_cols, count_col='num_highlights'):
    """
    Fit a two-part hurdle model:
      Part 1: Logistic regression for P(count > 0)
      Part 2: Truncated negative binomial for E[count | count > 0]

    Parameters
    ----------
    df : DataFrame with score columns and count_col
    score_cols : list of predictor column names
    count_col : name of the count outcome variable

    Returns
    -------
    dict with 'logistic' and 'truncated_nb' model results
    """
    y = df[count_col].values

    # Part 1: Binary model
    X_bin = df[score_cols].values
    y_bin = (y > 0).astype(int)

    logistic = LogisticRegression(random_state=42)
    logistic.fit(X_bin, y_bin)

    # Part 2: Truncated count model (fit on positive counts only)
    pos_mask = y > 0
    X_pos = sm.add_constant(df.loc[pos_mask, score_cols].values)
    y_pos = y[pos_mask]

    # Use NB on positive counts. The truncation adjustment is minor for
    # practical purposes when the zero probability under NB is small.
    # For formal correctness, use the offset approach or a truncated NB
    # package, but for exploratory backtesting this is sufficient.
    if len(y_pos) >= 10:
        try:
            nb_pos = sm.NegativeBinomial(y_pos, X_pos, loglike_method='nb2').fit(
                maxiter=100, disp=False
            )
        except Exception:
            # Fall back to Poisson if NB fails to converge
            nb_pos = sm.GLM(y_pos, X_pos, family=sm.families.Poisson()).fit()
    else:
        nb_pos = None
        print(f"  Warning: Only {len(y_pos)} positive observations. "
              f"Skipping count model.")

    return {
        'logistic': logistic,
        'truncated_nb': nb_pos,
        'n_zero': int((~pos_mask).sum()),
        'n_positive': int(pos_mask.sum()),
    }


# Fit the hurdle model
score_cols = ['info_score']
hurdle = fit_hurdle_model(df, score_cols)

print(f"Zeros: {hurdle['n_zero']}, Positives: {hurdle['n_positive']}")
print(f"\nPart 1 (Logistic) - Predicting any highlights:")
print(f"  Coefficient: {hurdle['logistic'].coef_[0][0]:.4f}")
print(f"  Odds ratio per 10 pts: {np.exp(hurdle['logistic'].coef_[0][0] * 10):.2f}")

if hurdle['truncated_nb'] is not None:
    print(f"\nPart 2 (Count | positive) - Predicting highlight count:")
    print(hurdle['truncated_nb'].summary())
```

### 5.4 Zero-Inflated Negative Binomial (ZINB) for Comparison

```python
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

X_zinb = sm.add_constant(df['info_score'])
y_zinb = df['num_highlights'].astype(int)

# The inflation model predicts the probability of being a "structural zero"
# Use reading_progress as the inflation predictor if available
if 'reading_progress' in df.columns:
    # Articles with low reading progress are more likely to be structural zeros
    X_inflate = sm.add_constant(1 - df['reading_progress'].fillna(0))
else:
    X_inflate = 'logit'  # use same predictors for inflation

try:
    zinb_model = ZeroInflatedNegativeBinomialP(
        y_zinb, X_zinb,
        exog_infl=X_inflate,
        p=2  # NB2 parameterization
    ).fit(maxiter=200, disp=False)
    print(zinb_model.summary())
except Exception as e:
    print(f"ZINB failed to converge: {e}")
    print("This is common with small samples. Use the hurdle model instead.")
```

### 5.5 Practical Filtering Strategy

If zero-inflated models prove unstable (common with small N), the pragmatic
alternative is to filter and run separate analyses:

```python
# Strategy A: Analyze only articles with reading_progress > 0
# This removes most structural zeros (articles never opened)
df_opened = df[df['reading_progress'] > 0]

# Strategy B: Analyze only articles with reading_progress >= 0.5
# Stricter filter: articles that were substantially read
df_read_half = df[df['reading_progress'] >= 0.5]

# Strategy C: Two-stage analysis
# Stage 1: Full data, binary outcome (any highlights vs none)
# Stage 2: Positive-only data, count outcome (how many highlights)
# This is effectively the hurdle model from Section 5.3
```

---

## 6. Sub-Dimension Analysis

### 6.1 Individual Dimension Correlations

Test each of the four sub-dimensions separately to see which drives engagement
the most:

```python
dimensions = ['specificity_score', 'novelty_score',
              'depth_score', 'actionability_score']

print("=" * 70)
print("SUB-DIMENSION CORRELATION ANALYSIS")
print("=" * 70)

dim_results = {}
for dim in dimensions:
    r_sp, p_sp = stats.spearmanr(df[dim], df['num_highlights'])
    tau, p_tau = stats.kendalltau(df[dim], df['num_highlights'])
    dim_results[dim] = {'spearman': r_sp, 'p_spearman': p_sp,
                        'kendall': tau, 'p_kendall': p_tau}
    print(f"{dim:>25s}: Spearman={r_sp:.3f} (p={p_sp:.4f})  "
          f"Kendall={tau:.3f} (p={p_tau:.4f})")

# Rank dimensions by correlation strength
ranked = sorted(dim_results.items(), key=lambda x: abs(x[1]['spearman']),
                reverse=True)
print(f"\nDimension ranking by |Spearman rho|:")
for i, (dim, vals) in enumerate(ranked, 1):
    sig = "*" if vals['p_spearman'] < 0.05 else ""
    print(f"  {i}. {dim}: {vals['spearman']:.3f}{sig}")
```

### 6.2 Multiple Regression with All Dimensions

```python
# Does using all 4 dimensions predict better than info_score alone?
X_multi = sm.add_constant(df[dimensions])
y_multi = df['num_highlights']

nb_multi = sm.NegativeBinomial(y_multi, X_multi, loglike_method='nb2').fit(
    maxiter=100, disp=False
)
print("\nNegative Binomial with all 4 sub-dimensions:")
print(nb_multi.summary())

# Compare AIC with the single info_score model
print(f"\nAIC comparison:")
print(f"  Single info_score:  {nb_model.aic:.1f}")
print(f"  Four sub-dimensions: {nb_multi.aic:.1f}")
print(f"  Lower AIC is better. Difference: {nb_model.aic - nb_multi.aic:.1f}")
```

### 6.3 Relative Importance via Dominance Analysis

Dominance analysis determines which predictor contributes most to the model by
examining all possible subsets:

```python
from itertools import combinations

def dominance_analysis_nb(df, predictors, outcome):
    """
    Approximate dominance analysis for Negative Binomial regression.

    For each predictor, compute the average marginal McFadden R-squared
    contribution across all subsets of the other predictors.
    """
    n_pred = len(predictors)
    marginal_contributions = {p: [] for p in predictors}

    # Null model (intercept only)
    X_null = sm.add_constant(np.ones(len(df)))
    y = df[outcome].values
    null_model = sm.NegativeBinomial(y, X_null, loglike_method='nb2').fit(
        maxiter=100, disp=False, start_params=None
    )
    llf_null = null_model.llf

    def mcfadden_r2(model):
        return 1 - (model.llf / llf_null)

    # Iterate over all subsets
    for size in range(n_pred):
        for subset in combinations(predictors, size):
            subset = list(subset)
            for p in predictors:
                if p in subset:
                    continue
                # Model without p
                if len(subset) > 0:
                    X_without = sm.add_constant(df[subset].values)
                    m_without = sm.NegativeBinomial(
                        y, X_without, loglike_method='nb2'
                    ).fit(maxiter=100, disp=False)
                    r2_without = mcfadden_r2(m_without)
                else:
                    r2_without = 0

                # Model with p
                subset_with = subset + [p]
                X_with = sm.add_constant(df[subset_with].values)
                m_with = sm.NegativeBinomial(
                    y, X_with, loglike_method='nb2'
                ).fit(maxiter=100, disp=False)
                r2_with = mcfadden_r2(m_with)

                marginal_contributions[p].append(r2_with - r2_without)

    # Average marginal contribution
    importance = {p: np.mean(vals) for p, vals in marginal_contributions.items()}
    total = sum(importance.values())
    relative = {p: v / total * 100 if total > 0 else 0
                for p, v in importance.items()}

    print("\nDominance Analysis Results:")
    print(f"{'Dimension':<25} {'Avg Marginal R2':>15} {'Relative %':>12}")
    for p in sorted(importance, key=importance.get, reverse=True):
        print(f"{p:<25} {importance[p]:>15.4f} {relative[p]:>11.1f}%")

    return importance, relative

# Run dominance analysis (warning: 2^4 = 16 models, may take a moment)
importance, relative = dominance_analysis_nb(df, dimensions, 'num_highlights')
```

---

## 7. Calibration Metrics

### 7.1 Monotonicity Check (Score Quintile vs Engagement)

The most intuitive calibration test: do higher score bins actually correspond to
higher engagement?

```python
def quintile_calibration(df, score_col, engagement_col, n_bins=5):
    """
    Check if score quintiles produce monotonically increasing engagement.

    Returns a DataFrame with one row per quintile showing mean score,
    mean engagement, and sample size.
    """
    df = df.copy()
    df['quintile'] = pd.qcut(df[score_col], q=n_bins, labels=False,
                              duplicates='drop')

    cal = df.groupby('quintile').agg(
        n=(engagement_col, 'count'),
        mean_score=(score_col, 'mean'),
        median_score=(score_col, 'median'),
        mean_engagement=(engagement_col, 'mean'),
        median_engagement=(engagement_col, 'median'),
        pct_with_highlights=('has_highlights', 'mean'),
    ).round(3)

    # Test for monotonic increase
    engagement_values = cal['mean_engagement'].values
    is_monotonic = all(engagement_values[i] <= engagement_values[i+1]
                       for i in range(len(engagement_values)-1))

    # Spearman correlation of quintile medians (should be ~1.0 if well-calibrated)
    r_quintile, p_quintile = stats.spearmanr(
        cal['median_score'], cal['median_engagement']
    )

    print(f"\nCalibration: {score_col} quintiles vs {engagement_col}")
    print(cal.to_string())
    print(f"\nMonotonic increase: {'Yes' if is_monotonic else 'No'}")
    print(f"Quintile-level Spearman: {r_quintile:.3f} (p={p_quintile:.4f})")

    return cal, is_monotonic, r_quintile

cal_results, monotonic, r_q = quintile_calibration(
    df, 'info_score', 'num_highlights'
)
```

### 7.2 Tier-Level Calibration (Using Existing Thresholds)

```python
def tier_calibration(df):
    """
    Test calibration using the existing tier thresholds (Low < 30, Medium 30-59,
    High >= 60).

    This is the most practically relevant test: do the operational thresholds
    actually separate different engagement levels?
    """
    tier_stats = df.groupby('score_tier', observed=False).agg(
        n=('num_highlights', 'count'),
        mean_highlights=('num_highlights', 'mean'),
        median_highlights=('num_highlights', 'median'),
        mean_highlighted_words=('total_highlighted_words', 'mean'),
        pct_with_highlights=('has_highlights', 'mean'),
        mean_highlight_density=('highlight_density', 'mean'),
    ).round(3)

    print("\nTier-Level Calibration:")
    print(tier_stats.to_string())

    # Kruskal-Wallis test: are engagement distributions different across tiers?
    groups = [group['num_highlights'].values
              for _, group in df.groupby('score_tier', observed=False)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis H={h_stat:.3f}, p={p_kw:.4f}")
        if p_kw < 0.05:
            print("  --> Tiers have significantly different engagement levels")
        else:
            print("  --> No significant difference between tiers")

        # Post-hoc pairwise Mann-Whitney U tests
        tier_names = [name for name, _ in df.groupby('score_tier', observed=False)]
        print("\nPairwise Mann-Whitney U tests:")
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if len(groups[i]) > 0 and len(groups[j]) > 0:
                    u_stat, p_mw = stats.mannwhitneyu(
                        groups[i], groups[j], alternative='two-sided'
                    )
                    print(f"  {tier_names[i]} vs {tier_names[j]}: "
                          f"U={u_stat:.0f}, p={p_mw:.4f}")

    return tier_stats

tier_stats = tier_calibration(df)
```

### 7.3 Concordance Index (C-statistic)

The concordance index measures how often a higher-scored article actually has
more highlights than a lower-scored article. It is essentially the AUC for
continuous outcomes.

```python
def concordance_index(scores, outcomes):
    """
    Compute Harrell's concordance index (C-index).

    For all pairs of articles (i, j) where outcome_i != outcome_j,
    what fraction have score_i > score_j when outcome_i > outcome_j?

    C = 0.5 means random, C = 1.0 means perfect concordance.
    """
    n = len(scores)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i+1, n):
            if outcomes[i] == outcomes[j]:
                continue  # Skip tied outcomes

            if outcomes[i] > outcomes[j]:
                if scores[i] > scores[j]:
                    concordant += 1
                elif scores[i] < scores[j]:
                    discordant += 1
                else:
                    tied += 1
            else:
                if scores[i] < scores[j]:
                    concordant += 1
                elif scores[i] > scores[j]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5

    c_index = (concordant + 0.5 * tied) / total
    return c_index

# Compute C-index
c_idx = concordance_index(df['info_score'].values, df['num_highlights'].values)
print(f"\nConcordance Index (C-statistic): {c_idx:.3f}")
print(f"  0.5 = random, 0.7+ = acceptable discrimination, 0.8+ = strong")

# Also compute for highlight_density (length-normalized)
c_idx_density = concordance_index(
    df['info_score'].values, df['highlight_density'].values
)
print(f"C-index vs highlight_density: {c_idx_density:.3f}")
```

Note: For large N the O(n^2) pairwise computation can be slow. A vectorized
alternative uses the relationship between the C-index and Somers' D (which is
related to Kendall's tau):

```python
# Fast C-index via Kendall's tau
tau, _ = stats.kendalltau(df['info_score'], df['num_highlights'])
c_index_fast = (tau + 1) / 2
print(f"C-index (via Kendall tau): {c_index_fast:.3f}")
```

### 7.4 McFadden's Pseudo-R-squared

For count models, standard R-squared is not meaningful. Use McFadden's
pseudo-R-squared instead:

```python
def mcfadden_pseudo_r2(fitted_model, y):
    """McFadden's pseudo R-squared for count models."""
    X_null = sm.add_constant(np.ones(len(y)))
    null_model = sm.NegativeBinomial(y, X_null, loglike_method='nb2').fit(
        maxiter=100, disp=False
    )
    r2 = 1 - (fitted_model.llf / null_model.llf)
    return r2

r2_mcf = mcfadden_pseudo_r2(nb_model, y)
print(f"McFadden's pseudo R-squared: {r2_mcf:.4f}")
print(f"  Note: Values of 0.2-0.4 in count models represent excellent fit")
```

---

## 8. Visualization Approaches

### 8.1 Overview Dashboard (4 Panels)

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Score vs Highlights scatter with jitter
ax1 = axes[0, 0]
jitter_y = df['num_highlights'] + np.random.normal(0, 0.15, len(df))
ax1.scatter(df['info_score'], jitter_y, alpha=0.4, s=30, edgecolor='none')
# Add LOWESS trend line
from statsmodels.nonparametric.smoothers_lowess import lowess
if len(df) >= 10:
    smooth = lowess(df['num_highlights'], df['info_score'], frac=0.5)
    ax1.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2.5, label='LOWESS')
ax1.set_xlabel('Info Score (0-100)')
ax1.set_ylabel('Number of Highlights')
ax1.set_title('Score vs Highlight Count')
ax1.legend()

# Panel 2: Score vs Highlight Density (length-normalized)
ax2 = axes[0, 1]
ax2.scatter(df['info_score'], df['highlight_density'], alpha=0.4, s=30,
            edgecolor='none')
if len(df) >= 10:
    smooth2 = lowess(df['highlight_density'], df['info_score'], frac=0.5)
    ax2.plot(smooth2[:, 0], smooth2[:, 1], 'r-', linewidth=2.5, label='LOWESS')
ax2.set_xlabel('Info Score (0-100)')
ax2.set_ylabel('Highlights per 1000 Words')
ax2.set_title('Score vs Highlight Density')
ax2.legend()

# Panel 3: Quintile calibration bar chart
ax3 = axes[1, 0]
cal_data = df.groupby('score_quintile').agg(
    mean_score=('info_score', 'mean'),
    mean_highlights=('num_highlights', 'mean'),
    pct_engaged=('has_highlights', 'mean'),
    n=('num_highlights', 'count'),
).reset_index()
bars = ax3.bar(cal_data['score_quintile'], cal_data['mean_highlights'],
               color=sns.color_palette("viridis", len(cal_data)), edgecolor='white')
ax3.set_xlabel('Score Quintile (0=lowest, 4=highest)')
ax3.set_ylabel('Mean Highlights')
ax3.set_title('Mean Highlights by Score Quintile')
# Add count labels
for bar, n in zip(bars, cal_data['n']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'n={n}', ha='center', va='bottom', fontsize=9)

# Panel 4: Distribution of highlights by score tier
ax4 = axes[1, 1]
tier_order = ['Low (<30)', 'Medium (30-59)', 'High (>=60)']
tier_colors = {'Low (<30)': '#e74c3c', 'Medium (30-59)': '#f39c12',
               'High (>=60)': '#2ecc71'}
for tier in tier_order:
    subset = df[df['score_tier'] == tier]
    if len(subset) > 0:
        ax4.hist(subset['num_highlights'], bins=range(0, int(df['num_highlights'].max())+2),
                 alpha=0.5, label=f'{tier} (n={len(subset)})',
                 color=tier_colors[tier])
ax4.set_xlabel('Number of Highlights')
ax4.set_ylabel('Count')
ax4.set_title('Highlight Distribution by Score Tier')
ax4.legend()

plt.tight_layout()
plt.savefig('backtest/plots/01_overview_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.2 Sub-Dimension Heatmap

```python
fig, ax = plt.subplots(figsize=(8, 5))

# Correlation matrix of sub-dimensions with engagement measures
corr_cols = dimensions + ['num_highlights', 'total_highlighted_words',
                           'highlight_density']
corr_matrix = df[corr_cols].corr(method='spearman')

# Show only sub-dimensions (rows) vs engagement (cols)
sub = corr_matrix.loc[dimensions,
                      ['num_highlights', 'total_highlighted_words',
                       'highlight_density']]

sns.heatmap(sub, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            vmin=-0.3, vmax=0.5, ax=ax,
            yticklabels=['Specificity', 'Novelty', 'Depth', 'Actionability'],
            xticklabels=['Highlights', 'Highlighted Words', 'Highlight Density'])
ax.set_title('Sub-Dimension Correlations with Engagement (Spearman)')
plt.tight_layout()
plt.savefig('backtest/plots/02_subdimension_heatmap.png', dpi=150,
            bbox_inches='tight')
plt.show()
```

### 8.3 ROC Curve for Binary Engagement Prediction

```python
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(7, 7))

# ROC for total score
fpr, tpr, thresholds = roc_curve(df['has_highlights'], df['info_score'])
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, linewidth=2, label=f'Total Score (AUC={roc_auc:.3f})')

# ROC for each sub-dimension
for dim, label in zip(dimensions, ['Specificity', 'Novelty', 'Depth', 'Actionability']):
    fpr_d, tpr_d, _ = roc_curve(df['has_highlights'], df[dim])
    auc_d = auc(fpr_d, tpr_d)
    ax.plot(fpr_d, tpr_d, linewidth=1.5, linestyle='--',
            label=f'{label} (AUC={auc_d:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC: Predicting Any Highlights from Score')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('backtest/plots/03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.4 Predicted vs Observed (Calibration Plot for Count Model)

```python
fig, ax = plt.subplots(figsize=(8, 6))

# Bin predictions from the NB model
predicted = nb_model.predict()
df_temp = pd.DataFrame({'predicted': predicted, 'observed': y})
df_temp['pred_bin'] = pd.qcut(df_temp['predicted'], q=10, duplicates='drop')

cal_plot_data = df_temp.groupby('pred_bin', observed=False).agg(
    mean_predicted=('predicted', 'mean'),
    mean_observed=('observed', 'mean'),
    n=('observed', 'count'),
).reset_index()

ax.scatter(cal_plot_data['mean_predicted'], cal_plot_data['mean_observed'],
           s=cal_plot_data['n'] * 3, alpha=0.7, edgecolor='black', zorder=5)

# Perfect calibration line
max_val = max(cal_plot_data['mean_predicted'].max(),
              cal_plot_data['mean_observed'].max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, label='Perfect calibration')

ax.set_xlabel('Mean Predicted Highlights (NegBin model)')
ax.set_ylabel('Mean Observed Highlights')
ax.set_title('Calibration Plot: Predicted vs Observed Highlight Counts')
ax.legend()

# Add bubble size legend
for n_val in [10, 25, 50]:
    ax.scatter([], [], s=n_val * 3, c='gray', alpha=0.5, label=f'n={n_val}')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('backtest/plots/04_calibration_plot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.5 Effect Size Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: IRR (Incidence Rate Ratios) with confidence intervals per sub-dimension
ax_left = axes[0]
irr_data = []
for dim in dimensions:
    X_d = sm.add_constant(df[dim])
    try:
        m = sm.NegativeBinomial(y, X_d, loglike_method='nb2').fit(
            maxiter=100, disp=False)
        irr = np.exp(m.params[1])
        ci_low = np.exp(m.conf_int().iloc[1, 0])
        ci_high = np.exp(m.conf_int().iloc[1, 1])
        irr_data.append({
            'dim': dim.replace('_score', ''),
            'irr': irr,
            'ci_low': ci_low,
            'ci_high': ci_high,
        })
    except Exception:
        pass

if irr_data:
    irr_df = pd.DataFrame(irr_data)
    y_pos = range(len(irr_df))
    ax_left.barh(y_pos, irr_df['irr'] - 1, left=1,
                 color=sns.color_palette("deep"), edgecolor='white', height=0.6)
    ax_left.errorbar(irr_df['irr'], y_pos,
                     xerr=[irr_df['irr'] - irr_df['ci_low'],
                           irr_df['ci_high'] - irr_df['irr']],
                     fmt='none', c='black', capsize=4)
    ax_left.axvline(x=1, color='red', linestyle='--', linewidth=0.8)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(irr_df['dim'])
    ax_left.set_xlabel('Incidence Rate Ratio (per 1-point increase)')
    ax_left.set_title('IRR by Sub-Dimension')

# Right: Correlation coefficients with CIs
ax_right = axes[1]
corr_data = []
for dim in dimensions:
    rng = np.random.default_rng(42)
    x_vals = df[dim].values
    y_vals = df['num_highlights'].values
    boot_r = np.zeros(2000)
    n = len(x_vals)
    for b in range(2000):
        idx = rng.integers(0, n, size=n)
        boot_r[b] = stats.spearmanr(x_vals[idx], y_vals[idx])[0]
    r_sp, _ = stats.spearmanr(x_vals, y_vals)
    ci = np.percentile(boot_r, [2.5, 97.5])
    corr_data.append({
        'dim': dim.replace('_score', ''),
        'r': r_sp, 'ci_low': ci[0], 'ci_high': ci[1]
    })

corr_df = pd.DataFrame(corr_data)
y_pos = range(len(corr_df))
ax_right.barh(y_pos, corr_df['r'],
              color=sns.color_palette("deep"), edgecolor='white', height=0.6)
ax_right.errorbar(corr_df['r'], y_pos,
                  xerr=[corr_df['r'] - corr_df['ci_low'],
                        corr_df['ci_high'] - corr_df['r']],
                  fmt='none', c='black', capsize=4)
ax_right.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
ax_right.set_yticks(y_pos)
ax_right.set_yticklabels(corr_df['dim'])
ax_right.set_xlabel('Spearman Correlation')
ax_right.set_title('Spearman rho with 95% Bootstrap CI')

plt.tight_layout()
plt.savefig('backtest/plots/05_effect_sizes.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 8.6 Reading Progress Confound Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Reading progress vs highlights (to validate the confound exists)
ax1 = axes[0]
ax1.scatter(df['reading_progress'], df['num_highlights'], alpha=0.4, s=30)
ax1.set_xlabel('Reading Progress (0-1)')
ax1.set_ylabel('Number of Highlights')
ax1.set_title('Reading Progress vs Highlights')

# Right: Score vs highlights, colored by reading progress
ax2 = axes[1]
scatter = ax2.scatter(df['info_score'], df['num_highlights'],
                      c=df['reading_progress'], cmap='RdYlGn',
                      alpha=0.6, s=40, edgecolor='none')
plt.colorbar(scatter, ax=ax2, label='Reading Progress')
ax2.set_xlabel('Info Score')
ax2.set_ylabel('Number of Highlights')
ax2.set_title('Score vs Highlights (colored by reading progress)')

plt.tight_layout()
plt.savefig('backtest/plots/06_reading_progress.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 9. Sample Size Considerations

### 9.1 Minimum Requirements by Analysis Type

| Analysis | Minimum N | Recommended N | Rationale |
|----------|-----------|---------------|-----------|
| Spearman correlation | 20 | 50+ | Detects rho >= 0.4 at alpha=0.05, power=0.80 |
| Negative Binomial regression (1 predictor) | 50 | 100+ | Rule of thumb: 10-20 observations per parameter. NB has 3 params (intercept, beta, alpha). |
| NB with 4 sub-dimensions | 100 | 200+ | 6 parameters. Need 15-30 obs per param. |
| Zero-Inflated NB | 150 | 300+ | Additional inflation parameters. Known convergence issues with small N. |
| Hurdle model | 100 | 200+ | Requires enough positive cases for the truncated count model. |
| Quintile calibration | 50 | 100+ | Need at least 10 per quintile for stable means. |
| Dominance analysis | 100 | 200+ | Fitting 2^4 = 16 models, each needs adequate data. |

### 9.2 Power Analysis for Correlation

```python
from scipy.stats import norm

def sample_size_for_correlation(r_target, alpha=0.05, power=0.80):
    """
    Required sample size to detect a Spearman correlation of r_target.

    Uses the asymptotic normal approximation for Fisher z-transformed
    correlation. This is approximate for Spearman but adequate for planning.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    z_r = np.arctanh(r_target)  # Fisher z-transform
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n))

print("Required sample sizes for Spearman correlation (alpha=0.05, power=0.80):")
for r in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    n_req = sample_size_for_correlation(r)
    print(f"  rho = {r:.2f}: N = {n_req}")
```

Expected output:
```
  rho = 0.15: N = 349
  rho = 0.20: N = 197
  rho = 0.25: N = 127
  rho = 0.30: N = 90
  rho = 0.40: N = 50
  rho = 0.50: N = 33
```

### 9.3 Practical Implications for This Study

Given the current database has 82 scored articles (none archived with scores),
we need to score archived articles. The `collect_highlights.py` script fetches
highlight data for all archived articles. The critical question is how many
archived articles exist and can be scored.

- **If N < 50**: Report only Spearman/Kendall correlations and the tier-level
  calibration table. Do not attempt regression modeling or zero-inflated models.
  Results should be treated as preliminary/exploratory.
- **If 50 <= N < 150**: Use Spearman correlations, simple NB regression,
  logistic regression for binary engagement, and quintile calibration. Skip
  ZINB/hurdle models and dominance analysis.
- **If N >= 150**: Full analysis pipeline is viable. Run all models including
  hurdle model and sub-dimension analysis.
- **If N >= 300**: Zero-inflated models become reliable. Dominance analysis and
  confounder-adjusted models are well-powered.

### 9.4 Bootstrap Confidence Intervals for Small Samples

When N is small, parametric confidence intervals may be unreliable. Use
bootstrap CIs for all key metrics:

```python
def bootstrap_metric(data, metric_func, n_boot=5000, ci=0.95, seed=42):
    """
    Compute a bootstrap confidence interval for an arbitrary metric.

    Parameters
    ----------
    data : array-like or DataFrame
    metric_func : callable that takes data and returns a scalar
    n_boot : number of bootstrap resamples
    ci : confidence level (default 0.95)
    seed : random seed

    Returns
    -------
    point_estimate, (ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    point = metric_func(data)
    boot_vals = np.zeros(n_boot)
    n = len(data)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if isinstance(data, pd.DataFrame):
            boot_vals[i] = metric_func(data.iloc[idx])
        else:
            boot_vals[i] = metric_func(data[idx])

    alpha = (1 - ci) / 2
    ci_low, ci_high = np.percentile(boot_vals, [alpha * 100, (1 - alpha) * 100])

    return point, (ci_low, ci_high)
```

---

## 10. Controlling for Confounders

### 10.1 Key Confounders

| Confounder | Why It Matters | How to Handle |
|------------|---------------|---------------|
| **Word count** | Longer articles have more opportunity for highlights AND may receive higher depth/specificity scores. This creates spurious correlation. | Include as offset in NB model; also analyze highlight_density. |
| **Reading progress** | Articles not fully read cannot be fully highlighted. 0 highlights may mean "unread" not "unengaging." | Filter to reading_progress > 0 or include in the zero-inflation component. |
| **Article category** | RSS feeds, emails, and articles may have fundamentally different highlight patterns. | Include as a fixed effect (dummy variable). |
| **Author** | Some authors' works may be highlighted more due to user preference, independent of content quality. | Include author_boost or a "known author" binary flag. |
| **Time** | Highlighting behavior may change over time (e.g., user becomes more selective). | Include a time covariate or test for temporal trends. |

### 10.2 Adjusted Negative Binomial Model

```python
# Full adjusted model
# Prepare covariates
df['log_word_count'] = np.log1p(df['word_count'].fillna(0))
df['has_author'] = df['author'].notna().astype(int)

# Category dummies (if there are multiple categories)
if df['category'].nunique() > 1:
    cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
    df = pd.concat([df, cat_dummies], axis=1)
    cat_cols = list(cat_dummies.columns)
else:
    cat_cols = []

# Adjusted model: score + confounders
adj_predictors = ['info_score', 'log_word_count', 'has_author'] + cat_cols
X_adj = sm.add_constant(df[adj_predictors].astype(float))
y_adj = df['num_highlights'].astype(int)

nb_adj = sm.NegativeBinomial(y_adj, X_adj, loglike_method='nb2').fit(
    maxiter=100, disp=False
)
print("Adjusted Negative Binomial Model:")
print(nb_adj.summary())

# Compare adjusted vs unadjusted coefficient for info_score
b_unadj = nb_model.params[1]
b_adj = nb_adj.params[1]
print(f"\ninfo_score coefficient:")
print(f"  Unadjusted: {b_unadj:.4f} (IRR={np.exp(b_unadj):.4f})")
print(f"  Adjusted:   {b_adj:.4f} (IRR={np.exp(b_adj):.4f})")
print(f"  Change: {(b_adj - b_unadj) / b_unadj * 100:.1f}%")
```

### 10.3 Word Count as Offset (Rate Model)

The strongest confounder is word count. When we want to model the *rate* of
highlighting (highlights per word) rather than the raw count, we use word count
as an exposure offset in the GLM:

```python
# Offset model: log(E[highlights]) = b0 + b1*score + log(word_count)
# This is equivalent to modeling: E[highlights/word_count] = exp(b0 + b1*score)
df_offset = df[df['word_count'] > 0].copy()
X_offset = sm.add_constant(df_offset['info_score'])
y_offset = df_offset['num_highlights'].astype(int)
offset = np.log(df_offset['word_count'].values)

nb_rate = sm.NegativeBinomial(
    y_offset, X_offset,
    loglike_method='nb2',
    offset=offset
).fit(maxiter=100, disp=False)

print("Negative Binomial Rate Model (word count as offset):")
print(nb_rate.summary())

b_rate = nb_rate.params[1]
print(f"\ninfo_score IRR (rate model): {np.exp(b_rate):.4f}")
print(f"  Interpretation: Per 1-point score increase, the highlight RATE "
      f"changes by {(np.exp(b_rate) - 1) * 100:.2f}%")
```

### 10.4 Partial Correlations

Remove the effect of word count from both score and highlights before computing
the correlation:

```python
def partial_spearman(x, y, z, data):
    """
    Partial Spearman correlation between x and y, controlling for z.

    Uses rank-based residualization.
    """
    from scipy.stats import rankdata

    rx = rankdata(data[x])
    ry = rankdata(data[y])
    rz = rankdata(data[z])

    # Residualize x and y on z using OLS on ranks
    from numpy.linalg import lstsq
    Z = np.column_stack([np.ones(len(rz)), rz])
    _, _, _, _ = lstsq(Z, rx, rcond=None)
    res_x = rx - Z @ lstsq(Z, rx, rcond=None)[0]
    res_y = ry - Z @ lstsq(Z, ry, rcond=None)[0]

    r, p = stats.pearsonr(res_x, res_y)  # Pearson on residuals = partial Spearman
    return r, p

r_partial, p_partial = partial_spearman(
    'info_score', 'num_highlights', 'word_count', df[df['word_count'] > 0]
)
print(f"Partial Spearman (controlling for word_count): "
      f"r={r_partial:.3f}, p={p_partial:.4f}")
```

---

## 11. Implementation Sequence

### Phase 1: Data Collection and Exploration

1. Run `collect_highlights.py` to fetch highlight data for archived articles.
2. Score the archived articles using the existing scorer (may require a new script
   that fetches archived article content and sends it through `InboxScorer`).
3. Run the data preparation code from Section 1 to create the merged dataset.
4. Run the EDA from Section 2. Record N, zero-inflation rate, and overdispersion ratio.
5. Decide analysis scope based on N (see Section 9.3).

### Phase 2: Core Analysis (Minimum Viable)

6. Compute all three correlation measures (Section 3).
7. Run tier-level calibration (Section 7.2).
8. Generate the overview dashboard (Section 8.1).

### Phase 3: Modeling (If N >= 50)

9. Fit Poisson baseline (Section 4.2).
10. Run overdispersion test (Section 4.4).
11. Fit Negative Binomial (Section 4.3).
12. Fit logistic regression for binary engagement (Section 4.6).
13. Compare models with AIC/BIC (Section 4.5).

### Phase 4: Advanced Analysis (If N >= 150)

14. Fit hurdle model (Section 5.3).
15. Run sub-dimension analysis (Section 6).
16. Fit adjusted models with confounders (Section 10).
17. Compute concordance index (Section 7.3).
18. Generate all visualization panels (Section 8).

### Phase 5: Interpretation and Recommendations

19. Compile results into a summary report.
20. Determine whether the scoring thresholds (30, 60) are well-calibrated.
21. Identify which sub-dimensions are most predictive and whether the weighting
    should be adjusted.
22. Recommend changes to the scoring prompt or thresholds if warranted.

---

## Summary of Key Recommendations

| Question | Recommendation | Rationale |
|----------|---------------|-----------|
| Primary correlation measure | Spearman rho | Robust to skew, captures monotonic relationships, widely understood |
| Secondary correlation | Kendall tau-b | More robust with ties and small samples; use as robustness check |
| Primary regression model | Negative Binomial GLM | Highlights are overdispersed counts (variance >> mean is expected) |
| Handling zeros | Hurdle model (logistic + truncated NB) | Zeros primarily from a single process (not reading); cleaner interpretation than ZINB |
| Zero-inflation fallback | Filter to reading_progress > 0 | Pragmatic alternative when N is too small for hurdle/ZINB models |
| Best calibration metric | Tier-level Kruskal-Wallis test + quintile monotonicity | Directly tests whether the operational score bins separate engagement levels |
| Discrimination metric | Concordance index (C-statistic) | Measures pairwise concordance, no distributional assumptions |
| Key confounder | Word count (use as offset in NB) | Longest articles have most highlighting opportunity; must be controlled |
| Minimum sample size | 50 for basic analysis, 150+ for full pipeline | Below 50, only descriptive statistics and correlations are reliable |
| Most informative plot | Quintile calibration bar chart | Immediately shows whether "higher score = more engagement" holds in practice |

---

## Dependencies

```
pip install scipy statsmodels scikit-learn matplotlib seaborn pandas numpy
```

All code is designed for Python 3.10+ and uses only well-maintained,
widely-available packages.

---

## References and Further Reading

- Cameron, A. C. & Trivedi, P. K. (1990). Regression-based tests for
  overdispersion in the Poisson model. *Journal of Econometrics*.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics*. (Original hurdle model paper)
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to
  defects in manufacturing. *Technometrics*. (Original ZIP paper)
- Steyerberg, E. W. et al. (2010). Assessing the performance of prediction
  models: a framework for some traditional and novel measures. *Epidemiology*.
- UVA Library guide on correlation measures:
  https://library.virginia.edu/data/articles/correlation-pearson-spearman-and-kendalls-tau
- UCLA OARC guide on zero-inflated models:
  https://stats.oarc.ucla.edu/r/dae/zinb/
- statsmodels ZeroInflatedNegativeBinomialP documentation:
  https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html
