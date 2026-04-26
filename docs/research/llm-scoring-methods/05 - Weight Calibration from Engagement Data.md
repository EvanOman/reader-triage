# Weight Calibration from Engagement Data

Methods for calibrating scoring weights against ground-truth engagement signals, specifically optimizing the mapping from LLM question responses to a composite score that maximally correlates with reader highlight behavior.

---

## Problem Statement

Our scoring system has:

- **Independent variables**: 8 categorical/binary answers from an LLM about article quality (standalone passages, novel framing, content type, author conviction, practitioner voice, content completeness, named framework, applicable ideas)
- **Dependent variable**: Number of highlights a reader makes in an article (proxy for engagement/value)
- **Current approach**: Expert-assigned point mappings per question, summed into 4 dimension sub-scores (0-25 each), totaling 0-100
- **Goal**: Set question weights so the weighted composite score maximally correlates with highlight behavior

With ~500+ scored articles and highlight counts from Readwise, we have enough data to begin empirical weight calibration. This document surveys the methods available, from simple regression to Bayesian updating, and recommends a practical implementation path.

---

## 1. Linear and Generalized Linear Regression

### 1a. OLS Linear Regression

The simplest approach: treat each LLM answer as a numeric feature (binary 0/1 or ordinal encoding for categorical responses) and fit a linear model predicting highlight count.

```python
import statsmodels.api as sm
import numpy as np

# X: matrix of encoded question responses (n_articles x n_features)
# y: highlight counts (n_articles,)
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# Learned weights
weights = model.params[1:]  # exclude intercept
```

**Strengths**: Interpretable coefficients that directly serve as question weights. Standard errors and p-values immediately reveal which questions are statistically significant predictors. R-squared tells you the overall explanatory power. The existing `cal_report.py` dimension analysis already runs OLS, so this extends naturally.

**Weaknesses**: Assumes a linear relationship between question responses and highlight counts. Highlight counts are non-negative integers (often zero-inflated), violating OLS assumptions about normally distributed residuals. Coefficients can go negative, which is interpretable (a "yes" on that question predicts fewer highlights) but may be confusing when converted to a scoring system.

**When to use**: As a quick diagnostic to identify which features matter. Not ideal as the final weight-learning method for count data, but excellent for initial exploration.

### 1b. Poisson Regression

Since highlight counts are non-negative integers, Poisson regression is the natural generalized linear model (GLM). It models log(E[highlights]) as a linear combination of features, ensuring predictions are always non-negative.

```python
from sklearn.linear_model import PoissonRegressor
import numpy as np

# X: binary/categorical features (n_articles x n_features)
# y: highlight counts
model = PoissonRegressor(alpha=1.0)  # alpha controls L2 regularization
model.fit(X, y)

# Coefficients as weights
weights = model.coef_
# Exponentiated coefficients give rate ratios:
# exp(coef) = multiplicative change in expected highlights per unit change in feature
rate_ratios = np.exp(weights)
```

**Strengths**: Respects the count nature of the dependent variable. Exponentiated coefficients have a clean interpretation: exp(w_i) is the multiplicative factor on expected highlight count when question i is answered "yes." Available in both scikit-learn (`PoissonRegressor`) and statsmodels (`GLM` with `Poisson` family).

**Weaknesses**: Assumes the variance equals the mean (equidispersion). Highlight data is likely overdispersed (many zeros, some articles with 10+ highlights), violating this assumption. This leads to underestimated standard errors and overconfident significance tests.

### 1c. Negative Binomial Regression

When highlight counts are overdispersed (variance > mean, which is typical for engagement data), negative binomial regression adds a dispersion parameter that Poisson regression lacks.

```python
import statsmodels.api as sm

# Using statsmodels for negative binomial
X_with_const = sm.add_constant(X)
model = sm.NegativeBinomial(y, X_with_const).fit()

weights = model.params[1:]  # exclude intercept
dispersion = model.params[0]  # alpha parameter
```

**Strengths**: Handles overdispersion properly, which is critical for highlight count data where most articles have 0 highlights and a few have many. Standard errors are more honest than Poisson regression. Cameron and Trivedi's auxiliary regression test can determine whether you need negative binomial over Poisson.

**Weaknesses**: Slightly more complex to fit and interpret. Two common parameterizations (NB1 and NB2) exist; statsmodels implements NB2 by default. Still assumes a log-linear relationship.

### 1d. Logistic Regression (Binarized Outcome)

If the goal is really "does this article get highlighted at all?" rather than "how many highlights?", binarize the outcome and use logistic regression. This is the approach outlined in the existing `weighting-strategies.md` document.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Binarize: engaged = 1 if any highlights
y_binary = (y > 0).astype(int)

model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y_binary)

# Coefficients as weights
weights = model.coef_[0]

# Convert to a 0-100 scoring system
def score_article(responses):
    raw = np.dot(weights, responses) + model.intercept_[0]
    return int(100 * (1 / (1 + np.exp(-raw))))  # sigmoid scaling
```

**Strengths**: Well-understood, robust to outliers in highlight counts. The sigmoid naturally maps to a 0-100 probability-like score. Regularization (L1, L2, ElasticNet) handles multicollinearity between correlated questions. This is the most commonly recommended approach in the scoring literature.

**Weaknesses**: Discards information about engagement intensity (an article with 1 highlight is treated the same as one with 20). The binarization threshold is arbitrary. With ~500 articles, we need at least 50+ in each class (highlighted / not highlighted) for stable estimates.

### Recommendation for Regression Approach

**Start with logistic regression** on a binarized outcome (highlighted >= 1) for weight learning. Use negative binomial regression as a secondary analysis to verify that the rank ordering of weights is similar when accounting for engagement intensity. The logistic regression weights can be directly converted to a point-based scoring system compatible with the existing 0-100 scale.

**Regularization**: Use L2 (Ridge) regularization by default. With only 8 features and 500+ observations, overfitting risk is moderate but real. L1 (Lasso) is useful if you suspect some questions are pure noise and should be zeroed out. Cross-validate the regularization strength C.

---

## 2. Spearman/Kendall Correlation Optimization

### The Approach

Instead of fitting a parametric model, directly optimize the weights to maximize rank correlation between the weighted composite score and highlight counts. This is appealing because Spearman's rho is already the primary calibration metric used in `cal_report.py`.

```python
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr
import numpy as np

def neg_spearman(weights, X, y):
    """Objective: negative Spearman rho (minimize = maximize correlation)."""
    scores = X @ weights
    rho, _ = spearmanr(scores, y)
    return -rho

# X: feature matrix (n_articles x n_features)
# y: highlight counts

# Method 1: Nelder-Mead (gradient-free, good for non-smooth objectives)
result = minimize(
    neg_spearman,
    x0=np.ones(X.shape[1]) / X.shape[1],  # equal weights as starting point
    args=(X, y),
    method='Nelder-Mead',
    options={'maxiter': 10000, 'xatol': 1e-6}
)
optimal_weights = result.x

# Method 2: Differential evolution (global optimization, avoids local minima)
bounds = [(-1, 10)] * X.shape[1]  # allow negative weights for penalty questions
result = differential_evolution(
    neg_spearman,
    bounds=bounds,
    args=(X, y),
    seed=42,
    maxiter=1000,
    tol=1e-8
)
optimal_weights = result.x
```

### Why Spearman Over Pearson

Spearman's rho measures monotonic (not just linear) association. Since we care about rank ordering articles correctly (higher score = more highlights), not about predicting the exact highlight count, rank correlation is the right objective. Highlight counts are also highly skewed, making Pearson correlation sensitive to outliers.

### Kendall's Tau as an Alternative

Kendall's tau is based on concordant vs. discordant pairs and is more robust than Spearman's rho with small samples. It has a smaller gross error sensitivity and smaller asymptotic variance. However, it is computationally more expensive to optimize (O(n^2) per evaluation vs. O(n log n) for Spearman). For our dataset size (~500 articles), either is feasible, but Spearman is the pragmatic choice given it is already embedded in the calibration toolkit.

### Practical Considerations

**Non-smoothness**: Spearman's rho is a rank-based statistic, so it is not differentiable with respect to the weights. Gradient-based optimizers (BFGS, L-BFGS) will not work. Use derivative-free methods: Nelder-Mead for local optimization, differential evolution or basin-hopping for global optimization.

**Local minima**: The Spearman objective surface can have multiple local optima, especially with correlated features. Differential evolution is recommended over Nelder-Mead for this reason, though it is slower.

**Constraint handling**: You may want to constrain weights to be non-negative (if you want all questions to contribute positively) or to sum to a target value. Differential evolution handles box constraints naturally via the `bounds` parameter. For sum constraints, reparameterize using a softmax or simplex projection.

**Normalization**: After optimization, rescale weights so the maximum achievable score maps to 100, preserving compatibility with the existing tier thresholds (High >= 60, Medium 30-59, Low < 30).

### Comparison with Regression

| Aspect | Regression (Logistic/Poisson) | Spearman Optimization |
|--------|------------------------------|----------------------|
| Objective | Maximize likelihood | Maximize rank correlation |
| Interpretability | Coefficients have probabilistic meaning | Weights are arbitrary scale |
| Standard errors | Yes (confidence intervals on weights) | No (need bootstrap) |
| Handles ties | Naturally | Spearman handles ties adequately |
| Overfitting risk | Controlled via regularization | No built-in regularization |
| Computational cost | Fast (closed-form or iterative) | Slower (derivative-free optimization) |
| Robustness to outliers | Moderate (logistic) to low (OLS) | High (rank-based) |

**Verdict**: Use regression for weight estimation and Spearman correlation for evaluation. The regression coefficients provide statistically grounded weights with confidence intervals; the Spearman correlation of the resulting composite score against highlights is the calibration health metric.

---

## 3. Bayesian Approaches

### 3a. Bayesian Ridge Regression

Bayesian linear regression places prior distributions on the weights and updates them with observed data. This naturally handles the expert-to-data transition: expert-assigned weights become informative priors.

```python
from sklearn.linear_model import BayesianRidge
import numpy as np

# X: feature matrix, y: highlight counts (or log-transformed)
model = BayesianRidge(
    alpha_1=1e-6, alpha_2=1e-6,  # prior on noise precision
    lambda_1=1e-6, lambda_2=1e-6,  # prior on weight precision
    fit_intercept=True
)
model.fit(X, y)

# Posterior mean weights
weights = model.coef_

# Posterior uncertainty on each weight
# (useful for identifying which weights are still uncertain)
weight_std = np.sqrt(np.diag(model.sigma_))  # posterior covariance
```

**Strengths**: Produces both point estimates and uncertainty intervals for each weight. Weights with wide posterior intervals indicate questions where more data would be most valuable. Scikit-learn's `BayesianRidge` is simple to use and computationally cheap.

**Weaknesses**: The scikit-learn implementation uses a conjugate Gaussian prior, which does not naturally encode domain knowledge like "this question should have weight between 5 and 10." For count data, a Bayesian GLM (Poisson or negative binomial) would be more appropriate but requires libraries like PyMC or Stan.

### 3b. Informative Priors from Expert Weights

The real power of Bayesian methods for this problem: encode expert-assigned weights as prior beliefs and let data update them.

```python
import pymc as pm
import numpy as np

# Expert-assigned weights (from weighting-strategies.md)
expert_weights = np.array([8, 6, 8, 6, 8, 5, 8, 6])  # example
expert_uncertainty = np.array([3, 3, 3, 3, 3, 3, 3, 3])  # how confident

with pm.Model() as weight_model:
    # Priors centered on expert weights
    w = pm.Normal('weights',
                  mu=expert_weights,
                  sigma=expert_uncertainty,
                  shape=len(expert_weights))

    # Linear predictor
    mu = pm.math.dot(X, w)

    # Likelihood (Poisson for count data)
    highlights = pm.Poisson('highlights', mu=pm.math.exp(mu), observed=y)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Posterior mean weights
posterior_weights = trace.posterior['weights'].mean(dim=['chain', 'draw']).values
```

**Strengths**: Formally combines expert judgment with data. The posterior gradually shifts from prior (expert) to likelihood (data) as more observations accumulate, which is exactly the phased approach recommended in `weighting-strategies.md`. The posterior standard deviation on each weight tells you where expert and data disagree, highlighting questions that may need redesign.

**Weaknesses**: Requires PyMC or Stan, adding dependency complexity. MCMC sampling is slower than frequentist methods (though ~500 observations with 8 features is very manageable). The choice of prior distribution shape and scale requires thought.

### 3c. Bayesian Scoring Systems (Rudin et al.)

Research by Rudin and colleagues specifically addresses Bayesian methods for learning interpretable scoring systems from data. Their approach constrains coefficients to be small integers (compatible with point-based scoring), places priors on what "reasonable" coefficient values look like, and uses MCMC to explore the space of scoring systems.

This is directly relevant: we want integer-valued point assignments per question, constrained to a human-interpretable range, informed by both expert judgment and data. However, the implementation complexity is significant, and the benefit over simpler regression + rounding is marginal for our problem size.

### Recommendation

For a system with ~8 features and ~500 observations, **Bayesian Ridge regression from scikit-learn** is the pragmatic choice. It provides uncertainty estimates without requiring MCMC infrastructure. If you later want to encode strong expert priors or move to a proper count model, PyMC provides a clear upgrade path.

---

## 4. Bootstrap Weight Estimation

### Purpose

Bootstrapping answers the question: "How stable are the learned weights?" If small perturbations to the training data cause large changes in a weight, that weight is unreliable and should be treated with caution.

### Implementation

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def bootstrap_weights(X, y, n_bootstrap=1000, random_state=42):
    """Learn weights on bootstrap resamples to assess stability."""
    rng = np.random.RandomState(random_state)
    n = len(y)
    all_weights = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X[indices], y[indices]

        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X_boot, y_boot)
        all_weights.append(model.coef_[0])

    all_weights = np.array(all_weights)

    # Point estimate: mean across bootstraps
    mean_weights = all_weights.mean(axis=0)

    # Stability: coefficient of variation (std / |mean|)
    std_weights = all_weights.std(axis=0)
    cv = std_weights / (np.abs(mean_weights) + 1e-10)

    # 95% confidence intervals
    ci_lower = np.percentile(all_weights, 2.5, axis=0)
    ci_upper = np.percentile(all_weights, 97.5, axis=0)

    return mean_weights, std_weights, cv, ci_lower, ci_upper
```

### Interpreting Results

| Coefficient of Variation | Interpretation | Action |
|-------------------------|----------------|--------|
| CV < 0.3 | Stable weight | Trust the learned value |
| CV 0.3 - 0.7 | Moderate stability | Use but monitor with new data |
| CV > 0.7 | Unstable weight | Consider fixing to expert value or dropping question |

### Sign Stability

Beyond magnitude stability, check whether the bootstrap weight changes sign across resamples:

```python
sign_consistency = (all_weights > 0).mean(axis=0)
# sign_consistency close to 1.0 or 0.0 = stable direction
# sign_consistency near 0.5 = weight direction is unreliable
```

A question whose weight flips between positive and negative across bootstraps is not reliably predictive and should be flagged for review.

### Weighted Bootstrap for Small Samples

With ~500 observations, standard bootstrap works well. If the dataset were smaller (< 100), a weighted bootstrap could improve representation of the underlying population by incorporating importance weights, reducing bias from limited sample sizes.

### Practical Use

Run bootstrap analysis **after** fitting the primary regression model. Report the confidence intervals alongside the point estimates to stakeholders. If a weight's 95% CI includes zero, that question is not a statistically significant predictor at the 5% level.

---

## 5. Online Learning

### Motivation

As new articles are scored and engagement data arrives over time, weights should adapt without requiring a full re-fit. Online learning methods update weights incrementally with each new observation.

### 5a. Stochastic Gradient Descent (SGD)

The simplest online approach: after each new article's engagement is observed, perform a gradient update on the weights.

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class OnlineWeightLearner:
    def __init__(self, n_features, initial_weights=None, learning_rate=0.01):
        if initial_weights is not None:
            self.weights = np.array(initial_weights, dtype=float)
        else:
            self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate
        self.n_updates = 0

    def predict_score(self, features):
        """Predict engagement probability (0-1)."""
        raw = np.dot(self.weights, features) + self.bias
        return sigmoid(raw)

    def update(self, features, engaged):
        """Update weights after observing engagement.

        Args:
            features: binary question responses (array of 0/1)
            engaged: 1 if article was highlighted, 0 otherwise
        """
        predicted = self.predict_score(features)
        error = engaged - predicted

        # Gradient update (logistic loss)
        self.weights += self.lr * error * features
        self.bias += self.lr * error
        self.n_updates += 1

    def get_score(self, features):
        """Convert to 0-100 score."""
        return int(100 * self.predict_score(features))
```

**Strengths**: Simple, no need to store historical data. Can start from expert weights and gradually adapt. Each update is O(n_features), essentially instant.

**Weaknesses**: Learning rate is a critical hyperparameter. Too high and weights oscillate; too low and adaptation is glacially slow. No convergence guarantees without a decreasing learning rate schedule. Sensitive to the order in which observations arrive.

### 5b. Exponentially Weighted Moving Regression (EWMR)

A more principled online approach from Boyd et al. that generalizes EWMA to regression. Each historical observation is weighted by lambda^(t-i), where lambda is a forgetting factor (typically 0.95-0.99).

```python
import numpy as np

class EWMRWeightLearner:
    """Exponentially weighted moving regression for weight learning.

    Uses a forgetting factor (half_life) to downweight older observations,
    allowing weights to adapt to changing engagement patterns.
    """
    def __init__(self, n_features, half_life=100):
        self.n = n_features
        self.lam = 0.5 ** (1.0 / half_life)  # forgetting factor
        # Sufficient statistics (can be updated recursively)
        self.S_xx = np.eye(self.n) * 0.01  # regularization
        self.S_xy = np.zeros(self.n)
        self.weights = np.zeros(self.n)

    def update(self, features, highlight_count):
        """Update with a new observation."""
        x = np.array(features, dtype=float)
        y = float(highlight_count)

        # Exponential decay of existing statistics
        self.S_xx = self.lam * self.S_xx + np.outer(x, x)
        self.S_xy = self.lam * self.S_xy + x * y

        # Re-solve for weights
        self.weights = np.linalg.solve(self.S_xx, self.S_xy)

    @property
    def half_life_articles(self):
        """Number of articles for influence to decay by half."""
        return int(np.log(0.5) / np.log(self.lam))
```

**Strengths**: Principled forgetting factor that downweights stale data. Recursive computation means no need to store history. The half-life parameter is intuitive: "how many articles until old data is half as important?" For a reader processing ~50 articles/month, a half-life of 100 means roughly 2 months of recency weighting.

**Weaknesses**: Assumes a linear model (not ideal for count data). The forgetting factor means effective sample size is always bounded, even with lots of data, so estimates never fully converge. Sensitive to the half-life choice.

### 5c. Practical Online Update Cadence

For our system, true per-article online updates are impractical because engagement data (highlights) arrives with a significant lag: a reader may not finish an article for days or weeks. A more realistic approach:

1. **Weekly batch**: Re-fit the model on all data with a recency weighting (EWMR-style) every week via a scheduled job.
2. **Monthly full refit**: Run a full logistic regression on all data monthly, producing new weights with bootstrap confidence intervals.
3. **Quarterly review**: Compare learned weights against expert weights, flag major divergences for human review.

This hybrid approach gets the benefits of adaptation without the instability risks of true online learning.

---

## 6. Cold-Start Problem

### How Many Labeled Examples Are Needed?

The literature provides several rules of thumb for minimum sample sizes in regression with binary predictors.

**Events-per-variable (EPV) rule**: The traditional guideline is at least 10 events (highlighted articles) per predictor variable. With 8 features, that means at least 80 highlighted articles. Some researchers argue 15-20 EPV is safer, especially when predictor effects are weak.

However, research by Austin and Steyerberg has shown that "there is no single rule based on EPP that would guarantee an accurate estimation of logistic regression parameters." The required N depends on the magnitude of predictor effects, the overall outcome rate, and the distribution of predictors.

**Empirical findings from regression research**: With very low variance in the data, N >= 8 can identify patterns, but N >= 25 is required for high-variance scenarios. For multiple regression with covariates, the common rules of thumb range from N > 50 + 8p (where p is number of predictors) to N > 50p, depending on the source.

**For our specific problem** (8 features, binary/categorical): With ~500 scored articles, assuming ~30-40% have at least one highlight, we have approximately 150-200 highlighted articles. This gives us ~19-25 EPV, which is in the adequate range for stable weight estimation. We are past the cold-start phase.

### Cold-Start Strategy

For systems that are just starting out:

| Phase | N (scored articles) | Highlighted | Approach |
|-------|-------------------|-------------|----------|
| Bootstrap | 0-50 | < 15 | Expert weights only |
| Early signal | 50-100 | 15-30 | Compute per-question correlations, no regression |
| Minimum viable | 100-200 | 30-60 | Logistic regression with strong regularization (C=0.1) |
| Stable estimation | 200-500 | 60-150 | Standard logistic regression, bootstrap CIs |
| Mature | 500+ | 150+ | Full toolkit: regression, Spearman optimization, Bayesian |

### Feature-Level Diagnostics at Each Phase

Even before you have enough data for full regression, you can compute point-biserial correlations between each binary question and the engagement outcome:

```python
from scipy.stats import pointbiserialr

for i, question_name in enumerate(question_names):
    corr, p = pointbiserialr(X[:, i], y_binary)
    print(f"{question_name}: r={corr:.3f}, p={p:.3f}")
```

This tells you which individual questions are predictive, even with small samples. Questions with near-zero or negative correlations are candidates for redesign regardless of sample size.

---

## 7. Expert-Set vs. Learned Weights: Tradeoffs

### The Evidence

Research on expert judgment vs. data-driven weights consistently finds that data-driven approaches outperform expert judgment in aggregate, but with important caveats.

**Data-driven advantages**:
- Linear models of expert judgment outperform the experts' own holistic judgments (Dawes, 1979; the "broken leg" problem).
- Human decision-makers weight predictors inconsistently across instances, while a fixed model applies weights uniformly.
- Supervised ML models "significantly outperform professional jury consultants under identical informational constraints."

**Expert judgment advantages**:
- Experts can encode domain knowledge about causal mechanisms that data alone cannot reveal.
- Experts can anticipate regime changes (e.g., "this question will matter more as my reading focus shifts").
- Expert weights require no historical data and work from day one.
- Expert weights are transparent and debuggable: "this article scored low because it failed the practitioner voice check."

**Neither is universally superior**: "The better method varies as a function of factors such as availability, quality, extent and format of data, suggesting that the two approaches can complement each other."

### A Hybrid Framework

The LLM-Rubric paper (ACL 2024) provides a compelling model for combining expert rubric design with learned calibration. Their approach:

1. **Experts design the rubric**: questions, dimensions, and response options (our current system)
2. **LLM answers each question**: produces distributions over responses (our categorical scoring)
3. **A calibration network combines responses**: a small feed-forward network with learned weights maps question-response distributions to a calibrated score
4. **Training**: the network weights are learned from labeled data (human judgments or, in our case, highlight counts)

The calibration network achieved 2x improvement in RMSE over the uncalibrated baseline (from 0.90 to 0.42 RMSE), demonstrating that even well-designed rubrics benefit substantially from weight calibration.

### Practical Recommendations

**Keep expert weights as the deployed default**. They are interpretable, require no data, and provide a stable baseline. Use learned weights for three purposes:

1. **Validation**: Do learned weights agree with expert intuition? If logistic regression assigns near-zero weight to "named framework" but the expert weighted it at 8 points, investigate why.

2. **Discovery**: Learned weights may reveal that a question the expert considered supplementary is actually the strongest predictor. Use this to inform expert weight revisions.

3. **Bounded overrides**: If a learned weight is significantly different from the expert weight (outside the bootstrap 95% CI of the expert value), consider adjusting the expert weight toward the learned value, but not all the way. A blended weight like `0.7 * expert + 0.3 * learned` preserves interpretability while incorporating data signal.

---

## 8. Multi-Objective Calibration

### The Problem

We have multiple potential engagement signals, not just highlight count:

- **Highlight count**: Number of distinct highlights made
- **Highlighted word count**: Total words highlighted (captures "depth" of engagement)
- **Reading completion**: How much of the article was read (from Readwise `reading_progress`)
- **Time in reader**: How long the article was open (if available)

Optimizing weights for one signal may hurt correlation with another.

### Approach 1: Weighted Sum of Objectives

Combine multiple engagement signals into a single composite target, then optimize weights against that.

```python
import numpy as np

# Normalize each signal to 0-1 range
hl_norm = highlight_counts / highlight_counts.max()
words_norm = highlighted_words / highlighted_words.max()
progress_norm = reading_progress  # already 0-1

# Weighted composite engagement score
engagement = 0.5 * hl_norm + 0.3 * words_norm + 0.2 * progress_norm
```

**Strengths**: Reduces to a single-objective problem. Simple to implement. The weights on engagement signals encode your belief about what "engagement" means.

**Weaknesses**: The engagement signal weights are themselves expert-assigned and uncalibrated. You are now calibrating weights against an already-weighted target.

### Approach 2: Multi-Objective Pareto Optimization

Find the set of weight vectors that are Pareto-optimal: no other weight vector improves correlation with one signal without hurting correlation with another.

```python
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr
import numpy as np

def multi_objective(weights, X, y_highlights, y_words, y_progress):
    """Return negative correlations with multiple engagement signals."""
    scores = X @ weights
    rho_hl, _ = spearmanr(scores, y_highlights)
    rho_words, _ = spearmanr(scores, y_words)
    rho_progress, _ = spearmanr(scores, y_progress)
    # Weighted sum (adjust weights based on priority)
    return -(0.5 * rho_hl + 0.3 * rho_words + 0.2 * rho_progress)
```

For true Pareto optimization (without pre-specifying signal weights), use a multi-objective evolutionary algorithm like NSGA-II. However, for our problem with 2-3 objectives and 8 decision variables, the Pareto front is likely narrow -- the weight vectors that maximize highlight-count correlation will also do reasonably well on highlighted-word correlation, since these signals are themselves correlated.

### Approach 3: Primary + Constraint

Optimize for the primary signal (highlight count) subject to a constraint that correlation with secondary signals stays above a threshold:

```python
# Pseudo-optimization:
# maximize Spearman(score, highlight_count)
# subject to: Spearman(score, reading_progress) >= 0.10
```

This ensures the scoring system does not become pathologically tuned to one engagement signal at the expense of others.

### Recommendation

**Use Approach 1 with highlight count as the dominant signal** (weight 0.6-0.8). Highlight count is the strongest proxy for "this article was valuable enough to save passages from," which directly matches the scoring system's stated goal. Use reading progress as a secondary signal (weight 0.2-0.3) primarily to filter the calibration dataset: exclude articles with < 10% reading progress, since zero highlights on an unread article is not informative about article quality.

Do not add highlighted word count as a separate signal unless analysis shows it diverges meaningfully from highlight count. In practice, the two are highly correlated.

---

## 9. Implementation Recommendations

### Immediate Actions (With Current ~500 Article Dataset)

**Step 1: Baseline diagnostic** (extends existing `cal-dimensions` command)

```python
# For each LLM question, compute:
# 1. Point-biserial correlation with highlight_count
# 2. Point-biserial correlation with binary engagement (any highlight)
# 3. Proportion of "yes" answers (base rate)
```

Questions with base rate > 0.9 or < 0.1 have low discrimination power and may need redesign regardless of weight.

**Step 2: Logistic regression weight learning**

```python
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

# Encode current categorical responses as binary features
# (some questions are already binary; multi-class ones get one-hot encoded)

model = LogisticRegressionCV(
    Cs=10,           # test 10 regularization values
    cv=5,            # 5-fold cross-validation
    scoring='roc_auc',
    penalty='l2',
    max_iter=1000,
    random_state=42
)
model.fit(X, y_binary)

print(f"Best C: {model.C_[0]:.4f}")
print(f"Cross-validated AUC: {model.scores_[1].mean(axis=0).max():.3f}")

for name, weight in zip(feature_names, model.coef_[0]):
    print(f"  {name:30s}: {weight:+.3f}")
```

**Step 3: Bootstrap stability analysis**

Run 1000 bootstrap iterations. Report the 95% CI for each weight. Flag any weight whose CI spans zero.

**Step 4: Compare learned vs. expert weights**

Create a comparison table showing:

| Question | Expert Weight | Learned Weight | Bootstrap 95% CI | Agreement |
|----------|--------------|----------------|-------------------|-----------|
| standalone_passages | 9/17/25 (ordinal) | +2.3 | [1.1, 3.5] | Aligned |
| novel_framing | 15 | +0.8 | [-0.2, 1.8] | Expert overweights |
| ... | ... | ... | ... | ... |

**Step 5: Evaluate with Spearman correlation**

After deriving learned weights, compute the Spearman correlation of the learned-weight composite score against highlights. Compare against the current expert-weight composite score. The improvement (or lack thereof) determines whether to adopt learned weights.

### Integration with Existing Calibration Toolkit

Add a new `just cal-weights` command that:

1. Loads the calibration dataset (reusing `cal_data.load_dataset`)
2. Encodes LLM question responses as features
3. Fits logistic regression with cross-validation
4. Runs bootstrap analysis
5. Compares learned vs. current (expert) weights
6. Reports Spearman correlation improvement

This fits naturally alongside the existing `cal-report`, `cal-dimensions`, `cal-misses`, and `cal-trends` commands.

### When to Update Weights

- **Monthly**: Re-run `cal-weights` to check if learned weights have shifted significantly
- **After scoring version changes**: When the rubric questions change, old weight estimates are invalid; reset to expert weights and re-enter the cold-start cycle
- **After engagement pattern shifts**: If `cal-trends` shows calibration degrading, re-fit weights on recent data (last 3-6 months)

---

## 10. Grid Search as a Simple Alternative

For a system with only 8 questions and expert-assigned weights in discrete tiers (e.g., 0/3/5/8/10/12/15 points per question), exhaustive grid search over reasonable weight combinations is feasible:

```python
from itertools import product
from scipy.stats import spearmanr
import numpy as np

# Define candidate weights for each question
weight_options = {
    'standalone_passages': [5, 8, 10, 15],
    'novel_framing': [5, 8, 10, 15],
    'content_type': [3, 5, 8, 10],
    'author_conviction': [5, 8, 10, 12],
    'practitioner_voice': [3, 5, 8, 10],
    'completeness': [2, 3, 5],
    'named_framework': [5, 8, 10, 12],
    'applicable_ideas': [5, 8, 10, 13],
}

best_rho = -1
best_weights = None

for combo in product(*weight_options.values()):
    weights = np.array(combo)
    scores = X @ weights
    rho, _ = spearmanr(scores, y)
    if rho > best_rho:
        best_rho = rho
        best_weights = dict(zip(weight_options.keys(), combo))

print(f"Best Spearman rho: {best_rho:.3f}")
print(f"Best weights: {best_weights}")
```

With 4 options per question and 8 questions, this is 4^8 = 65,536 combinations -- trivially fast. This approach has the advantage of producing integer weights that are immediately interpretable and deployable, without the need to round or rescale regression coefficients.

**Enhancement**: Add a cross-validation loop around the grid search. Split data into 5 folds, find the best weights on 4 folds, evaluate on the held-out fold. This prevents overfitting to the specific data sample.

---

## Summary of Approaches

| Method | Complexity | Data Required | Produces | Best For |
|--------|-----------|---------------|----------|----------|
| OLS Linear Regression | Low | 100+ articles | Continuous weights | Quick diagnostic |
| Logistic Regression | Low | 100+ articles, 50+ engaged | Binary-calibrated weights | Primary weight learning |
| Poisson/NegBin Regression | Medium | 100+ articles | Count-calibrated weights | When engagement intensity matters |
| Spearman Optimization | Medium | 200+ articles | Rank-optimal weights | When rank ordering is all you care about |
| Bayesian Ridge | Medium | 50+ articles | Weights + uncertainty | Incorporating expert priors |
| Full Bayesian (PyMC) | High | 100+ articles | Posterior distributions | Expert-to-data transition |
| Bootstrap Analysis | Low | Same as base method | Confidence intervals | Weight stability assessment |
| Online SGD | Low | Streaming | Adaptive weights | Continuous adaptation |
| EWMR | Medium | Streaming | Recency-weighted | Adapting to preference drift |
| Grid Search | Low | 200+ articles | Integer weights | Interpretable, deployable weights |

### Recommended Implementation Order

1. **Now**: Logistic regression + bootstrap (validates or challenges expert weights)
2. **Next month**: Grid search over integer weight candidates (produces deployable weights)
3. **Ongoing**: Monthly re-evaluation via `cal-weights` command
4. **If needed**: Bayesian updating for smooth expert-to-data transition
5. **Long-term**: EWMR or periodic re-fit for preference drift adaptation

---

## References and Sources

- [LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts (ACL 2024)](https://arxiv.org/html/2501.00274v1)
- [Microsoft LLM-Rubric GitHub Repository](https://github.com/microsoft/LLM-Rubric)
- [A Solution to Minimum Sample Size for Regressions (Jenkins & Quintana-Ascencio, 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7034864/)
- [Minimum Sample Size for Developing a Multivariable Prediction Model (Riley et al., 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6519266/)
- [No Rationale for 1 Variable per 10 Events Criterion for Binary Logistic Regression](https://link.springer.com/article/10.1186/s12874-016-0267-3)
- [Exponentially Weighted Moving Models (Luxenberg & Boyd, 2024)](https://arxiv.org/html/2404.08136v1)
- [A Bayesian Approach to Learning Scoring Systems (Rudin et al.)](https://pubmed.ncbi.nlm.nih.gov/27441407/)
- [Bayesian Ridge Regression - scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- [PoissonRegressor - scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- [NegativeBinomial - statsmodels Documentation](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html)
- [SciPy Differential Evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
- [SciPy Spearman Correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
- [An Importance Weighted Feature Selection Stability Measure (JMLR)](https://www.jmlr.org/papers/volume22/20-366/20-366.pdf)
- [Bootstrap Assessment of the Stability of Multivariable Models](https://journals.sagepub.com/doi/pdf/10.1177/1536867X0900900403)
- [Weighted Correlation in Python (matthijsz/weightedcorr)](https://github.com/matthijsz/weightedcorr)
- [Comparison of Weighting Methods Used in Multicriteria Decision Analysis Frameworks](https://becarispublishing.com/doi/10.2217/cer-2018-0102)
- [A Survey of Human Judgement and Quantitative Forecasting Methods (Royal Society)](https://royalsocietypublishing.org/rsos/article/8/2/201187/95920/A-survey-of-human-judgement-and-quantitative)
