# Active Learning and Feedback Loops for Article Scoring

## Executive Summary

A single-user article triage system can improve substantially over time by combining implicit engagement signals (highlights, reading time) with explicit feedback (ratings, flagging) in a closed-loop learning system. The key challenge is operating in a low-data regime -- hundreds of articles, not millions -- which rules out deep learning approaches and favors lightweight online models with Bayesian uncertainty estimation. The most practical architecture is a hybrid system: an LLM-based scorer provides the baseline, while a lightweight statistical model learns user-specific calibration weights from accumulated feedback, with time-decay weighting to handle shifting interests. Active learning (selecting which articles to ask the user to rate) accelerates model improvement by targeting the articles where the model is most uncertain, and bandit-style explore/exploit logic prevents the system from becoming a self-reinforcing filter bubble.

---

## 1. Cold Start Strategies: Initial Model with Very Little Data

### The Problem

With fewer than ~50 labeled articles, there is insufficient data to train a reliable discriminative model. The system must produce useful rankings from day one while collecting the feedback needed to improve.

### Approaches

**Phase 0: LLM-as-judge baseline (zero labeled data).** The current architecture already handles this: Claude scores articles on four dimensions (quotability, surprise, argument quality, applicable insight). This is effectively a zero-shot content quality model that requires no user-specific training data. Research on LLM-as-judge reliability shows binary and low-cardinality judgments are more consistent than fine-grained numeric scores ([Arize AI, 2025](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/)), which aligns with the system's current design of four 0-25 sub-scores.

**Phase 1: Prior-informed calibration (5-50 labeled articles).** Once the user has engaged with a handful of articles (highlighted some, skipped others), fit a simple Bayesian logistic regression that maps the four LLM dimension scores to a binary "user found this valuable" label. A Bayesian approach (e.g., `scikit-learn`'s `BayesianRidge` or a lightweight PyMC model) provides uncertainty estimates even with tiny datasets, and the prior acts as regularization to prevent overfitting on sparse data.

```python
# Conceptual: Bayesian recalibration of LLM scores
from sklearn.linear_model import BayesianRidge

# Features: [quotability, surprise, argument_quality, applicable_insight]
# Target: binary (1 = user highlighted/engaged, 0 = skipped)
model = BayesianRidge(alpha_init=1.0, lambda_init=1.0)
model.fit(X_train, y_train)

# Prediction includes uncertainty
y_pred, y_std = model.predict(X_new, return_std=True)
```

**Phase 2: Feature expansion (50-200 labeled articles).** Add content-derived features beyond the four LLM scores: article length, source domain, topic embeddings (from a sentence transformer), time-of-day, day-of-week. The Bayesian calibration model can incorporate these without overfitting because the prior constrains the weights.

**Phase 3: Full personalized model (200+ labeled articles).** At this scale, a gradient-boosted model (LightGBM, XGBoost) or a linear model with interaction features becomes viable. The LLM scores become features rather than the sole signal.

**Key insight from cold-start research:** Meta-learning approaches that learn "how to learn from few examples" are powerful but add substantial complexity ([Hao et al., 2020](https://xiaojingzi.github.io/publications/APWeb20-HAO-et-al-FewshotRec.pdf)). For a single-user system, the simpler staged approach above is more practical -- the LLM already provides a strong prior that a collaborative filtering cold-start approach would lack.

---

## 2. Active Learning: Which Articles to Ask the User to Rate

### Why Active Learning Matters Here

The user's labeling budget is severely limited -- they will only rate a few articles per day voluntarily. Choosing which articles to present for explicit feedback has an outsized effect on model improvement speed.

### Query Strategies

**Uncertainty sampling** selects articles where the model's prediction is most uncertain. For a Bayesian calibration model, this is directly available as the posterior predictive variance. Articles near the decision boundary (predicted score ~50 with high variance) are the most informative to label ([Settles, 2012](https://burrsettles.com/pub/settles.activelearning.pdf); [Lilian Weng, 2022](https://lilianweng.github.io/posts/2022-02-20-active-learning/)).

```python
# Select articles with highest predictive uncertainty
y_pred, y_std = model.predict(X_unlabeled, return_std=True)
most_uncertain_idx = np.argsort(-y_std)[:k]  # top-k most uncertain
```

**Diversity sampling** ensures selected articles span different topics and sources rather than clustering in one uncertain region. This is critical early on when the model's uncertainty is poorly calibrated. A practical approach: embed articles with a sentence transformer, cluster them, then pick the most uncertain article from each cluster.

**Cold-start hybrid strategy.** Research shows that diversity-based methods outperform uncertainty sampling when labeled data is scarce, because early-stage uncertainty estimates are unreliable ([Hacohen et al., 2022](https://arxiv.org/abs/2403.03728)). The recommended schedule:

| Labeled Articles | Strategy | Rationale |
|---|---|---|
| 0-30 | Diversity sampling (stratified by topic/source) | Uncertainty estimates unreliable |
| 30-100 | 70% uncertainty + 30% diversity | Uncertainty becoming useful |
| 100+ | 90% uncertainty + 10% random exploration | Model calibrated, avoid blind spots |

**Expected model change** is an alternative strategy that selects articles whose labels would most change the model's parameters. For a linear model, this is proportional to the gradient magnitude. This is more expensive to compute but can be more efficient than uncertainty sampling in some regimes ([Settles, 2012](https://burrsettles.com/pub/settles.activelearning.pdf)).

### Practical Implementation

Present active learning queries naturally in the UI:

- After the user reads an article scored in the 40-60 range: "Was this article worth your time? [Yes / No]"
- In a weekly digest: "We're least sure about these 5 articles. Quick rating?" with thumbs up/down
- On the triage page: highlight 2-3 articles with a subtle "Help us learn" badge

The key constraint is user friction. Active learning works best when feedback is binary (valuable / not valuable) and requires a single click. Multi-dimensional ratings generate richer signal but lower response rates.

---

## 3. Online Learning: Incremental Updates vs. Periodic Retraining

### The Spectrum

| Approach | Update Frequency | Pros | Cons |
|---|---|---|---|
| Full retraining | Weekly/monthly | Clean model, no accumulated errors | Wasteful, slow to adapt |
| Mini-batch updates | Daily | Balanced freshness and stability | Requires careful learning rate |
| True online learning | Per-article | Fastest adaptation | Risk of instability, catastrophic forgetting |

### Recommendation: Hybrid Approach

For a single-user system processing ~10-50 articles/day, the practical sweet spot is:

1. **Per-feedback incremental update** of the Bayesian calibration weights using online Bayesian updating or `partial_fit()`. This keeps the model responsive to new signals.

2. **Weekly full retraining** from all historical data (with time-decay weighting). This corrects any drift in the online updates and incorporates any new features or model architecture changes.

3. **LLM re-scoring is expensive** and should not be repeated for already-scored articles unless the prompt or model changes. The LLM scores are treated as cached features.

### Implementation with scikit-learn

`SGDClassifier` and `SGDRegressor` support `partial_fit()` for incremental updates ([scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)):

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', learning_rate='invscaling',
                      eta0=0.1, power_t=0.25, warm_start=True)

# Initial training
model.partial_fit(X_initial, y_initial, classes=[0, 1])

# Incremental update when new feedback arrives
model.partial_fit(X_new, y_new)
```

The `invscaling` learning rate decays as more data is seen, providing natural stabilization. The `power_t` parameter controls the decay rate.

### Implementation with River

For true streaming/online learning, the [River library](https://riverml.xyz/) provides algorithms designed for one-sample-at-a-time updates ([Montiel et al., 2021](https://jmlr.csail.mit.edu/papers/volume22/20-1380/20-1380.pdf)):

```python
from river import linear_model, preprocessing, optim

model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
    optimizer=optim.SGD(lr=0.01)
)

# Learn one article at a time
for x, y in stream:
    y_pred = model.predict_proba_one(x)
    model.learn_one(x, y)
```

River's advantage is that all its algorithms are designed for streaming from the ground up, with built-in concept drift detection. Its disadvantage is a smaller ecosystem and less battle-tested than scikit-learn for batch operations.

### Versioning and Rollback

Maintain model snapshots (weekly) so that if online updates degrade performance, the system can roll back:

```python
import pickle
from datetime import datetime

def save_snapshot(model, metrics):
    snapshot = {
        'model': pickle.dumps(model),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'n_samples': metrics['total_feedback_count']
    }
    # Store in SQLite alongside article data
```

---

## 4. Concept Drift Detection and Handling

### Types of Drift in This System

- **Gradual interest shift:** The user slowly moves from reading about ML infrastructure to reading about climate policy. This is the most common type and manifests as slowly declining model accuracy.

- **Sudden interest shift:** The user starts a new job or hobby, abruptly changing what they find valuable. Model accuracy drops sharply.

- **Recurring patterns:** Seasonal interests (tax articles in March, holiday recipes in November) that the model should remember rather than treat as drift.

- **Quality standard drift:** The user becomes more sophisticated in a domain, raising their bar for what counts as "valuable" even on the same topics.

### Detection Methods

**Performance monitoring** is the simplest approach. Track rolling accuracy (or log-loss) of the model's predictions against actual user engagement. A sustained drop signals drift.

```python
from collections import deque

class DriftMonitor:
    def __init__(self, window_size=50, alert_threshold=0.15):
        self.recent_errors = deque(maxlen=window_size)
        self.baseline_error_rate = None
        self.alert_threshold = alert_threshold

    def update(self, predicted_valuable, actually_valuable):
        error = int(predicted_valuable != actually_valuable)
        self.recent_errors.append(error)

        if len(self.recent_errors) == self.recent_errors.maxlen:
            current_error_rate = sum(self.recent_errors) / len(self.recent_errors)
            if self.baseline_error_rate is None:
                self.baseline_error_rate = current_error_rate
            elif current_error_rate - self.baseline_error_rate > self.alert_threshold:
                return "drift_detected"
            # Slowly update baseline
            self.baseline_error_rate = (
                0.95 * self.baseline_error_rate + 0.05 * current_error_rate
            )
        return "ok"
```

**ADWIN (Adaptive Windowing)** is a more sophisticated algorithm that maintains a variable-length window and detects drift by comparing statistical properties of sub-windows. It automatically adapts the window size based on the rate of change, making it effective for both gradual and sudden drift ([Bifet and Gavalda, 2007](https://www.researchgate.net/publication/220907178_Learning_from_Time-Changing_Data_with_Adaptive_Windowing)). Available in River:

```python
from river import drift

adwin = drift.ADWIN(delta=0.002)

for error in error_stream:
    adwin.update(error)
    if adwin.drift_detected:
        trigger_retraining()
```

**Page-Hinkley test** detects abrupt changes in the mean of a signal, complementing ADWIN's strength with gradual drift. Using both provides coverage across drift types.

**Feature distribution monitoring:** Track the distribution of LLM scores and article topics over time. If the input distribution shifts (the user is now seeing different types of articles), the model may need recalibration even if its error rate hasn't changed yet. This is data drift preceding concept drift.

### Handling Strategies

**Time-decay weighting** (see Section 7) is the first line of defense -- it naturally down-weights old data, allowing the model to adapt.

**Triggered retraining:** When drift is detected, retrain the calibration model on a shorter lookback window (e.g., last 60 days instead of all time). This sacrifices sample size for relevance.

**Ensemble with recency weighting:** Maintain two models -- one trained on all historical data, one on recent data only. Blend their predictions, increasing the weight of the recent model when drift is detected:

```python
alpha = 0.7 if drift_detected else 0.3  # weight on recent model
final_score = alpha * recent_model.predict(x) + (1 - alpha) * full_model.predict(x)
```

**Periodic interest profiling:** Monthly, generate a topic distribution summary (via LLM or topic model) from the user's recent highlights. Compare it to the previous month's profile. Large shifts trigger proactive model updates and can be surfaced to the user: "Your reading interests seem to have shifted toward X. Should I adjust scoring?"

---

## 5. Feedback Mechanisms: Implicit vs. Explicit

### Implicit Feedback Signals

Implicit feedback is collected automatically with zero user effort, but requires careful interpretation ([Hu et al., 2008](https://ieeexplore.ieee.org/document/4781121)):

| Signal | Strength | Noise Level | Available in Readwise Reader |
|---|---|---|---|
| Highlighted passage | Strong positive | Low | Yes |
| Number of highlights | Strong positive | Low | Yes |
| Saved to permanent notes | Strong positive | Low | Partial (via API) |
| Opened/read | Weak positive | High | Yes (read status) |
| Time spent reading | Moderate positive | Medium | No (not exposed) |
| Archived without highlights | Moderate negative | Medium | Yes |
| Never opened | Ambiguous (might not have seen it) | Very high | Yes |

**Key insight:** The absence of interaction is not the same as negative feedback. An article the user never saw cannot be treated the same as one they opened and abandoned. This is the "missing not at random" problem in implicit feedback systems.

**Confidence-weighted implicit feedback:** Assign confidence levels to implicit signals rather than treating them as binary labels:

```python
def compute_implicit_label(article):
    """Returns (label, confidence) tuple."""
    if article.highlight_count > 0:
        return (1.0, 0.6 + min(0.4, article.highlight_count * 0.1))
    elif article.is_read and article.highlight_count == 0:
        # Read but not highlighted -- likely not valuable
        return (0.0, 0.4)
    elif article.is_archived and not article.is_read:
        # Archived without reading -- weak negative
        return (0.0, 0.2)
    else:
        # Never interacted -- don't use as training data
        return (None, 0.0)
```

### Explicit Feedback Signals

Explicit feedback is unambiguous but expensive to collect:

| Signal | Information Value | User Effort | Recommended |
|---|---|---|---|
| Binary: valuable / not valuable | High | Very low | Yes -- primary |
| Thumbs up / thumbs down on score | High | Very low | Yes -- for disagreements |
| "Why was this scored high/low?" free text | Very high | High | No -- too costly |
| Re-rate on 1-5 scale | Moderate | Low | No -- adds cognitive load |
| Flag as "definitely wrong" | Very high for edge cases | Very low | Yes -- for outliers |

### Combining Implicit and Explicit

The recommended approach treats explicit feedback as ground truth and implicit feedback as a noisy proxy:

```python
def get_training_signal(article):
    # Explicit feedback always wins
    if article.explicit_rating is not None:
        return article.explicit_rating, 1.0  # label, confidence

    # Fall back to implicit
    return compute_implicit_label(article)
```

When training, use sample weights proportional to confidence:

```python
labels, confidences = zip(*[get_training_signal(a) for a in articles])
# Filter out None labels (no signal at all)
valid = [(l, c) for l, c in zip(labels, confidences) if l is not None]
model.fit(X, y, sample_weight=confidences)
```

### Feedback Loop Risks

A critical risk identified in the RecSys literature is the **feedback loop** or **filter bubble**: the model scores articles high, the user sees and engages with those articles, this engagement reinforces the model's beliefs, and the model becomes increasingly narrow ([Navigating the Feedback Loop in Recommender Systems, ACM RecSys 2023](https://dl.acm.org/doi/10.1145/3604915.3610246)). The explore/exploit strategies in Section 6 directly address this.

---

## 6. Bandit Approaches: Explore/Exploit for Article Recommendation

### Why Bandits Apply Here

The article triage system faces a classic explore/exploit dilemma:

- **Exploit:** Show articles the model is confident are high-value. Maximizes short-term user satisfaction.
- **Explore:** Show articles the model is uncertain about. Generates training signal and prevents filter bubbles.

Pure exploitation leads to filter bubbles. Pure exploration wastes the user's time. Bandit algorithms provide a principled balance.

### Epsilon-Greedy (Simplest)

Reserve a fraction epsilon of article slots for exploration:

```python
import random

def select_articles(scored_articles, n=20, epsilon=0.1):
    n_explore = max(1, int(n * epsilon))
    n_exploit = n - n_explore

    # Exploit: top-scoring articles
    exploit = sorted(scored_articles, key=lambda a: a.score, reverse=True)[:n_exploit]

    # Explore: random selection from remaining
    remaining = [a for a in scored_articles if a not in exploit]
    explore = random.sample(remaining, min(n_explore, len(remaining)))

    result = exploit + explore
    random.shuffle(result)  # Don't reveal which are exploration
    return result
```

Epsilon should decrease over time as the model improves. A simple annealing schedule: `epsilon = max(0.05, 0.3 * (0.99 ** n_feedback_received))`.

### Thompson Sampling (Recommended)

Thompson sampling naturally balances exploration and exploitation through posterior sampling. For each article, sample a score from the model's posterior predictive distribution, then rank by sampled scores. Articles with high uncertainty will occasionally sample high, getting explored ([Agrawal and Goyal, 2013](https://arxiv.org/abs/1209.3352)):

```python
def thompson_sampling_rank(articles, model):
    """Rank articles using Thompson sampling from posterior."""
    rankings = []
    for article in articles:
        x = article.features
        # Bayesian model gives mean and std
        mean, std = model.predict(x.reshape(1, -1), return_std=True)
        # Sample from posterior
        sampled_score = np.random.normal(mean[0], std[0])
        rankings.append((article, sampled_score))

    return sorted(rankings, key=lambda x: x[1], reverse=True)
```

Thompson sampling is competitive with or superior to UCB and epsilon-greedy in article recommendation settings ([Li et al., 2010](https://arxiv.org/abs/1003.0146)). Its key advantage is that exploration is automatically focused on articles where the model is uncertain, and it diminishes as the model becomes confident.

### Upper Confidence Bound (UCB)

UCB selects articles by adding an exploration bonus to the predicted score:

```
ucb_score = predicted_score + beta * uncertainty
```

where `beta` controls the exploration-exploitation trade-off. Higher beta means more exploration. UCB is deterministic (unlike Thompson sampling) and provides theoretical regret bounds, but in practice Thompson sampling tends to perform similarly or better for recommendation ([Chapelle and Li, 2011](https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)).

### Contextual Bandits

The above approaches treat each article independently. Contextual bandits condition the explore/exploit decision on article features (topic, source, length), allowing the system to be more exploratory in unfamiliar topic areas while exploiting in well-understood ones.

The [contextualbandits](https://github.com/david-cortes/contextualbandits) Python library provides production-ready implementations that wrap scikit-learn classifiers:

```python
from contextualbandits.online import BootstrappedTS

model = BootstrappedTS(
    base_algorithm=SGDClassifier(loss='log_loss'),
    nchoices=2,  # valuable / not valuable
    nsamples=10,
    batch_train=True
)
```

### Practical Integration

In the triage UI, exploration articles can be presented in a dedicated "Discovery" section rather than mixed into the main feed. This sets user expectations and may increase tolerance for lower-quality suggestions.

Track exploration vs. exploitation outcomes separately in metrics:

```python
class ExploreExploitTracker:
    def __init__(self):
        self.exploit_hits = 0   # Exploited article user liked
        self.exploit_misses = 0
        self.explore_hits = 0   # Explored article user liked (discovery!)
        self.explore_misses = 0

    @property
    def explore_hit_rate(self):
        total = self.explore_hits + self.explore_misses
        return self.explore_hits / total if total > 0 else 0.0

    @property
    def discovery_value(self):
        """Articles found through exploration that exploit would have missed."""
        return self.explore_hits
```

---

## 7. Time-Decay Weighting: Recent Preferences Matter More

### The Case for Time Decay

User interests evolve. An article about Kubernetes that would have scored 90 six months ago might score 40 today because the user has moved on to a different technology. Time-decay weighting addresses this by reducing the influence of older feedback on the current model.

### Exponential Decay Function

The standard approach uses an exponential decay function to weight training samples ([Ding and Li, 2005](https://www.researchgate.net/publication/221162224_Dynamic_item-based_recommendation_algorithm_with_time_decay)):

```python
import math
from datetime import datetime, timedelta

def time_decay_weight(feedback_date, half_life_days=90):
    """Exponential decay with configurable half-life.

    half_life_days=90 means feedback from 90 days ago
    has half the weight of today's feedback.
    """
    age_days = (datetime.now() - feedback_date).days
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return max(decay, 0.01)  # Floor to prevent zero weights
```

The `half_life_days` parameter is the key tunable:

| Half-life | Behavior | Good for |
|---|---|---|
| 30 days | Aggressive decay, highly responsive | Rapidly shifting interests |
| 90 days | Moderate decay (recommended default) | Typical reading patterns |
| 180 days | Slow decay, more stable | Stable, deep interests |
| Infinite | No decay (all data weighted equally) | Unchanging preferences |

### Personalized Decay Rate

Different topics may warrant different decay rates. Technical topics might decay faster (frameworks change), while philosophical interests decay slower. A more sophisticated approach estimates the decay rate per topic cluster:

```python
def personalized_decay(feedback_date, topic_cluster, decay_rates):
    """Per-topic decay rates learned from historical engagement patterns."""
    half_life = decay_rates.get(topic_cluster, 90)  # default 90 days
    age_days = (datetime.now() - feedback_date).days
    return math.exp(-math.log(2) * age_days / half_life)
```

Research on personalized time decay functions confirms that user-specific decay rates outperform global ones ([Koren, 2009](https://dl.acm.org/doi/10.1145/1557019.1557072); [Liu et al., 2020](https://www.researchgate.net/publication/340361119_A_collaborative_filtering_recommendation_system_with_dynamic_time_decay)).

### Integration with Training

Apply time-decay weights as sample weights during model training:

```python
weights = [time_decay_weight(a.feedback_date) for a in training_articles]
model.fit(X_train, y_train, sample_weight=weights)
```

For online updates, combine time decay with feedback confidence:

```python
final_weight = time_decay_weight(article.feedback_date) * feedback_confidence
```

### EWMA for Score Calibration

Exponentially Weighted Moving Average (EWMA) provides a lightweight alternative to full retraining for tracking score calibration drift. If the model consistently over- or under-predicts, an EWMA correction factor adjusts predictions ([Boyd and Luxenberg, 2024](https://arxiv.org/html/2404.08136v1)):

```python
class EWMACalibrator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.bias_estimate = 0.0

    def update(self, predicted, actual):
        error = actual - predicted
        self.bias_estimate = self.alpha * error + (1 - self.alpha) * self.bias_estimate

    def calibrate(self, prediction):
        return prediction + self.bias_estimate
```

---

## 8. Practical Implementation Patterns for a Single-User System

### Architecture Overview

```
                    +------------------+
                    |  New Articles    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  LLM Scoring     |  (Claude: 4 dimension scores)
                    |  (cached once)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Calibration     |  (Bayesian model: LLM scores -> user value)
                    |  Model           |
                    +--------+---------+
                             |
               +-------------+-------------+
               |                           |
    +----------v----------+     +----------v----------+
    |  Exploit: Show      |     |  Explore: Show      |
    |  top-scored articles|     |  uncertain articles  |
    +----------+----------+     +----------+----------+
               |                           |
               +-------------+-------------+
                             |
                    +--------v---------+
                    |  User Reads &    |
                    |  Engages         |
                    +--------+---------+
                             |
               +-------------+-------------+
               |                           |
    +----------v----------+     +----------v----------+
    |  Implicit Feedback  |     |  Explicit Feedback  |
    |  (highlights, read) |     |  (ratings, flags)   |
    +----------+----------+     +----------+----------+
               |                           |
               +-------------+-------------+
                             |
                    +--------v---------+
                    |  Update Model    |
                    |  (online + batch)|
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Drift Detection |
                    +------------------+
```

### Database Schema Additions

The existing schema needs minimal additions to support feedback loops:

```sql
-- Explicit feedback from user
ALTER TABLE articles ADD COLUMN user_rating INTEGER;  -- NULL, 0, or 1
ALTER TABLE articles ADD COLUMN user_flag TEXT;        -- 'false_positive', 'false_negative', NULL
ALTER TABLE articles ADD COLUMN rating_timestamp DATETIME;

-- Model metadata
CREATE TABLE model_snapshots (
    id INTEGER PRIMARY KEY,
    created_at DATETIME NOT NULL,
    model_blob BLOB NOT NULL,
    n_training_samples INTEGER,
    accuracy REAL,
    log_loss REAL,
    notes TEXT
);

-- Drift detection log
CREATE TABLE drift_events (
    id INTEGER PRIMARY KEY,
    detected_at DATETIME NOT NULL,
    detector TEXT NOT NULL,       -- 'adwin', 'performance', 'manual'
    severity TEXT NOT NULL,       -- 'low', 'medium', 'high'
    description TEXT,
    action_taken TEXT
);
```

### Implementation Roadmap

**Phase 1: Implicit feedback collection (minimal effort).**
- On the nightly sync, check which articles have been highlighted in Readwise Reader.
- Store highlight count and read status.
- Compute binary engagement labels with confidence weights.
- No model changes yet -- just accumulate data.

**Phase 2: Bayesian calibration model.**
- Once 30+ articles have engagement data, train a `BayesianRidge` mapping LLM scores to engagement.
- Use the calibrated scores for ranking instead of raw LLM scores.
- Run weekly batch retraining.

**Phase 3: Active learning UI.**
- Add a "Rate this article" prompt for uncertain predictions.
- Implement Thompson sampling for article ordering.
- Track explore/exploit metrics.

**Phase 4: Online learning and drift detection.**
- Switch from weekly batch to daily incremental updates with weekly full retraining.
- Add ADWIN drift detection on model error stream.
- Implement time-decay weighting.
- Add model snapshotting and rollback.

**Phase 5: Explicit feedback loop.**
- Add thumbs up/down to the triage UI.
- Add "Flag as wrong" for obvious misscores.
- Weight explicit feedback higher than implicit.
- Surface drift alerts to the user.

### Configuration

Expose key hyperparameters as environment variables or a config table:

```python
FEEDBACK_CONFIG = {
    "time_decay_half_life_days": 90,
    "exploration_epsilon": 0.1,
    "min_samples_for_calibration": 30,
    "drift_detection_window": 50,
    "drift_alert_threshold": 0.15,
    "online_learning_rate": 0.01,
    "retraining_schedule": "weekly",
    "implicit_highlight_confidence": 0.7,
    "implicit_read_no_highlight_confidence": 0.4,
    "explicit_feedback_confidence": 1.0,
}
```

### Monitoring and Observability

Track these metrics over time (compatible with the existing OTLP setup):

- **Model accuracy** (rolling 50-article window): Are predictions matching engagement?
- **Calibration** (predicted probability vs. actual engagement rate): Is the model overconfident?
- **Exploration rate**: What fraction of shown articles are explorative?
- **Discovery rate**: What fraction of exploration articles turn out to be valuable?
- **Feedback coverage**: What fraction of scored articles have engagement data?
- **Drift detector state**: Current ADWIN window size, error rate trend.

### Avoiding Common Pitfalls

1. **Don't retrain on filtered data.** The user only sees top-scored articles, so engagement data is biased toward high scores. Include the user's explicit "this should have been scored higher" flags to get signal on false negatives.

2. **Don't over-weight recent feedback in small datasets.** With only 200 articles, aggressive time decay can reduce the effective training set to ~50 samples. Start with a long half-life (180 days) and shorten it as data accumulates.

3. **Don't treat "not read" as "not valuable."** The user may not have seen the article yet. Only treat articles as negative examples after sufficient time has passed (e.g., 7 days in the queue without interaction).

4. **Don't forget to evaluate.** Hold out 20% of recent feedback for evaluation. If the calibrated model isn't beating the raw LLM scores, something is wrong -- check for data leakage or overfitting.

5. **Don't chase noise.** A single bad prediction is not drift. Require sustained performance degradation (5+ articles in a row) before triggering retraining.

---

## Sources

### Active Learning and Query Strategies
- [Learning with not Enough Data Part 2: Active Learning -- Lilian Weng](https://lilianweng.github.io/posts/2022-02-20-active-learning/)
- [Active Learning Overview: Strategies and Uncertainty Measures -- Zakarya Rouzki](https://medium.com/data-science/active-learning-overview-strategies-and-uncertainty-measures-521565e0b0b)
- [A personalized active learning strategy with enhanced user satisfaction for recommender systems -- ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425023838)
- [Active learning algorithm for alleviating the user cold start problem -- Nature Scientific Reports](https://www.nature.com/articles/s41598-025-09708-2)

### Online Learning and Incremental Updates
- [River: Machine learning for streaming data in Python -- JMLR](https://jmlr.csail.mit.edu/papers/volume22/20-1380/20-1380.pdf)
- [River library documentation](https://riverml.xyz/)
- [SGDClassifier -- scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [Incremental Learning with Scikit-learn -- GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/incremental-learning-with-scikit-learn/)
- [Retraining Model During Deployment: Continuous Training -- Neptune.ai](https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing)
- [Online Learning vs Batch Retraining -- APXML](https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-4-automated-retraining-updates/online-learning-vs-batch)

### Concept Drift
- [Best Practices for Dealing With Concept Drift -- Neptune.ai](https://neptune.ai/blog/concept-drift-best-practices)
- [Modelling Concept Drift in Dynamic Data Streams for Recommender Systems -- ACM](https://dl.acm.org/doi/10.1145/3707693)
- [Learning from Time-Changing Data with Adaptive Windowing (ADWIN) -- Bifet and Gavalda](https://www.researchgate.net/publication/220907178_Learning_from_Time-Changing_Data_with_Adaptive_Windowing)
- [ADWIN -- River documentation](https://riverml.xyz/dev/api/drift/ADWIN/)
- [Learning under Concept Drift: an Overview -- Zliobaite](https://arxiv.org/pdf/1010.4784)

### Feedback Loops and Implicit/Explicit Feedback
- [Navigating the Feedback Loop in Recommender Systems -- ACM RecSys 2023](https://dl.acm.org/doi/10.1145/3604915.3610246)
- [Handling Feedback Loops in Recommender Systems: Deep Bayesian Bandits -- Towards Data Science](https://towardsdatascience.com/handling-feedback-loops-in-recommender-systems-deep-bayesian-bandits-e83f34e2566a/)
- [How does implicit feedback differ from explicit feedback in recommendations? -- Milvus](https://milvus.io/ai-quick-reference/how-does-implicit-feedback-differ-from-explicit-feedback-in-recommendations)

### Bandit Algorithms
- [Thompson Sampling for Contextual Bandits with Linear Payoffs -- Agrawal and Goyal](https://arxiv.org/abs/1209.3352)
- [Beyond A/B Testing: A Practical Guide to Multi-Armed Bandits -- Shaped](https://www.shaped.ai/blog/multi-armed-bandits)
- [contextualbandits Python library -- David Cortes](https://github.com/david-cortes/contextualbandits)
- [A Tutorial on Thompson Sampling -- Stanford](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [Introduction to Bandits in Recommender Systems -- ACM RecSys 2020](https://dl.acm.org/doi/abs/10.1145/3383313.3411547)

### Time Decay and Temporal Dynamics
- [Exponential Decay Function-Based Time-Aware Recommender System -- IJACSA](https://thesai.org/Downloads/Volume13No10/Paper_71-Exponential_Decay_Function_Based_Time_Aware_Recommender_System.pdf)
- [Adaptive Collaborative Filtering with Personalized Time Decay Functions -- arXiv](https://arxiv.org/pdf/2308.01208)
- [A collaborative filtering recommendation system with dynamic time decay -- ResearchGate](https://www.researchgate.net/publication/340361119_A_collaborative_filtering_recommendation_system_with_dynamic_time_decay)
- [Exponentially Weighted Moving Models -- Boyd and Luxenberg, Stanford](https://arxiv.org/html/2404.08136v1)

### Cold Start
- [Few-Shot Representation Learning for Cold-Start Users and Items -- Hao et al.](https://xiaojingzi.github.io/publications/APWeb20-HAO-et-al-FewshotRec.pdf)
- [Awesome Cold-Start Recommendation -- GitHub](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation)
- [Cold-Start Recommendation with Knowledge-Guided RAG -- arXiv](https://arxiv.org/html/2505.20773v1)
