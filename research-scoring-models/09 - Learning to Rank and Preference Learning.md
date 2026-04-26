# Learning to Rank and Preference Learning for Article Prioritization

## Executive Summary

Learning to rank (LTR) offers a fundamentally different framing from regression-based scoring for article prioritization: instead of predicting an absolute engagement score, LTR learns a function that orders articles by predicted interest. This distinction matters because ranking metrics like NDCG directly optimize what we care about -- putting the best articles at the top -- while regression optimizes prediction accuracy across the entire score range, including articles we will never read. For a single-user system with hundreds of articles and highlight counts as implicit preference signals, pairwise and listwise LTR approaches (particularly LambdaMART via XGBoost's `XGBRanker`) provide a practical, well-supported path that can outperform regression while requiring less calibration of absolute scores. The main challenge is dataset size: with only hundreds of labeled examples, careful feature engineering and pair generation strategy matter more than algorithm choice.

---

## 1. How LTR Differs from Regression for This Problem

Our current system predicts an absolute score (0-100) for each article using Claude. A regression-based ML model would attempt to reproduce these scores by minimizing mean squared error (or similar). This creates a subtle misalignment: regression treats an error of 50 vs 55 on a mediocre article as equally costly to an error of 85 vs 90 on a top article, yet only the latter matters for prioritization.

Learning to rank resolves this by optimizing the *ordering* of articles directly. Key differences:

| Aspect | Regression | Learning to Rank |
|--------|-----------|-----------------|
| **Objective** | Minimize prediction error (MSE, MAE) | Minimize ranking inversions / maximize NDCG |
| **What's optimized** | Absolute score accuracy | Relative ordering of items |
| **Top-of-list focus** | Equal weight across all score ranges | Can weight top positions more heavily |
| **Score meaning** | Scores are calibrated predictions | Scores are arbitrary; only ordering matters |
| **Evaluation** | R-squared, RMSE | NDCG@k, MAP, Kendall tau |

As [OpenSource Connections](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/) explains, "a model that regularly swaps exactly-relevant and meh items but can accurately predict lower value relevance grades that end up on page 50 isn't that great." LTR avoids this problem by directly optimizing the ranking metric.

For our use case -- surfacing the 5-10 best articles from a batch of 50-100 -- LTR's focus on top-of-list accuracy is a natural fit. We do not care if a low-interest article scores 12 or 18; we care that high-interest articles sort above medium-interest ones.

---

## 2. Pointwise LTR: Regression-Based Ranking

Pointwise methods are the simplest LTR approach: they treat each article independently and predict a relevance score or class. In practice, pointwise LTR is nearly identical to standard regression or classification -- the "ranking" emerges by sorting items by predicted score.

### 2.1 How It Works

Given a feature vector `x_i` for article `i` and a relevance label `y_i` (e.g., highlight count or binned relevance grade), a pointwise model minimizes:

```
L = sum_i loss(f(x_i), y_i)
```

where `loss` is MSE, cross-entropy, or ordinal regression loss. The predicted scores `f(x_i)` are then sorted to produce a ranking.

### 2.2 RankNet (Pointwise Variant)

RankNet, introduced by [Burges et al. (2005)](https://www.semanticscholar.org/paper/Learning-to-rank-using-gradient-descent-Burges-Shaked/63aaf12163fe9735dfe9a69114937c4fa34f303a), was originally a neural network architecture but its core idea applies broadly. In its simplest form, RankNet can be viewed as a pointwise scorer: a neural network learns a scoring function `f(x)` that maps article features to a relevance score. The network is trained on pairs, but the scoring function itself operates on individual items.

The architecture uses a feed-forward neural network (originally three layers) that produces a single scalar score per document. For the pointwise interpretation, you can simply train it to minimize squared error against relevance labels and sort by predicted score.

### 2.3 Practical Applicability

Pointwise methods are the baseline approach and are trivially implementable with any regression framework (scikit-learn, XGBoost in regression mode, a small neural network). Their main limitation is that they weight errors uniformly across relevance levels. For our problem, if most articles have 0-1 highlights and a few have 5+, the model will optimize heavily for the low-engagement majority rather than for correctly ordering the high-engagement minority.

**When to use pointwise:** As a baseline, or when labels are sparse and you cannot form enough meaningful pairs for pairwise methods.

---

## 3. Pairwise LTR: Learning Relative Preferences

Pairwise methods reframe ranking as a binary classification problem: given two articles, predict which one should rank higher. This is a more natural fit for engagement data, where we often have stronger confidence in *relative* preferences ("article A got 5 highlights, article B got 0") than in absolute scores.

### 3.1 The Pairwise Transform

The key insight, demonstrated by [Bianp (2012)](https://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/), is that any ranking problem can be converted into binary classification:

1. For each pair of articles `(i, j)` within the same batch where `y_i > y_j`:
   - Create a difference vector: `x_i - x_j`
   - Assign label: +1 (article `i` should rank higher)
   - Optionally also create `x_j - x_i` with label -1 for symmetry

2. Train any binary classifier on these difference vectors.

3. At inference time, use the underlying scoring function `f(x)` to score individual articles, then sort.

This approach achieved a Kendall tau of 0.84 on test data compared to 0.71 for standard linear regression in the scikit-learn demonstration, showing meaningful improvement from the pairwise formulation alone.

### 3.2 RankNet Loss

RankNet's pairwise loss function is the cross-entropy between predicted and actual pairwise orderings. For articles `i` and `j` where `i` should rank higher:

```
P_ij = sigma(f(x_i) - f(x_j))    # predicted probability that i ranks above j
L_ij = -log(P_ij)                  # cross-entropy loss
```

where `sigma` is the logistic sigmoid. This loss is differentiable and can be optimized by gradient descent. The scoring function `f` learns to produce scores where higher-relevance items consistently score above lower-relevance items.

In XGBoost, this is available as the `rank:pairwise` objective ([XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)).

### 3.3 RankSVM

[Ranking SVM](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html), introduced by Joachims, applies the SVM framework to pairwise ranking. It learns a weight vector `w` such that for all pairs where `y_i > y_j`:

```
w^T x_i > w^T x_j + margin
```

The optimization maximizes the margin between correctly ordered pairs, analogous to standard SVM classification on the pairwise difference vectors `(x_i - x_j)`. RankSVM is particularly effective with small datasets due to SVM's strong regularization properties and can be implemented in scikit-learn using the pairwise transform with `LinearSVC` or `SVC`.

### 3.4 LambdaRank

LambdaRank, developed by [Burges et al. at Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf), extends RankNet with a crucial insight: not all pairwise swaps matter equally. Swapping the #1 and #2 articles affects ranking quality more than swapping #45 and #46.

LambdaRank multiplies the RankNet gradient by `|delta_NDCG|` -- the change in NDCG that would result from swapping the two documents:

```
lambda_ij = sigma(-s_ij) * |delta_NDCG_ij|
```

where `s_ij = f(x_i) - f(x_j)`. This "lambda gradient" is not derived from a formal loss function but works as a practical gradient that directly optimizes NDCG. As explained by [Shaped](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank), "the magnitude of the push or pull applied to a document's score during training depends not just on the pairwise error but also on how much that swap impacts the overall listwise metric."

### 3.5 Computational Considerations

Pairwise methods scale as O(n^2) in the number of articles per query group, since all pairs must be considered. For our problem (50-100 articles per batch), this produces 1,225-4,950 pairs per batch -- entirely manageable. XGBoost's pair sampling strategies (`lambdarank_num_pair_per_sample`) can further control this.

---

## 4. Listwise LTR: Optimizing the Entire Ranking

Listwise methods operate on the complete list of articles simultaneously, either by directly optimizing a ranking metric or by defining a loss over permutations.

### 4.1 LambdaMART

LambdaMART is the most widely deployed LTR algorithm and the de facto standard in production ranking systems. It combines LambdaRank's metric-aware gradients with gradient boosted decision trees (MART = Multiple Additive Regression Trees).

As detailed by [Software Doug](https://softwaredoug.com/blog/2022/01/17/lambdamart-in-depth), the algorithm proceeds iteratively:

1. **Score**: Current ensemble predicts scores for all articles in a batch
2. **Pair**: Form all pairs `(i, j)` where article `i` has higher relevance than `j`
3. **Compute rho**: `rho_ij = 1 / (1 + exp(score_i - score_j))` -- how wrong the current ranking is
4. **Compute delta**: `delta_ij = (discount_i - discount_j) * (gain_i - gain_j)` -- NDCG impact of swapping
5. **Lambda**: `lambda_ij = delta_ij * rho_ij` -- the gradient signal
6. **Accumulate**: Sum lambdas per article to get per-article gradient targets
7. **Fit tree**: Train a regression tree to predict these lambda targets from article features
8. **Update**: Add the new tree to the ensemble with learning rate damping

This bridges pairwise and listwise optimization: pairs generate the gradients, but the NDCG delta term ensures the model optimizes a listwise metric. LambdaMART is available in XGBoost (`rank:ndcg`), LightGBM (`lambdarank`), and CatBoost.

### 4.2 ListNet

[ListNet (Cao et al., 2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) defines a probability distribution over permutations and minimizes the cross-entropy between the predicted distribution and the ground-truth distribution.

For computational tractability, ListNet uses the "top-one probability" -- the probability that each article appears at the top of the ranking:

```
P(article_i at top) = exp(s_i) / sum_j exp(s_j)
```

This is essentially a softmax over relevance scores. The loss is:

```
L = -sum_i P_ground_truth(i) * log(P_predicted(i))
```

ListNet is conceptually clean but requires neural network training and is less commonly used in practice than LambdaMART.

### 4.3 ListMLE

[ListMLE (Xia et al., 2008)](https://icml.cc/Conferences/2008/papers/167.pdf) maximizes the likelihood of the ground-truth permutation under a Plackett-Luce model. For a ranked list `[d_1, d_2, ..., d_n]`:

```
P(ranking | scores) = product_i exp(s_{d_i}) / sum_{k>=i} exp(s_{d_k})
```

The loss is the negative log-likelihood of the correct ranking. ListMLE has been shown to be consistent with the permutation-level 0-1 loss, giving it desirable theoretical properties. However, like ListNet, it is less commonly deployed than LambdaMART in gradient boosting frameworks.

---

## 5. Deriving Pairwise Preferences from Highlight Count Data

Our implicit preference signal is highlight counts per article. Here is how to convert this into training data for LTR.

### 5.1 Direct Graded Relevance

The simplest approach: treat highlight count as a relevance grade.

```python
# Relevance grades from highlights
# 0 highlights -> grade 0
# 1-2 highlights -> grade 1
# 3-5 highlights -> grade 2
# 6+ highlights -> grade 3
```

Most LTR algorithms (XGBRanker, LambdaMART) accept graded relevance labels directly and generate pairs internally. This is the recommended starting point.

### 5.2 Explicit Pairwise Preference Extraction

For algorithms requiring explicit pairs (RankSVM, custom neural networks), extract preferences:

```python
def extract_pairs(articles, highlights):
    """Generate pairwise preferences from highlight counts.

    For articles within the same time-batch, create pairs where
    the more-highlighted article should rank higher.
    """
    pairs = []
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if highlights[i] != highlights[j]:
                # Article with more highlights should rank higher
                if highlights[i] > highlights[j]:
                    pairs.append((i, j, +1))  # i > j
                else:
                    pairs.append((i, j, -1))  # j > i
    return pairs
```

### 5.3 Handling Ties and Noise

Highlight counts are noisy: an article with 0 highlights might be unread (not irrelevant) rather than deliberately skipped. Strategies to handle this:

- **Only form pairs between read articles**: Exclude articles the user never opened from pairwise comparisons. An unread article is not necessarily worse than a read-but-not-highlighted one.
- **Margin-based filtering**: Only form pairs where the highlight difference exceeds a threshold (e.g., `|h_i - h_j| >= 2`) to reduce noise from borderline cases.
- **Time-decay weighting**: Weight recent preferences more heavily, since reading interests drift.
- **Binary simplification**: Collapse to binary relevance -- "highlighted at least once" vs "not highlighted" -- to reduce noise from small count differences.

### 5.4 Defining Query Groups

LTR algorithms require a "query" grouping -- articles are only compared within the same group. For article prioritization, natural groupings include:

- **Time-based batches**: Articles added in the same week or triage session
- **All articles**: Treat the entire corpus as one query group (simplest, but loses temporal context)
- **Category/topic groups**: Compare articles only within the same topic area

For a single user with hundreds of articles, treating all articles as one query group or using weekly batches are both reasonable. The XGBoost documentation recommends the `mean` pair sampling strategy for small datasets, which generates `|query| * num_pair_per_sample` pairs via random sampling ([XGBoost LTR docs](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)).

### 5.5 Interpreting Implicit Feedback as Preferences

Following the foundational work of [Joachims et al. (2005)](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf) on interpreting click data as pairwise preferences, we can apply similar reasoning to highlights:

- A highlighted article is **preferred over** a read-but-not-highlighted article (analogous to "click > skip above")
- An article with 5 highlights is **preferred over** one with 1 highlight (graded preference)
- An unread article has **unknown** preference (not negative preference)

This framing is more robust than treating highlight counts as absolute scores, because it handles the inherent noise in engagement signals.

---

## 6. XGBoost for Ranking (XGBRanker)

XGBoost provides first-class support for learning to rank through the `XGBRanker` API, making it the most practical starting point for our system.

### 6.1 Configuration

```python
import xgboost as xgb
import numpy as np

# Prepare data
# X: feature matrix (n_articles, n_features)
# y: relevance grades (n_articles,) -- e.g., binned highlight counts
# qid: query group IDs (n_articles,) -- e.g., batch/week IDs

ranker = xgb.XGBRanker(
    objective="rank:ndcg",        # LambdaMART with NDCG optimization
    tree_method="hist",           # Fast histogram-based tree building
    n_estimators=100,             # Number of boosting rounds
    max_depth=4,                  # Shallow trees for small data
    learning_rate=0.1,
    lambdarank_num_pair_per_sample=8,  # Pairs per document
    lambdarank_pair_method="mean",     # Random sampling (good for small data)
)

ranker.fit(X_train, y_train, qid=qid_train)

# Predict relevance scores (higher = more relevant)
scores = ranker.predict(X_test)
```

### 6.2 Objective Functions

Per the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html):

| Objective | Description | Best For |
|-----------|-------------|----------|
| `rank:ndcg` | LambdaMART optimizing NDCG (default) | Multi-level relevance grades |
| `rank:map` | LambdaMART optimizing MAP | Binary relevance (highlighted/not) |
| `rank:pairwise` | RankNet loss without metric scaling | Baseline / comparison |

**Recommendation for our problem**: Start with `rank:ndcg` and graded relevance labels. If simplifying to binary relevance, try `rank:map`.

### 6.3 Small Dataset Guidance

The XGBoost documentation specifically addresses small datasets:

> "When you have a comparatively small amount of training data, select NDCG or the RankNet loss (`rank:pairwise`) and choose the `mean` strategy for generating document pairs, to obtain more effective pairs."

Additional practices for small data:

- **Increase `lambdarank_num_pair_per_sample`**: Generate more pairs per document to extract more signal from limited data
- **Use strong regularization**: Higher `min_child_weight`, lower `max_depth` (3-4), `subsample` < 1.0
- **Cross-validation**: Use `GroupKFold` to split by query group, never splitting a query group across train/test
- **Feature importance**: With limited data, fewer well-chosen features outperform many noisy ones

### 6.4 Feature Engineering for Article Ranking

The feature matrix `X` should capture signals predictive of engagement:

```python
features = {
    # Content features (from embeddings or text analysis)
    "embedding_dim_0": ...,  # Article embedding components
    "word_count": ...,
    "reading_time_minutes": ...,

    # Source features
    "source_avg_highlights": ...,   # Historical engagement with this source
    "source_article_count": ...,

    # Topic features
    "topic_similarity_to_past_highlights": ...,  # Cosine sim to highlighted articles

    # Metadata features
    "days_since_published": ...,
    "has_images": ...,

    # Current AI scores (as features, not targets)
    "quotability_score": ...,
    "surprise_score": ...,
    "argument_quality_score": ...,
    "applicable_insight_score": ...,
}
```

### 6.5 Evaluation

```python
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
ndcg_scores = []

for train_idx, test_idx in gkf.split(X, y, groups=qid):
    ranker.fit(X[train_idx], y[train_idx], qid=qid[train_idx])
    y_pred = ranker.predict(X[test_idx])

    # Evaluate per query group
    for group in np.unique(qid[test_idx]):
        mask = qid[test_idx] == group
        if len(np.unique(y[test_idx][mask])) > 1:  # Need variation
            ndcg = ndcg_score([y[test_idx][mask]], [y_pred[mask]], k=10)
            ndcg_scores.append(ndcg)

print(f"Mean NDCG@10: {np.mean(ndcg_scores):.3f}")
```

---

## 7. Modern Neural LTR with Embeddings

For systems already producing article embeddings (as ours does via Claude or sentence transformers), neural LTR methods can operate directly on embeddings.

### 7.1 Two-Stage Architecture

A practical neural LTR pipeline for article ranking:

1. **Embedding stage**: Encode articles into dense vectors using a pre-trained language model (e.g., `all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`)
2. **Ranking stage**: A small neural network (2-3 layers) maps embeddings to relevance scores, trained with a pairwise or listwise loss

```python
import torch
import torch.nn as nn

class ArticleRanker(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embeddings):
        return self.scorer(embeddings).squeeze(-1)

    def pairwise_loss(self, scores_i, scores_j):
        """RankNet pairwise loss: i should rank above j."""
        return torch.log1p(torch.exp(-(scores_i - scores_j))).mean()
```

### 7.2 Cross-Encoder Reranking

[Cross-encoders](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) process a query-document pair jointly through a transformer, producing a relevance score. For article ranking, the "query" could be a representation of the user's preferences:

```python
from sentence_transformers import CrossEncoder

# Train a cross-encoder to predict:
# "Given this user's reading history, how relevant is this article?"
# Input: [user_preference_summary, article_text]
# Output: relevance score
```

Cross-encoders are more accurate than bi-encoders but computationally expensive, making them suitable for reranking a small candidate set (our 50-100 articles per batch).

### 7.3 LLM-Based Zero-Shot Ranking

Recent research by [Hou et al. (2023)](https://arxiv.org/abs/2305.08845) demonstrates that LLMs can function as zero-shot rankers for recommendation. The approach constructs prompts containing user interaction history and candidate items, asking the LLM to rank them.

Key findings relevant to our system:

- LLMs achieve strong zero-shot ranking performance (NDCG@20 of 50% on MovieLens-1M)
- **Position bias** is a significant concern: items listed later in the prompt are systematically ranked lower. Mitigation: bootstrap by ranking multiple times with shuffled candidate order.
- **Recency-focused prompting** (emphasizing recent interactions) outperforms chronological listing
- Optimal candidate set size is ~20 items per ranking call

This approach is already partially implemented in our system (Claude scores articles individually). A listwise extension would ask Claude to rank a batch of article summaries directly, which could capture comparative judgments that absolute scoring misses.

---

## 8. Converting Rankings Back to Scores for Display

A pure ranking model produces ordered lists, not the 0-100 scores our UI displays. Several strategies bridge this gap.

### 8.1 Min-Max Normalization of Raw Scores

LTR models output raw scores (unbounded). Normalize within each batch:

```python
def normalize_scores(raw_scores, target_min=0, target_max=100):
    """Convert raw ranking scores to display scores."""
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max == s_min:
        return np.full_like(raw_scores, 50.0)
    normalized = (raw_scores - s_min) / (s_max - s_min)
    return normalized * (target_max - target_min) + target_min
```

**Limitation**: Scores are relative to the current batch. A "100" in a weak batch may be less interesting than a "70" in a strong batch.

### 8.2 Rank-Based Scoring

Assign scores based on rank position rather than raw model output:

```python
def rank_to_score(ranks, n_total):
    """Convert ranks to scores via inverse rank weighting."""
    return 100 * (1 - (ranks - 1) / max(n_total - 1, 1))
```

This guarantees even spacing and is immune to score distribution quirks, but loses information about *how much* better the top article is versus the second.

### 8.3 Calibrated Scoring via Isotonic Regression

Train an isotonic regression model that maps raw ranking scores to calibrated engagement probabilities, then scale to 0-100:

```python
from sklearn.isotonic import IsotonicRegression

# Fit on historical data: raw_scores -> actual_highlights
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(historical_raw_scores, historical_highlights)

# Apply to new predictions
calibrated = calibrator.predict(new_raw_scores)
display_scores = np.clip(calibrated / max_highlights * 100, 0, 100)
```

### 8.4 Hybrid Approach (Recommended)

Use the LTR model for *ordering* but retain the existing AI scores (or a calibrated regression model) for *display scores*. This gives the best of both worlds:

1. LTR determines the rank order (optimized for top-of-list accuracy)
2. A separate regression model or the existing Claude scores provide meaningful absolute scores for the UI
3. If the LTR ranking disagrees with the absolute scores, the LTR ranking wins for sort order

---

## 9. Evaluation Metrics for Ranking

### 9.1 NDCG (Normalized Discounted Cumulative Gain)

The primary metric for graded relevance. NDCG@k evaluates ranking quality for the top-k positions:

```
DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

where IDCG@k is the DCG of the ideal (perfect) ranking. NDCG ranges from 0 to 1, with 1 being perfect. The logarithmic discount gives disproportionate credit to getting the top positions right, which aligns with our goal ([Evidently AI](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)).

### 9.2 MAP (Mean Average Precision)

Best for binary relevance (highlighted vs. not). MAP averages the precision at each relevant item's position, rewarding systems that place all relevant items near the top ([Shaped](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg)).

### 9.3 Kendall Tau / Spearman Rank Correlation

Measures overall rank agreement between predicted and actual orderings. Useful for comparing models but does not emphasize top-of-list accuracy.

### 9.4 Practical Choice

For our system with graded relevance (highlight counts), **NDCG@5 or NDCG@10** is the most appropriate primary metric, since we want the top 5-10 articles to be correctly identified. Use MAP as a secondary metric if simplifying to binary relevance.

---

## 10. Practical Recommendations for a Single-User Ranking System

### 10.1 Recommended Architecture

Given our constraints (single user, hundreds of articles, highlight counts as signal), here is a phased implementation plan:

**Phase 1: XGBRanker Baseline**

```python
import xgboost as xgb

# Features: existing AI scores + metadata + embedding similarity
# Labels: binned highlight counts (0, 1, 2, 3)
# Groups: weekly batches

ranker = xgb.XGBRanker(
    objective="rank:ndcg",
    tree_method="hist",
    n_estimators=50,          # Conservative for small data
    max_depth=3,              # Prevent overfitting
    learning_rate=0.1,
    min_child_weight=5,       # Regularization
    subsample=0.8,
    lambdarank_pair_method="mean",
    lambdarank_num_pair_per_sample=10,
)
```

This uses the four existing AI scores (quotability, surprise, argument quality, applicable insight) as features alongside metadata, letting the ranker learn which dimensions best predict actual engagement.

**Phase 2: Embedding Features**

Add article embeddings (or PCA-reduced embeddings) as features. Compute similarity features between each article and the user's historically highlighted articles.

**Phase 3: Neural Reranker** (if data grows sufficiently)

Train a small neural network on embeddings with pairwise loss, using the XGBRanker as a teacher signal for data augmentation.

### 10.2 Handling Small Data

With ~hundreds of articles, overfitting is the primary risk. Mitigations:

- **Strong regularization**: Low `max_depth` (3-4), high `min_child_weight`, `subsample` < 1
- **Few features**: Start with 5-10 well-chosen features, not hundreds of embedding dimensions
- **Leave-one-out or GroupKFold CV**: Evaluate on held-out batches
- **Feature selection**: Use XGBoost's built-in feature importance to prune
- **Online learning**: Retrain periodically as new data arrives, using all historical data

### 10.3 Cold Start Strategy

When the system has very few labeled articles (< 50), pure LTR will not work well. A practical cold-start pipeline:

1. **Start with Claude's AI scores** as the sole ranking signal (current approach)
2. **Collect engagement data** (highlights, reading time) passively
3. **At ~100 articles with highlights**, train an initial XGBRanker using AI scores as features
4. **Compare NDCG**: Does the XGBRanker reranking improve over raw AI score sorting?
5. **Gradually add features** as data grows

### 10.4 LTR vs. Regression Decision Framework

| Condition | Recommendation |
|-----------|---------------|
| < 50 labeled articles | Use regression or raw AI scores |
| 50-200 labeled articles | Try XGBRanker with `rank:pairwise` (simpler) |
| 200-500 labeled articles | Use XGBRanker with `rank:ndcg` (LambdaMART) |
| 500+ labeled articles | Consider neural LTR with embeddings |
| Need calibrated display scores | Hybrid: LTR for ordering + regression for display |

### 10.5 Key Implementation Considerations

1. **Temporal leakage**: Never use future engagement data to rank past articles. Split train/test by time.
2. **Position bias**: If the current UI shows articles in AI-score order, users see top articles first and may highlight them more simply due to exposure. Consider randomizing order occasionally to collect unbiased labels.
3. **Feedback loops**: The ranker affects what users see, which affects engagement, which affects training. Periodically inject exploration (show some lower-ranked articles) to prevent echo chambers.
4. **Retraining frequency**: With a slowly-growing dataset, retrain weekly or monthly. Use the entire history each time (not just new data).
5. **A/B evaluation**: Compare ranker output against the current AI scoring by measuring NDCG on held-out data where you know the actual highlights.

### 10.6 Why LTR May Be Worth It Despite Small Data

Even with limited data, LTR offers advantages over regression:

- **Robustness to label noise**: Pairwise comparisons ("article A is better than B") are more robust than absolute labels ("article A deserves score 73")
- **Metric alignment**: Optimizing NDCG directly aligns training with evaluation, avoiding the regression-ranking mismatch
- **Feature combination**: The ranker can learn non-obvious interactions between AI scores and metadata (e.g., "high quotability + short article = higher engagement")
- **Automatic calibration**: No need to hand-tune score thresholds (60 = high, 30 = medium); the ranking handles this implicitly

---

## Sources

- [XGBoost Learning to Rank Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) -- Official guide covering objectives, pair generation, and configuration
- [Burges et al., "Learning to Rank using Gradient Descent" (RankNet, 2005)](https://www.semanticscholar.org/paper/Learning-to-rank-using-gradient-descent-Burges-Shaked/63aaf12163fe9735dfe9a69114937c4fa34f303a) -- Foundational paper on pairwise neural ranking
- [Burges, "From RankNet to LambdaRank to LambdaMART: An Overview" (2010)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) -- Microsoft Research technical report covering the evolution of the LambdaRank family
- [Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach" (ListNet, 2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) -- Original ListNet paper
- [Xia et al., "Listwise Approach to Learning to Rank: Theory and Algorithm" (ListMLE, 2008)](https://icml.cc/Conferences/2008/papers/167.pdf) -- ListMLE paper with theoretical consistency proofs
- [Joachims, "SVM-Rank: Support Vector Machine for Ranking"](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) -- RankSVM implementation and documentation
- [Joachims et al., "Accurately Interpreting Clickthrough Data as Implicit Feedback" (2005)](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf) -- Foundational work on deriving pairwise preferences from implicit feedback
- [Software Doug, "LambdaMART in Depth" (2022)](https://softwaredoug.com/blog/2022/01/17/lambdamart-in-depth) -- Detailed walkthrough of LambdaMART implementation
- [Shaped, "LambdaMART Explained" (2024)](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank) -- Accessible explanation of LambdaMART mechanics
- [Shaped, "Learning to Rank for Recommender Systems: A Practical Guide"](https://www.shaped.ai/blog/learning-to-rank-for-recommender-systems) -- Practical LTR guide for recommendation contexts
- [Bianp, "Learning to Rank with scikit-learn: the pairwise transform" (2012)](https://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/) -- Implementation of pairwise transform for ranking with scikit-learn
- [Forecastegy, "How to Use XGBoost for Learning to Rank in Python"](https://forecastegy.com/posts/xgboost-learning-to-rank-python/) -- Practical XGBRanker tutorial
- [OLX Engineering, "From RankNet to LambdaMART: Leveraging XGBoost"](https://tech.olx.com/from-ranknet-to-lambdamart-leveraging-xgboost-for-enhanced-ranking-models-cf21f33350fb) -- Production experience with XGBoost ranking
- [Hou et al., "Large Language Models are Zero-Shot Rankers for Recommender Systems" (ECIR 2024)](https://arxiv.org/abs/2305.08845) -- Using LLMs for zero-shot recommendation ranking
- [NVIDIA, "Learning to Rank with XGBoost and GPU"](https://developer.nvidia.com/blog/learning-to-rank-with-xgboost-and-gpu/) -- GPU-accelerated ranking implementation
- [Evidently AI, "NDCG Metric Explained"](https://www.evidentlyai.com/ranking-metrics/ndcg-metric) -- Clear explanation of NDCG evaluation metric
- [Shaped, "Evaluating Recommendation Systems: MAP, MRR, NDCG"](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg) -- Comparison of ranking evaluation metrics
- [OpenSource Connections, "How is Search Different Than Other ML Problems?"](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/) -- Why ranking optimization differs from regression
- [Wikipedia, "Learning to Rank"](https://en.wikipedia.org/wiki/Learning_to_rank) -- General overview and taxonomy
- [Sentence Transformers, "Retrieve & Re-Rank"](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) -- Cross-encoder reranking with sentence transformers
- [Pinecone, "Rerankers and Two-Stage Retrieval"](https://www.pinecone.io/learn/series/rag/rerankers/) -- Practical guide to two-stage retrieval with rerankers
