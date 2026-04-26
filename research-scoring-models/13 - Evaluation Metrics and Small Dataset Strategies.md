# 13 - Evaluation Metrics and Small Dataset Strategies

## Executive Summary

When training engagement prediction models on small datasets (~hundreds of articles with zero-inflated highlight counts), the choice of evaluation metrics and validation strategy is critical to avoid misleading performance estimates. Because the end goal is surfacing the top N most interesting articles, **ranking metrics** (NDCG, Spearman rho) should be the primary evaluation criteria, not regression accuracy (RMSE). Cross-validation on small datasets requires careful stratification and repetition -- repeated stratified 5-fold CV with bootstrap confidence intervals provides the best balance of bias, variance, and reliability. Zero-inflation demands special handling: standard accuracy metrics will be dominated by the majority class (zero-engagement articles), so threshold-independent metrics (AUC-ROC, AUC-PR) and top-heavy ranking metrics (NDCG@k, Precision@k) are essential for honest evaluation.

---

## 1. Regression Metrics

When the model outputs a continuous engagement score, regression metrics quantify how close predictions are to actual values. However, for the task of surfacing top articles, not all regression metrics are equally useful.

### 1.1 RMSE (Root Mean Squared Error)

RMSE penalizes large errors disproportionately due to squaring. For zero-inflated engagement data, this means a model that badly mispredicts a high-engagement article will be penalized more than one that slightly mispredicts many zero-engagement articles.

```
RMSE = sqrt(1/n * SUM((y_i - y_hat_i)^2))
```

**When it matters:** RMSE is useful when large prediction errors on high-engagement articles are particularly costly. However, it does not directly measure ranking quality -- a model could have low RMSE by predicting near-zero for everything (which works well when most articles have zero engagement) while completely failing to identify the few high-engagement articles.

**Verdict for our use case:** Secondary metric. Low RMSE does not guarantee good ranking.

### 1.2 MAE (Mean Absolute Error)

MAE treats all errors equally regardless of magnitude:

```
MAE = 1/n * SUM(|y_i - y_hat_i|)
```

**When it matters:** MAE is more robust to outliers than RMSE and gives a more interpretable average error. However, like RMSE, it is dominated by the majority class in zero-inflated data. If 70% of articles have zero highlights, a model predicting zero everywhere achieves a low MAE while being useless for surfacing.

**Verdict for our use case:** Secondary metric. Same ranking-blindness problem as RMSE.

### 1.3 Spearman Rank Correlation (rho)

Spearman's rho measures the monotonic relationship between predicted and actual rankings. It operates on ranks rather than raw values, making it robust to non-linear relationships and outliers ([UVA Library, "Correlation: Pearson, Spearman, and Kendall's tau"](https://library.virginia.edu/data/articles/correlation-pearson-spearman-and-kendalls-tau)).

```
rho = 1 - (6 * SUM(d_i^2)) / (n * (n^2 - 1))
```

where `d_i` is the difference between the rank of the predicted score and the rank of the actual score for article i.

**When it matters:** Directly measures whether the model's ordering agrees with the true ordering. A Spearman rho of 0.7+ would indicate strong agreement between predicted and actual engagement rankings. This is exactly what we need -- the model does not need to predict exact highlight counts, just rank articles correctly.

**Verdict for our use case:** **Primary metric.** Directly measures ranking quality across the full list.

### 1.4 Kendall's Tau

Kendall's tau measures ordinal association based on concordant and discordant pairs. For every pair of articles (i, j), a pair is concordant if the model ranks them in the same relative order as their true engagement, and discordant otherwise ([Statistics Solutions, "Kendall's Tau and Spearman's Rank Correlation Coefficient"](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/kendalls-tau-and-spearmans-rank-correlation-coefficient/)).

```
tau = (concordant_pairs - discordant_pairs) / (n * (n - 1) / 2)
```

**When it matters:** Kendall's tau is more robust than Spearman's rho with small sample sizes and produces more accurate p-values. It is also less sensitive to outlier rank differences. However, it tends to produce smaller absolute values than Spearman's rho, which can be confusing when comparing across studies. Kendall's tau should be preferred when the dataset is small and there are many tied ranks ([ScienceDirect, "Effective use of Spearman's and Kendall's correlation coefficients for association between two measured traits"](https://www.sciencedirect.com/science/article/abs/pii/S0003347215000196)).

**Verdict for our use case:** **Primary metric**, especially given our small dataset. Report both Spearman and Kendall; prefer Kendall for statistical testing.

---

## 2. Classification Metrics

When engagement is binarized (e.g., "highlighted at least once" vs. "not highlighted"), or when we frame the problem as "identify articles worth reading," classification metrics apply.

### 2.1 Precision, Recall, and F1

With zero-inflated data, the choice of threshold dramatically affects these metrics:

- **Precision:** Of articles the model flagged as high-engagement, how many actually were? High precision means fewer false alarms -- users trust the recommendations.
- **Recall:** Of all truly high-engagement articles, how many did the model catch? High recall means fewer missed gems.
- **F1:** Harmonic mean of precision and recall. Useful as a single summary, but hides the precision-recall tradeoff.

For surfacing articles, **precision matters more than recall** -- showing 5 great articles is better than showing 20 articles where only 5 are great. However, if the pool is small (~hundreds), missing a truly excellent article is also costly.

**Class imbalance warning:** If only 20% of articles have any engagement, a model predicting "no engagement" for everything achieves 80% accuracy. Accuracy is meaningless here. F1 on the positive (engaged) class is the minimum viable classification metric ([Google Developers, "Datasets: Class-imbalanced datasets"](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)).

### 2.2 AUC-ROC (Area Under the Receiver Operating Characteristic)

AUC-ROC measures the model's ability to discriminate between positive and negative classes across all possible thresholds. A value of 0.5 indicates random guessing; 1.0 indicates perfect discrimination.

**Strengths:** Threshold-independent, provides a single number summarizing discrimination ability. Not affected by class imbalance in the same way accuracy is.

**Weaknesses:** Can be overly optimistic with severe class imbalance because the ROC curve incorporates the true negative rate, which is inflated when negatives dominate. For our zero-inflated data (many zero-engagement articles), the model gets "credit" for correctly ranking the abundant negative examples against each other.

**Verdict for our use case:** Report it, but do not rely on it alone. AUC-PR (below) is more informative for imbalanced data.

### 2.3 AUC-PR (Area Under the Precision-Recall Curve)

AUC-PR focuses exclusively on the positive class and is more informative than AUC-ROC when the positive class is rare. A random classifier achieves an AUC-PR equal to the prevalence of the positive class (e.g., 0.2 if 20% of articles are engaged), not 0.5 as with AUC-ROC.

**Verdict for our use case:** **Primary classification metric.** Directly measures how well the model identifies engaged articles regardless of threshold choice.

### 2.4 Precision@k

Precision@k answers the direct operational question: "If we show the user the top k articles, how many are actually worth reading?" ([Evidently AI, "Precision and recall at K in ranking and recommendations"](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k))

```
Precision@k = (relevant items in top k) / k
```

**When it matters:** This is the most operationally relevant metric for our use case. If we surface the top 10 articles each day, Precision@10 tells us what fraction are actually high-engagement. Choose k to match the actual number of articles you plan to surface.

**Limitations:** Does not consider the ordering within the top k (unlike NDCG). Also, the "relevant" threshold must be defined -- e.g., "at least 1 highlight" or "at least 3 highlights."

**Verdict for our use case:** **Primary operational metric.** Set k to match the number of articles surfaced in practice (e.g., 5, 10, 20).

---

## 3. Ranking Metrics

These metrics are purpose-built for evaluating ranked lists and are the most directly applicable to "did we surface the right articles in the right order?"

### 3.1 NDCG@k (Normalized Discounted Cumulative Gain)

NDCG is the gold standard for evaluating ranked lists with graded relevance -- exactly our scenario, where articles have varying highlight counts rather than binary relevant/not-relevant labels ([Evidently AI, "Normalized Discounted Cumulative Gain (NDCG) explained"](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)).

**How it works:**

1. **DCG@k** (Discounted Cumulative Gain): Sum of relevance scores discounted by position:
   ```
   DCG@k = SUM(i=1 to k) [ rel_i / log2(i + 1) ]
   ```
   Items at position 1 get full credit; items further down are logarithmically discounted.

2. **IDCG@k** (Ideal DCG): The DCG of the perfect ranking (items sorted by true relevance).

3. **NDCG@k** = DCG@k / IDCG@k, normalized to [0, 1].

**How to apply to engagement prediction:** Use highlight counts (or binned engagement levels) as the graded relevance score. The model produces predicted engagement scores, which define the ranking. Compare against the ideal ranking defined by actual highlight counts.

**Example with our data:**
- Model ranks articles: [A(3 highlights), B(0), C(5), D(1)]
- Ideal ranking: [C(5), A(3), D(1), B(0)]
- DCG@4 = 3/log2(2) + 0/log2(3) + 5/log2(4) + 1/log2(5) = 3.0 + 0 + 2.5 + 0.43 = 5.93
- IDCG@4 = 5/log2(2) + 3/log2(3) + 1/log2(4) + 0/log2(5) = 5.0 + 1.89 + 0.5 + 0 = 7.39
- NDCG@4 = 5.93 / 7.39 = 0.80

**Verdict for our use case:** **The single most important metric.** Directly measures whether high-engagement articles appear at the top of the ranked list, with graded relevance that maps naturally to highlight counts. Use NDCG@5, NDCG@10, and NDCG@20 to match different surfacing cutoffs.

### 3.2 MAP@k (Mean Average Precision)

MAP computes precision at each position where a relevant item appears, then averages. It requires binary relevance labels (relevant or not), so you must threshold engagement (e.g., "relevant" = 1+ highlights) ([Shaped, "Evaluating recommendation systems (mAP, MMR, NDCG)"](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg)).

```
AP@k = (1 / min(R, k)) * SUM(i=1 to k) [ Precision@i * rel_i ]
```

where R is the total number of relevant items.

**How to apply:** Define a relevance threshold (e.g., >= 1 highlight or >= 3 highlights), compute AP for each ranking (or each time period's batch), then average across batches.

**Verdict for our use case:** Useful secondary metric. Easier to interpret than NDCG ("the area under the precision-recall curve at each rank position"), but loses the graded relevance information that NDCG preserves. Since we have actual highlight counts, NDCG is preferred.

### 3.3 MRR (Mean Reciprocal Rank)

MRR measures how quickly the first relevant item appears in the ranked list ([Evidently AI, "Mean Reciprocal Rank (MRR) explained"](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)):

```
MRR = (1 / |Q|) * SUM(q=1 to |Q|) [ 1 / rank_q ]
```

where rank_q is the position of the first relevant item for query q.

**When it matters:** MRR is ideal when finding *one* great article quickly is the goal. It completely ignores articles beyond the first relevant one. MRR = 1.0 means the best article is always ranked first.

**Verdict for our use case:** Useful as a supplementary metric if the question is "does the single best article each day end up at the top?" Less useful if we care about the quality of the entire top-10 list (use NDCG for that).

### Ranking Metric Summary

| Metric | Graded Relevance | Position-Aware | Best For |
|--------|:---:|:---:|---------|
| NDCG@k | Yes | Yes | Overall ranking quality with graded engagement |
| MAP@k | No (binary) | Yes | Ranking quality with binary engage/not-engage |
| MRR | No (binary) | Yes (first only) | Finding the single best article quickly |
| Precision@k | No (binary) | No | "What fraction of top-k are relevant?" |
| Spearman rho | Yes | N/A (full list) | Overall rank agreement across all items |

---

## 4. Cross-Validation Strategies for Small Datasets

With ~hundreds of examples, the validation strategy matters as much as the model. Poor validation leads to either optimistic bias (overfitting to the validation set) or high variance (unstable performance estimates).

### 4.1 Stratified K-Fold

Standard k-fold randomly partitions data into k folds and rotates through them. **Stratified** k-fold ensures that each fold preserves the class distribution of the target variable ([scikit-learn, "Cross-validation: evaluating estimator performance"](https://scikit-learn.org/stable/modules/cross_validation.html)).

For regression with highlight counts, stratification can be done by binning engagement into categories (e.g., 0, 1-2, 3+) and ensuring each fold has similar proportions.

**Configuration for ~hundreds of articles:**
- **k = 5** is recommended over k = 10 for small datasets. With 300 articles and k = 10, each test fold has only 30 articles -- too few for stable metric estimates. k = 5 gives 60 articles per fold.
- Stratify by engagement bin to ensure each fold contains both zero-engagement and high-engagement articles.

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Bin highlights for stratification
engagement_bins = np.digitize(highlight_counts, bins=[1, 3, 7])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, engagement_bins):
    # train and evaluate
    pass
```

### 4.2 Repeated Stratified K-Fold

A single 5-fold split is sensitive to the random partition. **Repeating** the process with different random seeds and averaging results reduces variance ([MachineLearningMastery, "A Gentle Introduction to k-fold Cross-Validation"](https://machinelearningmastery.com/k-fold-cross-validation/)).

**Recommendation for ~hundreds of articles:** Use **5x5 CV** (5 repetitions of stratified 5-fold) or **10x5 CV** for a total of 25-50 train/test splits. This provides a distribution of performance estimates rather than a single point estimate.

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = []
for train_idx, test_idx in rskf.split(X, engagement_bins):
    # train, predict, compute NDCG
    scores.append(ndcg_score)

print(f"NDCG@10: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
```

### 4.3 Leave-One-Out Cross-Validation (LOOCV)

LOOCV uses n-1 samples for training and 1 for testing, repeated n times. It is the least biased estimator (trains on nearly all data) but has high variance and is computationally expensive ([MachineLearningMastery, "LOOCV for Evaluating Machine Learning Algorithms"](https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/)).

**Problems for our use case:**
- **High variance:** Each test set is a single article. The performance estimate across n single-article evaluations has high variance, especially for ranking metrics that require multiple items to compute.
- **Ranking metrics are undefined:** You cannot compute NDCG or Precision@k on a single test item. LOOCV is fundamentally incompatible with ranking evaluation.
- **Correlated training sets:** Adjacent folds share n-2 training examples, leading to highly correlated model fits.

**Verdict:** **Do not use LOOCV for ranking evaluation.** It is only appropriate when per-item regression error (RMSE, MAE) is the sole concern and the dataset is extremely small (< 50 items).

### 4.4 Time-Series Split

If articles arrive chronologically and engagement patterns shift over time (e.g., user interests evolve, sources change), temporal ordering should be respected ([scikit-learn, "TimeSeriesSplit"](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).

**How it works:** Train on articles from time 0 to t, test on articles from t to t+delta. Slide forward and repeat.

```
Split 1: Train [Jan-Mar] -> Test [Apr]
Split 2: Train [Jan-Apr] -> Test [May]
Split 3: Train [Jan-May] -> Test [Jun]
...
```

**When to use:** Use time-series split when:
- Article topics or user behavior change over time
- The model will be deployed to predict engagement on *future* articles
- There is concern about temporal leakage (training on articles that are topically related to test articles from the same period)

**Tradeoff:** With ~hundreds of articles, time-series split produces fewer usable splits (early splits have very small training sets), and metrics are computed on small test sets. Combine with expanding window (growing training set) rather than sliding window (fixed-size training set).

**Practical hybrid approach:** Use stratified k-fold as the primary validation strategy, but run a time-series split as a secondary check to verify that performance does not degrade on recent articles.

### 4.5 Grouped K-Fold

If multiple articles come from the same source (e.g., same author, same publication), they may share characteristics that inflate performance if split across train and test. **Grouped k-fold** ensures all articles from the same group stay together in either train or test.

```python
from sklearn.model_selection import GroupKFold

# group_labels = source/author for each article
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=group_labels):
    pass
```

---

## 5. Bootstrap Confidence Intervals

Point estimates of model performance (e.g., "NDCG@10 = 0.72") are insufficient on small datasets. The confidence interval tells you how much that estimate might change with different data samples.

### 5.1 The Bootstrap Procedure

1. From the test set of size n, draw n samples with replacement (a bootstrap sample).
2. Compute the metric of interest (e.g., NDCG@10) on the bootstrap sample.
3. Repeat B times (B >= 2000 for stable intervals, ideally 10000).
4. Take the 2.5th and 97.5th percentiles of the B metric values as the 95% confidence interval.

This is the **percentile bootstrap**, the simplest and most common approach ([Sebastian Raschka, "Creating Confidence Intervals for Machine Learning Classifiers"](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html)).

### 5.2 Implementation

```python
import numpy as np

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for any metric."""
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return np.mean(scores), lower, upper

# Example usage
mean_ndcg, lower, upper = bootstrap_ci(
    y_true, y_pred,
    lambda yt, yp: ndcg_score([yt], [yp], k=10)
)
print(f"NDCG@10: {mean_ndcg:.3f} [{lower:.3f}, {upper:.3f}]")
```

### 5.3 The .632+ Bootstrap

For small datasets, the standard bootstrap tends to be pessimistic because each bootstrap sample omits ~36.8% of the data (the probability of any single item not being selected in n draws is (1 - 1/n)^n which approaches 1/e). The **.632+ bootstrap** corrects for this by blending the bootstrap estimate with the apparent (training set) error:

```
Err_632+ = (1 - w) * Err_apparent + w * Err_bootstrap
```

where w is adjusted based on the amount of overfitting. This provides a less biased estimate for small samples ([PMC, "Machine learning algorithm validation with a limited sample size"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6837442/)).

### 5.4 Practical Guidance for ~Hundreds of Articles

- **Always report confidence intervals**, not just point estimates. With 300 articles, a 95% CI on NDCG@10 might span 0.10-0.15, which is enough to make two models indistinguishable.
- Use at least **B = 5000** bootstrap samples for ranking metrics (they are noisier than classification metrics).
- When comparing two models, bootstrap the **difference** in their metrics directly. If the 95% CI of the difference includes zero, the models are not reliably different.
- Consider the **BCa (bias-corrected and accelerated) bootstrap** for more accurate intervals, especially when the sampling distribution is skewed (common with ranking metrics that are bounded at 1.0).

---

## 6. Handling Class Imbalance (Zero-Inflated Engagement)

With many articles having zero highlights and a long tail of engagement, standard approaches fail in predictable ways.

### 6.1 The Problem

If 70% of articles have zero highlights:
- A model predicting zero everywhere achieves 70% accuracy, low RMSE, and low MAE.
- Precision/recall at any fixed threshold are brittle.
- Regression models are pulled toward predicting near-zero values.

### 6.2 Two-Stage Modeling Approach

Zero-inflated models (from count data statistics) decompose the problem into two parts ([Springer, "A systematic approach for learning imbalanced data: enhancing zero-inflated models through boosting"](https://link.springer.com/article/10.1007/s10994-024-06558-3)):

1. **Stage 1 (Classification):** Will this article receive *any* engagement? (Binary: 0 vs. 1+)
2. **Stage 2 (Regression/Ranking):** Among engaged articles, how much engagement?

This maps naturally to our evaluation:
- Evaluate Stage 1 with AUC-PR and F1 on the positive class.
- Evaluate Stage 2 with Spearman rho and NDCG on the engaged subset.
- Evaluate the combined pipeline with NDCG@k and Precision@k on the full dataset.

### 6.3 Evaluation Strategies for Imbalanced Data

**Metrics to use:**
- **AUC-PR** over AUC-ROC: The precision-recall curve is more sensitive to performance on the minority class.
- **F1 on the positive class** (or F-beta with beta > 1 if recall matters more).
- **NDCG@k** naturally handles imbalance: zero-engagement articles at the bottom of the ranking do not affect the score.

**Metrics to avoid:**
- **Accuracy:** Meaningless with class imbalance.
- **AUC-ROC alone:** Can be misleadingly high. An AUC-ROC of 0.85 might correspond to an AUC-PR of only 0.45.
- **RMSE/MAE alone:** Dominated by the majority (zero) class.

**Stratification during CV:**
Always stratify folds by engagement bin. Without stratification, some folds may contain very few positive examples, making metric estimates unstable.

### 6.4 Resampling Approaches

For training (not evaluation), consider:
- **SMOTE or oversampling** of the engaged class during training.
- **Cost-sensitive learning:** Weight the loss function to penalize misclassification of engaged articles more heavily.
- **Undersampling** the zero-engagement class during training.

Never resample the test set -- evaluation must reflect the true class distribution.

---

## 7. Calibration

Calibration measures whether predicted scores correspond to actual engagement probabilities. A well-calibrated model that predicts a 30% probability of engagement should be correct about 30% of the time for articles receiving that score.

### 7.1 Why Calibration Matters

Even if the model ranks articles perfectly (high NDCG), miscalibrated scores cause problems:
- Users cannot interpret raw scores ("is 0.6 good?").
- Thresholds based on predicted scores become arbitrary.
- Combining scores from different models or features requires calibration.

### 7.2 Reliability Diagrams

A reliability diagram (calibration curve) plots predicted probability vs. observed frequency ([scikit-learn, "Probability Calibration curves"](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)):

1. Bin predicted scores into groups (e.g., 10 bins).
2. For each bin, compute the mean predicted score (x-axis) and the actual fraction of engaged articles (y-axis).
3. A perfectly calibrated model follows the diagonal.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_proba, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability Diagram')
```

### 7.3 Calibration Metrics

- **Brier Score:** Mean squared difference between predicted probability and actual outcome. Lower is better. Decomposes into calibration, refinement, and uncertainty components.
  ```
  Brier = (1/n) * SUM((p_i - o_i)^2)
  ```

- **Expected Calibration Error (ECE):** Weighted average of per-bin calibration errors. More directly measures calibration than Brier score.

- **Log Loss:** Heavily penalizes confident wrong predictions. Sensitive to calibration.

### 7.4 Calibration on Small Datasets

**Warning:** Post-hoc calibration methods (Platt scaling, isotonic regression) require held-out calibration data. With ~hundreds of articles, this is problematic:

- **Platt scaling** fits a logistic curve to map raw scores to calibrated probabilities. Requires minimal data (fits 2 parameters), so it is feasible on small datasets.
- **Isotonic regression** is nonparametric and highly flexible. It will overfit on small datasets -- avoid unless using cross-validated calibration.

**Recommendation:** Use **cross-validated calibration** (scikit-learn's `CalibratedClassifierCV` with `cv=5`) rather than a separate calibration holdout. For our dataset size, Platt scaling is preferred over isotonic regression. Evaluate calibration with the Brier score, but keep in mind that the reliability diagram bins will be noisy with few samples per bin.

---

## 8. Statistical Tests for Comparing Models

When comparing Model A vs. Model B, we need to determine whether the performance difference is statistically significant or just noise.

### 8.1 Paired Tests on Cross-Validation Folds

**The problem:** Standard paired t-tests on k-fold CV scores violate the independence assumption because training sets across folds overlap. This inflates Type I error (false positives -- concluding models differ when they do not) ([Demsar, "Statistical Comparisons of Classifiers over Multiple Datasets," JMLR 2006](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)).

**Corrected 5x2 CV t-test:** Dietterich (1998) proposed running 5 repetitions of 2-fold CV and using a modified t-statistic that accounts for the non-independence. This is the recommended approach for comparing two models on a single dataset ([MachineLearningMastery, "Statistical Significance Tests for Comparing Machine Learning Algorithms"](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)).

```python
from mlxtend.evaluate import paired_ttest_5x2cv

t, p = paired_ttest_5x2cv(estimator1=model_a, estimator2=model_b,
                           X=X, y=y, random_seed=42)
print(f"t-statistic: {t:.3f}, p-value: {p:.4f}")
```

### 8.2 Wilcoxon Signed-Rank Test

A nonparametric alternative to the paired t-test. Does not assume normally distributed differences, making it more appropriate for small samples where normality is hard to verify ([Demsar, 2006](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)).

```python
from scipy.stats import wilcoxon

# scores_a and scores_b are per-fold metric values
stat, p = wilcoxon(scores_a, scores_b)
```

**Minimum sample size:** Wilcoxon requires at least 6 paired observations to detect significance at the 0.05 level. With 5-fold CV, you have only 5 pairs -- use 10-fold or repeated CV to get more pairs.

### 8.3 McNemar's Test

For classification models, McNemar's test compares the disagreements between two classifiers on the same test set. It does not require cross-validation and works well when data is limited ([MachineLearningMastery, "How to Calculate McNemar's Test to Compare Two Machine Learning Classifiers"](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)).

The test statistic is computed from a 2x2 contingency table:

|  | Model B Correct | Model B Wrong |
|--|:---:|:---:|
| **Model A Correct** | a | b |
| **Model A Wrong** | c | d |

```
chi2 = (|b - c| - 1)^2 / (b + c)
```

McNemar's test is most appropriate when you can only evaluate on a single test set (no cross-validation), which can be the case when data is extremely limited.

### 8.4 Bootstrap Test for Comparing Models

The most flexible approach: bootstrap the *difference* in metrics between two models.

```python
def bootstrap_comparison(y_true, pred_a, pred_b, metric_fn, n_bootstrap=10000):
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        score_a = metric_fn(y_true[idx], pred_a[idx])
        score_b = metric_fn(y_true[idx], pred_b[idx])
        diffs.append(score_a - score_b)

    p_value = np.mean(np.array(diffs) <= 0)  # one-sided test: A > B
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    return p_value, ci_lower, ci_upper
```

**Advantages:** Works with any metric (including NDCG), does not assume normality, provides both a p-value and a confidence interval for the difference. This is the recommended approach for comparing ranking metrics on small datasets.

### 8.5 Practical Significance

With ~hundreds of articles, statistical significance is necessary but not sufficient. Also assess **practical significance**: is the performance difference large enough to matter? A 0.02 improvement in NDCG@10 may be statistically significant but operationally meaningless.

Define minimum meaningful differences upfront:
- NDCG@10: delta >= 0.05
- Spearman rho: delta >= 0.05
- Precision@10: delta >= 0.10 (one additional relevant article in top 10)

---

## 9. Nested Cross-Validation

When both hyperparameter tuning and performance estimation happen on the same data, the performance estimate is optimistically biased. Nested CV solves this ([scikit-learn, "Nested versus non-nested cross-validation"](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).

### 9.1 Structure

```
Outer loop (performance estimation): 5-fold stratified CV
  Inner loop (hyperparameter tuning): 3-fold stratified CV within each outer training set
```

For each outer fold:
1. The inner loop tries all hyperparameter combinations on the outer training set using 3-fold CV.
2. The best hyperparameters are selected based on inner CV performance.
3. A model is trained on the full outer training set with those hyperparameters.
4. The model is evaluated on the outer test fold -- this is the unbiased performance estimate.

### 9.2 Computational Cost

With 5 outer folds and 3 inner folds, every hyperparameter combination is evaluated 5 x 3 = 15 times. For a grid of 50 hyperparameter combinations, that is 750 model fits. This is manageable for small models on hundreds of examples.

### 9.3 Implementation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Inner loop: hyperparameter tuning
clf = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='average_precision',  # AUC-PR
    refit=True
)

# Outer loop: unbiased performance estimation
nested_scores = cross_val_score(
    clf, X, engagement_bins,
    cv=outer_cv,
    scoring='average_precision'
)

print(f"Nested CV AUC-PR: {nested_scores.mean():.3f} +/- {nested_scores.std():.3f}")
```

### 9.4 When to Skip Nested CV

Nested CV is most valuable when:
- The hyperparameter search space is large.
- You are comparing models with different numbers of hyperparameters (e.g., a simple linear model vs. a tuned random forest).

It may be overkill when:
- You are using a model with few or no hyperparameters (e.g., logistic regression with default regularization).
- Computational budget is severely limited.
- The dataset is so small (< 100) that the inner folds become unreliable.

For ~hundreds of articles, nested CV is recommended but use **5 outer x 3 inner** folds (not 10 x 10) to keep the inner training sets large enough ([Ploomber, "Model selection done right: A gentle introduction to nested cross-validation"](https://ploomber.io/blog/nested-cv/)).

---

## 10. Practical Recommendations for ~Hundreds of Articles

### 10.1 Primary Evaluation Framework

Use this hierarchy of metrics, in order of importance:

1. **NDCG@k** (k = 5, 10, 20): The primary metric. Directly measures ranking quality with graded relevance from highlight counts.
2. **Spearman rho / Kendall tau:** Full-list ranking correlation. Validates that the overall ordering is sensible.
3. **Precision@k:** Operational metric. "How many of the top k are actually good?"
4. **AUC-PR:** Threshold-independent classification quality on the positive (engaged) class.
5. **RMSE / MAE:** Supplementary. Useful for understanding prediction error magnitude but not for ranking quality.

### 10.2 Validation Protocol

```
Primary:    Repeated Stratified 5-Fold CV (10 repeats = 50 evaluations)
Secondary:  Time-series split (to check for temporal degradation)
Comparison: Bootstrap test on metric differences (B = 10000)
Confidence: Bootstrap CIs on all reported metrics (B = 5000)
Tuning:     Nested CV (5 outer x 3 inner) when hyperparameters are involved
```

### 10.3 Avoiding Overfitting Checklist

- [ ] **Never evaluate on training data.** Even for "quick checks," always use held-out data.
- [ ] **Stratify all splits** by engagement bin (0, 1-2, 3+, 7+).
- [ ] **Use repeated CV** to average out split-specific variance.
- [ ] **Regularize aggressively.** With hundreds of features and hundreds of examples, prefer L1/L2 regularization, limit tree depth, or use dimensionality reduction.
- [ ] **Prefer simpler models.** With hundreds of examples, logistic regression or small random forests often outperform deep networks.
- [ ] **Track train-test gap.** If training NDCG is 0.95 and test NDCG is 0.60, the model is overfitting.
- [ ] **Feature selection inside CV.** Never select features on the full dataset before CV -- this causes data leakage. Feature selection must happen inside each fold.
- [ ] **Monitor learning curves.** Plot performance vs. training set size. If the test curve is still rising steeply, more data would help more than model tuning.

### 10.4 Reporting Template

When reporting model performance, include all of the following:

```
Model: [name, key hyperparameters]
Dataset: N = [total], N_engaged = [articles with 1+ highlight], N_zero = [articles with 0]
Validation: Repeated Stratified 5-Fold CV (10 repeats)

Ranking Metrics (mean +/- std [95% CI]):
  NDCG@5:     0.XX +/- 0.XX [0.XX, 0.XX]
  NDCG@10:    0.XX +/- 0.XX [0.XX, 0.XX]
  NDCG@20:    0.XX +/- 0.XX [0.XX, 0.XX]
  Spearman:   0.XX +/- 0.XX [0.XX, 0.XX]
  Kendall:    0.XX +/- 0.XX [0.XX, 0.XX]

Classification Metrics (threshold = [X]):
  Precision@5:  0.XX +/- 0.XX
  Precision@10: 0.XX +/- 0.XX
  AUC-PR:       0.XX +/- 0.XX [0.XX, 0.XX]
  AUC-ROC:      0.XX +/- 0.XX

Regression Metrics:
  RMSE:  0.XX +/- 0.XX
  MAE:   0.XX +/- 0.XX

Calibration:
  Brier Score: 0.XX

Baselines:
  Random ranking NDCG@10:        0.XX
  Popularity-based NDCG@10:      0.XX
  Current LLM scoring NDCG@10:   0.XX
```

### 10.5 Baseline Models

Always compare against these baselines:
1. **Random ranking:** Establishes the floor. NDCG@10 for random is approximately log2(k+1) * prevalence / k (varies by data).
2. **Popularity-based:** Rank by source popularity or recency. This is the "does the ML model beat simple heuristics?" check.
3. **Current LLM scoring:** The existing system's performance. This is the bar to beat.
4. **Always-predict-mean:** For regression, predicting the mean highlight count for all articles. If the model cannot beat this, it has learned nothing.

### 10.6 Dataset Size Thresholds

| Dataset Size | Recommended Approach |
|---|---|
| < 50 articles | LOOCV for regression only; bootstrap CI; do not trust ranking metrics |
| 50-200 articles | Repeated stratified 5-fold CV; bootstrap CI; focus on Spearman/NDCG@5 |
| 200-500 articles | Repeated stratified 5-fold CV; nested CV for tuning; NDCG@5/10/20 |
| 500-1000 articles | Stratified 5-fold or 10-fold CV; all metrics become reliable |
| 1000+ articles | Standard train/val/test split is viable |

---

## Sources

- [scikit-learn: Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: Probability Calibration curves](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)
- [Evidently AI: NDCG explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)
- [Evidently AI: Precision and recall at K](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
- [Evidently AI: Mean Reciprocal Rank (MRR) explained](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)
- [Evidently AI: 10 metrics to evaluate recommender and ranking systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)
- [Shaped: Evaluating recommendation systems (mAP, MMR, NDCG)](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg)
- [Sebastian Raschka: Creating Confidence Intervals for Machine Learning Classifiers](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html)
- [Sebastian Raschka: Model evaluation, model selection, and algorithm selection in ML](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html)
- [MachineLearningMastery: A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)
- [MachineLearningMastery: LOOCV for Evaluating Machine Learning Algorithms](https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/)
- [MachineLearningMastery: Statistical Significance Tests for Comparing ML Algorithms](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)
- [MachineLearningMastery: How to Calculate McNemar's Test](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)
- [MachineLearningMastery: Bootstrap Confidence Intervals for ML Results](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
- [Demsar: Statistical Comparisons of Classifiers over Multiple Datasets, JMLR 2006](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)
- [PMC: Machine learning algorithm validation with a limited sample size](https://pmc.ncbi.nlm.nih.gov/articles/PMC6837442/)
- [PMC: Machine learning models and over-fitting considerations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8905023/)
- [PMC: Toward Generalizable ML Models in Speech, Language, and Hearing Sciences](https://pmc.ncbi.nlm.nih.gov/articles/PMC11005022/)
- [Springer: A systematic approach for learning imbalanced data (zero-inflated models)](https://link.springer.com/article/10.1007/s10994-024-06558-3)
- [Google Developers: Class-imbalanced datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)
- [UVA Library: Correlation: Pearson, Spearman, and Kendall's tau](https://library.virginia.edu/data/articles/correlation-pearson-spearman-and-kendalls-tau)
- [Statistics Solutions: Kendall's Tau and Spearman's Rank Correlation Coefficient](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/kendalls-tau-and-spearmans-rank-correlation-coefficient/)
- [ScienceDirect: Effective use of Spearman's and Kendall's correlation coefficients](https://www.sciencedirect.com/science/article/abs/pii/S0003347215000196)
- [Ploomber: Model selection done right: A gentle introduction to nested cross-validation](https://ploomber.io/blog/nested-cv/)
- [Neptune.ai: Cross-validation in Machine Learning: How to Do It Right](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
- [Pinecone: Evaluation Measures in Information Retrieval](https://www.pinecone.io/learn/offline-evaluation/)
- [Weaviate: Evaluation Metrics for Search and Recommendation Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [Wikipedia: Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Wikipedia: Cross-validation (statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
