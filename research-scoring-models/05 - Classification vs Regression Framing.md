# Classification vs. Regression Framing for Engagement Prediction

## Executive Summary

For predicting reader engagement (highlight counts) with ~hundreds of articles where >50% have zero highlights, the optimal approach is a **two-stage hurdle model**: a binary classifier predicting P(any engagement), followed by a conditional regression model predicting engagement intensity given engagement occurs. This framing directly addresses the zero-inflation problem that plagues naive regression, outperforms pure classification by preserving fine-grained engagement signal, and is well-supported by both the content recommendation literature and the statistical modeling literature on zero-inflated count data. The two-stage architecture also maps cleanly onto the existing 0-100 scoring output requirement: the binary stage gates whether an article crosses the "worth reading" threshold, while the regression stage calibrates where within the engaged range it falls.

---

## 1. Regression Approaches

### 1.1 Direct Regression (MSE Loss)

The most straightforward framing: train a model to predict a continuous engagement score (0-100) directly, minimizing mean squared error.

**Strengths:**
- Simple to implement and interpret
- Produces the continuous 0-100 output natively
- Works well when the target distribution is roughly symmetric and unimodal

**Weaknesses with our data:**
- MSE is highly sensitive to outliers. A single article with 20 highlights in a dataset where the median is 0 will dominate the loss landscape.
- Zero-inflated distributions violate the implicit Gaussian assumption. The model will predict moderate scores for zero-engagement articles (regression to the mean) rather than confidently predicting zero ([Towards Data Science: Zero-Inflated Regression](https://towardsdatascience.com/zero-inflated-regression-c7dfc656d8af/)).
- With >50% zeros, the optimal MSE prediction for a mediocre model is to predict near-zero for everything, which is uninformative for ranking.

### 1.2 MAE (L1 Loss)

Mean absolute error is more robust to outliers than MSE because it penalizes errors linearly rather than quadratically.

**Strengths:**
- Less distorted by high-engagement outliers
- The optimal prediction under MAE is the conditional median, which for zero-inflated data may actually be zero -- correctly reflecting the modal outcome

**Weaknesses:**
- Non-differentiable at zero, which can cause optimization instability
- Still does not explicitly model the zero-generation process
- Predicting the median (zero) for most articles provides no discriminative power among the majority class

### 1.3 Huber Loss

Huber loss combines MSE for small errors (|error| < delta) and MAE for large errors, controlled by a threshold parameter delta ([Wikipedia: Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)).

**Strengths:**
- Balances sensitivity to small differences (quadratic region) with robustness to outliers (linear region)
- Differentiable everywhere, unlike MAE
- Delta parameter can be tuned: for zero-inflated data, a smaller delta emphasizes the quadratic region around zero, helping the model discriminate among low-engagement articles

**Weaknesses:**
- Still a single-model approach that does not explicitly decompose the zero-inflation
- Requires careful tuning of delta, which adds a hyperparameter

### 1.4 Quantile Regression

Instead of predicting the conditional mean or median, quantile regression estimates specific percentiles of the conditional distribution (e.g., the 75th percentile of engagement given article features).

**Strengths:**
- Can characterize the full conditional distribution, not just a point estimate
- Naturally handles asymmetric distributions
- Multiple quantile predictions can be combined to produce confidence intervals

**Weaknesses:**
- Choosing which quantile to report as "the score" is an additional design decision
- For zero-inflated data, low quantiles (below the zero-inflation rate) will all predict zero, providing no useful signal
- More complex to train and interpret than point regression

### 1.5 Tweedie Regression

The Tweedie family of distributions is specifically designed for zero-inflated continuous data. It is parameterized by a power parameter p (1 < p < 2 for zero-inflated continuous data) and naturally produces a point mass at zero plus a continuous right-skewed distribution for positive values. All major gradient boosting libraries (XGBoost, LightGBM, CatBoost) support Tweedie loss natively.

**Strengths:**
- Directly models the zero-inflated, right-skewed distribution in a single model
- No need for an explicit two-stage architecture
- Well-studied in insurance actuarial science for claim prediction, which has an analogous zero-inflation structure ([Arxiv: Zero-Inflated Tweedie Boosted Trees](https://arxiv.org/html/2406.16206v1))
- Recent research shows zero-inflated Tweedie models with gradient boosting substantially outperform standard Poisson and Tweedie GLMs, with CatBoost ZIPB2 achieving pseudo R-squared of 0.520 vs 0.046 for standard Poisson on insurance claim data ([Arxiv: Enhanced Gradient Boosting for Zero-Inflated Claims](https://arxiv.org/html/2307.07771v3))

**Weaknesses:**
- Assumes a specific distributional form that may not match highlight count data
- The power parameter adds complexity
- Less interpretable than a two-stage model where P(engage) has a clear meaning

---

## 2. Binary Classification: "Engaged" vs. "Not Engaged"

### 2.1 Core Approach

Reduce the problem to: given article features, predict whether the reader will highlight *anything* (1) or not (0). This is the simplest possible framing.

**Strengths:**
- Clean, well-understood problem with abundant tooling
- Directly addresses the dominant mode in the data (zero vs. non-zero)
- Many production recommendation systems use this framing: YouTube's original DNN ranking system used weighted logistic regression for watch time prediction, where the core model is essentially a binary classifier (clicked/not-clicked) with positive examples weighted by observed watch time ([Covington et al., 2016](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf))
- With hundreds of examples and >50% zeros, this may be the most statistically robust framing -- binary outcomes require the fewest parameters to estimate

**Weaknesses:**
- Discards all information about *how much* engagement occurred. An article with 1 highlight and an article with 15 highlights are treated identically.
- The predicted probability P(engage) serves as a ranking score, but it conflates "likely to engage at all" with "likely to engage deeply" -- these are correlated but not identical.
- Does not directly produce a 0-100 score; requires a calibration/mapping step.

### 2.2 Threshold Selection

The binarization threshold is a critical design decision:

- **Zero vs. non-zero** (any highlights): Most natural boundary. Maximizes sample size in both classes. Directly models the dominant data characteristic (zero-inflation).
- **Median split**: Guarantees balanced classes but is arbitrary and shifts with data.
- **Domain-informed threshold**: e.g., "3+ highlights = engaged." This captures meaningful engagement but reduces the positive class size, potentially worsening class imbalance.

**Recommendation:** Use zero vs. non-zero as the binary threshold. This aligns with the hurdle model interpretation (did the reader engage *at all*?) and makes the most of limited data.

### 2.3 Weighted Logistic Regression (YouTube Approach)

Covington et al. (2016) at YouTube showed that binary classification can implicitly predict continuous engagement by weighting positive examples by their engagement magnitude. Positive impressions are weighted by observed watch time; negative impressions receive unit weight. At serving time, the model's logistic output approximates E[watch time] rather than P(click) ([The Morning Paper: YouTube Recommendations](https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/)).

This is a compelling hybrid: train as classification, serve as regression. However, it requires significant scale (YouTube had billions of examples) to work well. With hundreds of examples, the weighting scheme may introduce excessive variance.

---

## 3. Ordinal Classification: Ordered Tiers

### 3.1 Core Approach

Instead of binary (engage/not) or continuous (0-100), define ordered categories:

- **Tier 0:** No engagement (0 highlights)
- **Tier 1:** Low engagement (1-2 highlights)
- **Tier 2:** Medium engagement (3-5 highlights)
- **Tier 3:** High engagement (6+ highlights)

Train an ordinal classifier that respects the ordering: predicting Tier 0 when truth is Tier 3 is a worse error than predicting Tier 2.

### 3.2 Statistical Efficiency Argument

Frank Harrell has shown that ordinal outcomes provide substantially more statistical information than binary outcomes for the same sample size. Specifically, with the same number of subjects and desired power, ordinal outcomes with 5 categories can detect an odds ratio of 0.618 vs. 0.5 for binary, meaning ordinal outcomes require smaller samples to achieve the same statistical power ([Harrell: Information Gain From Ordinal vs. Binary Outcomes](https://www.fharrell.com/post/ordinal-info/)). This is directly relevant to our small dataset scenario.

Recent work on student engagement measurement has applied supervised contrastive learning to ordinal engagement classification, representing engagement levels as ordered variables rather than distinct categories ([Arxiv: Supervised Contrastive Learning for Ordinal Engagement](https://arxiv.org/html/2505.20676)).

### 3.3 Methods

**Cumulative Link Models (Proportional Odds):** The classic ordinal regression approach assumes a latent continuous variable with learned thresholds dividing it into ordered categories. The proportional odds model estimates a single effect per predictor across all thresholds, reducing parameter count ([Arxiv: Cumulative Link Models for Deep Ordinal Classification](https://arxiv.org/abs/1905.13392)).

**Binary Decomposition:** Train K-1 binary classifiers for K categories. Classifier k predicts P(Y > k). The ordinal prediction is derived from these cumulative probabilities. This approach is simple and works with any binary classifier ([Frank & Hall, 2001](https://www.researchgate.net/publication/226877154_A_Simple_Approach_to_Ordinal_Classification)).

**Neural Ordinal Regression:** Extend the cumulative link model into a neural network by placing a single output neuron (the latent variable) and learning thresholds as additional parameters. This has been shown to work well for deep ordinal classification tasks.

### 3.4 Mapping Back to Continuous Scores

To produce a 0-100 score from ordinal predictions:

1. **Expected value method:** Assign each tier a representative score (e.g., Tier 0 = 10, Tier 1 = 35, Tier 2 = 60, Tier 3 = 85). Compute E[score] = sum of P(tier_k) * score_k. This produces a smooth continuous output.
2. **Latent variable extraction:** In cumulative link models, the latent variable value before thresholding is a continuous score. Normalize it to 0-100 directly.
3. **Calibrated probability:** Use P(Tier >= 1) * 100 as a simple engagement likelihood score.

### 3.5 Weaknesses

- Tier boundary selection is arbitrary and affects model performance; recent watch-time prediction research shows that "model performance is highly contingent on the method of time interval segmentation" ([Arxiv: Generative Regression for Watch Time](https://arxiv.org/html/2412.20211v1))
- With hundreds of examples split across 4 tiers, higher tiers may have very few examples (potentially <20), making training unstable
- The proportional odds assumption may not hold for engagement data

---

## 4. Two-Stage (Hurdle) Models

### 4.1 Core Architecture

A hurdle model decomposes the prediction into two sequential stages:

1. **Stage 1 (Binary Classifier):** Predict P(engage > 0) -- will the reader highlight anything?
2. **Stage 2 (Conditional Regressor):** Given engagement > 0, predict E(engagement | engagement > 0) -- how much will they engage?

The final prediction is: **E[engagement] = P(engage > 0) * E[engagement | engage > 0]**

This decomposition is the standard statistical approach for zero-inflated count data and is well-established in the econometrics and biostatistics literature ([PMC: Zero-Inflated and Hurdle Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC3238139/)).

### 4.2 Hurdle vs. Zero-Inflated Models

These are often confused but differ in their assumptions about zeros:

- **Hurdle models** assume all zeros come from a single process (the "hurdle" of whether to engage at all). Once the hurdle is crossed, a separate process determines the count. This is the more natural framing for article engagement: the decision to highlight *anything* is qualitatively different from how many passages to highlight.

- **Zero-inflated models** assume zeros come from two processes: "structural zeros" (articles the reader *could never* engage with, e.g., wrong language) and "sampling zeros" (articles they could engage with but didn't this time). This distinction requires a latent mixture, which is harder to estimate with small samples.

A comparison study found that hurdle models are generally more robust, particularly because they can handle zero-deflation at certain covariate levels, while zero-inflated models cannot ([PMC: Comparison of ZI and Hurdle Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570364/)). For our use case, the hurdle framing is more appropriate: the question is whether the reader engages, not whether the article is structurally un-engageable.

### 4.3 Evidence from Recommendation Literature

The two-stage approach has strong precedent in production recommendation systems:

- **User-item consumption rates:** Lund & Bhaskar (2018) showed that zero-inflated Poisson regression models systematically outperformed matrix factorization methods across music, restaurant, and social media datasets for predicting sparse user-item consumption rates ([ACM: Prediction of Sparse User-Item Consumption Rates with ZIP Regression](https://dl.acm.org/doi/abs/10.1145/3178876.3186153)).

- **Insurance claims:** The actuarial literature has extensively validated two-stage models for zero-inflated claim data. CatBoost with zero-inflated Poisson boosted trees (ZIPB2) achieved pseudo R-squared of 0.520 compared to 0.046 for standard Poisson models -- an order of magnitude improvement ([Arxiv: Enhanced Gradient Boosting for Zero-Inflated Claims](https://arxiv.org/html/2307.07771v3)).

- **Demand forecasting:** A 2025 study on zero-inflated data demonstrated that a two-fold modeling approach (classifier + regressor) significantly outperformed single-model regression, increasing weighted Precision by 39%, Recall by 49%, F1 by 88%, and AUC by 48% ([ScienceDirect: Two-Fold ML Approach for Zero-Inflated Data](https://www.sciencedirect.com/science/article/pii/S0952197625003392)).

### 4.4 Practical Implementation

**Stage 1 options:**
- Logistic regression (interpretable, works well with small data)
- Gradient boosted classifier (XGBoost/LightGBM, more expressive)
- For our LLM-based system: the existing scoring dimensions could feed into a simple binary classifier

**Stage 2 options (trained only on engaged articles):**
- Linear regression or gradient boosted regressor on log(highlights)
- Poisson or negative binomial regression (natural for count data)
- Truncated distribution models (only model positive counts)

**Score mapping:**
```
final_score = P(engage) * (conditional_score_normalized_to_0_100)
```
Where `conditional_score_normalized_to_0_100` maps the conditional engagement prediction to the 0-100 range using min-max scaling on the training set's non-zero engagement distribution. Alternatively:
```
final_score = P(engage) * 100  # Simple version using just the binary probability
```

### 4.5 Advantages for Small Datasets

The two-stage decomposition is particularly valuable with ~hundreds of examples:

- **Stage 1** has the full dataset with balanced-ish classes (~50/50 split), maximizing effective sample size
- **Stage 2** has a smaller but homogeneous dataset (only engaged articles), avoiding the zero-inflation problem entirely
- Each stage can use a simpler model that is less prone to overfitting
- The architecture is interpretable: you can examine what drives engagement likelihood vs. engagement depth separately

---

## 5. Learning-to-Rank

### 5.1 Core Idea

Instead of predicting an absolute score, predict the relative ordering of articles. The model learns: "Article A should rank higher than Article B." This sidesteps the question of what the score *means* and focuses on what matters for triage: relative priority.

### 5.2 Approaches

**Pointwise:** Train a regression/classification model on individual items. Ranking is implicit from predicted scores. This is what our current system does.

**Pairwise (RankNet, LambdaRank):** Train on pairs of articles where one has more engagement than the other. The loss function penalizes incorrect orderings. RankNet uses a neural network to predict P(article_i > article_j) from feature differences ([Burges et al.: From RankNet to LambdaRank to LambdaMART](https://www.researchgate.net/publication/228936665_From_ranknet_to_lambdarank_to_lambdamart_An_overview)).

**Listwise (LambdaMART, ListNet):** Optimize directly for ranking metrics like NDCG. LambdaMART combines gradient boosted trees with metric-aware optimization and is widely considered the strongest non-neural LTR method ([Shaped: LambdaMART Explained](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank)). XGBoost has native LTR support.

**Hybrid Pointwise-Pairwise:** Recent work on news recommendation shows that combining pointwise relevance prediction with pairwise reranking improves AUC by up to 7.95% on certain datasets, with the largest gains when the pointwise model is weak ([Arxiv: Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation](https://arxiv.org/abs/2409.17711)).

### 5.3 Advantages

- Directly optimizes for the ranking task, which is what article triage actually needs
- Naturally handles zero-inflation: zero-highlight articles should rank below any highlighted article, and the pairwise loss captures this
- Does not require a well-calibrated absolute score -- only relative ordering
- Robust to label noise and outliers in engagement counts

### 5.4 Weaknesses

- Requires mapping back to 0-100 scores (can normalize the model's raw scores)
- Pairwise training creates O(n^2) pairs from n articles, but with hundreds of articles this is manageable (~tens of thousands of pairs)
- "GBDT and neural LTR can be quite data hungry" with small datasets; an LLM re-ranker may be more effective in low-data regimes ([Shaped: LambdaMART Explained](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank))
- The model does not learn the semantics of "a score of 75 means X" -- only relative ordering

### 5.5 Relevance to Our Setting

Our current system already functions as an implicit pointwise ranker (LLM assigns scores, articles are ranked by score). A learning-to-rank framing could improve this by training on pairwise comparisons of articles with known engagement labels, then using the trained model to rerank or calibrate the LLM's raw scores.

---

## 6. Handling Zero-Inflation Across Framings

| Framing | How It Handles Zeros | Effectiveness |
|---------|---------------------|---------------|
| **Direct Regression (MSE)** | Does not handle; predicts moderate scores for zero articles | Poor |
| **Direct Regression (MAE)** | Median prediction may be zero; uninformative | Poor |
| **Huber Loss** | Somewhat robust to outliers but no zero-specific handling | Mediocre |
| **Tweedie Regression** | Natively models point mass at zero + right-skewed positive | Good |
| **Binary Classification** | Explicitly models zero vs. non-zero | Good for stage 1 |
| **Ordinal Classification** | Zero is the lowest tier; other tiers model non-zero levels | Moderate |
| **Hurdle Model** | Explicitly decomposes into zero/non-zero + conditional amount | Excellent |
| **Zero-Inflated Model** | Two latent zero processes + count process | Good but complex |
| **Learning-to-Rank** | Zeros rank below non-zeros; pairwise loss handles naturally | Good |

The key insight is that zero-inflation is not just a statistical nuisance -- it reflects a genuine two-process phenomenon. The decision to engage at all (open the article, start reading carefully) is qualitatively different from the depth of engagement (how many passages warrant highlighting). Models that explicitly represent this distinction consistently outperform those that treat it as a single continuous prediction.

---

## 7. Evidence from Content Recommendation Literature

### 7.1 YouTube (Covington et al., 2016)

YouTube's seminal recommendation paper frames engagement prediction as weighted logistic regression: a binary classification model where positive examples are weighted by observed watch time. At inference time, the model's output approximates expected watch time rather than click probability. This works at YouTube's scale (billions of examples) but may not translate to small datasets ([Covington et al., 2016](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf)).

### 7.2 YouTube (2024: Generative Regression)

A December 2024 paper from Kuaishou (a TikTok competitor) compares direct value regression, ordinal regression, and a novel "generative regression" approach for watch time prediction. Key finding: ordinal regression outperforms direct value regression but is sensitive to discretization. Generative regression (treating the prediction as a sequence generation problem) achieved 3-4% MAE improvement and 1-3% XAUC improvement across multiple datasets, including a 0.129% increase in video consumption time in online A/B tests with 10M+ users ([Arxiv: Generative Regression for Watch Time](https://arxiv.org/html/2412.20211v1)).

### 7.3 Netflix

Netflix's recommendation system uses Tweedie regression for engagement optimization. Offline simulations and online A/B tests revealed that Tweedie regression substantially enhanced viewing time and revenue metrics compared to standard logistic loss ([Netflix Research](https://research.netflix.com/research-area/recommendations)). This is notable because Netflix has the engineering resources to compare many approaches and settled on a distribution-aware regression method.

### 7.4 Sparse Consumption Rates

Lund & Bhaskar (2018) directly addressed the problem of predicting sparse user-item consumption rates (analogous to article highlight counts). Their zero-inflated Poisson regression models outperformed matrix factorization across music, restaurant review, and social media datasets. The key advantage was explicitly modeling the probability that a user-item pair generates any consumption at all ([ACM: ZIP Regression for Consumption Rates](https://dl.acm.org/doi/abs/10.1145/3178876.3186153)).

### 7.5 Click-Through Rate and Post-Click Engagement

Modern recommendation systems increasingly use multi-stage prediction: first predict P(click), then predict P(conversion | click) or E(engagement | click). This two-stage architecture mirrors the hurdle model and is standard practice at companies like Alibaba, Google, and Meta ([MDPI: A Review of CTR Prediction Using Deep Learning](https://www.mdpi.com/2079-9292/14/18/3734)).

---

## 8. Practical Recommendations

### Primary Recommendation: Two-Stage Hurdle Model

**For our specific setting** (~hundreds of articles, >50% zero highlights, need 0-100 output), the recommended approach is:

#### Stage 1: Binary Classification
- **Model:** Logistic regression or small gradient-boosted classifier
- **Target:** Any highlights (1) vs. no highlights (0)
- **Features:** Text embeddings, article metadata, source reputation, content type, length
- **Output:** P(engage) -- calibrated probability of any engagement

#### Stage 2: Conditional Regression
- **Model:** Linear regression, Poisson regression, or small gradient-boosted regressor
- **Training data:** Only articles with highlights > 0
- **Target:** log(highlight_count) or highlight_count directly
- **Output:** E[engagement | engaged] -- expected engagement depth

#### Score Composition
```
raw_score = P(engage) * normalize_to_100(E[engagement | engaged])
```

Where `normalize_to_100` maps the conditional prediction to a 0-100 scale. Articles with P(engage) < 0.3 map to the low tier (score < 30), P(engage) between 0.3-0.6 with moderate conditional engagement map to medium (30-59), and P(engage) > 0.6 with high conditional engagement map to high (60+).

### Why Not the Alternatives?

| Alternative | Reason Against |
|------------|----------------|
| Direct regression | Zero-inflation causes poor calibration; MSE dominated by zeros |
| Binary classification only | Discards engagement magnitude; all engaged articles scored the same |
| Ordinal classification | Tier boundaries are arbitrary; small per-tier sample sizes |
| Tweedie regression | Good option but less interpretable; harder to debug two failure modes separately |
| Learning-to-rank | Best for pure ranking but we need calibrated scores; data-hungry for pairwise training |

### Fallback: Tweedie Regression

If the two-stage architecture proves too complex for the current system (e.g., maintaining two models is burdensome), **Tweedie regression with gradient boosting** is the strongest single-model alternative. It natively handles zero-inflation, is supported by XGBoost/LightGBM/CatBoost, and has strong empirical backing from Netflix and the insurance actuarial literature.

### Integration with Current LLM Scoring

The existing 4-dimension LLM scoring system can serve as a **feature generator** for either approach:

1. Use the four LLM dimension scores (quotability, surprise, argument quality, applicable insight) as input features to the hurdle model
2. Train Stage 1 to predict P(engage) from these features + article metadata
3. Train Stage 2 to predict engagement depth from the same features
4. The ML model learns which LLM dimensions are actually predictive of real engagement, and how to weight them -- replacing the current equal-weighted sum with a data-driven combination

This hybrid approach (LLM for feature extraction, ML for calibration) is likely to outperform both pure-LLM scoring and pure-ML-on-raw-text, especially with limited labeled data.

### Sample Size Considerations

With ~hundreds of articles:
- Binary classification (Stage 1): ~50% positive class gives ~100-200 examples per class. This is adequate for logistic regression with <20 features and marginal for gradient boosting.
- Conditional regression (Stage 2): ~50-100 positive examples. This is tight for complex models but adequate for linear regression or Poisson regression with <10 features.
- Cross-validation (5-fold) is essential given the small dataset; held-out test sets alone will have high variance.
- Consider regularization (L1/L2 for regression, early stopping for boosting) to prevent overfitting.

---

## Sources

- [Towards Data Science: Zero-Inflated Regression](https://towardsdatascience.com/zero-inflated-regression-c7dfc656d8af/) -- Overview of zero-inflated regression approaches and meta-model architecture
- [ScienceDirect: Two-Fold ML Approach for Zero-Inflated Data (2025)](https://www.sciencedirect.com/science/article/pii/S0952197625003392) -- Two-fold modeling approach performance improvements
- [Towards Data Science: Zero-Inflated Data Model Comparison](https://towardsdatascience.com/zero-inflated-data-comparison-of-regression-models/) -- Comparison of regression models for zero-inflated data
- [ACM: Prediction of Sparse User-Item Consumption Rates with ZIP Regression (2018)](https://dl.acm.org/doi/abs/10.1145/3178876.3186153) -- ZIP regression outperforming matrix factorization for consumption prediction
- [PMC: Zero-Inflated and Hurdle Models of Count Data (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3238139/) -- Foundational reference on hurdle vs. ZI models
- [PMC: Comparison of Zero-Inflated and Hurdle Models (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570364/) -- Detailed comparison showing when each model type is preferred
- [Wikipedia: Hurdle Model](https://en.wikipedia.org/wiki/Hurdle_model) -- Formal definition and statistical properties
- [UCLA: Zero-Inflated and Hurdle Models](https://stats.oarc.ucla.edu/r/seminars/zero-inflated-and-hurdle-models-for-count-data-in-r/) -- Practical implementation guide
- [Covington et al.: Deep Neural Networks for YouTube Recommendations (2016)](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf) -- YouTube's weighted logistic regression approach
- [The Morning Paper: YouTube Recommendations](https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/) -- Accessible summary of YouTube's approach
- [Arxiv: Generative Regression for Watch Time Prediction (2024)](https://arxiv.org/html/2412.20211v1) -- Comparison of regression, ordinal, and generative approaches for engagement
- [Arxiv: Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation (2024)](https://arxiv.org/abs/2409.17711) -- Hybrid LTR approach for news recommendation
- [Shaped: LambdaMART Explained](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank) -- Overview of LambdaMART and learning-to-rank
- [Burges et al.: From RankNet to LambdaRank to LambdaMART](https://www.researchgate.net/publication/228936665_From_ranknet_to_lambdarank_to_lambdamart_An_overview) -- Evolution of pairwise LTR methods
- [Wikipedia: Learning to Rank](https://en.wikipedia.org/wiki/Learning_to_rank) -- Taxonomy of pointwise, pairwise, and listwise approaches
- [Wikipedia: Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) -- Definition and properties of Huber loss
- [Harrell: Information Gain From Ordinal vs. Binary Outcomes](https://www.fharrell.com/post/ordinal-info/) -- Statistical efficiency of ordinal vs. binary outcomes
- [Arxiv: Supervised Contrastive Learning for Ordinal Engagement (2025)](https://arxiv.org/html/2505.20676) -- Ordinal classification for engagement measurement
- [Arxiv: Cumulative Link Models for Deep Ordinal Classification (2019)](https://arxiv.org/abs/1905.13392) -- Neural network ordinal regression
- [Frank & Hall: A Simple Approach to Ordinal Classification (2001)](https://www.researchgate.net/publication/226877154_A_Simple_Approach_to_Ordinal_Classification) -- Binary decomposition for ordinal classification
- [Arxiv: Zero-Inflated Tweedie Boosted Trees with CatBoost (2024)](https://arxiv.org/html/2406.16206v1) -- Zero-inflated Tweedie with gradient boosting
- [Arxiv: Enhanced Gradient Boosting for Zero-Inflated Claims (2023)](https://arxiv.org/html/2307.07771v3) -- Comparison of XGBoost, LightGBM, CatBoost on zero-inflated data
- [Netflix Research: Recommendations](https://research.netflix.com/research-area/recommendations) -- Netflix's recommendation system overview
- [MDPI: A Review of CTR Prediction Using Deep Learning (2024)](https://www.mdpi.com/2079-9292/14/18/3734) -- Survey of multi-stage engagement prediction
- [Google: Framing an ML Problem](https://developers.google.com/machine-learning/problem-framing/ml-framing) -- Google's guidance on classification vs. regression framing
