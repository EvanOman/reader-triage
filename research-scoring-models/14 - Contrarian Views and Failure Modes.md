# 14 - Contrarian Views and Failure Modes

## When Embedding-Based Engagement Prediction Fails

**Date:** 2026-02-18
**Purpose:** Steel-man the opposition to embedding-based ML scoring. Honest assessment of failure modes, fundamental limitations, and radically different alternatives.

---

## Executive Summary

Embedding-based regression for predicting article engagement faces several fundamental challenges that may make it worse than simpler alternatives in our specific context: a single user, a few hundred labeled examples, a noisy target signal (highlights), and rapidly shifting interests. The curse of dimensionality is severe when projecting 1536-dimensional embeddings onto a scalar engagement score with only hundreds of training samples. Recent research (Rybinski et al., 2025) demonstrates that compressed embeddings can outperform full-dimensional ones on noisy regression tasks, suggesting the signal-to-noise ratio in our problem may be too low for high-dimensional representations to help. Before investing in ML infrastructure, we should honestly evaluate whether a calibrated LLM scorer, simple metadata heuristics, or a multi-armed bandit approach would deliver better results with far less engineering effort.

---

## 1. Embeddings Capture Semantic Similarity, Not Engagement

### The core conflation

Embeddings are trained to place semantically similar texts near each other in vector space. But semantic similarity and engagement-worthiness are fundamentally different properties. Two articles about "monetary policy" will have similar embeddings regardless of whether one is a brilliant contrarian analysis and the other is a dry press release. The embedding knows they are *about* the same thing; it has no representation of whether either is *worth reading*.

### Evidence

Google DeepMind's 2025 paper "On the Theoretical Limitations of Embedding-Based Retrieval" ([Weller et al., 2025](https://arxiv.org/abs/2508.21038)) demonstrates that single-vector embeddings have fundamental geometric constraints on what relationships they can represent. The paper proves that for any embedding dimension *d*, there exist combinations of documents that cannot be distinguished by any query. This is not a training problem---it is a mathematical limitation of the representation itself.

In practical terms: embeddings excel at answering "what is this article about?" but struggle with "is this article good?" The latter depends on writing quality, argument structure, novelty of framing, and personal resonance---properties that may not be linearly separable in embedding space.

### The regression assumption

When we regress engagement on embeddings, we assume there exists a hyperplane (or nonlinear surface) in embedding space that separates high-engagement from low-engagement articles. But if the engagement signal depends on properties orthogonal to the semantic axes that dominate the embedding, no amount of training data will find that hyperplane because it does not exist in that space.

---

## 2. Highlights Are a Noisy, Biased Signal

### What highlights actually measure

Highlighting behavior is not a clean measure of article value. It is confounded by:

- **Device**: Reading on a phone discourages highlighting; reading on a tablet encourages it
- **Time of day**: Late-night reading produces fewer highlights due to fatigue
- **Reading context**: Commute reading vs. focused desk reading
- **Article length**: Longer articles have more opportunity for highlights, creating a mechanical correlation
- **Highlighting habit**: Some periods involve more active annotation than others
- **Reading mode**: Skimming vs. deep reading changes highlight behavior independently of article quality

### Research on annotation noise

Research on text highlighting as a data collection method ([ScienceDirect, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0950329323000988)) shows that highlighting behavior varies significantly based on context and instruction, not just content quality. Studies on highlights as predictors of comprehension ([Winchell et al., 2020](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12901)) found that "the amount of highlighted material is unrelated to quiz performance," suggesting highlights capture attention or interest rather than deep engagement or value.

### Label noise compounds dimensionality problems

When your target variable is noisy, the effective signal-to-noise ratio drops. Rybinski et al. ([2025](https://arxiv.org/html/2502.02199v1)) demonstrate this directly: on their noisiest regression task (stock return prediction), optimal embedding dimensionality was only 8 dimensions (compressed from 768). The noisier the target, the fewer embedding dimensions are useful before overfitting dominates. Our highlight signal is likely closer to "noisy" than "clean" on this spectrum.

---

## 3. Small Dataset + High Dimensionality = Overfitting Trap

### The math is brutal

Standard embeddings from models like OpenAI's `text-embedding-3-small` are 1536-dimensional. Even with PCA reduction to 50 dimensions, the classical "one in ten" rule ([Wikipedia](https://en.wikipedia.org/wiki/One_in_ten_rule)) suggests needing at least 500 training samples for a regression with 50 predictors. With only a few hundred highlighted articles, we are right at or below this threshold.

More conservative estimates from the statistical literature suggest 15-20 events per variable ([Riley et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6519266/)), which would require 750-1000 samples for 50 PCA components. With the noisy target variable, the effective requirement is even higher.

### Empirical evidence on embedding compression

The "When Dimensionality Hurts" paper ([Rybinski et al., 2025](https://arxiv.org/abs/2502.02199)) tested regression on embeddings across three tasks with varying noise levels:

| Task | Signal-to-Noise | Optimal Dimensions (from 768) |
|------|-----------------|-------------------------------|
| Stock return prediction | Very low | 8 |
| Writing quality assessment | Medium | 32 |
| Review scoring | High | 768 (no compression needed) |

Their key finding: "compressed representations of text can yield better performance in LLM-based regression tasks" for noisy problems. This directly applies to our scenario---highlight counts are a noisy proxy for engagement, suggesting we would need aggressive dimensionality reduction, at which point we are left with so few features that simpler approaches may work just as well.

### Cross-validation theater

With small datasets, cross-validation can give a false sense of security. Leave-one-out or k-fold CV on 300 samples with 50+ features will show "good" performance that does not generalize. The variance of the CV estimate itself is high with small n, meaning the confidence interval around reported metrics is wide enough to be practically useless.

---

## 4. User Interests Change; A Static Model Degrades

### Concept drift is real

Research on concept drift in recommendation systems ([ACM, 2025](https://dl.acm.org/doi/10.1145/3707693)) documents that user interests shift over time due to life events, seasonal patterns, professional changes, and evolving intellectual interests. A model trained on six months of reading history captures *who you were*, not who you are becoming.

### The decay rate matters

For a single user's reading preferences, interest drift can be rapid. A new job, a new hobby, a geopolitical event, or discovering a new author can shift interests within weeks. Evidently AI's documentation on concept drift ([evidentlyai.com](https://www.evidentlyai.com/ml-in-production/concept-drift)) notes that "the exact speed of this decay varies and heavily depends on the modeled process" and can be "hours or days for fast-changing user preferences."

### Retraining is expensive for embeddings

Each retraining cycle requires re-embedding all articles (if the embedding model changes) or re-fitting the regression. With a small, shifting dataset, the model may never stabilize. By the time you have enough data to train reliably, the oldest data is stale.

### The LLM advantage here

An LLM-based scorer does not suffer from concept drift in the same way. It evaluates each article on its intrinsic properties (argument quality, novelty, quotability). If properly calibrated, it should be robust to interest shifts because it is not predicting *your* engagement---it is predicting *article quality*. The user's interests act as a filter upstream (what you choose to read), while the LLM scores within that filtered set.

---

## 5. The Best Predictor Might Be Author/Source, Not Content

### The metadata hypothesis

Before building a content-based model, consider that the strongest predictor of whether you will highlight an article might be:

- **Author**: You consistently highlight certain writers regardless of topic
- **Source/publication**: Articles from specific publications match your engagement style
- **Word count**: Articles of a certain length are more likely to be deeply read
- **Time in feed**: How long the article sat before you read it
- **Topic tags/categories**: Coarse topic labels from the source

### Why this matters

If author and source explain 60%+ of engagement variance, a simple lookup table outperforms any embedding model. This is testable immediately with existing data---run a feature importance analysis on metadata features before touching embeddings.

Amazon's research on metadata in recommendation systems ([AWS Blog](https://aws.amazon.com/blogs/machine-learning/selecting-the-right-metadata-to-build-high-performing-recommendation-models-with-amazon-personalize/)) shows that selecting the right metadata features is critical and often more impactful than sophisticated content analysis.

### A concrete test

Before any ML work, compute the following baselines:
1. Mean highlight count by author (top 20 authors)
2. Mean highlight count by source/publication
3. Correlation between word count and highlight count
4. Author frequency as a predictor (do you highlight more from authors you read frequently?)

If any of these simple features achieves correlation > 0.3 with engagement, metadata-based approaches deserve serious investigation before content-based ML.

---

## 6. Simple Heuristics Might Outperform ML

### The baseline problem

Many ML projects skip rigorous baseline comparisons. For article engagement prediction, plausible heuristics include:

- **Heuristic A**: `score = 0.4 * author_history + 0.3 * source_history + 0.2 * length_bucket + 0.1 * recency`
- **Heuristic B**: `score = historical_highlight_rate_for_this_author`
- **Heuristic C**: `score = 1 if (word_count > 1000 AND source in top_sources) else 0`

### Evidence that simple rules compete

Capital One's analysis of rules vs. ML ([capitalone.com](https://www.capitalone.com/tech/machine-learning/rules-vs-machine-learning/)) concludes that "if either the cost of a decision produced by your model is too high or your model requires explanatory power, the rule-based approach will be a better option." In our context, the "cost" is not financial but engineering complexity and maintenance burden.

Research from multiple domains shows that "directionally correct models with clear constraints often outperform brittle 'perfect' predictors" and that for less complicated problems, "if the rule-based system is giving performance comparable to a machine learning system, then it is advisable to avoid the use of a machine learning system" ([Towards Data Science](https://towardsdatascience.com/when-not-to-use-machine-learning-14ec62daacd7/)).

### The maintenance tax

A heuristic system requires zero retraining, no embedding infrastructure, no vector database, and no model monitoring. It is inspectable, debuggable, and adjustable in minutes. The ML system requires all of the above plus ongoing evaluation against concept drift. For a personal tool, this maintenance burden is a real cost.

---

## 7. The LLM-Based Approach Might Actually Be Superior If Calibrated Properly

### Why current LLM scoring underperforms

The current system's weak correlation with engagement may not be an indictment of the LLM approach itself, but of the specific prompt and scoring rubric. Common issues include:

- **Rubric misalignment**: The four scoring dimensions (quotability, surprise, argument quality, applicable insight) may not map to actual highlighting behavior
- **Score clustering**: LLMs tend to score in narrow ranges (40-70) rather than using the full 0-100 scale
- **Prompt sensitivity**: Small changes in prompt wording can shift scores significantly
- **Missing personalization**: The LLM does not know your specific preferences

### The calibration opportunity

The LLM-RUBRIC framework ([ACL 2024](https://aclanthology.org/2024.acl-long.745.pdf)) demonstrates that calibration networks trained on human judge responses can significantly improve LLM scoring alignment. Applied to our context: rather than replacing the LLM with embeddings, we could:

1. Collect ground truth engagement data (which we already have)
2. Train a lightweight calibration layer on top of LLM scores
3. Map the four dimension scores to actual engagement via isotonic regression or a small neural net

This hybrid approach preserves the LLM's ability to assess article quality while learning the mapping from quality dimensions to personal engagement. It requires far fewer parameters than embedding regression and is more interpretable.

### Cost-benefit analysis

| Approach | Parameters to Learn | Data Needed | Interpretability | Drift Robustness |
|----------|-------------------|-------------|-----------------|-----------------|
| Embedding regression | 50-1536 | 500-1500+ | Low | Low |
| LLM + calibration | 4-8 (dimension weights) | 50-100 | High | Medium |
| Simple heuristics | 3-5 (hand-tuned) | 0 | Very high | Medium |

The calibrated LLM approach is dramatically more sample-efficient because it learns only the mapping from 4 quality dimensions to engagement, not the mapping from raw text to engagement.

---

## 8. Cold Start: New Articles Have No Engagement History

### The irony of content recommendation

The entire purpose of article scoring is to predict engagement *before* reading. But an embedding regression model trained on past engagement is fundamentally backward-looking. For new articles from new authors on new topics, the model has no relevant training signal.

### How cold start manifests

- **New author**: No historical engagement data for this writer
- **New topic**: If you have never highlighted articles about quantum computing, the model has no basis for prediction
- **New format**: A podcast transcript vs. a longform essay vs. a newsletter---different formats may engage differently
- **Trending topics**: Breaking news or emerging topics have no historical pattern

### Content-based approaches handle cold start better

This is where LLM-based scoring has a structural advantage: it evaluates the article's intrinsic properties, not its similarity to previously-highlighted articles. A bandit algorithm ([ACM RecSys, 2019](https://dl.acm.org/doi/10.1145/3298689.3346956)) that balances exploration (showing unseen article types) with exploitation (showing similar-to-highlighted articles) would handle cold start more gracefully than pure regression.

---

## 9. Selection Bias: You Only Highlight Articles You Chose to Read

### The fundamental data gap

Your engagement data has a massive selection bias problem: you only have highlight data for articles you chose to read. Articles you skipped---which might have been highly engaging had you read them---are invisible to the model.

Research on selection bias in recommendation systems ([Chen et al., 2023](https://dl.acm.org/doi/10.1145/3564284)) identifies this as a core challenge: "users are free to choose which items to rate, so that the observed ratings are not a representative sample of all ratings." The data is Missing Not At Random (MNAR), violating a key assumption of most regression approaches.

### The feedback loop

This creates a pernicious feedback loop documented in recommendation systems research ([Correcting the User Feedback-Loop Bias, 2021](https://arxiv.org/abs/2109.06037)):

1. You read articles matching your current interests
2. You highlight passages in some of them
3. The model learns that "articles like these" are engaging
4. The model recommends more articles like these
5. You read those and highlight some
6. The model becomes increasingly confident in a narrow band of content

This loop actively suppresses discovery of new interests and creates the illusion of high model performance within an ever-narrowing content window.

### What you cannot measure

The model literally cannot learn about:
- Articles you would have loved but never saw
- Topics you would enjoy but have not yet encountered
- Authors you would consistently highlight but have never read
- Formats (interviews, data journalism, personal essays) you have not been exposed to

Any model trained on this biased sample will systematically undervalue novel content and overvalue familiar patterns.

---

## 10. Engagement Does Not Equal Value (Goodhart's Law)

### The clickbait trap

Optimizing for engagement (highlights, reading time, etc.) is not the same as optimizing for value. Research from the ACM Web Conference 2024 ([Hron et al., 2024](https://arxiv.org/abs/2401.09804)) demonstrates that "engagement-based optimization can perform worse in terms of user utility than a baseline with random recommendations." The content with the highest predicted engagement "very often has low scores by various measures of objective quality."

### Goodhart's Law applied

Goodhart's Law states: "When a measure becomes a target, it ceases to be a good measure" ([Wikipedia](https://en.wikipedia.org/wiki/Goodhart%27s_law)). If we optimize our scoring system for highlight prediction, we are implicitly optimizing for the kind of content that triggers highlighting behavior, which may differ from content that actually improves your thinking, informs your decisions, or provides lasting value.

Strong Goodhart effects occur when "over-optimizing the metric is *harmful* for the true goal" ([Arxiv, 2024](https://arxiv.org/abs/2410.09638)). A model that maximizes predicted highlights might surface:
- Articles with many quotable one-liners but shallow arguments
- Provocative hot takes that trigger highlighting for disagreement
- Well-written but ultimately forgettable listicles
- Content that confirms existing beliefs (highlighting = agreement, not learning)

### What you actually want

The original scoring system attempts to measure *article quality* (quotability, surprise, argument quality, applicable insight). These are intrinsic properties of the text. Engagement (highlights) is an extrinsic, behavioral measure confounded by context, mood, device, and time. Replacing quality assessment with engagement prediction may be optimizing the wrong objective entirely.

---

## Alternative Approaches That Might Work Better

### A. Calibrated LLM Scoring (recommended first attempt)

Instead of replacing the LLM scorer, fix it:

1. **Rubric realignment**: Adjust the four scoring dimensions based on which ones actually correlate with engagement
2. **Score calibration**: Use isotonic regression to map raw LLM scores to calibrated probabilities
3. **Personalized weighting**: Learn per-dimension weights from engagement data (only 4 parameters to fit)
4. **Prompt iteration**: Systematically test prompt variations against held-out engagement data

**Data needed**: 50-100 labeled examples.
**Engineering effort**: Low (prompt changes + lightweight calibration).
**Interpretability**: High (you can inspect each dimension's contribution).

### B. Metadata-Based Heuristics

Build a simple scoring model using only structured features:

```
score = w1 * author_avg_highlights
      + w2 * source_avg_highlights
      + w3 * word_count_bucket
      + w4 * topic_match_score
      + w5 * reading_time_estimate
```

**Data needed**: Sufficient history per author/source (20+ articles per author for stable estimates).
**Engineering effort**: Very low.
**Interpretability**: Very high.

### C. Multi-Armed Bandit for Exploration

Instead of predicting engagement, use a contextual bandit to balance showing high-confidence-good articles with exploring uncertain ones:

- **Thompson Sampling** or **LinUCB** with article metadata as context
- Naturally handles cold start by exploring new content
- Automatically adapts to interest drift
- No retraining required---updates incrementally

**Data needed**: Ongoing implicit feedback (read/skip, highlight/no-highlight).
**Engineering effort**: Medium.
**Key advantage**: Handles exploration-exploitation tradeoff that static models cannot.

### D. Two-Stage Pipeline

Combine approaches:

1. **Stage 1 (Filter)**: LLM scores articles on intrinsic quality dimensions
2. **Stage 2 (Rank)**: Lightweight model reranks based on personal engagement patterns using metadata + LLM scores as features

This keeps the LLM's ability to assess quality while adding personalization through a small, trainable component. The reranking model has only 5-10 input features (4 LLM scores + metadata), making it feasible to train on hundreds of examples.

### E. Direct Preference Learning

Instead of predicting highlight counts, frame the problem as pairwise preference:

- "Would I prefer to read article A or article B?"
- Train on pairs where one article was highlighted and the other was not
- Uses contrastive learning, which is more data-efficient than regression
- Naturally handles the ordinal nature of engagement (more vs. less, not exact counts)

---

## Conditions Under Which Embedding-Based Approaches Definitely Fail

1. **Fewer than 500 labeled examples with full-dimensional embeddings** (1536-d): The model will memorize noise, guaranteed.

2. **Noisy target variable without dimensionality reduction**: The "When Dimensionality Hurts" paper shows that noisy regression tasks need aggressive compression (to 8-32 dimensions). Without it, performance degrades below simpler baselines.

3. **Rapid concept drift without retraining infrastructure**: If your interests shift faster than your retraining cadence, the model is perpetually stale.

4. **When metadata features dominate**: If author and source explain most of the variance, embeddings add complexity without improving prediction.

5. **Single-user scenario with no collaborative signal**: Embeddings shine when you can learn from many users' patterns. With one user, you are learning an idiosyncratic mapping with minimal data.

6. **When the true signal is not in the text**: If engagement depends more on context (time of day, reading device, current mood, what else you read that day) than on content, no content-based approach will work well.

---

## Conditions Under Which Embedding-Based Approaches Are Likely to Succeed

1. **1000+ labeled examples with reliable engagement signal**: Enough data to train a regularized model on compressed embeddings (PCA to 30-50 dimensions).

2. **Stable interests over the training window**: If your reading interests are consistent over 6+ months, the training data is more representative.

3. **Clean engagement signal**: If highlights are a consistent, deliberate annotation practice (not casual or device-dependent), the target variable is less noisy.

4. **Metadata features have been exhausted**: If simple features (author, source, length) have been tried and leave significant unexplained variance, content embeddings may capture the residual signal.

5. **Combined with dimensionality reduction and regularization**: PCA to 30-50 components, Ridge or ElasticNet regression, proper cross-validation with temporal splits.

6. **As features alongside other signals, not standalone**: Embeddings as one input to a model that also includes metadata, LLM scores, and contextual features.

---

## The Honest Minimum Viable Dataset

### For embedding regression (PCA-compressed)

| Embedding Dims After PCA | Minimum Samples (10x rule) | Recommended Samples (20x rule) | With Noisy Target (30x rule) |
|--------------------------|---------------------------|-------------------------------|------------------------------|
| 10 | 100 | 200 | 300 |
| 30 | 300 | 600 | 900 |
| 50 | 500 | 1000 | 1500 |
| 100 | 1000 | 2000 | 3000 |

### For calibrated LLM scoring

| Parameters | Minimum Samples | Recommended Samples |
|------------|----------------|-------------------|
| 4 (dimension weights) | 40 | 80-100 |
| 8 (with interaction terms) | 80 | 160-200 |

### For metadata heuristics

| Features | Minimum Samples | Notes |
|----------|----------------|-------|
| 5 | 50 | But need coverage across authors/sources |

### Critical caveat

These are *minimums for fitting*. For reliable *evaluation* (estimating how well the model will perform on new articles), you need a held-out test set of at least 50-100 examples, meaning the total dataset requirement is the training minimum + 100. With temporal splits (required to avoid data leakage), the effective requirement increases further because you need enough data in each time period.

---

## Frank Assessment: When Should You NOT Use ML For This Problem?

### Do not use embedding-based ML if:

1. **You have fewer than 500 articles with engagement data.** The model will overfit, and you will not have enough held-out data to know it is overfitting.

2. **You have not tried calibrating the LLM scorer first.** The LLM approach has structural advantages (handles cold start, robust to drift, interpretable) and may only need better calibration, not replacement.

3. **You have not tested metadata baselines.** If author + source + word count achieves correlation > 0.3, start there. ML should beat a strong baseline, not a straw man.

4. **You cannot commit to ongoing evaluation.** A model without monitoring is a model that silently degrades. If you will not build drift detection and periodic retraining, do not deploy the model.

5. **The engineering cost exceeds the value.** This is a personal reading tool. If the scoring system goes from "okay" to "slightly better" but costs 40 hours of engineering, the ROI is negative. The LLM scorer can be improved with 2 hours of prompt engineering.

6. **You are trying to solve a taste problem with a content model.** If the real issue is that the scoring rubric does not match your values (e.g., you value humor but the rubric does not measure it), fix the rubric.

### Consider ML if:

1. You have 1000+ engaged articles with consistent highlighting behavior
2. Metadata and LLM calibration have been tried and plateaued
3. You are willing to build retraining and evaluation infrastructure
4. The use case justifies ongoing maintenance (e.g., expanding to multiple users)

---

## Recommended Next Steps (In Priority Order)

1. **Measure baseline correlation**: Compute Spearman correlation between current LLM scores (total and per-dimension) and engagement metrics. This is the number to beat.

2. **Test metadata heuristics**: Compute per-author and per-source average engagement. Check if these simple features outperform LLM scores.

3. **Calibrate LLM scores**: Fit isotonic regression from LLM total score to engagement. Fit per-dimension weights. This requires only 4 parameters and 50-100 examples.

4. **Only then consider embeddings**: If steps 1-3 plateau, add PCA-compressed embeddings (30 components) as additional features alongside calibrated LLM scores and metadata.

5. **If building ML, use temporal cross-validation**: Train on months 1-4, validate on month 5, test on month 6. Never use random splits for time-series engagement data.

---

## Sources

- [Weller et al. (2025) - On the Theoretical Limitations of Embedding-Based Retrieval (Google DeepMind)](https://arxiv.org/abs/2508.21038)
- [Rybinski et al. (2025) - When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks](https://arxiv.org/abs/2502.02199)
- [Hron et al. (2024) - Clickbait vs. Quality: How Engagement-Based Optimization Shapes the Content Landscape](https://arxiv.org/abs/2401.09804)
- [The Shaped Blog - The Vector Bottleneck: Limitations of Embedding-Based Retrieval](https://www.shaped.ai/blog/the-vector-bottleneck-limitations-of-embedding-based-retrieval)
- [Open Source For You (2025) - When Embeddings Miss the Point: The Quiet Crisis in Embedding Models](https://www.opensourceforu.com/2025/06/when-embeddings-miss-the-point-the-quiet-crisis-in-embedding-models/)
- [Riley et al. (2019) - Minimum Sample Size for Developing a Multivariable Prediction Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC6519266/)
- [Wikipedia - One in Ten Rule](https://en.wikipedia.org/wiki/One_in_ten_rule)
- [Chen et al. (2023) - Bias and Debias in Recommender System: A Survey](https://dl.acm.org/doi/10.1145/3564284)
- [Correcting the User Feedback-Loop Bias for Recommendation Systems (2021)](https://arxiv.org/abs/2109.06037)
- [ACM (2025) - Modelling Concept Drift in Dynamic Data Streams for Recommender Systems](https://dl.acm.org/doi/10.1145/3707693)
- [Evidently AI - Concept Drift Best Practices](https://www.evidentlyai.com/ml-in-production/concept-drift)
- [LLM-RUBRIC: A Multidimensional, Calibrated Approach (ACL 2024)](https://aclanthology.org/2024.acl-long.745.pdf)
- [Capital One - A Modern Dilemma: When to Use Rules vs. Machine Learning](https://www.capitalone.com/tech/machine-learning/rules-vs-machine-learning/)
- [Towards Data Science - When Not to Use Machine Learning](https://towardsdatascience.com/when-not-to-use-machine-learning-14ec62daacd7/)
- [Springer (2022) - When Not to Use Machine Learning: A Perspective on Potential and Limitations](https://link.springer.com/article/10.1557/s43577-022-00417-z)
- [Winchell et al. (2020) - Highlights as an Early Predictor of Student Comprehension and Interests](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12901)
- [ScienceDirect (2023) - Text Highlighting: Three Methodological Studies](https://www.sciencedirect.com/science/article/abs/pii/S0950329323000988)
- [ACM RecSys (2019) - Bandit Algorithms in Recommender Systems](https://dl.acm.org/doi/10.1145/3298689.3346956)
- [AWS Blog - Selecting the Right Metadata for Recommendation Models](https://aws.amazon.com/blogs/machine-learning/selecting-the-right-metadata-to-build-high-performing-recommendation-models-with-amazon-personalize/)
- [Goodhart's Law - Wikipedia](https://en.wikipedia.org/wiki/Goodhart%27s_law)
- [Goodhart's Law Applied to Value Alignment (2024)](https://arxiv.org/abs/2410.09638)
- [CNN - Zillow's Home-Buying Debacle Shows How Hard It Is to Use AI to Value Real Estate](https://edition.cnn.com/2021/11/09/tech/zillow-ibuying-home-zestimate)
- [Pinecone - Straightforward Guide to Dimensionality Reduction](https://www.pinecone.io/learn/dimensionality-reduction/)
- [GeeksforGeeks - The Relationship Between High Dimensionality and Overfitting](https://www.geeksforgeeks.org/the-relationship-between-high-dimensionality-and-overfitting/)
- [Mitigating Selection Bias in Recommendation Systems (MDPI, 2025)](https://www.mdpi.com/2076-3417/15/8/4170)
- [Google Developers - Collaborative Filtering Advantages and Disadvantages](https://developers.google.com/machine-learning/recommendation/collaborative/summary)
