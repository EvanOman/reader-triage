# Dimensionality Reduction and Embedding Quality

## Executive Summary

Using 768-dimensional embeddings from nomic-embed-text-v1.5 as features for regression on a dataset of hundreds of articles presents a classic high-dimensional small-sample problem: with ~300-500 samples and 768 features, the model is severely overparameterized, violating the common rule of thumb requiring at least 5-10 samples per dimension. Dimensionality reduction is not optional here -- it is essential to avoid overfitting. The most practical path forward is to exploit nomic-embed-text-v1.5's native Matryoshka support to truncate embeddings to 128 or 256 dimensions, then apply PCA to reduce further to 20-50 components, and feed those into a regularized regression model. Before committing to any pipeline, diagnostic analysis (intrinsic dimensionality estimation, nearest-neighbor probes, and PCA variance analysis) should confirm that the embedding space actually encodes engagement-relevant signal.

---

## 1. The Curse of Dimensionality with Small Datasets

### The Core Problem

The curse of dimensionality states that as the number of features grows, the volume of the feature space increases exponentially, making available data increasingly sparse ([Wikipedia](https://en.wikipedia.org/wiki/Curse_of_dimensionality)). For regression, this means:

- **768 dimensions with ~300 samples** puts you at a ratio of ~0.4 samples per dimension. The commonly cited rule of thumb is at least 5 training examples per dimension ([Zilliz](https://zilliz.com/glossary/curse-of-dimensionality-in-machine-learning)), meaning you would need ~3,840 samples to use 768 features directly.
- **Distance concentration**: In high-dimensional spaces, Euclidean distances between points converge to similar values. The ratio of the maximum to minimum distance approaches 1.0, making nearest-neighbor relationships unreliable ([GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/curse-of-dimensionality-in-machine-learning/)).
- **Overfitting risk**: A flexible model (neural net, kernel regression) can perfectly memorize training data in 768 dimensions with only hundreds of samples, producing near-zero training error with catastrophic generalization.

### Why Embeddings Are Partially Exempt

Embeddings from transformer models are not arbitrary 768-dimensional points -- they lie on a much lower-dimensional manifold within the ambient space ([TheLinuxCode](https://thelinuxcode.com/the-curse-of-dimensionality-in-machine-learning-and-how-i-deal-with-it-in-practice/)). The effective degrees of freedom are far fewer than 768, because:

1. The model was trained to produce semantically meaningful representations, creating correlated structure across dimensions.
2. Many dimensions encode redundant or near-zero-variance information.
3. Research suggests text embedding manifolds have intrinsic dimensionality on the order of 10-50, far below the ambient 768 ([arxiv: Measuring Intrinsic Dimension of Token Embeddings](https://arxiv.org/abs/2503.02142)).

However, "partially exempt" is not "exempt." The redundant dimensions still create noise that regression models must navigate, and regularization alone may not be sufficient.

---

## 2. PCA for Embeddings

### How PCA Works on Embedding Spaces

PCA identifies orthogonal directions of maximum variance and projects data onto the top-k such directions. For text embeddings, this means:

- The first few principal components capture broad semantic distinctions (topic, genre, length).
- Middle components capture finer-grained distinctions (writing style, argument structure).
- Later components capture noise, encoding artifacts, or information irrelevant to any downstream task.

### How Much Variance Is Needed?

The standard advice of "retain 95% of variance" is a starting point but not always optimal for supervised tasks.

**Key findings from the literature:**

- For 768-dimensional DPR embeddings, approximately 99% of variance and mutual information is captured in the first ~256 dimensions, and PCA compression can achieve a 48x reduction with less than 4% drop in retrieval effectiveness ([PCA-RAG, arxiv](https://arxiv.org/html/2504.08386v1)).
- For downstream regression, the components with highest variance are not necessarily the most predictive. A direction with low variance (explaining 0.1% of total variance) might correlate strongly with the target variable. This is the fundamental limitation of unsupervised dimensionality reduction for supervised tasks ([scikit-learn: PCR vs PLS](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)).

### Optimal Number of Components

For a dataset of hundreds of samples, practical guidance:

| Approach | Components | Rationale |
|----------|-----------|-----------|
| Conservative | 20-30 | Maintains ~5-10:1 sample-to-feature ratio with 300 samples |
| Moderate | 50-80 | Still below sample count; needs regularization |
| Aggressive | 100-150 | Requires strong regularization (Ridge, ElasticNet) |
| Variance threshold (95%) | ~100-200 | May retain too many components for small datasets |

**Recommended approach for this project:**

```python
from sklearn.decomposition import PCA
import numpy as np

# Fit PCA on all embeddings
pca = PCA(n_components=100)
X_pca = pca.fit_transform(embeddings_768d)

# Examine cumulative explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
# Look for the "elbow" and note how many components reach 90%, 95%, 99%

# Use cross-validation to select optimal k
# (not just variance threshold)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

best_score = -np.inf
for k in [10, 20, 30, 50, 80, 100]:
    X_k = X_pca[:, :k]
    scores = cross_val_score(RidgeCV(), X_k, y, cv=5, scoring='r2')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k
```

The critical insight: **select components by cross-validated downstream performance, not by explained variance** ([scikit-learn PCA docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)).

### PCA Limitations for Supervised Tasks

PCA maximizes variance, not predictive power. Two important alternatives:

1. **Partial Least Squares (PLS)**: A supervised dimensionality reduction that finds directions of maximum covariance between features and the target. PLS typically needs fewer components than PCR to achieve the same predictive performance ([Medium: Dimensionality Reduction in Supervised Framework](https://medium.com/analytics-vidhya/dimensionality-reduction-in-supervised-framework-and-partial-least-square-regression-b557a4c6c049)).

2. **Calibrated Principal Component Regression (CPCR)**: Addresses the truncation bias of standard PCR by adding a Tikhonov regularization step after PCA, recovering signal from dropped low-variance components ([arxiv](https://arxiv.org/html/2510.19020)).

---

## 3. UMAP and t-SNE: Visualization vs. Prediction

### t-SNE: Visualization Only

t-SNE is designed exclusively for visualization. It:
- Has no stable inverse mapping (cannot transform new test data).
- Does not preserve global distances -- only local neighborhood structure.
- Produces different layouts with different random seeds.
- Should never be used as a feature extraction step for regression.

Its only role in this pipeline is diagnostic: visualizing whether articles with high engagement cluster together in embedding space.

### UMAP: Potentially Useful for Prediction, with Major Caveats

UMAP preserves more global structure than t-SNE and does support transforming new data. It also has a supervised mode that incorporates target labels ([UMAP docs: Supervised](https://umap-learn.readthedocs.io/en/latest/supervised.html)).

**However, supervised UMAP has severe overfitting problems with small datasets:**

- Multiple users report that supervised UMAP "produces impressive separation on training sets but shows huge overfitting on test data" ([GitHub issue #504](https://github.com/lmcinnes/umap/issues/504)).
- When the unsupervised structure is weak, "the supervised aspect dominates the embedding," creating spurious separations ([GitHub issue #148](https://github.com/lmcinnes/umap/issues/148)).
- The UMAP author recommends reducing `target_weight` and `n_neighbors` to regularize, but this weakens the signal you are trying to capture ([GitHub issue #1116](https://github.com/lmcinnes/umap/issues/1116)).

**Practical guidance:**

| Use Case | Recommendation |
|----------|---------------|
| Exploring clusters in embedding space | Use unsupervised UMAP (2-3 components), color by engagement |
| Feature extraction for regression | Avoid supervised UMAP with <500 samples |
| Feature extraction at scale (>5000 samples) | Supervised UMAP with cross-validation may help |
| Parametric embedding for new data | Use Parametric UMAP if needed, but PCA is simpler and more stable ([arxiv: Parametric UMAP](https://arxiv.org/abs/2009.12981)) |

**For this project (hundreds of samples): use UMAP only for visualization/diagnostics, not as a feature extraction step.**

---

## 4. Random Projection and Johnson-Lindenstrauss

### The Johnson-Lindenstrauss Lemma

The JL lemma guarantees that any set of n points in high-dimensional space can be projected to k dimensions while preserving all pairwise distances within a factor of (1 +/- epsilon), where:

```
k >= O(log(n) / epsilon^2)
```

More precisely, the minimum number of dimensions required is approximately:

```
k >= 4 * log(n) / (epsilon^2 / 2 - epsilon^3 / 3)
```

([Wikipedia: JL Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma))

For practical values:

| n (samples) | epsilon | Min dimensions (k) |
|-------------|---------|-------------------|
| 300 | 0.1 | ~11,000 (higher than 768!) |
| 300 | 0.3 | ~1,200 |
| 300 | 0.5 | ~460 |
| 1000 | 0.1 | ~14,000 |
| 1000 | 0.5 | ~580 |

### When Random Projection Helps

Random projection is designed for the case where the original dimensionality is very high (10,000+) and you need a fast, data-independent reduction. For 768-dimensional embeddings with hundreds of samples:

- **Random projection is not the right tool.** The JL bound with reasonable epsilon gives dimensions comparable to or larger than 768. You would need to accept high distortion (epsilon > 0.5) for meaningful reduction.
- Random projection does not exploit the structure of the data -- it treats all directions equally. PCA will always produce lower reconstruction error for the same number of dimensions.
- The computational advantage of random projection (avoiding the O(d^2*n) cost of PCA) is negligible at d=768 with n=300 -- PCA completes in milliseconds.

**Random projection has one valid use case here:** as a baseline to compare against PCA. If PCA does not substantially outperform random projection in downstream regression, it suggests the embedding structure is not well-aligned with the prediction task.

```python
from sklearn.random_projection import GaussianRandomProjection

# Baseline comparison
rp = GaussianRandomProjection(n_components=50, random_state=42)
X_rp = rp.fit_transform(embeddings_768d)
# Compare cross-validated regression on X_rp vs X_pca
```

([scikit-learn: Random Projection](https://scikit-learn.org/stable/modules/random_projection.html))

---

## 5. Matryoshka Embeddings: The Best First Step

### What Matryoshka Representation Learning Is

Matryoshka Representation Learning (MRL), introduced by Kusupati et al. (2022), trains models so that the first m dimensions of a d-dimensional embedding form a valid, independently useful embedding at dimension m ([arxiv: MRL](https://arxiv.org/abs/2205.13147)). Unlike post-hoc PCA, the model is explicitly optimized so that earlier dimensions carry more information.

The training loss is a sum of individual losses computed at each nested dimension:

```
L_MRL = sum(C_m * L(z[1:m], y)) for m in M
```

where M is a set of target dimensions (e.g., {64, 128, 256, 512, 768}) and C_m are importance weights ([NeurIPS 2022 paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)).

### nomic-embed-text-v1.5 Matryoshka Performance

nomic-embed-text-v1.5 natively supports Matryoshka truncation at dimensions 768, 512, 256, 128, and 64. MTEB benchmark scores ([HuggingFace model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)):

| Dimensions | MTEB Score | Relative to 768d |
|-----------|-----------|-------------------|
| 768 | 62.28 | baseline |
| 512 | 61.96 | -0.5% |
| 256 | 61.04 | -2.0% |
| 128 | 59.34 | -4.7% |
| 64 | 56.10 | -9.9% |

Key observations:

- **512 dimensions lose only 0.5%** of overall MTEB performance -- essentially free dimensionality reduction.
- **256 dimensions lose 2.0%** -- still excellent, and a 3x reduction in feature count.
- **128 dimensions lose 4.7%** -- meaningful but modest; this brings you to a 2.3:1 sample-to-feature ratio with 300 samples.
- **64 dimensions** start showing real degradation but still outperform many models at their full dimensionality.

### Why Matryoshka Truncation Beats Post-Hoc PCA

Matryoshka truncation is strictly preferable to PCA truncation for the first reduction step:

1. **Training-time optimization**: The model was explicitly trained to pack maximum information into the first k dimensions. PCA can only linearly rearrange the existing 768 dimensions after the fact.
2. **No fitting required**: Matryoshka truncation is a simple slice operation (`embedding[:256]`). PCA requires fitting a covariance matrix on your data, which is noisy with small samples.
3. **No distribution shift**: The truncated Matryoshka embedding is what the model was trained to produce. PCA produces a different basis that may not align with the model's learned representations.
4. **Composable**: You can apply PCA after Matryoshka truncation for further reduction (e.g., 768 -> 256 via Matryoshka, then 256 -> 30 via PCA).

### Practical Recommendation

If your embeddings are already stored in Qdrant at 768d, truncation is trivial:

```python
# Matryoshka truncation -- just slice the first k dimensions
embeddings_256d = embeddings_768d[:, :256]
embeddings_128d = embeddings_768d[:, :128]
```

This is possible because Matryoshka-trained models front-load information into early dimensions by design. No transformation matrix is needed.

---

## 6. Diagnostic Tools: Does the Embedding Space Capture Engagement Signal?

Before building any regression model, you need to verify that the embedding space actually encodes information relevant to engagement (highlight counts). Here are concrete diagnostic procedures.

### 6.1 Nearest-Neighbor Engagement Correlation

If embeddings capture engagement-relevant signal, articles that are close in embedding space should have similar engagement levels.

```python
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import numpy as np

def nn_engagement_correlation(embeddings, engagement_scores, k=5):
    """Check if nearest neighbors have similar engagement."""
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    correlations = []
    for i in range(len(embeddings)):
        neighbor_idx = indices[i, 1:]  # exclude self
        neighbor_scores = engagement_scores[neighbor_idx]
        correlations.append(np.mean(neighbor_scores))

    rho, p = spearmanr(engagement_scores, correlations)
    return rho, p

# Run at multiple dimensions
for dim in [768, 256, 128, 64]:
    emb = embeddings_768d[:, :dim]
    rho, p = nn_engagement_correlation(emb, highlight_counts)
    print(f"dim={dim}: Spearman rho={rho:.3f}, p={p:.4f}")
```

**Interpretation:**
- rho > 0.3 with p < 0.05: Embedding space encodes meaningful engagement signal.
- rho ~ 0.0: Engagement is orthogonal to the semantic structure captured by embeddings. May need a different approach entirely.
- rho < 0: Something is wrong with data preprocessing.

([Medium: Evaluating Embedding Quality](https://medium.com/@shailsharma2001/evaluating-embedding-quality-key-benchmarks-and-metrics-367ddac3ca41))

### 6.2 UMAP Visualization Colored by Engagement

```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(embeddings_256d)

plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
            c=highlight_counts, cmap='viridis', alpha=0.7)
plt.colorbar(label='Highlight Count')
plt.title('Article Embeddings (256d Matryoshka -> UMAP 2D)')
plt.savefig('embedding_engagement_umap.png', dpi=150)
```

**What to look for:**
- Smooth gradients of color: engagement varies smoothly across the manifold (good signal).
- Random color distribution: engagement is not captured by semantic similarity.
- Distinct high-engagement clusters: some topics/styles consistently drive engagement.

### 6.3 PCA Variance Analysis

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

pca = PCA(n_components=min(len(embeddings), 200))
pca.fit(embeddings_256d)

cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumvar)+1), cumvar)
plt.axhline(y=0.90, color='r', linestyle='--', label='90%')
plt.axhline(y=0.95, color='g', linestyle='--', label='95%')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance (256d Matryoshka Embeddings)')
plt.legend()
plt.savefig('pca_variance_curve.png', dpi=150)
```

**What to look for:**
- A sharp elbow at 20-50 components suggests the data lies on a low-dimensional manifold.
- A gradual curve with no clear elbow suggests variance is spread across many dimensions (harder to reduce without loss).

### 6.4 Correlation Between PCA Components and Target

```python
from scipy.stats import pearsonr
import numpy as np

X_pca = pca.transform(embeddings_256d)
correlations = []
for i in range(X_pca.shape[1]):
    r, p = pearsonr(X_pca[:, i], highlight_counts)
    correlations.append((i, r, p))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("Top 10 most predictive PCA components:")
for comp, r, p in correlations[:10]:
    var_pct = pca.explained_variance_ratio_[comp] * 100
    print(f"  PC{comp}: r={r:.3f}, p={p:.4f}, variance={var_pct:.1f}%")
```

**Critical insight:** If the most predictive PCA components are not the highest-variance ones, standard PCR will discard signal. This is exactly when PLS regression or target-aware component selection matters.

---

## 7. Intrinsic Dimensionality Estimation

### Why It Matters

The intrinsic dimensionality (ID) of the embedding space tells you the true degrees of freedom in the data, independent of the ambient dimension. If your 768d embeddings have an intrinsic dimensionality of 25, then 25 features should suffice to capture all the structure -- reducing further loses information, retaining more adds noise.

### Methods

**TwoNN (Two-Nearest Neighbors):**

TwoNN estimates ID by analyzing the ratio of distances to each point's first and second nearest neighbors. It is robust to curvature and density variation ([arxiv: Facco et al., 2017](https://arxiv.org/abs/1803.06992)).

**MLE (Maximum Likelihood Estimation):**

The Levina-Bickel MLE estimator computes local intrinsic dimension at each point and aggregates. It is more sensitive to hyperparameters but can capture varying dimensionality across the manifold ([PMC: scikit-dimension](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8534554/)).

### Practical Implementation

```python
# Using the scikit-dimension package
# pip install scikit-dimension
import skdim

# TwoNN estimator
twonn = skdim.id.TwoNN()
id_twonn = twonn.fit_transform(embeddings_256d)
print(f"TwoNN intrinsic dimension: {id_twonn:.1f}")

# MLE estimator
mle = skdim.id.MLE()
id_mle = mle.fit_transform(embeddings_256d)
print(f"MLE intrinsic dimension: {id_mle:.1f}")

# Also try the simple approach: PCA eigenvalue analysis
pca_full = PCA().fit(embeddings_256d)
# Participation ratio (effective dimensionality)
eigenvalues = pca_full.explained_variance_
participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
print(f"PCA participation ratio: {participation_ratio:.1f}")
```

### Expected Results for Text Embeddings

- Research on LLM token embeddings finds intrinsic dimensions ranging from 10-50 depending on the model and corpus ([arxiv: Measuring Intrinsic Dimension of Token Embeddings](https://arxiv.org/abs/2503.02142)).
- Document-level embeddings (what you have) likely have lower intrinsic dimensionality than token-level, since averaging/pooling over tokens smooths out variation.
- For a corpus of hundreds of articles on varied topics, expect intrinsic dimensionality in the range of 15-40.

### How to Use the Estimate

The estimated intrinsic dimension provides an upper bound on useful PCA components:

- If ID = 25, then 25-30 PCA components should capture nearly all the structure.
- Using 50 components when ID = 25 adds noise dimensions that hurt regression.
- Using 10 components when ID = 25 loses meaningful signal.

---

## 8. Practical Pipeline: From 768d to Optimal Features

Here is the recommended end-to-end pipeline for this project, ordered by priority and expected impact.

### Step 1: Matryoshka Truncation (768d -> 256d)

```python
# Free dimensionality reduction with <2% information loss
embeddings = embeddings_768d[:, :256]
```

**Rationale:** This is the single most impactful step. It uses the model's own training to perform dimensionality reduction without any data-dependent fitting, eliminating any risk of overfitting in the reduction step itself.

### Step 2: Diagnostic Analysis

Before proceeding, run the diagnostics from Section 6:

1. Nearest-neighbor engagement correlation (to verify signal exists).
2. Intrinsic dimensionality estimation (to guide PCA component count).
3. PCA variance curve (to identify the elbow).
4. Component-target correlation analysis (to check if variance aligns with signal).

**Decision point:** If nearest-neighbor engagement correlation is near zero, the embedding space does not capture engagement-relevant information. In that case:
- Consider alternative features (article metadata, source, topic categories).
- Consider fine-tuning or a different embedding model.
- Consider that engagement may be inherently unpredictable from content alone.

### Step 3: PCA Reduction (256d -> 20-50d)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize before PCA (important for regression, optional for cosine-similar embeddings)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Fit PCA -- use n_components based on diagnostics
pca = PCA(n_components=50)
X_pca = pca.fit_transform(embeddings_scaled)
```

**Important:** PCA should be fit only on training data in a cross-validation loop to avoid data leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('ridge', Ridge(alpha=1.0)),
])

scores = cross_val_score(pipe, embeddings, highlight_counts, cv=5, scoring='r2')
print(f"R^2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Step 4: Compare Against PLS Regression

```python
from sklearn.cross_decomposition import PLSRegression

# PLS finds components that maximize covariance with target
pls_scores = []
for n in [5, 10, 15, 20, 30]:
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, embeddings, highlight_counts, cv=5, scoring='r2')
    pls_scores.append((n, scores.mean()))
    print(f"PLS n_components={n}: R^2={scores.mean():.3f}")
```

PLS is likely to outperform PCR here because it explicitly seeks components that predict the target, not just components that explain variance in the features ([scikit-learn: PCR vs PLS](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)).

### Step 5: Model Selection with Hyperparameter Sweep

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Compare multiple reduction strategies and models
param_grid = {
    'pca__n_components': [10, 20, 30, 50],
    'model': [Ridge(alpha=1.0), Ridge(alpha=10.0), ElasticNet(alpha=0.1)],
}

# Or use a simpler approach:
results = {}
for dim_trunc in [64, 128, 256]:
    emb = embeddings_768d[:, :dim_trunc]
    for n_pca in [10, 20, 30, 50, None]:
        pipe_steps = [('scaler', StandardScaler())]
        if n_pca and n_pca < dim_trunc:
            pipe_steps.append(('pca', PCA(n_components=n_pca)))
        pipe_steps.append(('ridge', Ridge(alpha=1.0)))
        pipe = Pipeline(pipe_steps)
        scores = cross_val_score(pipe, emb, highlight_counts, cv=5, scoring='r2')
        results[(dim_trunc, n_pca)] = scores.mean()
        print(f"Matryoshka={dim_trunc}, PCA={n_pca}: R^2={scores.mean():.3f}")
```

### Step 6: Validate and Interpret

Once you have a best configuration:

1. **Learning curve analysis**: Plot training vs. validation R^2 as a function of sample size. If they are far apart, you are still overfitting.
2. **Permutation test**: Shuffle the target variable and re-run regression. The real R^2 should be significantly higher than permuted R^2.
3. **Feature importance**: Examine which PCA components contribute most to predictions. Map back to original embedding dimensions to understand what semantic axes drive engagement.

---

## 9. Summary of Recommendations

| Method | Recommended? | When to Use | Notes |
|--------|-------------|-------------|-------|
| Matryoshka truncation | **Yes, always** | First step, 768 -> 128 or 256 | Free, no fitting needed |
| PCA | **Yes** | After Matryoshka, 256 -> 20-50 | Fit only on training folds |
| PLS regression | **Yes, compare** | Alternative to PCA + regression | Supervised; often better |
| Ridge/ElasticNet | **Yes, always** | Regression model | Regularization essential |
| Random projection | No | Only as diagnostic baseline | JL bounds unfavorable at n=300 |
| UMAP (unsupervised) | Diagnostic only | Visualizing engagement patterns | Do not use as feature step |
| UMAP (supervised) | No | Overfits with <500 samples | Major overfitting risk |
| t-SNE | Diagnostic only | Visualization | No transform for new data |
| Intrinsic dimension | **Yes** | Guides component count | Run once, early in pipeline |

### Expected Pipeline for This Project

```
nomic-embed-text-v1.5 (768d)
  |
  v  Matryoshka truncation (slice first 256 dims)
Embeddings (256d)
  |
  v  StandardScaler + PCA (fit on training folds only)
Features (20-50d, tuned by cross-validation)
  |
  v  Ridge or PLS regression
Predicted engagement (highlight count)
```

---

## Sources

- [Wikipedia: Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
- [Zilliz: Curse of Dimensionality in Machine Learning](https://zilliz.com/glossary/curse-of-dimensionality-in-machine-learning)
- [GeeksforGeeks: Curse of Dimensionality in Machine Learning](https://www.geeksforgeeks.org/machine-learning/curse-of-dimensionality-in-machine-learning/)
- [TheLinuxCode: The Curse of Dimensionality in Machine Learning](https://thelinuxcode.com/the-curse-of-dimensionality-in-machine-learning-and-how-i-deal-with-it-in-practice/)
- [scikit-learn: PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: Random Projection](https://scikit-learn.org/stable/modules/random_projection.html)
- [scikit-learn: PCR vs PLS Regression](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)
- [PCA-RAG: Principal Component Analysis for Efficient RAG (arxiv)](https://arxiv.org/html/2504.08386v1)
- [Effective Dimensionality Reduction for Word Embeddings (ACL)](https://aclanthology.org/W19-4328.pdf)
- [Apple Research: Single Training Dimension Selection for Word Embedding with PCA](https://machinelearning.apple.com/research/single-training-dimension-selection-for-word-embedding-with-pca)
- [Wikipedia: Johnson-Lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
- [Kusupati et al., 2022: Matryoshka Representation Learning (NeurIPS)](https://arxiv.org/abs/2205.13147)
- [NeurIPS 2022 Paper PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)
- [HuggingFace: nomic-embed-text-v1.5 Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Nomic Blog: Nomic Embed Matryoshka](https://www.nomic.ai/news/nomic-embed-matryoshka)
- [HuggingFace Blog: Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka)
- [Sentence Transformers: Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html)
- [UMAP Documentation: Supervised Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/supervised.html)
- [Parametric UMAP: Learning Embeddings (arxiv)](https://arxiv.org/abs/2009.12981)
- [GitHub: UMAP Overfitting Issue #504](https://github.com/lmcinnes/umap/issues/504)
- [GitHub: UMAP Overfitting Issue #148](https://github.com/lmcinnes/umap/issues/148)
- [GitHub: UMAP Regularization Issue #1116](https://github.com/lmcinnes/umap/issues/1116)
- [Tang et al., 2024: Understanding LLM Embeddings for Regression (arxiv)](https://arxiv.org/abs/2411.14708)
- [Facco et al., 2017: Intrinsic Dimension Estimation via TwoNN (arxiv)](https://arxiv.org/abs/1803.06992)
- [Measuring Intrinsic Dimension of Token Embeddings (arxiv)](https://arxiv.org/abs/2503.02142)
- [scikit-dimension: Python Package for Intrinsic Dimension Estimation (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8534554/)
- [Evaluating Embedding Quality: Key Benchmarks and Metrics (Medium)](https://medium.com/@shailsharma2001/evaluating-embedding-quality-key-benchmarks-and-metrics-367ddac3ca41)
- [Unsupervised Embedding Quality Evaluation (ICML 2023)](https://arxiv.org/abs/2305.16562)
- [Calibrated Principal Component Regression (arxiv)](https://arxiv.org/html/2510.19020)
- [Benchmarking Random Projections and PCA (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11838541/)
- [Dimensionality Reduction in Supervised Framework and PLS (Medium)](https://medium.com/analytics-vidhya/dimensionality-reduction-in-supervised-framework-and-partial-least-square-regression-b557a4c6c049)
- [Pinecone: Straightforward Guide to Dimensionality Reduction](https://www.pinecone.io/learn/dimensionality-reduction/)
