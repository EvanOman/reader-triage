# XGBoost and Gradient Boosting on Embeddings

## Executive Summary

Gradient boosting methods (XGBoost, LightGBM, CatBoost) can work on embedding vectors, but the setup -- hundreds of samples with 768 features -- is squarely in the "curse of dimensionality" danger zone where n_features >> n_samples. Tree-based models handle high-dimensional dense features less gracefully than linear models because each split only examines one feature at a time, making it difficult to exploit the geometric structure that embeddings encode. For this specific problem (predicting article engagement from nomic-embed-text-v1.5 embeddings with ~hundreds of samples), a regularized linear model (Ridge/ElasticNet) or a simple neural network with dropout is likely to outperform gradient boosting out of the box. If gradient boosting is used, aggressive dimensionality reduction (PCA to 50-128 components or Matryoshka truncation to 256d) and heavy regularization are essential.

---

## 1. How Gradient Boosting Handles High-Dimensional Dense Embeddings

### The Fundamental Mismatch

Tree-based models make axis-aligned splits: each decision node tests whether `feature_i > threshold`. This works brilliantly for heterogeneous tabular data where individual features have distinct meanings (age, income, category). Embeddings are fundamentally different -- they encode meaning in the *direction* of the vector, not in individual coordinates. The semantic content is distributed across all 768 dimensions simultaneously.

This creates several problems:

1. **Axis-aligned splits fragment geometric structure.** A single tree split on `dim_42 > 0.15` carves the embedding space with a hyperplane perpendicular to one axis. To approximate the kind of angular/distance-based separations that embeddings naturally encode, trees need many sequential splits -- each of which consumes data and risks overfitting. ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815))

2. **Feature importance is diluted.** With 768 roughly-equally-informative embedding dimensions, no single feature stands out as a strong splitter. XGBoost's greedy split search evaluates all 768 features at each node, but each individual dimension carries only a small fraction of the total signal.

3. **Correlated features waste splits.** Embedding dimensions are heavily correlated (they arise from the same learned representation). Tree ensembles do not explicitly model correlation -- they may split on redundant dimensions instead of exploring complementary ones.

### What Actually Happens in Practice

Despite these theoretical concerns, gradient boosting on embeddings *does* produce usable models, particularly when:

- The dataset is large enough (thousands+ of samples)
- Embeddings are combined with handcrafted tabular features
- Dimensionality reduction is applied first

The Keras/TensorFlow Decision Forests tutorial demonstrates this clearly: a Gradient Boosted Tree model using Universal Sentence Encoder embeddings (512d) achieved **81.6% test accuracy** on tweet classification, compared to 54.4% without embeddings -- a 27-point improvement on ~6,800 training samples. ([Keras: Text Classification with Decision Forests](https://keras.io/examples/nlp/tweet-classification-using-tfdf/))

However, with only hundreds of samples and 768 features, the situation is much more precarious.

---

## 2. The n_samples << n_features Problem

With ~300-500 articles and 768 embedding dimensions, the feature-to-sample ratio is approximately 1.5:1 to 2.5:1. This is a well-studied regime in statistical learning:

- **Overfitting is the dominant risk.** The model has enough degrees of freedom to memorize the training set. For tree ensembles, each tree can find splits that appear predictive by chance in high-dimensional space.
- **Cross-validation variance is high.** With few samples, different CV folds can yield wildly different performance estimates.
- **Spurious feature importance.** SHAP values and feature importance metrics become unreliable because many dimensions will appear important due to random correlations in small samples. ([McElfresh et al., NeurIPS 2023](https://arxiv.org/abs/2305.02997))

### The Empirical Rule of Thumb

A rough heuristic: tree-based models start struggling when `n_features / n_samples > 0.5` without regularization. At a 1.5:1 ratio, aggressive regularization or dimensionality reduction is mandatory.

---

## 3. Dimensionality Reduction Strategies

### 3a. Matryoshka Truncation (Recommended First Step)

nomic-embed-text-v1.5 is trained with [Matryoshka Representation Learning](https://www.nomic.ai/blog/posts/nomic-embed-matryoshka), which means the first N dimensions of the embedding are explicitly trained to be useful on their own. This is a *free* dimensionality reduction that preserves more task-relevant information than PCA.

Performance by dimension for nomic-embed-text-v1.5 ([HuggingFace model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)):

| Dimensions | Relative Performance |
|------------|---------------------|
| 768 | 100% (full) |
| 512 | ~99% (outperforms text-embedding-3-small) |
| 256 | ~97% |
| 128 | ~94% |
| 64 | ~88% |

**Recommendation:** Start with 256d truncation. This gives a 3:1 reduction (768 -> 256) with minimal information loss, bringing the feature-to-sample ratio below 1:1 for datasets of 300+ articles.

```python
import numpy as np

def truncate_matryoshka(embeddings: np.ndarray, dim: int = 256) -> np.ndarray:
    """Truncate Matryoshka embeddings to target dimensionality."""
    truncated = embeddings[:, :dim]
    # Re-normalize after truncation (important for Matryoshka embeddings)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / norms
```

### 3b. PCA After Truncation

PCA can further reduce dimensions after Matryoshka truncation. The standard practice is to retain enough components to explain 90-95% of variance.

**Important caveats for PCA + tree models:**

- Some evidence suggests PCA-reduced features perform *worse* than raw features with XGBoost, because PCA rotates the coordinate system and tree models make axis-aligned splits. The principal components may not align with the decision boundaries XGBoost needs to learn. ([Nature: Empirical Evaluation of Dimensionality Reduction](https://www.nature.com/articles/s41598-025-30537-w))
- Counter-argument: when n_features >> n_samples, the variance reduction from fewer features outweighs the representational cost of rotation.
- Always compare PCA-reduced vs. Matryoshka-truncated vs. full embeddings on your validation set.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),  # retain 95% variance
])
```

### 3c. UMAP (For Clustering-Oriented Tasks)

UMAP preserves local structure better than PCA and can reduce to very low dimensions (10-30) while retaining neighborhood relationships. However:

- UMAP is non-deterministic and slower to fit
- It does not have a clean `transform()` for new data (parametric UMAP exists but adds complexity)
- It is better suited for clustering/visualization than for regression features

([BERTopic: Dimensionality Reduction](https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html))

**Recommendation:** Use UMAP only if you specifically need cluster-based features. For regression, stick with Matryoshka truncation + optional PCA.

### 3d. Supervised Dimensionality Reduction

An alternative approach: use a small neural network as a dimensionality reduction step trained on the target variable, then feed the bottleneck layer into XGBoost. This preserves task-relevant information but adds complexity.

---

## 4. Hyperparameter Tuning Guidance

### 4a. XGBoost Settings for Embedding Features

When using 768d (or reduced) embeddings with hundreds of samples, the priority is **preventing overfitting**. Recommended starting configuration:

```python
import xgboost as xgb

params = {
    # Tree complexity -- keep shallow
    'max_depth': 3,              # Start at 3, search [2, 3, 4, 5]
    'min_child_weight': 10,      # Higher than default (1) to prevent tiny leaves

    # Subsampling -- aggressive
    'subsample': 0.7,            # Row subsampling per tree
    'colsample_bytree': 0.3,     # Critical: sample only 30% of 768 features per tree
    'colsample_bylevel': 0.7,    # Additional column sampling per level

    # Regularization
    'reg_alpha': 1.0,            # L1 regularization (feature selection effect)
    'reg_lambda': 5.0,           # L2 regularization (shrinkage)

    # Learning rate
    'learning_rate': 0.05,       # Low learning rate + early stopping
    'n_estimators': 1000,        # Will be pruned by early stopping

    # Objective
    'objective': 'reg:squarederror',  # For regression
    'eval_metric': 'rmse',
}
```

**Key parameters for this setup:**

| Parameter | Why It Matters | Recommended Range |
|-----------|---------------|-------------------|
| `max_depth` | Shallow trees prevent memorization | 2-5 |
| `colsample_bytree` | Most important for high-d: forces feature diversity | 0.1-0.5 |
| `min_child_weight` | Prevents splits on tiny subsets | 5-20 |
| `reg_alpha` | L1 penalty induces sparsity in feature usage | 0.1-10 |
| `reg_lambda` | L2 penalty shrinks leaf weights | 1-10 |
| `subsample` | Row subsampling reduces overfitting | 0.5-0.8 |
| `learning_rate` | Lower = more robust, needs more trees | 0.01-0.1 |

([XGBoost Parameter Tuning Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html); [Neptune.ai: LightGBM Parameters Guide](https://neptune.ai/blog/lightgbm-parameters-guide))

### 4b. LightGBM Settings

LightGBM's histogram-based approach and leaf-wise growth can be faster, but leaf-wise growth is more prone to overfitting on small datasets. Key differences from XGBoost tuning:

```python
import lightgbm as lgb

params = {
    'num_leaves': 15,            # Lower than default 31; controls complexity
    'max_depth': 4,
    'min_data_in_leaf': 20,      # Critical for small datasets
    'feature_fraction': 0.3,     # Equivalent to colsample_bytree
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 5.0,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'objective': 'regression',
    'metric': 'rmse',
}
```

### 4c. CatBoost Native Embedding Support

CatBoost has native embedding feature support via the `embedding_features` parameter, but with significant limitations:

- It transforms embeddings using **LDA** (for classification: Gaussian likelihood values per class) or **KNN** (counts of target classes among neighbors, or average target for regression) rather than using raw coordinates
- **Embedding features do not work on CPU** -- GPU training is required
- **Regression support for native embeddings is limited** -- the documentation states embeddings work best for classification
- CatBoost recommends also passing embedding coordinates as separate numerical features for best quality

([CatBoost: Embedding Features](https://catboost.ai/docs/en/features/embeddings-features); [CatBoost: Embedding to Numeric Transformation](https://catboost.ai/docs/en/concepts/algorithm-main-stages_embedding-to-numeric))

**Recommendation:** For a regression task, CatBoost's native embedding support is not the best fit. Either pass embedding dimensions as regular numerical features (with the same regularization guidance as XGBoost) or use XGBoost/LightGBM directly.

### 4d. Hyperparameter Optimization with Optuna

Bayesian optimization is strongly recommended over grid search given the high-dimensional parameter space:

```python
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'objective': 'reg:squarederror',
    }

    model = xgb.XGBRegressor(**params, random_state=42)

    # Use repeated k-fold for stable estimates with small datasets
    from sklearn.model_selection import RepeatedKFold
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
```

([Optuna: Hyperparameter Optimization](https://optuna.org/); [XGBoosting: Bayesian Optimization with Optuna](https://xgboosting.com/bayesian-optimization-of-xgboost-hyperparameters-with-optuna/))

---

## 5. Comparison with Linear Models and Neural Networks

### 5a. Linear Models: The Strong Baseline

For embeddings with small datasets, **regularized linear models are extremely competitive** and should always be the first baseline.

**Why linear models work well on embeddings:**

- Embeddings are pre-trained to encode semantic similarity as vector geometry (cosine similarity, dot products). Linear models directly exploit this structure.
- Ridge regression / logistic regression with proper regularization handles n_features > n_samples gracefully through the L2 penalty.
- A Bank of England working paper found that penalized logistic regression on LLM embeddings "often matches or exceeds the performance of GPT-4" for text classification with as few as 60-75 training samples per class. ([Buckmann & Hill, 2025](https://www.bankofengland.co.uk/working-paper/2025/improving-text-classification-logistic-regression-llms-tens-of-shot-classifiers))
- The SentEval benchmark suite evaluates sentence embeddings by training logistic regression classifiers on top of them -- this is the *standard* evaluation protocol, precisely because it is a strong and stable baseline.

```python
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, RepeatedKFold

# Strong baseline -- try this first
ridge = Ridge(alpha=1.0)
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(ridge, X_embeddings, y_scores, cv=cv,
                         scoring='neg_root_mean_squared_error')
print(f"Ridge RMSE: {-scores.mean():.3f} +/- {scores.std():.3f}")

# ElasticNet for automatic feature selection
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
scores = cross_val_score(enet, X_embeddings, y_scores, cv=cv,
                         scoring='neg_root_mean_squared_error')
print(f"ElasticNet RMSE: {-scores.mean():.3f} +/- {scores.std():.3f}")
```

### 5b. Neural Networks: Small MLP

A 1-2 hidden layer MLP with dropout is the natural neural approach for this task:

```python
import torch
import torch.nn as nn

class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

**Advantages over gradient boosting for this task:**

- The linear layers can learn arbitrary linear combinations of embedding dimensions in a single operation (vs. many sequential tree splits)
- Dropout provides effective regularization for small datasets
- The model naturally respects the dense, correlated structure of embeddings

**Disadvantages:**

- Requires more careful training (learning rate scheduling, early stopping, batch size tuning)
- Less interpretable than tree-based models
- Noisier performance estimates on very small datasets

### 5c. Head-to-Head Comparison Summary

| Approach | Strengths for This Task | Weaknesses | Expected Performance |
|----------|------------------------|------------|---------------------|
| Ridge/ElasticNet | Simple, fast, exploits embedding geometry, regularization handles p > n | Cannot capture nonlinear patterns | Strong baseline |
| XGBoost (raw 768d) | Handles mixed features, feature importance | Overfitting risk, axis-aligned splits | Likely worse than Ridge without tuning |
| XGBoost (256d Matryoshka + tuning) | Better feature ratio, still has tree benefits | Requires careful tuning | Competitive with Ridge |
| LightGBM | Faster training, histogram-based | Same fundamental issues as XGBoost | Similar to XGBoost |
| Small MLP | Respects embedding structure, flexible | Training complexity, less interpretable | Potentially best, with careful training |
| CatBoost (native embeddings) | Built-in embedding processing | GPU-only, regression support limited | Untested for this exact setup |

The NeurIPS 2022 paper "Why do tree-based models still outperform deep learning on typical tabular data?" found that tree-based models dominate on *heterogeneous tabular features*, but embeddings are not heterogeneous tabular features -- they are dense, correlated, and geometrically structured. This distinction matters. ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815))

The follow-up NeurIPS 2023 paper found that "for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs." ([McElfresh et al., NeurIPS 2023](https://arxiv.org/abs/2305.02997))

---

## 6. Handling Zero-Inflated Engagement Data

Many articles have zero highlights, creating a zero-inflated distribution. This is a common problem in insurance claims modeling, and the solutions transfer well.

### 6a. Two-Stage Hurdle Model

The most effective approach for zero-inflated data: decompose the problem into two stages.

1. **Stage 1 -- Binary classifier:** "Will this article receive *any* highlights?" (XGBoost classifier or logistic regression)
2. **Stage 2 -- Regressor:** "Given that it received highlights, how many?" (trained only on non-zero samples)

The final prediction: `P(any) * E[count | count > 0]`

This approach consistently outperforms single-model approaches on zero-inflated data. ([Enhanced Gradient Boosting for Zero-Inflated Insurance Claims, 2023](https://arxiv.org/abs/2307.07771); [ScienceDirect: Two-Fold ML for Zero-Inflated Data, 2025](https://www.sciencedirect.com/science/article/pii/S0952197625003392))

```python
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class HurdleRegressor(BaseEstimator, RegressorMixin):
    """Two-stage hurdle model for zero-inflated regression."""

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        # Stage 1: binary classification (any engagement?)
        y_binary = (y > 0).astype(int)
        self.classifier.fit(X, y_binary)

        # Stage 2: regression on positive samples only
        pos_mask = y > 0
        if pos_mask.sum() > 10:  # need enough positive samples
            self.regressor.fit(X[pos_mask], y[pos_mask])

        return self

    def predict(self, X):
        p_positive = self.classifier.predict_proba(X)[:, 1]
        y_if_positive = self.regressor.predict(X)
        return p_positive * np.maximum(y_if_positive, 0)
```

### 6b. Tweedie Regression

XGBoost supports Tweedie regression (`objective='reg:tweedie'`), which naturally handles zero-inflated continuous data by modeling the response as a Tweedie distribution. Set `tweedie_variance_power` between 1 and 2 (1 = Poisson-like, 2 = Gamma-like). This is simpler than the hurdle model but less flexible.

### 6c. Classification as Alternative

Instead of regression, consider reframing as classification:

- **Binary:** "High-value article (score >= 60) or not?"
- **Ordinal:** "Low / Medium / High" based on score thresholds

Classification avoids the zero-inflation issue entirely and may be more practically useful for triage.

---

## 7. Real-World Benchmarks and Examples

### Keras TF-DF Tweet Classification

- **Dataset:** 6,852 train / 761 test samples, binary classification
- **Embeddings:** Universal Sentence Encoder (512d)
- **Result:** GBT + embeddings = 81.6% accuracy, AUC 0.87; GBT without embeddings = 54.4%
- **Source:** [Keras Official Tutorial](https://keras.io/examples/nlp/tweet-classification-using-tfdf/)

### Twitter Text Classification with XGBoost

- XGBoost on text embeddings for tweet classification showed that the embedding + XGBoost pipeline "bridges deep learning's ability to capture complex semantic patterns and gradient boosting's strength in structured data modeling."
- **Source:** [Ghosh, Medium](https://medium.com/@suvro.dgp/leveraging-text-embeddings-for-twitter-text-classification-with-xgboost-6a4a1a89371c)

### Biomedical Embeddings + Gradient Boosting

- XGBoost and LightGBM with BioBERT embeddings achieved test accuracies of 0.91 and 0.90 respectively with AUC up to 0.92 for biomedical text classification.
- **Source:** [Springer: Advanced ML for Breast Cancer Diagnostics, 2025](https://link.springer.com/article/10.1007/s12672-025-02111-3)

### Bank of England: Logistic Regression on Embeddings

- Penalized logistic regression on small LLM embeddings matched or exceeded GPT-4 zero-shot classification with only 60-75 labeled examples per class.
- Demonstrates that simple linear models on embeddings are a formidable baseline.
- **Source:** [Buckmann & Hill, Bank of England Working Paper, 2025](https://www.bankofengland.co.uk/working-paper/2025/improving-text-classification-logistic-regression-llms-tens-of-shot-classifiers)

### NeurIPS Large-Scale Tabular Benchmarks

- Trees outperform NNs on heterogeneous tabular data with ~10K samples (45 datasets, NeurIPS 2022)
- On 176 datasets, performance differences between GBDTs and NNs are often negligible (NeurIPS 2023)
- **Sources:** [Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815); [McElfresh et al., 2023](https://arxiv.org/abs/2305.02997)

---

## 8. Complete Pipeline Example

### Full Pipeline: Matryoshka Truncation + PCA + XGBoost with Optuna

```python
import numpy as np
import xgboost as xgb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import Ridge


def truncate_matryoshka(embeddings: np.ndarray, dim: int = 256) -> np.ndarray:
    """Truncate and re-normalize Matryoshka embeddings."""
    truncated = embeddings[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.where(norms > 0, norms, 1.0)


# --- Data preparation ---
# embeddings: np.ndarray of shape (n_articles, 768)
# y: np.ndarray of shape (n_articles,) -- engagement scores
X = truncate_matryoshka(embeddings, dim=256)

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)


# --- Baseline 1: Ridge regression (always try this first) ---
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0)),
])
ridge_scores = cross_val_score(
    ridge_pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error'
)
print(f"Ridge RMSE: {-ridge_scores.mean():.3f} (+/- {ridge_scores.std():.3f})")


# --- Baseline 2: XGBoost with default-ish settings ---
xgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64, random_state=42)),
    ('xgb', xgb.XGBRegressor(
        max_depth=3,
        n_estimators=300,
        learning_rate=0.05,
        colsample_bytree=0.5,
        subsample=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=10,
        random_state=42,
    )),
])
xgb_scores = cross_val_score(
    xgb_pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error'
)
print(f"XGBoost RMSE: {-xgb_scores.mean():.3f} (+/- {xgb_scores.std():.3f})")


# --- Optuna-tuned XGBoost ---
def objective(trial):
    n_components = trial.suggest_int('n_components', 16, 128)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('xgb', xgb.XGBRegressor(
            max_depth=trial.suggest_int('max_depth', 2, 5),
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            learning_rate=trial.suggest_float('lr', 0.01, 0.1, log=True),
            colsample_bytree=trial.suggest_float('colsample', 0.3, 0.8),
            subsample=trial.suggest_float('subsample', 0.5, 0.8),
            reg_alpha=trial.suggest_float('alpha', 0.01, 10.0, log=True),
            reg_lambda=trial.suggest_float('lambda', 0.1, 10.0, log=True),
            min_child_weight=trial.suggest_int('min_child_weight', 5, 30),
            random_state=42,
        )),
    ])
    scores = cross_val_score(
        pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error'
    )
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)
print(f"Best RMSE: {-study.best_value:.3f}")
print(f"Best params: {study.best_params}")


# --- LightGBM alternative ---
import lightgbm as lgb

lgb_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64, random_state=42)),
    ('lgb', lgb.LGBMRegressor(
        num_leaves=15,
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        feature_fraction=0.5,
        bagging_fraction=0.7,
        bagging_freq=5,
        lambda_l1=1.0,
        lambda_l2=5.0,
        min_data_in_leaf=20,
        random_state=42,
        verbosity=-1,
    )),
])
lgb_scores = cross_val_score(
    lgb_pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error'
)
print(f"LightGBM RMSE: {-lgb_scores.mean():.3f} (+/- {lgb_scores.std():.3f})")
```

### Hurdle Model for Zero-Inflated Scores

```python
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb

# Option A: Linear hurdle
hurdle_linear = HurdleRegressor(
    classifier=LogisticRegression(C=1.0, max_iter=1000),
    regressor=Ridge(alpha=1.0),
)

# Option B: XGBoost hurdle (with PCA-reduced features)
from sklearn.pipeline import make_pipeline

hurdle_xgb = HurdleRegressor(
    classifier=make_pipeline(
        StandardScaler(),
        PCA(n_components=64),
        xgb.XGBClassifier(max_depth=3, n_estimators=200,
                           colsample_bytree=0.3, reg_lambda=5.0),
    ),
    regressor=make_pipeline(
        StandardScaler(),
        PCA(n_components=64),
        xgb.XGBRegressor(max_depth=3, n_estimators=200,
                          colsample_bytree=0.3, reg_lambda=5.0),
    ),
)
```

---

## 9. Specific Recommendations for This Project

Given the constraints (nomic-embed-text-v1.5 embeddings, ~hundreds of articles, zero-inflated highlight counts, 0-100 score target):

### Do First (Quick Wins)

1. **Truncate to 256d using Matryoshka.** This is free -- just slice the first 256 dimensions and re-normalize. It cuts the feature space by 3x with negligible information loss.

2. **Start with Ridge regression as the baseline.** It will be surprisingly hard to beat on this dataset size. Use `RepeatedKFold(n_splits=5, n_repeats=5)` for stable CV estimates.

3. **Try the hurdle model approach** if predicting raw highlight counts. Separate "any engagement?" from "how much engagement?" -- this handles the zero-inflation naturally.

### Then Experiment

4. **Try XGBoost with PCA(64) + aggressive regularization.** Use the parameter ranges in Section 4a. If it does not clearly beat Ridge, stick with Ridge.

5. **Try LightGBM as an XGBoost alternative.** LightGBM's histogram-based feature binning may handle the dense features more efficiently, but expect similar performance.

6. **Consider a small MLP (768 -> 128 -> 64 -> 1 with dropout=0.3).** This may outperform both linear and tree models by learning nonlinear combinations of embedding dimensions. Use early stopping on validation loss.

### Avoid

7. **Do not use CatBoost's native embedding features for this regression task.** The native support is classification-oriented and GPU-only. If using CatBoost, pass embedding dimensions as regular numerical features.

8. **Do not use raw 768d embeddings with XGBoost without dimensionality reduction.** The overfitting risk at this sample size is too high.

9. **Do not over-invest in hyperparameter tuning before establishing baselines.** The NeurIPS 2023 finding that "light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs" suggests diminishing returns from extensive tuning.

### Architecture Decision Summary

```
Recommended evaluation order:

1. Ridge(alpha=1.0) on 256d Matryoshka        -- baseline (minutes to run)
2. ElasticNet on 256d Matryoshka               -- with L1 feature selection
3. XGBoost on PCA(64) of 256d Matryoshka       -- with Optuna (hours to run)
4. LightGBM on PCA(64) of 256d Matryoshka      -- compare with XGBoost
5. Small MLP on 256d Matryoshka                 -- if above are insufficient
6. Hurdle model (any of above as components)    -- if zero-inflation hurts
```

---

## Sources

- [Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS 2022](https://arxiv.org/abs/2207.08815)
- [McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" NeurIPS 2023](https://arxiv.org/abs/2305.02997)
- [Han et al., "A Closer Look at Deep Learning on Tabular Data," 2024](https://arxiv.org/html/2407.00956v1)
- [Keras: Text Classification using Decision Forests and Pretrained Embeddings](https://keras.io/examples/nlp/tweet-classification-using-tfdf/)
- [CatBoost: Embedding Features Documentation](https://catboost.ai/docs/en/features/embeddings-features)
- [CatBoost: Transforming Embedding Features to Numerical Features](https://catboost.ai/docs/en/concepts/algorithm-main-stages_embedding-to-numeric)
- [Enhanced Gradient Boosting for Zero-Inflated Insurance Claims (CatBoost vs XGBoost vs LightGBM), 2023](https://arxiv.org/abs/2307.07771)
- [ScienceDirect: Dealing with Zero-Inflated Data via Two-Fold ML, 2025](https://www.sciencedirect.com/science/article/pii/S0952197625003392)
- [Nomic Blog: Nomic Embed v1.5 -- Matryoshka Representation Learning](https://www.nomic.ai/blog/posts/nomic-embed-matryoshka)
- [nomic-ai/nomic-embed-text-v1.5 on HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Buckmann & Hill, "Improving Text Classification: Logistic Regression Makes Small LLMs Strong," Bank of England, 2025](https://www.bankofengland.co.uk/working-paper/2025/improving-text-classification-logistic-regression-llms-tens-of-shot-classifiers)
- [Ghosh, "Leveraging Text Embeddings for Twitter Text Classification with XGBoost"](https://medium.com/@suvro.dgp/leveraging-text-embeddings-for-twitter-text-classification-with-xgboost-6a4a1a89371c)
- [MachineLearningMastery: Combining XGBoost and Embeddings](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)
- [Springer: Advanced ML for Breast Cancer Diagnostics with Embeddings, 2025](https://link.springer.com/article/10.1007/s12672-025-02111-3)
- [XGBoost Parameter Tuning Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [Neptune.ai: LightGBM Parameters Guide](https://neptune.ai/blog/lightgbm-parameters-guide)
- [Neptune.ai: XGBoost vs LightGBM](https://neptune.ai/blog/xgboost-vs-lightgbm)
- [Optuna: Hyperparameter Optimization Framework](https://optuna.org/)
- [XGBoosting: Bayesian Optimization with Optuna](https://xgboosting.com/bayesian-optimization-of-xgboost-hyperparameters-with-optuna/)
- [Nature: Empirical Evaluation of Dimensionality Reduction and Class Balancing for Medical Text Classification, 2025](https://www.nature.com/articles/s41598-025-30537-w)
- [BERTopic: Dimensionality Reduction](https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html)
- [scikit-learn: PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Gradient Boosting Mapping for Dimensionality Reduction, 2024](https://arxiv.org/abs/2405.08486)
