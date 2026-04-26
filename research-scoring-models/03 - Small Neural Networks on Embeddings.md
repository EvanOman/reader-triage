# Small Neural Networks on Embeddings for Engagement Regression

## Executive Summary

When predicting continuous engagement scores from frozen 768-dimensional text embeddings with only hundreds of training examples, the architecture choice matters far less than the regularization strategy. Ridge regression is the strongest baseline and is surprisingly hard to beat at this scale; a one-hidden-layer MLP (768 -> 128 -> 1) with aggressive dropout (0.4-0.5) and weight decay (1e-2 to 1e-1) represents the practical ceiling of complexity. Multi-task learning---jointly predicting a binary "worth reading" flag and a continuous score---can improve performance by acting as an implicit regularizer, and is particularly well-suited to our zero-inflated highlight distribution. Deeper or attention-based architectures are not justified until the dataset reaches thousands of examples.

---

## 1. Baselines: Linear and Ridge Regression

Before reaching for neural networks, establish strong linear baselines. With 768 features and a few hundred samples, the regime is firmly p >> n-adjacent, which is where regularized linear models shine.

### Why Ridge Regression Is Hard to Beat

Ridge regression (L2-penalized least squares) is the canonical approach when dimensionality approaches or exceeds sample count. It shrinks coefficient magnitudes toward zero, preventing the model from overfitting to noise in high-dimensional embedding space. With cross-validated alpha selection, Ridge can capture the linear signal in embeddings without any risk of the optimization instabilities that plague neural networks on small data ([scikit-learn Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)).

Key advantages at our scale:
- **No architecture decisions**: no hidden sizes, activations, or dropout rates to tune.
- **Closed-form solution**: `RidgeCV` with leave-one-out cross-validation is essentially free ([scikit-learn RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)).
- **Stable gradients**: no vanishing/exploding gradient problems.
- **Interpretable coefficients**: can inspect which embedding dimensions drive predictions.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import numpy as np

# X: (n_samples, 768) embedding matrix
# y: (n_samples,) engagement scores 0-100

alphas = np.logspace(-2, 4, 50)
ridge = RidgeCV(alphas=alphas, cv=None)  # None = efficient LOO-CV
ridge.fit(X_train, y_train)

print(f"Best alpha: {ridge.alpha_:.2f}")
print(f"LOO R^2: {ridge.score(X_train, y_train):.3f}")

# Cross-validated evaluation
scores = cross_val_score(ridge, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"5-fold RMSE: {np.sqrt(-scores.mean()):.2f}")
```

### Lasso and ElasticNet

Lasso (L1 penalty) performs implicit feature selection, zeroing out irrelevant embedding dimensions. ElasticNet combines L1 and L2. These are worth testing as baselines but typically perform similarly to Ridge when the features are dense embeddings rather than sparse features.

```python
from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=alphas, cv=5)
enet.fit(X_train, y_train)
print(f"Non-zero coefficients: {np.sum(enet.coef_ != 0)} / 768")
```

---

## 2. MLP Architecture Patterns for Embedding Regression

### The Core Design Principle: Aggressive Bottlenecking

With 768 input dimensions and hundreds of samples, the network must compress information quickly. The guiding principle is that each layer should dramatically reduce dimensionality, funneling the embedding through a tight bottleneck before producing the scalar output.

### Recommended Architecture: One Hidden Layer

For ~hundreds of samples, a single hidden layer is the right complexity level ([Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data," NeurIPS 2021](https://arxiv.org/pdf/2106.11959)). Research on tabular deep learning consistently shows that MLPs outperform transformer-based architectures when data is scarce.

```
Input (768) -> Linear(768, 128) -> LayerNorm -> GELU -> Dropout(0.4) -> Linear(128, 1)
```

**Why this configuration:**
- **128 hidden units**: A 6:1 compression ratio. Wide enough to capture non-linear interactions, narrow enough to prevent memorization. With ~100 weights per training example, this is at the edge of what the data can support.
- **LayerNorm over BatchNorm**: Layer normalization computes statistics per-sample rather than per-batch, making it batch-size-independent ([PyTorch LayerNorm docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)). This matters because small datasets mean small batches.
- **GELU activation**: Smoother gradients than ReLU, no dead neuron problem. GELU is the default activation in BERT and modern transformer architectures for good reason---it weights inputs by their percentile rather than gating by sign ([Hendrycks & Gimpel, 2016](https://arxiv.org/pdf/2305.12073)). For small networks, the difference from ReLU is marginal, but GELU is the modern default.
- **Dropout(0.4)**: Aggressively regularize the hidden representation. Higher than the typical 0.1-0.2 used in large-data regimes.

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class EmbeddingRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

### Two Hidden Layers (Use With Caution)

Only consider two hidden layers if you have 500+ samples and cross-validation shows improvement over the single-layer variant:

```
Input (768) -> Linear(768, 256) -> LN -> GELU -> Dropout(0.3)
            -> Linear(256, 64)  -> LN -> GELU -> Dropout(0.3)
            -> Linear(64, 1)
```

The additional layer adds ~16K parameters (256*64). With 500 samples, that is roughly one parameter per 8 samples in the second layer---marginal but potentially workable with strong regularization.

### scikit-learn MLPRegressor (Quick Prototyping)

For fast experimentation without writing a training loop:

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128,),
        activation="relu",  # scikit-learn doesn't support GELU
        alpha=0.01,          # L2 regularization (weight decay)
        learning_rate="adaptive",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=1000,
        random_state=42,
    )),
])
mlp.fit(X_train, y_train)
```

Note: scikit-learn's `MLPRegressor` uses L2 regularization via the `alpha` parameter ([scikit-learn MLPRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)), which is conceptually equivalent to weight decay in PyTorch. It does not support dropout or layer normalization, making it less flexible but useful for quick baselines.

---

## 3. Regularization Strategies for Small Datasets

With hundreds of samples and 768-dimensional inputs, overfitting is the primary failure mode. Every design decision should be viewed through the lens of regularization.

### 3.1 Dropout

Dropout randomly zeros hidden units during training, forcing the network to learn redundant representations ([Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR 2014](http://d2l.ai/chapter_multilayer-perceptrons/dropout.html)).

**Recommended rates by dataset size:**

| Training Samples | Dropout Rate | Rationale |
|:-----------------|:-------------|:----------|
| < 200            | 0.5          | Maximum regularization; barely more expressive than linear |
| 200-500          | 0.3-0.4      | Standard heavy regularization |
| 500-1000         | 0.2-0.3      | Moderate regularization |
| > 1000           | 0.1-0.2      | Light regularization |

**Placement**: Apply dropout after the activation function in each hidden layer. Do not apply dropout to the output layer.

### 3.2 Weight Decay (L2 Regularization)

Weight decay penalizes large weights, shrinking the effective capacity of the network. In the small-data regime, use aggressive weight decay: 1e-2 to 1e-1 ([Loshchilov & Hutter, "Decoupled Weight Decay Regularization"](https://link.springer.com/chapter/10.1007/978-3-642-04921-7_4)).

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.05,  # Aggressive for small data
)
```

**AdamW vs Adam**: Use AdamW (decoupled weight decay) rather than Adam with L2 regularization. In Adam, the L2 penalty interacts with the adaptive learning rate in unintended ways; AdamW fixes this ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)).

### 3.3 Early Stopping

Monitor validation loss and stop training when it plateaus or increases ([MachineLearningMastery: Early Stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/)).

**Configuration for small datasets:**
- **Patience**: 15-30 epochs. Too low (5) risks premature stopping; too high (100) wastes time.
- **Validation split**: 15-20% of training data. With 300 samples, that is ~50 validation samples---marginal but workable.
- **Restore best weights**: Always checkpoint and restore the model from the epoch with lowest validation loss.

```python
best_val_loss = float("inf")
patience_counter = 0
patience = 20

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

model.load_state_dict(torch.load("best_model.pt"))
```

### 3.4 Data Augmentation in Embedding Space

Unlike images or raw text, embeddings don't have natural augmentation semantics. However, two techniques are effective:

**Gaussian noise injection**: Add small random perturbations to embeddings during training. This acts as a smoothing regularizer ([Xiao et al., "An integrated approach based on Gaussian noises-based data augmentation"](https://www.sciencedirect.com/science/article/abs/pii/S0022169421005576)).

```python
class NoisyEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, targets, noise_std=0.05, training=True):
        self.embeddings = embeddings
        self.targets = targets
        self.noise_std = noise_std
        self.training = training

    def __getitem__(self, idx):
        x = self.embeddings[idx]
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.targets[idx]

    def __len__(self):
        return len(self.embeddings)
```

**Mixup in embedding space**: Linearly interpolate pairs of embeddings and their labels ([Zhang et al., "mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/2405.17938)). This is particularly natural for embeddings, since the embedding space is designed such that linear interpolation is semantically meaningful.

```python
def mixup_batch(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * y + (1 - lam) * y[idx]
    return x_mixed, y_mixed
```

For regression, C-Mixup ([Yao et al., "C-Mixup: Improving Generalization in Regression"](https://arxiv.org/abs/2405.17938)) is a variant that selects pairs based on label proximity, preventing interpolation between distant targets (e.g., mixing a 0-score article with a 90-score article).

### 3.5 Dimensionality Reduction as Pre-processing

Reducing the 768 dimensions to 64-128 via PCA before feeding into the MLP can help, especially with very small datasets (< 200 samples). This reduces the effective parameter count of the first layer by 6-12x ([Neptune.ai: Dimensionality Reduction](https://neptune.ai/blog/dimensionality-reduction)).

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Reduce 768 -> 128, then feed into MLP
pca = PCA(n_components=128, random_state=42)
X_reduced = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

In our case, nomic-embed-text-v1.5 embeddings are L2-normalized and produced by a model trained with Matryoshka representation learning, meaning the first dimensions already carry disproportionate information. PCA may capture 95%+ of variance in 128 dimensions.

### 3.6 Spectral Normalization

An advanced technique that constrains the spectral norm (largest singular value) of weight matrices. Research from MIT shows that spectral norm scaling improves training stability and acts as an accuracy predictor for test accuracy, preventing a single outer product from dominating the learned transformation ([MIT Deep Learning Blog: Spectral Normalization in MLPs](https://deep-learning-mit.github.io/staging/blog/2023/WeightDecaySpecNormEffects/)). Worth trying if dropout + weight decay is insufficient.

```python
from torch.nn.utils.parametrizations import spectral_norm

model = nn.Sequential(
    spectral_norm(nn.Linear(768, 128)),
    nn.GELU(),
    nn.Dropout(0.3),
    spectral_norm(nn.Linear(128, 1)),
)
```

### Summary: Regularization Stack

For our regime (~hundreds of samples, 768d embeddings), stack these techniques:

1. **Always**: Weight decay (0.01-0.1), early stopping (patience 20)
2. **Default**: Dropout (0.3-0.5), Gaussian noise augmentation (std=0.05)
3. **If overfitting persists**: PCA pre-reduction (768 -> 128), Mixup (alpha=0.2)
4. **Advanced**: Spectral normalization on weight matrices

---

## 4. Multi-Task Learning: Binary Engagement + Continuous Score

### Why Multi-Task Learning Fits This Problem

Our dataset has a zero-inflated distribution: many articles receive zero highlights, while a smaller subset receives varying positive counts. This is a classic hurdle model scenario ([Wikipedia: Hurdle model](https://en.wikipedia.org/wiki/Hurdle_model)). Multi-task learning with a shared backbone naturally decomposes this into:

1. **Binary task**: "Will this article receive any highlights?" (classification)
2. **Continuous task**: "How many highlights will it receive?" (regression, conditional on engagement)

This decomposition serves dual purposes:
- **Better predictions**: The binary head provides a probability gate for the regression head.
- **Implicit regularization**: The shared representation must support both tasks, preventing overfitting to either ([Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018](https://arxiv.org/abs/1705.07115)).

### Architecture

```
                    +--> Binary Head: Linear(64, 1) -> Sigmoid
Input (768) -->     |
  Shared Backbone --+
  (768->128->64)    |
                    +--> Regression Head: Linear(64, 1) -> ReLU (or clamp)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskEngagementModel(nn.Module):
    """
    Predicts both binary engagement (any highlights?) and
    continuous engagement score (0-100) from a frozen embedding.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Task-specific heads
        self.binary_head = nn.Linear(bottleneck_dim, 1)
        self.regression_head = nn.Linear(bottleneck_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        binary_logit = self.binary_head(features).squeeze(-1)
        score = self.regression_head(features).squeeze(-1)
        # Clamp regression output to valid range
        score = torch.clamp(score, 0.0, 100.0)
        return binary_logit, score


class UncertaintyWeightedLoss(nn.Module):
    """
    Learns task-specific loss weights via homoscedastic uncertainty.

    From Kendall et al. (2018): "Multi-Task Learning Using
    Uncertainty to Weigh Losses for Scene Geometry and Semantics."

    The loss for two tasks becomes:
        L = (1/2σ₁²) * L₁ + (1/2σ₂²) * L₂ + log(σ₁) + log(σ₂)

    where σ are learnable parameters.
    """

    def __init__(self):
        super().__init__()
        # log(σ²) for numerical stability; initialized to 0 => σ = 1
        self.log_var_binary = nn.Parameter(torch.zeros(1))
        self.log_var_regression = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        binary_logit: torch.Tensor,
        binary_target: torch.Tensor,
        score_pred: torch.Tensor,
        score_target: torch.Tensor,
    ) -> torch.Tensor:
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            binary_logit, binary_target.float()
        )
        # MSE loss for regression
        mse_loss = F.mse_loss(score_pred, score_target)

        # Uncertainty-weighted combination
        precision_binary = torch.exp(-self.log_var_binary)
        precision_regression = torch.exp(-self.log_var_regression)

        loss = (
            precision_binary * bce_loss
            + self.log_var_binary
            + precision_regression * mse_loss
            + self.log_var_regression
        )
        return loss.squeeze()
```

### Training Loop

```python
def train_multitask(
    model: MultiTaskEngagementModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
):
    criterion = UncertaintyWeightedLoss()
    # Include uncertainty parameters in optimization
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for embeddings, binary_labels, scores in train_loader:
            optimizer.zero_grad()
            binary_logit, score_pred = model(embeddings)
            loss = criterion(binary_logit, binary_labels, score_pred, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for embeddings, binary_labels, scores in val_loader:
                binary_logit, score_pred = model(embeddings)
                val_loss = criterion(
                    binary_logit, binary_labels, score_pred, scores
                )
                val_losses.append(val_loss.item())

        mean_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss - 1e-4:
            best_val_loss = mean_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_multitask.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load("best_multitask.pt"))
    return model
```

### Inference: Combining the Two Heads

At inference time, combine the binary gate and regression score:

```python
def predict_engagement(
    model: MultiTaskEngagementModel, embedding: torch.Tensor
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        binary_logit, score = model(embedding.unsqueeze(0))
        engagement_prob = torch.sigmoid(binary_logit).item()
        raw_score = score.item()

    return {
        "engagement_probability": engagement_prob,
        "raw_score": raw_score,
        # Gated score: zero out predictions for low-probability articles
        "gated_score": raw_score * engagement_prob,
        "worth_reading": engagement_prob > 0.5,
    }
```

The `gated_score` (probability * raw_score) naturally handles zero-inflation: articles the model thinks won't be engaged with get scores pushed toward zero, even if the regression head predicts a moderate value.

### Alternative: Explicit Hurdle Model

Rather than uncertainty-weighted multi-task learning, you can implement a classical two-stage hurdle model with separate models ([ScienceDirect: Two-fold machine learning for zero-inflated data](https://www.sciencedirect.com/science/article/pii/S0952197625003392)):

```python
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
import numpy as np

# Stage 1: Binary classifier (any highlights?)
binary_labels = (y > 0).astype(int)
classifier = LogisticRegressionCV(cv=5, max_iter=1000)
classifier.fit(X_train, binary_labels_train)

# Stage 2: Regression on positive examples only
positive_mask = y_train > 0
regressor = RidgeCV(alphas=np.logspace(-2, 4, 50))
regressor.fit(X_train[positive_mask], y_train[positive_mask])

# Prediction: gate regression by classification probability
def predict_hurdle(x):
    prob = classifier.predict_proba(x.reshape(1, -1))[0, 1]
    score = regressor.predict(x.reshape(1, -1))[0]
    return prob * max(score, 0)
```

---

## 5. When Neural Networks Beat Simpler Models (and When They Don't)

### The Evidence

Research consistently shows that with small tabular datasets, simple models are competitive or superior:

- **Gorishniy et al. (2021)** found that a well-tuned ResNet-like MLP can rival transformers on tabular data, but that MLPs outperform transformer-based architectures when data is scarce ([Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959)).
- **Linear probes** (logistic regression / ridge on frozen embeddings) are "surprisingly strong" baselines, particularly in few-shot settings ([Huang et al., "LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP," CVPR 2024](https://arxiv.org/abs/2404.02285)).
- A comparison between dropout and weight decay for regularizing neural networks showed that with very limited data, both techniques narrow the gap with linear models but rarely create a decisive advantage ([University of Arkansas study](https://scholarworks.uark.edu/cgi/viewcontent.cgi?article=1028&context=csceuht)).

### Decision Framework

| Dataset Size | Non-linearity in Data | Recommended Approach |
|:-------------|:----------------------|:---------------------|
| < 100 samples | Any | Ridge regression. Neural networks will overfit. |
| 100-300 samples | Low | Ridge regression. MLP might match it but won't beat it. |
| 100-300 samples | High | 1-layer MLP (128 hidden) with heavy regularization. Cross-validate against Ridge. |
| 300-1000 samples | Any | 1-layer MLP is worth trying. Multi-task learning can help. |
| > 1000 samples | Any | 1-2 layer MLP likely wins. Consider two-layer architectures. |

### How to Detect Non-linearity

If Ridge regression achieves R^2 > 0.3 on your embeddings, there is meaningful linear signal. An MLP will only improve on this if there are non-linear interactions between embedding dimensions. Test this empirically:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
import numpy as np

ridge_scores = cross_val_score(
    RidgeCV(), X, y, cv=5, scoring="neg_mean_squared_error"
)
mlp_scores = cross_val_score(
    MLPRegressor(
        hidden_layer_sizes=(128,), alpha=0.01,
        early_stopping=True, max_iter=500,
    ),
    X, y, cv=5, scoring="neg_mean_squared_error",
)

ridge_rmse = np.sqrt(-ridge_scores.mean())
mlp_rmse = np.sqrt(-mlp_scores.mean())
print(f"Ridge RMSE: {ridge_rmse:.2f}")
print(f"MLP RMSE:   {mlp_rmse:.2f}")
print(f"MLP improvement: {(ridge_rmse - mlp_rmse) / ridge_rmse:.1%}")
```

If the MLP does not improve RMSE by at least 5-10%, use Ridge. The simplicity, stability, and interpretability are worth more than a marginal accuracy gain.

---

## 6. Attention-Based Architectures: Mostly Overkill

### Self-Attention on a Single Embedding Vector

A single 768d embedding is a vector, not a sequence. Self-attention operates on sequences of tokens. To apply attention, you would need to either:

1. **Chunk the embedding** into pseudo-tokens (e.g., 12 tokens of 64 dimensions each) and apply a transformer layer.
2. **Use cross-attention** with learned query vectors to extract task-relevant features.

Both approaches are architecturally interesting but add significant parameters and complexity that are not justified with hundreds of samples.

### When Attention Helps

Attention-based heads become relevant when:
- You have **multiple embeddings per article** (e.g., paragraph-level embeddings) and need to aggregate them.
- You are doing **multi-article comparison** (e.g., predicting relative engagement).
- Your dataset exceeds **thousands of samples**.

For a single 768d vector with hundreds of examples, the attention mechanism has more parameters than the data can support. Stick with MLPs.

### Lightweight Cross-Attention (For Reference)

If you eventually scale to paragraph-level embeddings (a sequence of vectors per article), here is a minimal cross-attention head:

```python
class CrossAttentionHead(nn.Module):
    """
    Attends over a sequence of paragraph embeddings using
    a learned query to produce a single article-level representation.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.3, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, paragraph_embeddings: torch.Tensor) -> torch.Tensor:
        # paragraph_embeddings: (batch, seq_len, 768)
        batch_size = paragraph_embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)
        attended, _ = self.attn(query, paragraph_embeddings, paragraph_embeddings)
        return self.head(attended.squeeze(1)).squeeze(-1)
```

---

## 7. Evaluation Strategy for Small Datasets

### K-Fold Cross-Validation

With hundreds of samples, a single train/test split is unreliable. Use 5-fold or 10-fold cross-validation for all model comparisons ([scikit-learn cross-validation docs](https://scikit-learn.org/stable/modules/cross_validation.html)).

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# For scikit-learn models
scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")

# For PyTorch models, implement manually
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    # ... train and evaluate
```

### Stratified Folds for Zero-Inflated Data

Since many articles have zero highlights, ensure each fold has a representative distribution. Bin the target variable and use stratified splitting:

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Bin targets for stratification
bins = np.digitize(y, bins=[0.5, 30, 60])  # 0, low, medium, high
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, bins)):
    # Guaranteed similar distribution of zeros in each fold
    pass
```

### Nested Cross-Validation for Hyperparameter Tuning

To avoid optimistic bias from tuning hyperparameters on the same folds used for evaluation, use nested CV ([Neptune.ai: Cross-Validation Done Right](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)):

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {"alpha": np.logspace(-2, 4, 20)}
ridge_cv = GridSearchCV(Ridge(), param_grid, cv=inner_cv, scoring="neg_mean_squared_error")

# Outer CV gives unbiased performance estimate
nested_scores = cross_val_score(
    ridge_cv, X, y, cv=outer_cv, scoring="neg_mean_squared_error"
)
print(f"Nested CV RMSE: {np.sqrt(-nested_scores.mean()):.2f}")
```

---

## 8. Specific Recommendations for This Project

Given the constraints---nomic-embed-text-v1.5 (768d), ~hundreds of articles, zero-inflated highlight counts scored 0-100---here is the recommended implementation plan:

### Phase 1: Establish Baselines (Do This First)

1. **Ridge regression** with `RidgeCV` (LOO cross-validation). This takes one line of code and gives you a number to beat.
2. **Hurdle model**: `LogisticRegressionCV` (any highlights?) + `RidgeCV` (how many, given positive). Combine with probability gating.
3. Evaluate both with 5-fold cross-validated RMSE and MAE.

### Phase 2: MLP Experiments (Only If Baselines Are Insufficient)

4. **Single-layer MLP** (768 -> 128 -> 1) with LayerNorm, GELU, Dropout(0.4), weight decay 0.05. Compare 5-fold CV against Ridge.
5. **Multi-task MLP** with binary + regression heads and uncertainty-weighted loss. Compare against single-task MLP and hurdle model.
6. Test **Gaussian noise augmentation** (std=0.05) and **Mixup** (alpha=0.2).

### Phase 3: Refinements (If Data Grows)

7. Once the dataset reaches 500+ articles, revisit two-layer MLP architectures.
8. Consider PCA pre-reduction (768 -> 128) if the first layer is the overfitting bottleneck.
9. Explore ensemble of Ridge + MLP predictions (simple averaging often helps).

### Parameter Budget Rule of Thumb

A useful heuristic: keep total trainable parameters below 10x the number of training samples. For 300 samples, that means < 3000 parameters.

| Architecture | Parameters | Viable at 300 Samples? |
|:-------------|:-----------|:-----------------------|
| Ridge (768 -> 1) | 769 | Yes (closed-form, no overfitting risk with CV alpha) |
| MLP (768 -> 64 -> 1) | 49,281 | Marginal (needs heavy regularization) |
| MLP (768 -> 128 -> 1) | 98,561 | Risky (dropout 0.4+ and weight decay 0.05+ required) |
| MLP (768 -> 256 -> 64 -> 1) | 213,057 | No (too many parameters for this data size) |
| Multi-task (768 -> 128 -> 64 -> 2 heads) | ~107,000 | Marginal (multi-task acts as regularizer, but still risky) |

The parameter counts above highlight why Ridge is such a strong baseline: it has 769 parameters compared to ~100K for even a modest MLP. The MLP must learn something substantially non-linear to justify that 100x increase in model complexity.

### Final Architecture Recommendation

```python
# For ~300 training samples, this is the sweet spot:
model = nn.Sequential(
    nn.Linear(768, 64),     # Aggressive compression
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),       # Total: 49,281 params
)
# With: AdamW(lr=1e-3, weight_decay=0.05), early stopping patience=20
# And: Gaussian noise augmentation (std=0.05)
```

For the multi-task variant, use the `MultiTaskEngagementModel` from Section 4 with `hidden_dim=64` and `bottleneck_dim=32` to keep the parameter count manageable.

---

## 9. Complete End-to-End Example

```python
"""
Complete pipeline for predicting article engagement from embeddings.
Compares Ridge baseline, single-task MLP, and multi-task MLP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    """Load embeddings and targets. Replace with actual data loading."""
    # X: (n_articles, 768) embeddings from nomic-embed-text-v1.5
    # y: (n_articles,) engagement scores 0-100
    X = np.load("embeddings.npy")
    y = np.load("scores.npy")
    return X, y


def evaluate_ridge(X, y, cv):
    """Ridge regression baseline."""
    alphas = np.logspace(-2, 4, 50)
    ridge = RidgeCV(alphas=alphas)
    rmses = []
    for train_idx, val_idx in cv.split(X):
        ridge.fit(X[train_idx], y[train_idx])
        preds = ridge.predict(X[val_idx])
        rmse = np.sqrt(np.mean((preds - y[val_idx]) ** 2))
        rmses.append(rmse)
    return np.mean(rmses), np.std(rmses)


def train_mlp_fold(X_train, y_train, X_val, y_val, max_epochs=200):
    """Train a single-task MLP on one fold."""
    model = nn.Sequential(
        nn.Linear(768, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.05
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        # Add Gaussian noise augmentation
        noise = torch.randn_like(X_t) * 0.05
        preds = model(X_t + noise).squeeze()
        loss = F.mse_loss(preds, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v).squeeze()
            val_loss = F.mse_loss(val_preds, y_v).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_preds = model(X_v).squeeze().numpy()
    return np.sqrt(np.mean((final_preds - y_val) ** 2))


def main():
    X, y = load_data()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline
    ridge_mean, ridge_std = evaluate_ridge(X, y, cv)
    print(f"Ridge RMSE: {ridge_mean:.2f} +/- {ridge_std:.2f}")

    # MLP
    mlp_rmses = []
    for train_idx, val_idx in cv.split(X):
        rmse = train_mlp_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        mlp_rmses.append(rmse)
    mlp_mean = np.mean(mlp_rmses)
    mlp_std = np.std(mlp_rmses)
    print(f"MLP RMSE:   {mlp_mean:.2f} +/- {mlp_std:.2f}")

    improvement = (ridge_mean - mlp_mean) / ridge_mean
    if improvement > 0.05:
        print(f"MLP improves by {improvement:.1%} -- worth using.")
    else:
        print(f"MLP improvement is only {improvement:.1%} -- stick with Ridge.")


if __name__ == "__main__":
    main()
```

---

## Sources

- [scikit-learn Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [scikit-learn RidgeCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
- [scikit-learn MLPRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [scikit-learn Cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data," NeurIPS 2021](https://arxiv.org/pdf/2106.11959)
- [Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR 2018](https://arxiv.org/abs/1705.07115)
- [Huang et al., "LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP," CVPR 2024](https://arxiv.org/abs/2404.02285)
- [Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)"](https://arxiv.org/pdf/2305.12073)
- [Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR 2014 (via D2L)](http://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
- [Dive into Deep Learning: Batch Normalization](http://d2l.ai/chapter_convolutional-modern/batch-norm.html)
- [Batch and Layer Normalization comparison (Pinecone)](https://www.pinecone.co/learn/batch-layer-normalization/)
- [MIT Deep Learning Blog: Spectral Normalization effects in MLPs and Residual Networks](https://deep-learning-mit.github.io/staging/blog/2023/WeightDecaySpecNormEffects/)
- [Nomic-embed-text-v1.5 model card (Hugging Face)](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Nomic Embed Technical Report](https://arxiv.org/html/2402.01613v2)
- [Neptune.ai: Dimensionality Reduction](https://neptune.ai/blog/dimensionality-reduction)
- [Neptune.ai: Cross-Validation Done Right](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
- [Zero-Inflated Regression (Towards Data Science)](https://towardsdatascience.com/zero-inflated-regression-c7dfc656d8af/)
- [Wikipedia: Hurdle model](https://en.wikipedia.org/wiki/Hurdle_model)
- [ScienceDirect: Two-fold ML approach for zero-inflated data](https://www.sciencedirect.com/science/article/pii/S0952197625003392)
- [MachineLearningMastery: Early Stopping](https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/)
- [Zhang et al., "mixup: Beyond Empirical Risk Minimization" / RC-Mixup](https://arxiv.org/abs/2405.17938)
- [Xiao et al., Gaussian noise augmentation for small datasets](https://www.sciencedirect.com/science/article/abs/pii/S0022169421005576)
- [GumGum Tech Blog: Multi-Task Learning in PyTorch](https://medium.com/gumgum-tech/an-easy-recipe-for-multi-task-learning-in-pytorch-that-you-can-do-at-home-1e529a8dfb7f)
- [University of Arkansas: Dropout vs Weight Decay comparison](https://scholarworks.uark.edu/cgi/viewcontent.cgi?article=1028&context=csceuht)
- [Regularization in Neural Networks (Pinecone)](https://www.pinecone.io/learn/regularization-in-neural-networks/)
- [MachineLearningMastery: Building MLPs in PyTorch](https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/)
