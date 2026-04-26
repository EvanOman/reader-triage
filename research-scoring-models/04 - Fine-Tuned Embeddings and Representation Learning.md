# Fine-Tuned Embeddings and Representation Learning

## Executive Summary

Fine-tuning embedding models for domain-specific prediction tasks is a well-established technique that, as of 2025-2026, has become increasingly accessible through libraries like sentence-transformers v3, SetFit, and PEFT/LoRA adapters. However, for a dataset of only hundreds of articles predicting engagement (highlight counts), full embedding fine-tuning carries substantial overfitting risk and is unlikely to outperform a well-designed pipeline using frozen general-purpose embeddings with a trained regression head. The recommended path is a staged approach: start with frozen nomic-embed-text-v1.5 embeddings plus a regularized regression head (the current approach), then explore lightweight adapter fine-tuning or SetFit-style contrastive tuning only after exhausting feature engineering improvements and accumulating at least 500-1,000 labeled examples. The most impactful near-term investment is likely data augmentation (synthetic pairs via LLM) and careful feature engineering on top of frozen embeddings, not embedding fine-tuning.

---

## 1. Fine-Tuning Approaches: A Taxonomy

### 1.1 Full Fine-Tuning

Full fine-tuning updates all parameters of the embedding model during training. For a model like nomic-embed-text-v1.5 (~137M parameters), this means adjusting every weight in every transformer layer.

**Advantages:**
- Maximum expressiveness; the model can learn to reshape its entire representation space
- Best in-distribution performance when sufficient data is available

**Disadvantages:**
- Requires thousands to tens of thousands of examples to avoid catastrophic overfitting ([Weaviate, 2024](https://weaviate.io/blog/fine-tune-embedding-model))
- Can "distort pretrained features and underperform out-of-distribution" ([Kumar et al., NeurIPS 2022](https://arxiv.org/abs/2202.10054))
- High compute cost; requires GPU and careful hyperparameter tuning
- Risk of catastrophic forgetting of general semantic knowledge

**Minimum data:** The consensus from multiple sources is 1,000-5,000 examples for narrow domains with simple tasks, scaling to 50,000-100,000 for complex tasks ([Zilliz, 2024](https://zilliz.com/ai-faq/how-many-examples-do-i-need-to-finetune-an-embedding-model-effectively)). Modal's empirical study found that model-dependent behavior matters enormously: `all-mpnet-base-v2` showed improvement with just 100 examples, while `bge-base-en-v1.5` needed ~200 to beat baseline, and `jina-embeddings-v2-small-en` never improved at all ([Modal, 2024](https://modal.com/blog/fine-tuning-embeddings)).

### 1.2 Linear Probe (Frozen Embeddings + Trained Head)

A linear probe keeps the embedding model completely frozen and trains only a lightweight prediction head (linear regression, MLP, or similar) on top of the fixed embeddings.

**Advantages:**
- No risk of distorting pretrained representations
- Works well with very small datasets (dozens to hundreds of examples)
- Fast to train; no GPU required
- Preserves out-of-distribution generalization ([Kumar et al., NeurIPS 2022](https://arxiv.org/abs/2202.10054))
- Easy to iterate: swap heads, add features, experiment quickly

**Disadvantages:**
- Cannot adapt the embedding space to the task; limited by the quality of the frozen representations
- Performance ceiling is determined by how well general-purpose embeddings capture the relevant signal

**When to use:** This is the default starting point for datasets under 1,000 examples. The linear probe establishes a performance baseline that any fine-tuning approach must beat to justify its complexity.

### 1.3 Adapter Layers (LoRA / PEFT)

Parameter-Efficient Fine-Tuning (PEFT) methods, particularly LoRA (Low-Rank Adaptation), insert small trainable matrices into the transformer layers while keeping the original weights frozen. This trains only 1-5% of total parameters.

**Advantages:**
- Dramatically reduces overfitting risk compared to full fine-tuning
- "LoRA often outperforms full fine-tuning by preventing the model from overfitting to limited examples" for datasets under 1,000 examples ([Sapien, 2024](https://www.sapien.io/blog/strategies-for-fine-tuning-llms-on-small-datasets))
- Adapter weights are tiny (2-3% of model size) and composable
- sentence-transformers v3 has native PEFT integration

**Disadvantages:**
- Still requires more data than a simple linear probe
- Adds training complexity (LoRA rank, alpha, target modules)
- Performance gains over frozen embeddings may be marginal for very small datasets

**Empirical result:** On the GooAQ benchmark, LoRA fine-tuning of BERT-base achieved 0.4705 NDCG@10 vs 0.4728 for full fine-tuning, using only 2.14% of parameters ([sentence-transformers docs](https://sbert.net/examples/sentence_transformer/training/peft/README.html)).

### 1.4 Linear Probing Then Fine-Tuning (LP-FT)

A hybrid approach from Kumar et al. (2022): first train a linear head on frozen features, then fine-tune the full model initialized with that head.

**Advantages:**
- Best of both worlds: 1% better in-distribution and 10% better out-of-distribution than standard fine-tuning ([Kumar et al., NeurIPS 2022](https://arxiv.org/abs/2202.10054); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/fcc22e5b7d5d2155d994da22d045f0a6-Paper-Conference.pdf))
- Preserves pretrained features while still allowing adaptation

**Disadvantages:**
- Two-stage training adds complexity
- Still requires enough data to fine-tune without overfitting

---

## 2. Sentence-Transformers Fine-Tuning API (v3+)

The sentence-transformers library (v3.0+, released mid-2024) introduced a modernized training API built on HuggingFace's Trainer. This is the primary tool for fine-tuning embedding models in the Python ecosystem.

### 2.1 Core Components

The training pipeline has five components ([HuggingFace blog](https://huggingface.co/blog/train-sentence-transformers)):

1. **Model**: `SentenceTransformer` wrapping any HuggingFace transformer
2. **Dataset**: HuggingFace `Dataset` with columns matching the loss function's requirements
3. **Loss function**: Determines what the model learns (see below)
4. **Training arguments**: Hyperparameters via `SentenceTransformerTrainingArguments`
5. **Evaluator**: Metrics computed during training (e.g., `EmbeddingSimilarityEvaluator`)

### 2.2 Loss Functions Relevant to Regression/Prediction

For predicting a continuous engagement score from text pairs, the most relevant loss functions are:

| Loss Function | Input Format | Use Case |
|---|---|---|
| `CosineSimilarityLoss` | (text_a, text_b, float_score) | Similarity regression |
| `CoSENTLoss` | (text_a, text_b, float_score) | Stronger signal than CosineSimilarity |
| `AnglELoss` | (text_a, text_b, float_score) | Angular distance optimization |
| `MSELoss` | (text_a, text_b, float_score) | Direct MSE on similarity |
| `MultipleNegativesRankingLoss` | (anchor, positive) | Contrastive; no explicit scores |
| `MatryoshkaLoss` | Wraps any other loss | Multi-resolution training |

For our engagement prediction task, the most natural framing would be: given pairs of articles with relative engagement labels (article A has more highlights than article B), use contrastive losses to push high-engagement articles into a distinguishable region of embedding space. Alternatively, pair each article with its engagement score and use `CoSENTLoss`.

### 2.3 Regression-Oriented Fine-Tuning Example

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import Dataset
import numpy as np

# Load base model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# Prepare dataset: pairs of articles with normalized engagement similarity
# For regression, create pairs where the "score" reflects engagement similarity
train_data = {
    "sentence1": article_texts_a,      # Article text (or summary)
    "sentence2": article_texts_b,      # Another article text
    "score": similarity_scores,          # Float in [0, 1] based on engagement similarity
}
train_dataset = Dataset.from_dict(train_data)

# CoSENTLoss produces stronger training signal than CosineSimilarityLoss
loss = losses.CoSENTLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned-nomic",
    num_train_epochs=3,                  # Keep low for small datasets
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained("./finetuned-nomic-engagement")
```

### 2.4 PEFT/LoRA Fine-Tuning Example

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from peft import LoraConfig, TaskType

# Load base model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# Add LoRA adapter - only trains ~2% of parameters
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=16,               # Rank of update matrices; lower = fewer params
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,    # Regularization
    target_modules=["q_proj", "v_proj"],  # Which attention matrices to adapt
)
model.add_adapter(peft_config)

loss = losses.CoSENTLoss(model)

# Train with same SentenceTransformerTrainer API
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
    args=training_args,
)
trainer.train()

# Save only the adapter weights (tiny file)
model.save_pretrained("./nomic-engagement-lora")
```

---

## 3. SetFit for Few-Shot Scenarios

### 3.1 How SetFit Works

SetFit ([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)) is a two-phase framework designed for few-shot learning:

**Phase 1 - Contrastive Embedding Fine-Tuning:**
- Generates positive pairs (texts from the same class) and negative pairs (texts from different classes) from the small labeled set
- From just 8 examples per class, combinatorics yield ~92 training pairs
- Fine-tunes the sentence transformer body using contrastive loss
- Training on an NVIDIA V100 with 8 examples takes ~30 seconds

**Phase 2 - Classifier Head Training:**
- Passes all training texts through the fine-tuned body to generate embeddings
- Trains a logistic regression classifier (sklearn) on these embeddings

**Key results:** With only 8 labeled examples per class, SetFit matches RoBERTa-Large fine-tuned on the full 3,000-example training set on Customer Reviews sentiment ([HuggingFace blog](https://huggingface.co/blog/setfit)).

### 3.2 SetFit for Regression

SetFit was designed primarily for **classification**, not regression. The default head is `LogisticRegression` from scikit-learn, which expects discrete class labels ([SetFit docs](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit)). However, there are workarounds:

1. **Binned classification:** Discretize engagement scores into bins (low/medium/high) and use standard SetFit classification
2. **Custom head replacement:** After Phase 1 (contrastive fine-tuning of the body), replace the classification head with a regression model:

```python
from setfit import SetFitModel, SetFitTrainer
from sklearn.linear_model import Ridge

# Phase 1: Contrastive fine-tuning (uses class labels)
model = SetFitModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

# Bin engagement scores into classes for contrastive training
binned_labels = pd.cut(highlight_counts, bins=3, labels=[0, 1, 2])

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,  # with binned labels
    num_iterations=20,
    batch_size=16,
)
trainer.train()

# Phase 2: Replace head with regression model
embeddings = model.model_body.encode(article_texts)
regression_head = Ridge(alpha=1.0)
regression_head.fit(embeddings, highlight_counts)  # continuous targets
```

3. **SetFit's differentiable head:** SetFit also offers `SetFitHead`, a PyTorch-based head that could theoretically be adapted for regression, though this is not well-documented for continuous targets.

### 3.3 SetFit Applicability Assessment

For our engagement prediction task with hundreds of articles:

- **Pro:** SetFit's contrastive Phase 1 can work with very few examples (8-64 per class), making it viable for small datasets
- **Pro:** The contrastive fine-tuning reshapes the embedding space to be more task-aware
- **Con:** SetFit is fundamentally classification-oriented; regression requires workarounds
- **Con:** Binning continuous engagement scores into classes loses information
- **Verdict:** SetFit is worth trying as a quick experiment (binned into 3-5 engagement tiers), but is not the natural fit for continuous regression

---

## 4. Minimum Dataset Sizes for Effective Fine-Tuning

### 4.1 Empirical Evidence

The research consensus on minimum dataset sizes, synthesized from multiple sources:

| Approach | Minimum Examples | Sweet Spot | Source |
|---|---|---|---|
| Linear probe on frozen embeddings | 50-100 | 200+ | General ML practice |
| SetFit (classification) | 8 per class | 16-64 per class | [Tunstall et al., 2022](https://arxiv.org/abs/2209.11055) |
| LoRA adapter fine-tuning | 500-1,000 | 2,000-5,000 | [Sapien, 2024](https://www.sapien.io/blog/strategies-for-fine-tuning-llms-on-small-datasets) |
| Full contrastive fine-tuning | 1,000-5,000 | 5,000-10,000 | [Zilliz, 2024](https://zilliz.com/ai-faq/how-many-examples-do-i-need-to-finetune-an-embedding-model-effectively); [Weaviate, 2024](https://weaviate.io/blog/fine-tune-embedding-model) |
| Full fine-tuning (complex tasks) | 10,000-50,000 | 50,000-100,000 | [Zilliz, 2024](https://zilliz.com/ai-faq/how-many-examples-do-i-need-to-finetune-an-embedding-model-effectively) |

### 4.2 Model-Specific Variation

Modal's empirical study ([Modal, 2024](https://modal.com/blog/fine-tuning-embeddings)) demonstrated that minimum dataset size depends heavily on the base model:

- **all-mpnet-base-v2:** Improved with just 100 examples, but plateaued quickly
- **bge-base-en-v1.5:** Needed ~200 examples to beat baseline, then continued improving all the way to 100,000
- **jina-embeddings-v2-small-en:** Never improved regardless of dataset size

This underscores that there is no universal minimum; the base model's architecture and pre-training data determine how efficiently it can be fine-tuned.

### 4.3 What "Hundreds of Articles" Means

With ~300-500 labeled articles:
- **Linear probe:** Fully viable. This is the recommended starting point.
- **SetFit (binned):** Viable if binned into 3-5 classes (60-100+ per class).
- **LoRA adapter:** Marginal. On the lower edge of viability; expect modest gains at best, with significant overfitting risk.
- **Full fine-tuning:** Not recommended. High overfitting risk with no clear path to improvement.

---

## 5. Risk of Overfitting with Small Datasets

### 5.1 Why Embedding Fine-Tuning Overfits Easily

Embedding models have millions of parameters (nomic-embed-text-v1.5 has ~137M). With hundreds of training examples, the ratio of parameters to examples is ~100,000:1 for full fine-tuning. This creates severe overfitting conditions:

- The model memorizes training examples rather than learning generalizable patterns
- Performance on training data diverges sharply from validation/test performance
- The model loses general-purpose semantic knowledge ("catastrophic forgetting")
- Fine-tuned features can "distort pretrained features" such that "fine-tuning leads to worse accuracy out-of-distribution" ([Kumar et al., 2022](https://arxiv.org/abs/2202.10054))

### 5.2 Mitigation Strategies

If fine-tuning is attempted on a small dataset, these techniques reduce (but do not eliminate) overfitting:

1. **LoRA/PEFT:** Reduce trainable parameters to 1-3% of total. LoRA with rank 8-16 adds only thousands of parameters instead of millions.
2. **Low learning rate:** Use 1e-5 to 2e-5, much lower than default.
3. **Few epochs:** Limit to 1-3 epochs. "More isn't always better -- overfitting can creep in if your dataset is small" ([Weaviate, 2024](https://weaviate.io/blog/fine-tune-embedding-model)).
4. **Weight decay:** Apply L2 regularization (0.01-0.1).
5. **Dropout:** Increase dropout during fine-tuning (0.1-0.2).
6. **Early stopping:** Monitor validation loss and stop when it starts increasing.
7. **Cross-validation:** Use k-fold CV to get robust performance estimates.
8. **Data augmentation:** Generate synthetic pairs using an LLM to expand the effective dataset size.

### 5.3 The Augmentation Escape Hatch

For small datasets, LLM-generated synthetic data can dramatically expand the effective training set. The approach from [Lu et al., 2024](https://arxiv.org/abs/2408.11868) demonstrated that starting from only 26 Q&A pairs, they generated 249,600 training samples using synthetic variations, achieving meaningful improvements over baseline. For our use case:

```python
# Pseudocode: Generate synthetic article pairs for contrastive training
for article in articles:
    # Use Claude to generate similar articles (positive pairs)
    similar = claude.generate(f"Write a short article on a similar topic to: {article.title}")

    # Use Claude to generate dissimilar articles (negative pairs)
    dissimilar = claude.generate(f"Write a short article on a very different topic from: {article.title}")

    # Create training pairs with engagement-based scores
    pairs.append((article.text, similar, high_similarity_score))
    pairs.append((article.text, dissimilar, low_similarity_score))
```

However, for a regression task (predicting highlight counts), synthetic augmentation is harder because you cannot synthesize the target variable (actual human engagement). Augmentation is more naturally suited to contrastive/retrieval tasks.

---

## 6. Matryoshka Embeddings and Their Relevance

### 6.1 What Are Matryoshka Embeddings?

Matryoshka Representation Learning (MRL) ([Kusupati et al., NeurIPS 2022](https://arxiv.org/abs/2205.13147)) trains embedding models so that the first d dimensions of a D-dimensional embedding remain meaningful when truncated. Like Russian nesting dolls, coarser semantic information is packed into earlier dimensions, while finer details occupy later ones.

For example, a 768-dimensional Matryoshka embedding can be truncated to 256, 128, or even 64 dimensions while retaining most of its semantic quality. nomic-embed-text-v1.5 was explicitly trained with Matryoshka loss and supports this truncation.

### 6.2 Fine-Tuning with MatryoshkaLoss

sentence-transformers provides `MatryoshkaLoss` as a wrapper around any other loss function:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

base_loss = CoSENTLoss(model)
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
    matryoshka_weights=[1, 1, 1, 1, 1],
)
```

During training, the loss is computed at each specified dimensionality and summed, ensuring the model maintains quality across all truncation levels.

### 6.3 Relevance to Our Use Case

Matryoshka embeddings are relevant in two ways:

1. **If fine-tuning:** Since nomic-embed-text-v1.5 already uses Matryoshka training, any fine-tuning should wrap the loss in `MatryoshkaLoss` to preserve this property. Otherwise, fine-tuning may degrade the quality of truncated embeddings.

2. **For the regression head:** The Matryoshka property means we can experiment with different embedding dimensionalities for the regression head. Lower dimensions (128 or 256) have fewer features, which acts as a natural regularizer for small datasets. This is a free parameter to tune with frozen embeddings, requiring no fine-tuning at all.

**Practical recommendation:** Before fine-tuning, try training the regression head on truncated embeddings (e.g., first 128 or 256 dimensions). The dimensionality reduction may improve generalization on small datasets by reducing the feature-to-sample ratio.

---

## 7. Practical Workflow: When to Fine-Tune vs. When to Use Frozen Embeddings

### 7.1 Decision Framework

```
START
  |
  v
Are frozen embeddings + regression head performing well enough?
  |-- YES --> Stop. Do not fine-tune.
  |-- NO
      |
      v
    Do you have >= 1,000 labeled examples?
      |-- NO --> Stay with frozen embeddings.
      |          Try: feature engineering, dimensionality reduction,
      |          ensemble methods, better regression heads, data augmentation.
      |-- YES
          |
          v
        Is the domain highly specialized (jargon, novel concepts)?
          |-- NO --> Frozen embeddings are probably fine.
          |          General-purpose models handle common language well.
          |-- YES
              |
              v
            Try LoRA adapter fine-tuning first (lowest risk).
            Evaluate with k-fold CV against frozen baseline.
              |
              v
            Did LoRA improve performance?
              |-- NO --> Frozen embeddings are sufficient for this task.
              |-- YES --> Consider full fine-tuning if you have >= 5,000 examples.
```

### 7.2 Evidence for the "Frozen First" Approach

The paper "Do We Need Domain-Specific Embedding Models?" ([Tang & Yang, 2024](https://arxiv.org/abs/2409.18511)) found that general-purpose embedding models exhibit significant performance drops on domain-specific benchmarks (e.g., 71.67 on MTEB vs 63.09 on FinMTEB). However, this study focused on retrieval tasks with specialized financial terminology. For our task -- predicting engagement from general-interest articles -- the domain gap is much smaller. Reader Triage articles span general technology, culture, science, and similar topics that are well-represented in nomic-embed-text-v1.5's training data.

The key question is not "do the embeddings capture the article's meaning?" (they almost certainly do for general-interest content) but rather "does the embedding space naturally separate high-engagement from low-engagement articles?" This is fundamentally a question about the regression head, not the embeddings.

### 7.3 What to Try Before Fine-Tuning

For a dataset of hundreds of articles, exhaust these options first:

1. **Optimize the regression head:** Try Ridge, Lasso, ElasticNet, SVR, gradient-boosted trees (XGBoost/LightGBM), or a small MLP. Each handles the high-dimensional embedding space differently.

2. **Dimensionality reduction:** Use PCA, UMAP, or Matryoshka truncation to reduce the 768-dim embeddings before regression. This reduces overfitting.

3. **Feature engineering:** Combine embeddings with hand-crafted features: article length, source domain, publication recency, readability scores, topic clusters.

4. **Ensemble embeddings:** Concatenate or average embeddings from multiple models (e.g., nomic-embed-text-v1.5 + a different model like bge-base-en-v1.5).

5. **Cross-validation:** Use 5-fold or 10-fold CV to get reliable performance estimates before concluding that embeddings are the bottleneck.

---

## 8. Code Examples for Complete Workflows

### 8.1 Baseline: Frozen Embeddings + Regression Head

```python
"""
Baseline approach: frozen nomic-embed-text-v1.5 + regression head.
This should be the first thing to try and the benchmark to beat.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load model and encode articles
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
embeddings = model.encode(
    ["search_document: " + text for text in article_texts],
    show_progress_bar=True,
)

# Try different Matryoshka truncation levels
for dim in [768, 512, 256, 128, 64]:
    truncated = embeddings[:, :dim]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])

    scores = cross_val_score(
        pipeline, truncated, highlight_counts,
        cv=5, scoring="neg_mean_squared_error"
    )
    rmse = np.sqrt(-scores.mean())
    print(f"dim={dim}: RMSE={rmse:.3f} (+/- {np.sqrt(scores.std()):.3f})")
```

### 8.2 SetFit-Style Contrastive Fine-Tuning (Binned Engagement)

```python
"""
SetFit approach: bin engagement into classes, contrastively fine-tune,
then train a regression head on the fine-tuned embeddings.
"""
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# Bin highlight counts into engagement tiers
bins = pd.qcut(highlight_counts, q=3, labels=["low", "medium", "high"])
label_map = {"low": 0, "medium": 1, "high": 2}
labels = [label_map[b] for b in bins]

# Create dataset
ds = Dataset.from_dict({
    "text": article_texts,
    "label": labels,
})
train_ds = ds.select(range(int(0.8 * len(ds))))
eval_ds = ds.select(range(int(0.8 * len(ds)), len(ds)))

# Phase 1: Contrastive fine-tuning of the embedding body
model = SetFitModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    num_iterations=20,       # Number of text pairs to generate
    batch_size=16,
    num_epochs=1,
)
trainer.train()

# Phase 2: Extract fine-tuned embeddings, train regression head
finetuned_embeddings = model.model_body.encode(article_texts)
ridge = Ridge(alpha=1.0)
# Use cross-validation to evaluate
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ridge, finetuned_embeddings, highlight_counts, cv=5,
                         scoring="neg_mean_squared_error")
print(f"SetFit + Ridge RMSE: {np.sqrt(-scores.mean()):.3f}")
```

### 8.3 LoRA Adapter Fine-Tuning with Sentence-Transformers

```python
"""
LoRA adapter fine-tuning for engagement prediction.
Only trains ~2% of parameters, reducing overfitting risk.
"""
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from peft import LoraConfig, TaskType
from datasets import Dataset
import numpy as np

# Create article pairs with engagement similarity scores
# Higher score = both articles have similar engagement levels
def create_pairs(texts, scores, n_pairs=5000):
    pairs = {"sentence1": [], "sentence2": [], "score": []}
    n = len(texts)
    for _ in range(n_pairs):
        i, j = np.random.randint(0, n, 2)
        # Normalize engagement difference to [0, 1] similarity
        max_score = max(scores)
        diff = abs(scores[i] - scores[j]) / max(max_score, 1)
        similarity = 1.0 - diff
        pairs["sentence1"].append(texts[i])
        pairs["sentence2"].append(texts[j])
        pairs["score"].append(float(similarity))
    return Dataset.from_dict(pairs)

train_dataset = create_pairs(article_texts, highlight_counts)

# Initialize model with LoRA
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,                  # Low rank for small dataset
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model.add_adapter(peft_config)

loss = losses.CoSENTLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./nomic-engagement-lora",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

# Extract fine-tuned embeddings for regression
finetuned_embeddings = model.encode(article_texts)
```

### 8.4 LlamaIndex Linear Adapter (Black-Box Embedding Adaptation)

```python
"""
LlamaIndex's approach: fine-tune a lightweight linear transformation
on top of any embedding model (even API-based ones).
"""
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.embeddings import resolve_embed_model

# Base embedding model (frozen)
base_embed = resolve_embed_model("local:nomic-ai/nomic-embed-text-v1.5")

# Fine-tune a linear adapter on query embeddings
finetune_engine = EmbeddingAdapterFinetuneEngine(
    base_embed_model=base_embed,
    train_dataset=train_dataset,       # (query, relevant_doc) pairs
    val_dataset=val_dataset,
    epochs=25,
    batch_size=16,
    learning_rate=1e-4,
    adapter_model="linear",            # or "two_layer" for MLP
)

finetune_engine.finetune()
adapted_model = finetune_engine.get_finetuned_model()
```

---

## 9. Specific Recommendations for a Dataset of Hundreds of Articles

### 9.1 Bottom Line

**Do not fine-tune the embedding model yet.** With hundreds of articles, the risk-reward ratio strongly favors frozen embeddings with a well-tuned regression pipeline. The potential gains from embedding fine-tuning are small and uncertain, while the risks (overfitting, wasted engineering time, degraded generalization) are concrete.

### 9.2 Recommended Action Plan

**Phase 1 (Now): Maximize frozen embedding performance**
1. Systematically evaluate regression heads: Ridge, SVR, XGBoost, LightGBM, small MLP
2. Test Matryoshka truncation: compare 768, 512, 256, 128 dimensions
3. Add auxiliary features: article length, source, word count, readability metrics
4. Use rigorous cross-validation (5-fold minimum) to measure each change
5. Establish a clear performance baseline with confidence intervals

**Phase 2 (500+ articles): Experiment with lightweight adaptation**
1. Try SetFit with binned engagement classes (3-5 tiers)
2. Compare fine-tuned SetFit embeddings + Ridge against the Phase 1 baseline
3. If SetFit shows improvement, try LoRA adapter fine-tuning with rank 8
4. Always compare against the frozen baseline with proper CV

**Phase 3 (1,000+ articles): Consider contrastive fine-tuning**
1. Generate article pairs with engagement-based similarity scores
2. Fine-tune with CoSENTLoss + MatryoshkaLoss wrapper
3. Use LoRA first; only try full fine-tuning if LoRA shows clear gains
4. Augment training data with LLM-generated synthetic pairs if needed

**Phase 4 (5,000+ articles): Full fine-tuning becomes viable**
1. At this scale, full fine-tuning with careful regularization can work
2. Use LP-FT (linear probe then fine-tune) for best generalization
3. Benchmark against all previous approaches

### 9.3 Expected Gains

Based on the literature:

- **Frozen embeddings + tuned head:** This is already a strong approach. The primary bottleneck is likely not the embeddings but the regression head and feature engineering.
- **LoRA fine-tuning (500-1,000 examples):** Expect 0-5% improvement in prediction accuracy, with high variance. Some models improve, others do not ([Modal, 2024](https://modal.com/blog/fine-tuning-embeddings)).
- **Full fine-tuning (5,000+ examples):** Expect 5-15% improvement for domain-specific tasks where general embeddings genuinely miss relevant patterns ([Weaviate, 2024](https://weaviate.io/blog/fine-tune-embedding-model)).
- **For general-interest articles:** The improvement from fine-tuning will likely be at the lower end, since general-purpose embeddings already handle this content well.

### 9.4 When Fine-Tuning Would Be Worth It

Fine-tuning would become a higher priority if:
- The article corpus contains highly specialized jargon not in the pre-training data
- Error analysis reveals that semantically similar articles get very different engagement scores, suggesting the embedding space conflates dimensions that matter for engagement
- The dataset grows to 1,000+ articles with reliable engagement labels
- A linear probe analysis (training a classifier on frozen embeddings to predict engagement bins) achieves low accuracy, suggesting the embedding space does not naturally encode engagement-relevant features

---

## Sources

- [Weaviate: Why, When and How to Fine-Tune a Custom Embedding Model](https://weaviate.io/blog/fine-tune-embedding-model) -- Comprehensive decision framework for embedding fine-tuning
- [Modal: Beating Proprietary Models with a Quick Fine-Tune](https://modal.com/blog/fine-tuning-embeddings) -- Empirical study of fine-tuning with 100 to 100,000 examples across three models
- [Zilliz: How Many Examples Do I Need to Fine-Tune an Embedding Model?](https://zilliz.com/ai-faq/how-many-examples-do-i-need-to-finetune-an-embedding-model-effectively) -- Dataset size guidelines by task complexity
- [HuggingFace: Training and Finetuning with Sentence Transformers v3](https://huggingface.co/blog/train-sentence-transformers) -- Official training API documentation
- [Sentence Transformers: Training Overview](https://sbert.net/docs/sentence_transformer/training_overview.html) -- Loss functions, evaluators, and trainer API
- [Sentence Transformers: PEFT Adapter Training](https://sbert.net/examples/sentence_transformer/training/peft/README.html) -- LoRA integration with sentence-transformers
- [Tunstall et al., 2022: Efficient Few-Shot Learning Without Prompts (SetFit)](https://arxiv.org/abs/2209.11055) -- SetFit paper
- [HuggingFace: SetFit Blog Post](https://huggingface.co/blog/setfit) -- SetFit overview and results
- [SetFit Conceptual Guide](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit) -- How SetFit works internally
- [Kumar et al., 2022: Fine-Tuning can Distort Pretrained Features](https://arxiv.org/abs/2202.10054) -- Linear probing then fine-tuning (LP-FT) approach
- [NeurIPS 2024: Understanding Linear Probing then Fine-tuning](https://proceedings.neurips.cc/paper_files/paper/2024/file/fcc22e5b7d5d2155d994da22d045f0a6-Paper-Conference.pdf) -- Extended analysis of LP-FT
- [Tang & Yang, 2024: Do We Need Domain-Specific Embedding Models?](https://arxiv.org/abs/2409.18511) -- Empirical evidence for domain-specific vs general-purpose embeddings
- [Lu et al., 2024: Improving Embedding with Contrastive Fine-Tuning on Small Datasets](https://arxiv.org/abs/2408.11868) -- Expert-augmented scoring for small-dataset fine-tuning
- [Kusupati et al., NeurIPS 2022: Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) -- Original Matryoshka embeddings paper
- [HuggingFace: Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka) -- Matryoshka practical guide
- [Sentence Transformers: MatryoshkaLoss](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html) -- MatryoshkaLoss implementation and examples
- [LlamaIndex: Fine-Tuning a Linear Adapter for Any Embedding Model](https://www.llamaindex.ai/blog/fine-tuning-a-linear-adapter-for-any-embedding-model-8dd0a142d383) -- Linear adapter approach
- [Sapien: Strategies for Fine-Tuning LLMs on Small Datasets](https://www.sapien.io/blog/strategies-for-fine-tuning-llms-on-small-datasets) -- PEFT strategies and LoRA for small data
- [Nomic: nomic-embed-text-v1.5 Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) -- Model architecture and training details
- [Databricks: Improving Retrieval and RAG with Embedding Model Finetuning](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning) -- Practical fine-tuning guide
- [LlamaIndex: Fine-Tuning Embeddings for RAG with Synthetic Data](https://www.llamaindex.ai/blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971) -- Synthetic data generation for fine-tuning
- [Aurelio AI: Fine-Tuning in Sentence Transformers 3](https://www.aurelio.ai/learn/sentence-transformers-fine-tuning) -- Practical sentence-transformers v3 fine-tuning guide
