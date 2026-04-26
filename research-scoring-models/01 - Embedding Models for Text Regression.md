# Embedding Models for Text Regression: Predicting Reader Engagement

## Executive Summary

General-purpose text embeddings from 2025-2026 models are viable input features for predicting reader engagement (highlights, words highlighted) via downstream regression, but model choice matters less than modeling strategy when datasets are small (~hundreds of articles). The current nomic-embed-text-v1.5 (768d) embeddings already stored in Qdrant are competitive for this use case and likely not the bottleneck -- the real challenge is the zero-inflated target distribution and limited training data, which demand a two-stage hurdle model (classify "any highlights?" then regress on magnitude) with aggressive dimensionality reduction and regularization. Switching to a higher-capacity embedding model (e.g., Voyage 3.5, Gemini Embedding, or Qwen3-Embedding) would yield marginal gains at the cost of re-embedding the entire corpus and increased API costs; the effort is better spent on modeling architecture and expanding the labeled dataset.

---

## 1. Landscape of Major 2025-2026 Embedding Models

### 1.1 Model Comparison Table

| Model | Provider | Params | Default Dims | Max Context | Matryoshka | MTEB (English) | Pricing (per 1M tokens) | Open Source |
|---|---|---|---|---|---|---|---|---|
| **nomic-embed-text-v1.5** | Nomic AI | 137M | 768 | 8,192 | Yes (64-768) | 62.28 | Free (self-hosted) | Yes |
| **nomic-embed-text-v2-moe** | Nomic AI | 475M (305M active) | 768 | 8,192 | Yes (256-768) | ~63-64 (est.) | Free (self-hosted) | Yes |
| **text-embedding-3-small** | OpenAI | Undisclosed | 1,536 | 8,191 | Yes (256-1536) | ~62.3 | $0.02 | No |
| **text-embedding-3-large** | OpenAI | Undisclosed | 3,072 | 8,191 | Yes (256-3072) | 64.6 | $0.13 | No |
| **embed-v4** | Cohere | Undisclosed | 768 (down from 1024) | 128,000 | Yes (64-1024) | 65.2 | $0.12 | No |
| **jina-embeddings-v3** | Jina AI | 570M | 1,024 | 8,192 | Yes (32-1024) | 65.52 | $0.02 (first 1M free) | Partial |
| **voyage-3.5** | Voyage AI | Undisclosed | 2,048 | 32,000 | Yes (256-2048) | ~67 (est.) | $0.06 | No |
| **voyage-3-large** | Voyage AI | Undisclosed | 2,048 | 32,000 | Yes (256-2048) | 66.8 | $0.18 | No |
| **gemini-embedding-001** | Google | Undisclosed | 3,072 | 2,048 | Yes (768-3072) | 68.32 | $0.15 | No |
| **Qwen3-Embedding-8B** | Alibaba | 8B | 4,096 | 32,000 | No | 70.58 (multilingual) | Free (self-hosted) | Yes |
| **text-embedding-005** | Google Cloud | Undisclosed (Gecko) | 768 | 2,048 | No | ~66.3 | $0.025 | No |

Sources: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), [Nomic HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [Voyage AI Blog](https://blog.voyageai.com/2025/05/20/voyage-3-5/), [Google Developers Blog](https://developers.googleblog.com/gemini-embedding-available-gemini-api/), [Qwen3 HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [Ailog MTEB Comparison](https://app.ailog.fr/en/blog/guides/choosing-embedding-models).

### 1.2 Architecture Notes

**nomic-embed-text-v1.5** is built on a modified BERT architecture (nomic-bert) with 12 transformer blocks, 12 attention heads, rotary positional embeddings (replacing absolute positional encodings), SwiGLU activation (replacing GeLU), and Flash Attention. At 137M parameters, it is one of the smallest competitive models, which makes it extremely fast -- over 100 queries/second on a MacBook M2. The model was trained in two stages: unsupervised contrastive learning on weakly-related text pairs (StackExchange Q&A, Amazon review titles, news summaries), followed by supervised fine-tuning with hard-example mining ([Nomic Technical Report](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf)).

**nomic-embed-text-v2-moe** upgrades to a Mixture-of-Experts architecture with 8 experts and top-2 routing, totaling 475M parameters but only 305M active during inference. Trained on 1.6 billion contrastive pairs across ~100 languages, it outperforms models in the same parameter class on both monolingual and multilingual benchmarks. It retains 768-dimensional output and Matryoshka support ([Nomic Blog](https://www.nomic.ai/blog/posts/nomic-embed-text-v2), [arxiv 2502.07972](https://arxiv.org/abs/2502.07972)).

**Jina-embeddings-v3** uses task-specific Low-Rank Adaptation (LoRA) adapters, meaning the same base model can activate different LoRA weights for retrieval, classification, clustering, or text matching. This is architecturally significant: the model explicitly optimizes for classification-quality embeddings when you request them, achieving the highest classification accuracy (82.58) and STS score (85.80) on MTEB among models in its class ([arxiv 2409.10173](https://arxiv.org/abs/2409.10173)).

**Voyage 3.5** was trained with explicit "tricky negatives" (A vs. not-A pairs) by Stanford researchers, with quantization-aware training enabling int8/binary representations that lose minimal quality. At $0.06/1M tokens, it outperforms OpenAI text-embedding-3-large by 8.26% across 100 benchmark datasets spanning 8 domains ([Voyage AI Blog](https://blog.voyageai.com/2025/05/20/voyage-3-5/)).

**Qwen3-Embedding-8B** is the current top performer on the MTEB multilingual leaderboard (70.58), built on the Qwen3 LLM backbone with 8B parameters and 4096-dimensional output. It excels at text retrieval, code retrieval, classification, clustering, and bitext mining. However, it requires significant compute for inference ([Qwen Blog](https://qwenlm.github.io/blog/qwen3-embedding/)).

---

## 2. Embedding Quality for Downstream Regression Tasks

### 2.1 What MTEB Classification Scores Actually Measure

MTEB's classification task evaluates embeddings by training a logistic regression classifier (a "linear probe") on top of frozen embeddings and measuring accuracy. This is directly relevant to our use case: if embeddings produce linearly separable representations for classification, those same representations carry predictive signal for regression tasks on the same or related targets.

Key MTEB classification results:
- **jina-embeddings-v3**: 82.58 (highest among mid-size models, using classification-specific LoRA)
- **OpenAI text-embedding-3-large**: ~81.7
- **nomic-embed-text-v1.5 (768d)**: Competitive -- e.g., AmazonPolarityClassification 91.81% accuracy, AmazonReviewsClassification 47.16%
- **Cohere embed-v4**: Strong but Cohere's strength is more in retrieval than classification

Source: [HuggingFace MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), [Nomic v1.5 Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

### 2.2 Evidence from "Understanding LLM Embeddings for Regression"

The most directly relevant research is Tang, Yang & Song (2024), "Understanding LLM Embeddings for Regression" ([arxiv 2411.14708](https://arxiv.org/abs/2411.14708)), published in TMLR 2025. Key findings:

1. **LLM embeddings outperform hand-crafted features** for high-dimensional regression tasks, particularly as problem dimensionality increases. The advantage grows with the number of degrees of freedom in the prediction target.

2. **Model size does not always improve regression performance.** Testing T5 (512d to 4096d) and Gemini 1.0 (1536d to 14336d), the authors found that larger models occasionally underperformed smaller counterparts. Within the T5 family, scaling showed consistent improvement, but Gemini variants displayed substantial variance.

3. **The key property is Lipschitz continuity**: LLM embeddings inherently preserve smooth mappings from input space to embedding space, meaning similar inputs produce similar embeddings. The authors introduced the Normalized Lipschitz Factor Distribution (NLFD) metric and found Pearson correlations of 0.60-0.88 between embedding smoothness and regression performance.

4. **Performance gaps narrow with more training data.** When sufficient training samples are available, the choice of embedding model matters less -- all reasonable embeddings converge to similar downstream performance.

5. **Methodology**: They used a two-layer MLP (256 hidden units, ReLU, MSE loss, AdamW) as the downstream predictor, with XGBoost as a baseline. Evaluation used Kendall-Tau ranking correlation.

**Implication for our use case**: With only hundreds of training examples, the specific embedding model matters less than it would with thousands. The bottleneck is data, not embedding quality.

### 2.3 Embeddings + Tree Ensembles for Prediction

Recent work on combining embeddings with XGBoost/Random Forest ([MachineLearningMastery](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/), [arxiv 2411.01645](https://arxiv.org/html/2411.01645v1)) demonstrates that embedding-based feature enrichment consistently boosts predictive power for tabular ML:

- Embeddings capture complex semantic patterns that manual feature engineering misses.
- **Dimensionality reduction is critical**: PCA, UMAP, or autoencoders should be applied to reduce high-dimensional embeddings (768+) before feeding them into tree-based models. This improves both speed and generalization.
- Combining embeddings with hand-crafted features (article length, reading time, source reputation, topic signals) often outperforms either alone ([ACM TOSEM 2023](https://dl.acm.org/doi/10.1145/3576039)).

---

## 3. Are 768d Nomic Embeddings Competitive?

### 3.1 Short Answer: Yes, for This Use Case

The nomic-embed-text-v1.5 embeddings already in Qdrant are competitive for downstream regression for several reasons:

1. **Sufficient dimensionality for small datasets.** Research shows that 99% of the variance and mutual information in 768d embeddings is captured in the first ~256 dimensions ([arxiv 2403.14001](https://arxiv.org/abs/2403.14001)). With only hundreds of training samples, using all 768 dimensions is already borderline excessive -- you will almost certainly need to reduce to 64-256 dimensions via PCA to avoid overfitting.

2. **Matryoshka performance is strong.** At 768d, nomic-embed-text-v1.5 scores 62.28 on MTEB; at 256d it still scores 61.04. The performance degradation from truncation is gentle, confirming that semantic information is well-distributed across early dimensions.

3. **Classification-quality embeddings.** The model achieves 91.8% accuracy on AmazonPolarityClassification, demonstrating that the embedding space captures document-level sentiment and quality signals relevant to engagement prediction.

4. **The gap to top models is small in classification.** While models like Jina v3 (82.58 classification) and OpenAI text-embedding-3-large (~81.7) score higher on classification benchmarks, these differences are measured on large test sets. With hundreds of training samples, the statistical noise from the small dataset will dominate any difference between embedding models.

### 3.2 When Switching Would Help

Switching to a higher-quality embedding model would be justified if:

- **The dataset grows to 2,000+ labeled articles**, where the embedding quality ceiling becomes the bottleneck rather than data scarcity.
- **The task requires cross-lingual or code understanding**, where specialized models (Qwen3, Gemini Embedding) have clear advantages.
- **You need task-specific optimization**, where Jina v3's LoRA-based classification adapter or Voyage's domain-tuned models could provide measurable lifts.
- **You want to combine article text with images/charts**, where Cohere embed-v4 (multimodal) becomes relevant.

### 3.3 Cost of Switching

Re-embedding the corpus in Qdrant with a new model requires:
- Re-processing all articles through the new embedding model API
- Updating the Qdrant collection schema (new dimension count)
- Re-indexing all vectors
- API costs (e.g., at $0.06/1M tokens with Voyage 3.5, a corpus of 1,000 articles averaging 2,000 tokens each = $0.12 total -- negligible)

The API cost is trivial; the engineering cost of updating the pipeline is the real consideration.

---

## 4. The Real Challenge: Zero-Inflated Targets and Small Data

### 4.1 Problem Structure

The target variable (number of highlights or total highlighted words) is zero-inflated: many articles have zero highlights, not because they are universally uninteresting, but because:
- The reader has not read them yet
- The reader read them but did not highlight
- The reader skimmed without deep engagement

This is a classic hurdle/zero-inflated problem. Standard regression (linear, XGBoost) will be pulled toward predicting zero for everything, producing poor discrimination among articles that would receive highlights.

### 4.2 Recommended Modeling Architecture: Two-Stage Hurdle Model

Based on literature on zero-inflated count data ([Springer 2021](https://link.springer.com/article/10.1186/s40488-021-00121-4), [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S0952197625003392)):

**Stage 1 -- Binary Classification**: "Will this article receive any highlights?"
- Input: PCA-reduced embeddings (64-128 dimensions) + hand-crafted features
- Model: Logistic regression with L2 regularization, or a small gradient-boosted classifier
- This stage handles the zero-inflation by separating "structural zeros" from potential engagement

**Stage 2 -- Count/Magnitude Regression**: "Given that this article receives highlights, how many / how many words?"
- Input: Same features, trained only on articles with > 0 highlights
- Model: Ridge regression, SVR, or a small XGBoost regressor
- Consider log-transforming the target (log(1 + highlights)) to handle skew

**Combined prediction**: P(any highlights) x E[highlights | highlights > 0]

This two-stage approach has been shown to significantly outperform single-stage regression on zero-inflated data in multiple domains ([MDPI Applied Sciences](https://www.mdpi.com/2076-3417/14/23/10790)).

### 4.3 Dimensionality Reduction is Essential

With ~hundreds of samples and 768 embedding dimensions, the feature-to-sample ratio is dangerously high. Strategies:

1. **PCA to 64-128 dimensions**: Retains 85-95% of variance in 768d embeddings. PCA to 256d retains ~99% of mutual information ([arxiv 2403.14001](https://arxiv.org/abs/2403.14001)).
2. **Use Matryoshka truncation**: nomic-embed-text-v1.5 supports truncation to 64d natively. Simply take the first N dimensions and L2-normalize. At 128d, MTEB score is 59.34 (only 3 points below 768d).
3. **Augment with hand-crafted features**: Article length, readability scores, source/feed, publication date, topic cluster, existing Claude-based score -- these are low-dimensional and complementary to embeddings.
4. **Regularization**: L1 (Lasso) for sparse feature selection, L2 (Ridge) for general shrinkage. Cross-validate the regularization strength.

### 4.4 Alternative: SetFit for Few-Shot Classification

SetFit ([HuggingFace](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit)) is a framework designed for exactly this scenario: fine-tuning sentence transformer embeddings with very few labeled examples. With only 8 labeled examples per class, SetFit can match models fine-tuned on 3,000+ examples. The approach:

1. Fine-tune the sentence transformer body using contrastive learning on your labeled pairs
2. Train a logistic regression head on the fine-tuned embeddings

For the binary "will this get highlighted?" task, SetFit could be trained on as few as 20-50 positive and 20-50 negative examples from your dataset. This would adapt the nomic-embed-text-v1.5 embedding space specifically to your engagement signal without needing to switch models.

---

## 5. Fine-Tuned vs. General-Purpose Embeddings

### 5.1 General Finding

General-purpose embeddings are sufficient for most downstream tasks but leave 5-15% performance on the table compared to task-specific fine-tuning ([Weaviate Blog](https://weaviate.io/blog/fine-tune-embedding-model), [Modal Blog](https://modal.com/blog/fine-tuning-embeddings)):

- Fine-tuning on as few as 6,300 samples can improve retrieval performance by ~7%.
- Domain-specific nuances (e.g., "highlight-worthy" vs. "generally interesting") are poorly captured by general-purpose models trained on web-scale contrastive pairs.
- The gap is largest on narrow domains and smallest on broad tasks.

### 5.2 When Fine-Tuning Makes Sense

For the reader engagement prediction task, fine-tuning the embedding model itself (not just training a head on top) makes sense when:

- You have 500+ labeled articles with highlight data (positive and negative examples)
- The general-purpose embeddings plateau in cross-validation performance
- You are willing to maintain a custom model artifact

### 5.3 Practical Fine-Tuning Options

| Approach | Data Needed | Effort | Expected Lift |
|---|---|---|---|
| **Frozen embeddings + linear probe** | 50-200 samples | Low | Baseline |
| **Frozen embeddings + XGBoost/SVR** | 100-500 samples | Low | +2-5% over linear |
| **SetFit contrastive fine-tuning** | 50-100 samples | Medium | +5-10% over frozen |
| **Full sentence transformer fine-tuning** | 500-2000 samples | High | +7-15% over frozen |
| **Custom embedding model training** | 5000+ samples | Very High | Domain-dependent |

Source: [SetFit Paper](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e/), [Databricks Blog](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning).

---

## 6. Specific Recommendations

### 6.1 Immediate (No Model Change Required)

1. **Export nomic-embed-text-v1.5 embeddings from Qdrant** for all articles with known highlight data.
2. **Apply PCA to reduce from 768d to 64-128d.** Fit PCA on the full corpus (not just labeled data) to learn the embedding subspace.
3. **Build a two-stage hurdle model:**
   - Stage 1: Logistic regression (L2-regularized) predicting any highlights (binary)
   - Stage 2: Ridge regression predicting log(1 + highlight_count) for articles with highlights > 0
4. **Combine with hand-crafted features:** article word count, Flesch-Kincaid readability, source feed, existing Claude score dimensions (quotability, surprise, argument, insight).
5. **Evaluate with leave-one-out or 5-fold cross-validation** given the small dataset. Report both the binary classification AUC and the regression RMSE/MAE separately.

### 6.2 Short-Term (If Baseline Underperforms)

6. **Try SetFit fine-tuning** on the binary "any highlights?" task using the nomic-embed-text-v1.5 model as the base. This adapts the embedding space to your specific engagement signal with minimal data.
7. **Try Matryoshka truncation at multiple dimensions** (64, 128, 256, 512, 768) and compare cross-validation performance. The optimal dimension is likely 128-256 for this dataset size.
8. **Upgrade to nomic-embed-text-v2-moe** if you want a drop-in improvement. Same 768d output, same Matryoshka support, better semantic quality from the MoE architecture, still fully open-source and self-hostable.

### 6.3 Medium-Term (If Dataset Grows to 1000+ Articles)

9. **Consider Jina-embeddings-v3** with the classification LoRA adapter. Its 82.58 classification accuracy on MTEB suggests it produces embeddings specifically optimized for predicting categorical properties of text, which is closest to engagement prediction.
10. **Consider Voyage 3.5** ($0.06/1M tokens) for its strong cross-domain performance and quantization-aware training. At 1024d with int8 precision, storage costs are minimal.
11. **Re-evaluate whether a direct LLM-based approach (current Claude scoring) outperforms the embedding+ML approach.** With more data, the ML approach should eventually win on calibration and consistency, but the crossover point depends on how well Claude's rubric captures the engagement signal.

### 6.4 What NOT to Do

- **Do not switch to a 3072d or 4096d model** (OpenAI text-embedding-3-large, Gemini, Qwen3-Embedding-8B) with the current dataset size. The additional dimensions will not help and will exacerbate overfitting.
- **Do not train a deep neural network head** on top of embeddings with only hundreds of samples. Use linear models, SVMs, or small tree ensembles.
- **Do not use raw embedding cosine similarity as the sole predictor.** Cosine similarity between an article and a "prototypical highlighted article" captures some signal but discards per-dimension information that regression models can exploit.

---

## 7. Summary of Model Rankings for This Use Case

For predicting reader engagement from text embeddings with a small labeled dataset:

| Rank | Model | Rationale |
|---|---|---|
| 1 | **nomic-embed-text-v1.5** (current) | Already embedded, 768d is sufficient, Matryoshka enables dimension reduction, free/self-hosted |
| 2 | **nomic-embed-text-v2-moe** | Drop-in upgrade, better semantics from MoE, same 768d, still free |
| 3 | **jina-embeddings-v3** | Classification LoRA adapter, highest MTEB classification score, 1024d |
| 4 | **voyage-3.5** | Best cost/performance ratio among commercial models, strong cross-domain |
| 5 | **Cohere embed-v4** | Useful only if multimodal (images in articles) becomes relevant |
| 6 | **text-embedding-3-large** | Adequate but overpriced for this task relative to alternatives |
| 7 | **gemini-embedding-001** | Top MTEB score but 2048 token limit is constraining for articles |
| 8 | **Qwen3-Embedding-8B** | Best absolute quality but 8B params is overkill for hundreds of articles |

---

## Sources

- [MTEB Leaderboard (HuggingFace)](https://huggingface.co/spaces/mteb/leaderboard)
- [nomic-embed-text-v1.5 Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [nomic-embed-text-v2-moe Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
- [Nomic Embed Text V2 Blog Post](https://www.nomic.ai/blog/posts/nomic-embed-text-v2)
- [Nomic Embed Technical Report (PDF)](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf)
- [Tang et al., "Understanding LLM Embeddings for Regression" (arxiv 2411.14708)](https://arxiv.org/abs/2411.14708)
- [jina-embeddings-v3 Paper (arxiv 2409.10173)](https://arxiv.org/abs/2409.10173)
- [Voyage 3.5 Announcement](https://blog.voyageai.com/2025/05/20/voyage-3-5/)
- [Voyage 3-Large Announcement](https://blog.voyageai.com/2025/01/07/voyage-3-large/)
- [Gemini Embedding GA Announcement](https://developers.googleblog.com/gemini-embedding-available-gemini-api/)
- [Qwen3-Embedding-8B Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [Qwen3 Embedding Blog Post](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Google text-embedding-005 / Gecko Paper (arxiv 2403.20327)](https://arxiv.org/abs/2403.20327)
- [Matryoshka Representation Learning (arxiv 2205.13147)](https://arxiv.org/abs/2205.13147)
- [Matryoshka Embeddings Introduction (HuggingFace Blog)](https://huggingface.co/blog/matryoshka)
- [Evaluating Unsupervised Dimensionality Reduction for Sentence Embeddings (arxiv 2403.14001)](https://arxiv.org/abs/2403.14001)
- [Enriching Tabular Data with LLM Embeddings (arxiv 2411.01645)](https://arxiv.org/html/2411.01645v1)
- [Combining XGBoost and Embeddings (MachineLearningMastery)](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)
- [SetFit: Sentence Transformer Fine-Tuning (HuggingFace)](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit)
- [SetFit: Outperforms GPT-3 on Few-Shot Classification (TDS)](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e/)
- [Fine-Tuning Embedding Models (Weaviate)](https://weaviate.io/blog/fine-tune-embedding-model)
- [Fine-Tuning Embeddings: Beating Proprietary Models (Modal)](https://modal.com/blog/fine-tuning-embeddings)
- [Zero-Inflated Regression Model Comparison (Springer)](https://link.springer.com/article/10.1186/s40488-021-00121-4)
- [Two-Fold ML for Zero-Inflated Data (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S0952197625003392)
- [ML-Based Hurdle Model for Zero-Inflated Data (MDPI)](https://www.mdpi.com/2076-3417/14/23/10790)
- [Embedding Models: OpenAI vs Gemini vs Cohere (AIMultiple)](https://research.aimultiple.com/embedding-models/)
- [Top Embedding Models on MTEB (Modal Blog)](https://modal.com/blog/mteb-leaderboard-article)
- [Best Embedding Models 2025 (Ailog)](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)
- [Improving Retrieval with Embedding Finetuning (Databricks)](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)
- [Embedding Dimensions Guide (Particula)](https://particula.tech/blog/embedding-dimensions-rag-vector-search)
