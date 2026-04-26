# 08 - Hybrid LLM and ML Scoring Approaches

## Executive Summary

Our current article scoring system uses Claude Sonnet 4 to produce four categorical dimension scores (Quotability, Surprise Factor, Argument Quality, Applicable Insight -- 0-25 each) but shows weak correlation with actual engagement (highlights/saves). This report investigates hybrid architectures that combine LLM-based scoring with embedding-based ML models to improve prediction accuracy. The research literature from 2024-2026 strongly supports hybrid approaches: LLM-derived features consistently rank among the most impactful predictors when fed into gradient-boosted ensemble models like XGBoost ([Koc et al., 2024](https://arxiv.org/abs/2411.01645)), and knowledge distillation can compress LLM scoring capabilities into models that are 4,000x smaller with only marginal accuracy loss ([Lee et al., 2024](https://arxiv.org/abs/2312.15842)). The recommended path forward is a staged pipeline: retain LLM scoring during a training data collection phase, train a lightweight ML model on LLM scores + embeddings + engagement signals, then transition to the ML model for inference with periodic LLM recalibration.

---

## 1. Using LLM Dimension Scores as Features in ML Models

### The Core Idea

Rather than treating the LLM's composite score as the final output, treat each of the four dimension scores (Quotability, Surprise, Argument Quality, Applicable Insight) as input features to a downstream ML model. The ML model then learns the mapping from these features -- potentially combined with other signals -- to actual engagement outcomes.

### Why This Works

LLM scores encode semantic judgments that are difficult to capture with traditional features or even embeddings alone. When used as features in gradient-boosted models, they provide a form of "expert knowledge injection." A 2024 ablation study on enriching tabular data with LLM-derived features found that XGBoost and CatBoost classifiers showed the most pronounced improvements when augmented with LLM-derived features, and that these features "frequently rank among the most impactful for the predictions" via SHAP analysis ([Koc et al., 2024](https://arxiv.org/abs/2411.01645)).

### Practical Implementation

```
Input Features for XGBoost/LightGBM:
├── LLM Dimension Scores (4 features)
│   ├── quotability_score (0-25)
│   ├── surprise_score (0-25)
│   ├── argument_quality_score (0-25)
│   └── applicable_insight_score (0-25)
├── Article Embeddings (N-dimensional, e.g., 384 or 768)
│   └── from sentence-transformers or voyage-3
├── Metadata Features
│   ├── word_count
│   ├── source_domain
│   ├── category/tags
│   └── reading_time
└── Historical Features
    ├── source_avg_engagement
    ├── topic_avg_engagement
    └── user_highlight_rate_by_category
```

A key practical consideration: XGBoost expects numerical input in a flat tabular format. Embedding vectors must be concatenated as individual features (e.g., `emb_0, emb_1, ..., emb_383`) alongside the scalar LLM scores and metadata ([MachineLearningMastery, 2024](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)).

### Expected Gains

The hybrid approach of LLM features + embeddings in ensemble classifiers has shown consistent improvements over either approach alone, particularly on datasets with class imbalance or limited features -- both characteristics of our engagement prediction task where high-engagement articles are the minority class ([Koc et al., 2024](https://arxiv.org/abs/2411.01645)).

### XGBoost vs Neural Network

For our scale (hundreds to low thousands of articles), XGBoost/LightGBM is likely the better choice over a neural network:
- More robust with small datasets
- Built-in handling of missing values (useful when LLM scores are unavailable for older articles)
- Native feature importance via SHAP for interpretability
- No GPU requirement for training or inference

A shallow neural network (2-3 layers) becomes worthwhile only once the training set exceeds ~10K labeled examples with engagement signals.

---

## 2. Knowledge Distillation: Training a Small Model to Approximate LLM Scoring

### Concept

Knowledge distillation (KD) trains a compact "student" model to replicate the behavior of a larger "teacher" model (in our case, Claude Sonnet 4). The student learns not just from hard labels but from the teacher's soft output distribution, which encodes richer information about the relative relationships between scoring dimensions.

### Relevant Research

The most directly applicable study is [Lee et al. (2024)](https://arxiv.org/abs/2312.15842), which distilled fine-tuned BERT models for automatic scoring of science assessments into a compact neural network. Their results:

| Model | Parameters | Inference Speed | Accuracy |
|-------|-----------|----------------|----------|
| Teacher (SciEdBERT) | 114M | Baseline | Baseline |
| Student (LSTM + Dense) | 0.03M | 10x faster | Within 3% of teacher |
| TinyBERT | ~15M | 3x faster | 2.5% below student |

The student model was **4,000x smaller** in parameters and **10x faster** in inference while maintaining competitive accuracy.

### Two-Phase Distillation for Our Use Case

**Phase 1: Distill LLM Scoring Capability**
- Use Claude's 4-dimension scores as soft labels for a student model
- Student input: article text (or embeddings thereof)
- Student output: predicted 4-dimension scores
- Loss function: MSE between student and teacher scores, possibly with temperature scaling

**Phase 2: Fine-tune on Engagement Signal**
- Take the Phase 1 student model and add an engagement prediction head
- Fine-tune on actual highlight/save data
- The student now produces both LLM-approximate scores AND engagement predictions
- This addresses the weak correlation problem: the model learns where the LLM's judgment aligns with actual user behavior and where it diverges

### Active Distillation to Reduce Costs

A 2025 paper, "LLM on a Budget" ([Sebok et al., 2025](https://arxiv.org/abs/2511.11574)), introduced M-RARU (Multi-class Randomized Accept/Reject Uncertainty Sampling), which selects only the most informative data points for LLM labeling. Their results showed **up to 80% reduction in sample requirements** compared to random sampling while maintaining classification accuracy. Student models tested included SVM, Random Forest, GBDT (gradient boosting), and DistilBERT.

Applied to our case: rather than scoring every article with Claude, we score a strategically selected subset and use those to train the student model. The student then scores the rest.

---

## 3. LLM-as-Feature-Extractor: Structured Features for ML Consumption

### Beyond Numeric Scores

Instead of limiting Claude to four numeric scores, we can prompt it to extract a richer set of structured features designed specifically for downstream ML consumption:

```json
{
  "quotability_score": 18,
  "surprise_score": 12,
  "argument_quality_score": 22,
  "applicable_insight_score": 15,
  "content_type": "research_synthesis",
  "rhetorical_strategy": "data_driven_argument",
  "topic_novelty": "incremental_advance",
  "writing_style": "accessible_technical",
  "key_claim_strength": "well_supported",
  "practical_takeaway_count": 3,
  "data_citation_density": "high",
  "contrarian_stance": false,
  "domain_tags": ["machine_learning", "production_systems"],
  "estimated_expertise_required": "intermediate",
  "emotional_valence": "neutral_informative"
}
```

### The FeatLLM Pattern

[Han et al. (2024)](https://arxiv.org/abs/2404.09491) introduced FeatLLM, an in-context learning framework that uses LLMs as feature engineers for tabular prediction tasks. The key insight: rather than sending every data point through the LLM at inference time, FeatLLM uses the LLM to **generate feature extraction rules** (as program code) from a few examples. These rules then produce features for the entire dataset without further LLM calls.

Applied to our domain: Claude could analyze a sample of high-engagement vs low-engagement articles and output programmatic feature extraction rules like:
- "Articles with >3 specific data points per 1000 words score higher on quotability"
- "Articles that reference >2 competing frameworks score higher on surprise"

These rules become deterministic feature extractors that run without LLM inference.

### LLM-FE: Automated Feature Engineering

A 2025 follow-up, [LLM-FE](https://arxiv.org/html/2503.14434v1), extends this concept by using LLMs as evolutionary optimizers for feature engineering on tabular data, iteratively generating and refining feature transformations.

### Practical Architecture

```
Article Text
    │
    ├──→ [Embedding Model] → 384-dim vector → ML features
    │
    ├──→ [Claude Structured Extraction] → 15+ categorical/numeric features
    │         (run at ingestion time, cached in DB)
    │
    ├──→ [Deterministic Feature Extractors] → word_count, reading_level, etc.
    │         (derived from FeatLLM-style rules)
    │
    └──→ [XGBoost] → engagement_prediction
```

---

## 4. Ensemble Approaches: Combining LLM Score with ML Score

### Stacking Ensemble

The most natural ensemble approach is stacking (also called blended ensembling):

**Level 0 (Base Models):**
- Model A: LLM composite score (current system)
- Model B: Embedding-based ML model (XGBoost on article embeddings + metadata)
- Model C: Collaborative filtering signal (engagement patterns from similar users/articles)

**Level 1 (Meta-learner):**
- A simple model (logistic regression or shallow gradient booster) learns optimal weights for combining Level 0 predictions

This approach is well-supported by the ensemble LLM literature. A 2025 survey on Ensemble Large Language Models ([Xu et al., 2025](https://www.mdpi.com/2078-2489/16/8/688)) catalogs methods ranging from simple majority voting to learned combination functions. A key finding: models from the same family tend to share failure modes, so diversity between ensemble members (LLM vs embedding-ML) provides more benefit than combining multiple LLM variants.

### Cascade Architecture

A cost-efficient alternative is a cascade (or "router") approach:

```
Article arrives
    │
    ├──→ [Fast ML Model] → confidence score
    │
    ├── If confidence > threshold_high → Use ML prediction (skip LLM)
    ├── If confidence < threshold_low → Use LLM scoring (expensive but reliable)
    └── If in between → Average ML + LLM scores
```

This is analogous to the multi-stage inference pipeline pattern described in [Agrawal et al. (2025)](https://arxiv.org/abs/2504.09775), where lighter models handle easy cases and heavier models are invoked only when needed. In production recommendation systems, this pattern is standard: YouTube's User Behavior Service pre-computes embeddings asynchronously and caches them, enabling experimentation with larger models without latency constraints ([Yan, 2025](https://eugeneyan.com/writing/recsys-llm/)).

### Weighted Score Fusion

The simplest ensemble: a learned weighted average.

```python
final_score = w1 * llm_composite_score + w2 * ml_predicted_score
```

Where `w1` and `w2` are learned from a validation set of articles with known engagement outcomes. This requires no architectural changes -- just a calibration step.

---

## 5. Cost-Benefit: When Is the LLM Call Worth It?

### Cost Comparison

| Approach | Cost per Article | Latency | Notes |
|----------|-----------------|---------|-------|
| Claude Sonnet 4 scoring | ~$0.003-0.01 | 2-5s | Depends on article length |
| Claude Haiku 4.5 scoring | ~$0.0005-0.002 | 0.5-1s | Cheaper, slightly less capable |
| Embedding generation (voyage-3 / text-embedding-3-small) | ~$0.00001-0.00005 | 50-100ms | Orders of magnitude cheaper |
| XGBoost inference | ~$0.000001 | <1ms | Effectively free |
| Combined: Embedding + XGBoost | ~$0.00005 | ~100ms | Near-free at scale |

Sources: [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI Embedding Pricing](https://costgoat.com/pricing/openai-embeddings), [Azimbaev, 2025](https://medium.com/@alex-azimbaev/embedding-models-in-2025-technology-pricing-practical-advice-2ed273fead7f).

Embedding models are roughly **100-1000x cheaper** per article than LLM scoring calls. At our current volume (~100-500 articles/day), the LLM cost is manageable (~$0.50-5.00/day for Sonnet, or ~$0.05-1.00/day for Haiku). But the latency difference matters more than cost: embedding + XGBoost inference completes in ~100ms vs 2-5 seconds for an LLM call.

### Cost Optimization for LLM Scoring

If retaining LLM scoring, two Anthropic features can significantly reduce costs:

1. **Prompt Caching**: Cached reads cost just 10% of normal input token rates. Since our scoring prompt (system instructions, rubric, examples) is identical across all articles, only the article content varies. This can reduce costs by ~68% ([Anthropic, 2024](https://www.anthropic.com/news/prompt-caching)).

2. **Batch API**: Provides a 50% discount on both input and output tokens for asynchronous processing. Combined with prompt caching, total savings can exceed 80% ([Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)).

### Decision Framework

| Scenario | Recommendation |
|----------|---------------|
| Building initial training set | Use LLM scoring on all articles (investment phase) |
| Mature model with >1K engagement-labeled articles | Use ML model for scoring; LLM for periodic recalibration |
| High-confidence ML predictions | Skip LLM entirely |
| Low-confidence ML predictions | Route to LLM for second opinion |
| New content domains / distribution shift | Temporarily increase LLM scoring rate |

---

## 6. Leveraging LLMs for Generating Training Data and Labels

### The Cold-Start Problem

Our core challenge: we need engagement labels (highlights/saves) to train a model, but engagement data is sparse and noisy. LLMs can help bootstrap this process in several ways.

### Approach 1: LLM as Weak Supervisor (Snorkel Pattern)

The [Snorkel](https://arxiv.org/abs/1711.10160) weak supervision framework treats noisy label sources as "labeling functions" whose outputs are combined via a learned generative model. LLMs can serve as labeling functions:

- **LF1**: Claude rates article quality on 4 dimensions (our current system)
- **LF2**: Claude predicts "would a thoughtful reader highlight passages from this article?" (binary)
- **LF3**: Claude identifies specific highlight-worthy passages and counts them
- **LF4**: Heuristic rules (source reputation, article length, topic match)

Snorkel's label model then aggregates these noisy signals into probabilistic training labels. This approach showed an average 19.5% error reduction on the WRENCH weak supervision benchmark when using LLM-based labeling functions ([Zhang et al., 2024](https://dl.acm.org/doi/10.1145/3617130)).

Snorkel's modern implementation, [Alfred](https://snorkel.ai/blog/alfred-data-labeling-with-foundation-models-and-weak-supervision/), directly supports defining labeling rules as natural language prompts to foundation models.

### Approach 2: LLM-Generated Synthetic Training Data

Rather than labeling real articles, use Claude to generate synthetic article summaries with known engagement properties:

```
"Generate 50 article summaries that a reader interested in [user's topics]
would find highly quotable. Then generate 50 that would be informative but
not highlight-worthy. Include realistic metadata."
```

This synthetic data augments the real engagement data, particularly for underrepresented classes (high-engagement articles are rare).

### Approach 3: LLM Label Quality and Agreement

Research from [Refuel.ai (2024)](https://www.refuel.ai/blog-posts/llm-labeling-technical-report) found that GPT-4 achieved 88.4% agreement with ground truth labels, compared to 86% for skilled human annotators, while being ~20x faster and ~7x cheaper. For our scoring task, this suggests LLM-generated engagement predictions can serve as reasonable proxy labels when actual engagement data is unavailable.

However, LLM labels have systematic biases. A Human-LLM collaborative annotation study ([Li et al., 2024](https://dl.acm.org/doi/10.1145/3613904.3641960)) developed a verifier model that identifies potentially erroneous LLM labels, focusing human correction effort where it matters most. For our pipeline, this means: use Claude to label the bulk of articles, but verify a strategic sample against actual engagement data to detect and correct systematic biases.

### Approach 4: Knowledge Distillation as Data Generation

From [Sebok et al. (2025)](https://arxiv.org/abs/2511.11574): use active learning to select the most informative articles for LLM scoring, then train a student model on the resulting labels. This achieves 80% reduction in required LLM calls while maintaining accuracy. The student models tested (including GBDT/XGBoost) performed well in this regime.

---

## 7. Practical Pipeline Architectures

### Architecture A: LLM-Enriched Feature Pipeline (Recommended Starting Point)

This is the lowest-risk, highest-information architecture. It preserves existing LLM scoring while adding ML prediction as a parallel signal.

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                              │
│  Article ──→ [Readwise API] ──→ Raw Article                 │
│                                      │                       │
│                    ┌─────────────────┼──────────────────┐    │
│                    │                 │                   │    │
│                    ▼                 ▼                   ▼    │
│            [Claude Sonnet]   [Embedding Model]   [Metadata]  │
│            4-dim scores      384/768-dim vec     word_count   │
│            + structured      (voyage-3 or       source_domain│
│              features        all-MiniLM-L6)     read_time    │
│                    │                 │                   │    │
│                    └────────┬────────┘───────────────────┘    │
│                             │                                │
│                             ▼                                │
│                     [Feature Store / DB]                      │
│                     (all features cached)                     │
│                             │                                │
│                             ▼                                │
│                     [XGBoost Model]                           │
│                     engagement_prediction                     │
│                             │                                │
│                             ▼                                │
│                     [Scoring & Ranking]                       │
│                     final_score = f(llm_score, ml_score)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                          │
│                                                              │
│  [Feature Store] + [Engagement Events] ──→ [Training Job]    │
│       (LLM scores, embeddings, metadata)    (weekly retrain) │
│                                                    │         │
│                                                    ▼         │
│                                            [Updated XGBoost] │
│                                            + SHAP analysis   │
└─────────────────────────────────────────────────────────────┘
```

**Advantages**: Preserves existing scoring, adds ML signal, enables A/B comparison, collects training data.

**Implementation effort**: Medium. Requires adding embedding generation at ingestion and training infrastructure.

### Architecture B: Distillation Pipeline (Cost Optimization Phase)

Once sufficient training data is collected (>1K articles with engagement labels), transition to a distilled model.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (ONE-TIME)                        │
│                                                              │
│  Historical Articles + LLM Scores + Engagement Data          │
│                         │                                    │
│                         ▼                                    │
│              [Distillation Training]                          │
│              Teacher: Claude scores (soft labels)             │
│              Student: DistilBERT or small transformer         │
│              + engagement fine-tuning head                    │
│                         │                                    │
│                         ▼                                    │
│              [Distilled Scoring Model]                        │
│              Outputs: 4-dim scores + engagement_pred          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                         │
│                                                              │
│  Article ──→ [Distilled Model] ──→ scores + engagement_pred  │
│                                                              │
│  Cost: ~$0.0001/article  |  Latency: ~50ms                  │
│  (vs $0.003-0.01 and 2-5s for Claude)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RECALIBRATION (PERIODIC)                   │
│                                                              │
│  Sample 10-20% of articles ──→ [Claude Scoring]              │
│                                       │                      │
│  Compare with distilled model ──→ [Drift Detection]          │
│                                       │                      │
│  If drift > threshold ──→ [Retrain distilled model]          │
└─────────────────────────────────────────────────────────────┘
```

**Advantages**: 100-1000x cost reduction, 20-50x latency reduction, self-contained inference.

**Risk**: Model drift if content distribution changes significantly.

### Architecture C: Cascade Router (Balanced Cost/Quality)

A hybrid that uses the ML model as a fast first pass and routes uncertain cases to the LLM.

```
Article ──→ [Embedding + XGBoost] ──→ confidence score
                                           │
                    ┌──────────────────────┼──────────────────┐
                    │                      │                   │
              confidence > 0.8      0.4 < conf < 0.8    conf < 0.4
                    │                      │                   │
                    ▼                      ▼                   ▼
              Use ML score          Average ML + LLM       Use LLM score
              (skip LLM)           (both contribute)       (full analysis)
                    │                      │                   │
                    └──────────────────────┴───────────────────┘
                                           │
                                           ▼
                                    Final Score
```

In production recommendation systems, this cascade pattern is standard. It reduces LLM usage by 60-80% while maintaining quality on difficult cases.

---

## 8. Specific Recommendations for Reader Triage

### Phase 1: Data Collection and Feature Enrichment (Weeks 1-4)

1. **Add embedding generation at article ingestion time.** Use `voyage-3-lite` or `all-MiniLM-L6-v2` (free, local). Store embeddings alongside existing LLM scores in the database.

2. **Expand Claude's structured output.** In addition to the four dimension scores, have Claude extract 10-15 categorical/binary features (content type, rhetorical strategy, contrarian stance, data density, etc.) as described in Section 3. Use structured output / JSON mode to ensure parseable responses.

3. **Instrument engagement tracking.** Ensure every highlight, save, and "mark as read without highlights" event is logged with article ID and timestamp. Negative signals (skipped articles, read-but-not-highlighted) are as valuable as positive ones.

4. **Enable prompt caching and Batch API.** If not already in use, enable both for the scoring pipeline. Expected cost reduction: 60-80%.

### Phase 2: Initial ML Model Training (Weeks 4-8)

5. **Train an XGBoost model** with the following feature groups:
   - LLM dimension scores (4 features)
   - LLM extracted categorical features (10-15 features, one-hot encoded)
   - Article embeddings (384 features if using MiniLM, or apply PCA to reduce dimensionality)
   - Metadata features (word count, source, reading time, etc.)
   - Target variable: binary engagement (highlighted = 1, read-without-highlight = 0)

6. **Run SHAP analysis** on the trained model to understand which features drive engagement predictions. This will reveal whether the LLM dimension scores actually contribute predictive power beyond embeddings alone, and which dimensions matter most.

7. **A/B compare** the ML model's ranking against the current LLM-only ranking. Use the ML model's predictions as a secondary sort signal alongside the existing LLM composite score.

### Phase 3: Model Refinement and Transition (Weeks 8-16)

8. **Based on SHAP results, decide the architecture:**
   - If LLM scores are among top features: Keep hybrid (Architecture A) or move to cascade (Architecture C)
   - If embeddings dominate and LLM scores add little: Move to distillation (Architecture B) to eliminate ongoing LLM costs
   - If neither is great alone but both together work: Keep the stacking ensemble

9. **Implement active learning.** Use the ML model's uncertainty to select which new articles should receive LLM scoring vs ML-only scoring. Target 80% reduction in LLM calls per [Sebok et al. (2025)](https://arxiv.org/abs/2511.11574).

10. **Set up weekly retraining.** As engagement data accumulates, retrain the ML model weekly. Monitor for concept drift by comparing SHAP feature importances across model versions.

### Phase 4: Production Optimization (Ongoing)

11. **Consider distilling to a local model.** If the hybrid approach works well, train a small DistilBERT or LSTM-based student model that approximates the full pipeline. This eliminates all API dependencies for scoring.

12. **Implement the FeatLLM pattern.** Periodically (monthly), have Claude analyze the latest batch of high-engagement vs low-engagement articles and generate updated feature extraction rules. These deterministic rules can supplement or replace some of the LLM-extracted features.

### Summary Decision Matrix

| Metric | Current (LLM Only) | Phase 2 (Hybrid) | Phase 3+ (Optimized) |
|--------|-------------------|-------------------|---------------------|
| Cost/article | $0.003-0.01 | $0.003-0.01 (same, adding ML is cheap) | $0.0001-0.002 (80%+ reduction) |
| Latency | 2-5s | 2-5s (ML adds <100ms) | 100ms-1s (most skip LLM) |
| Engagement correlation | Weak | Expected moderate | Expected strong |
| Interpretability | Dimension scores | SHAP + dimension scores | SHAP analysis |
| Cold-start capability | Full (any article) | Full | Requires periodic LLM recalibration |

---

## Sources

### Research Papers

- Koc, E., et al. (2024). "Enriching Tabular Data with Contextual LLM Embeddings: A Comprehensive Ablation Study for Ensemble Classifiers." [arXiv:2411.01645](https://arxiv.org/abs/2411.01645)
- Lee, M., et al. (2024). "Knowledge Distillation of LLM for Automatic Scoring of Science Education Assessments." [arXiv:2312.15842](https://arxiv.org/abs/2312.15842)
- Sebok, M., et al. (2025). "LLM on a Budget: Active Knowledge Distillation for Efficient Classification of Large Text Corpora." [arXiv:2511.11574](https://arxiv.org/abs/2511.11574)
- Han, S., et al. (2024). "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning (FeatLLM)." [arXiv:2404.09491](https://arxiv.org/abs/2404.09491)
- Xu, Y., et al. (2025). "Ensemble Large Language Models: A Survey." [MDPI Information 16(8):688](https://www.mdpi.com/2078-2489/16/8/688)
- Zhang, J., et al. (2024). "Language Models in the Loop: Incorporating Prompting into Weak Supervision." [ACM/IMS Journal of Data Science](https://dl.acm.org/doi/10.1145/3617130)
- Li, M., et al. (2024). "Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels." [CHI 2024](https://dl.acm.org/doi/10.1145/3613904.3641960)
- Atkinson, J. & Palma, D. (2025). "An LLM-based hybrid approach for enhanced automated essay scoring." [Scientific Reports / Nature](https://www.nature.com/articles/s41598-025-87862-3)
- Ratner, A., et al. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." [arXiv:1711.10160](https://arxiv.org/abs/1711.10160)
- Agrawal, N., et al. (2025). "Understanding and Optimizing Multi-Stage AI Inference Pipelines (HERMES)." [arXiv:2504.09775](https://arxiv.org/abs/2504.09775)
- Wang, L., et al. (2025). "What Matters in LLM-Based Feature Extractor for Recommender?" [arXiv:2509.14979](https://arxiv.org/html/2509.14979)
- Chen, G., et al. (2025). "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers." [arXiv:2503.14434](https://arxiv.org/html/2503.14434v1)
- Gu, Y., et al. (2023). "MiniLLM: Knowledge Distillation of Large Language Models." [arXiv:2306.08543](https://arxiv.org/pdf/2306.08543)
- Liu, H., et al. (2025). "Training an LLM-as-a-Judge Model (Themis): Pipeline, Insights, and Practical Lessons." [arXiv:2502.02988](https://arxiv.org/abs/2502.02988)
- Su, J., et al. (2025). "LLM Embeddings for Deep Learning on Tabular Data." [arXiv:2502.11596](https://arxiv.org/abs/2502.11596)

### Industry Resources

- Yan, E. (2025). "Improving Recommendation Systems & Search in the Age of LLMs." [eugeneyan.com](https://eugeneyan.com/writing/recsys-llm/)
- Anthropic. (2024). "Prompt caching with Claude." [anthropic.com](https://www.anthropic.com/news/prompt-caching)
- Anthropic. (2026). "Pricing." [platform.claude.com](https://platform.claude.com/docs/en/about-claude/pricing)
- Snorkel AI. (2024). "Alfred: Data labeling with foundation models and weak supervision." [snorkel.ai](https://snorkel.ai/blog/alfred-data-labeling-with-foundation-models-and-weak-supervision/)
- Refuel.ai. (2024). "LLMs can structure data as well as humans, but 100x faster." [refuel.ai](https://www.refuel.ai/blog-posts/llm-labeling-technical-report)
- MachineLearningMastery. (2024). "Combining XGBoost and Embeddings: Hybrid Semantic Boosted Trees?" [machinelearningmastery.com](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)
- MachineLearningMastery. (2024). "Feature Engineering with LLM Embeddings: Enhancing Scikit-learn Models." [machinelearningmastery.com](https://machinelearningmastery.com/feature-engineering-with-llm-embeddings-enhancing-scikit-learn-models/)
- Azimbaev, A. (2025). "Embedding Models in 2025 -- Technology, Pricing & Practical Advice." [Medium](https://medium.com/@alex-azimbaev/embedding-models-in-2025-technology-pricing-practical-advice-2ed273fead7f)
- Raschka, S. (2025). "The State of LLMs 2025." [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
