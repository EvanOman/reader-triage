# Literature Review: Binary-Weighted LLM Scoring

## 1. Binary vs. Multi-Class vs. Numeric Judgments in LLM Evaluation

### The Core Problem: Scale Instability

Research consistently demonstrates that LLMs are more reliable at binary (yes/no) judgments than at assigning numeric scores or making fine-grained categorical distinctions.

**Arize AI's empirical testing** (2025) compared binary, categorical (A-E rubric), and numeric (1-10) scoring formats across GPT-5-nano, Claude Opus, and Qwen3. Their findings were stark:

- Numeric scores produced "plateaus, discontinuous jumps, and long stretches where passages with very different error rates landed on the same number"
- All models "plateaued quickly after only a small amount of corruption, with scores saturating and collapsing into narrow bands"
- Score distributions were "very bimodal, suggesting binary values might give more consistent results"
- Binary labels "consistently separate clean from corrupted passages, with low variance across runs"
- Their conclusion: "discrete labels generalize more broadly, while numeric scores are only useful in tightly controlled and calibrated contexts"

Source: [Testing Binary vs Score Evals on the Latest Models](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/)

**Evidently AI** summarizes the practitioner consensus: "Binary evaluations, like 'Polite' vs. 'Impolite,' tend to be more reliable and consistent for both LLMs and human evaluators. It's easier to get accurate results with two simple choices rather than trying to decide if a specific response scores 73 vs. 82."

Source: [LLM-as-a-judge: a complete guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)

**Databricks** recommends "an integer scale of 0-3 or 0-4, with binary grading working well for simple metrics."

Source: [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)

### "Likert or Not" (Godfrey et al., 2025)

This paper on fine-grained ordinal relevance judgments found that the gap between pointwise scoring and listwise ranking shrinks with larger ordinal label spaces. However, the paper's focus on information retrieval relevance differs from our content quality use case. The key insight for us: absolute pointwise scoring can work when the label space is carefully designed, but simpler is generally more robust.

Source: [arxiv.org/abs/2505.19334](https://arxiv.org/abs/2505.19334)

### Evaluating the Consistency of LLM Evaluators (COLING 2025)

Found that output scores are less consistent across judges when moving from low-precision to high-precision scales. Cohen's Kappa for binary/categorical data showed better inter-evaluator agreement than correlation metrics on ordinal data.

Source: [aclanthology.org/2025.coling-main.710.pdf](https://aclanthology.org/2025.coling-main.710.pdf)

---

## 2. CheckEval: The Definitive Checklist-Based Binary Framework

**CheckEval** (Lee et al., 2024, published EMNLP 2025) is the most directly relevant framework for our purposes. It explicitly replaces Likert-scale LLM evaluation with decomposed binary questions.

### Methodology

Three-stage process:
1. **Define dimensions**: Humans select evaluation dimensions and sub-dimensions
2. **Generate checklist**: Create Boolean yes/no questions through seed writing, then augment via question diversification (exploring different perspectives), question elaboration (making questions more specific), and question filtering (removing redundant questions)
3. **Evaluate**: Model answers binary questions; final score = proportion of "yes" responses

### Key Results

- Improved average inter-evaluator agreement by **0.45** compared to G-Eval's Likert-scale approach
- Achieved Krippendorff's alpha of **0.67** on SummEval and Topical-Chat -- comparable to human rater agreement (~0.7)
- Dramatically reduced score variance across different evaluator models

### Why It Works

"Questions within the checklist are formatted in a Boolean QA style, allowing for binary responses (Yes/No), which improves the precision and clarity of evaluation, facilitating a more straightforward interpretation and offering an advantage over traditional 1-5 scale ratings."

The authors note that existing methods "achieved decent human correlation but showed low inter-evaluator agreement and high variance in evaluation results." Binary decomposition fixes this by making each judgment atomic and unambiguous.

Source: [CheckEval paper](https://arxiv.org/abs/2403.18771)

---

## 3. G-Eval and Token Probability Weighting

**G-Eval** (Liu et al., 2023) introduced using chain-of-thought to decompose evaluation criteria, then weighting scores by token-level log probabilities.

For our binary scoring approach, the relevant insight is: "Each step is judged by the model with a yes, no, or unsure answer, and final scores are computed by weighting these responses using token-level log probabilities." This produces more fine-grained, continuous scores from binary judgments.

However, log probability access requires specific API support (available via Anthropic's API but adds complexity). The simpler CheckEval-style aggregation (count of yes responses / total questions) may be sufficient for our use case.

Source: [G-Eval: The Definitive Guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)

---

## 4. QAG (Question Answer Generation) Scoring

**DeepEval's QAG approach** demonstrates a production-ready pattern for binary scoring:

- Uses "confined answers (usually either a 'yes' or 'no') to close-ended questions to compute a final metric score"
- "It is reliable because it does NOT use LLMs to directly generate scores. Instead, QAG constrains verdicts to a binary 'yes' or 'no' for close-ended questions, with very little room for stochasticity"
- Supports 'idk' as a third option when information is insufficient

This validates the core thesis: constraining LLM output to binary decisions produces more reliable evaluation than asking for scores.

Source: [DeepEval LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

---

## 5. DEBATE: Devil's Advocate for Score Inflation

**DEBATE** (Kim et al., 2024, ACL Findings) directly addresses score inflation through multi-agent debate:

### Framework

Three agents -- Commander, Scorer, and Critic -- where the Critic serves as Devil's Advocate:
- The Critic "criticizes the score as much as possible" and assesses whether evaluations are "accurate"
- Iterative debate continues until the Critic finds no issues or max iterations are reached

### Results

- Outperformed G-Eval by 6.4 percentage points (Spearman) on SummEval
- 11.9 points higher Pearson correlation on Topical-Chat
- "Single-agent LLM evaluation approaches introduce an inherent performance limitation due to biases in LLM agent responses"

### Relevance to Our Problem

While we likely won't implement multi-agent debate (cost and latency), the insight about designing adversarial/critical questions is directly applicable. We can embed the devil's advocate concept into our binary questions themselves -- questions designed to identify weaknesses rather than strengths.

Source: [DEBATE paper](https://arxiv.org/abs/2405.09935)

---

## 6. Criteria-Eval: Production Binary Checklist Scoring

**Criteria-Eval** (Samaya AI, 2025) demonstrates binary checklist scoring in a production financial analysis context:

- Uses "objective, yes-or-no checklists written by domain experts"
- "For every criterion, the LLM judge returns a binary judgment: Pass if the answer fully satisfies the criterion, or Fail"
- "The final score is simply the percentage of criteria met"
- Evaluated on 3,000 queries covering "straightforward factual queries to complex strategic analyses"

This validates the approach for complex, domain-specific content evaluation at scale.

Source: [Criteria-Eval blog post](https://samaya.ai/blog/criteria-eval)

---

## 7. ARES: Binary Classification for RAG Evaluation

**ARES** (Stanford, 2023) uses binary classification for evaluating retrieval-augmented generation:

- "For each metric, a separate LLM with a binary classifier head is fine-tuned to classify positive and negative examples"
- Three binary classifiers: context relevance, answer faithfulness, answer relevance
- Uses Prediction-Powered Inference (PPI) for statistical confidence

The key insight: even complex evaluation can be decomposed into multiple independent binary decisions, each with its own classifier/question.

Source: [ARES paper](https://arxiv.org/abs/2311.09476)

---

## 8. AlpacaEval: Binary Preference at Scale

**AlpacaEval** uses binary preference judgments (win/loss) with weighted aggregation:

- "An auto-annotator returns a preference judgment (a 'win' or 'loss') for the evaluated model"
- Win rates are computed as averages of binary probabilities across the dataset
- Length-controlled win rates address the "length bias" where longer responses are favored
- Achieves Spearman correlation of **0.98** with Chatbot Arena human preferences

The length-bias correction is analogous to our ceiling effect problem: certain surface features (length, vocabulary) inflate scores independent of actual quality.

Source: [AlpacaEval GitHub](https://github.com/tatsu-lab/alpaca_eval)

---

## 9. Practitioner Consensus: Hamel Husain and Eugene Yan

### Hamel Husain

Perhaps the strongest advocate for binary evaluation in production:

- "A common mistake is straying from binary pass/fail judgments. If your evaluations consist of metrics that LLMs score on a 1-5 scale or any other scale, you're doing it wrong."
- "The distinction between a '3' and a '4' is often subjective and inconsistent across different reviewers, leading to noisy, unreliable data."
- "Binary decisions force clarity and compel a domain expert to define a clear line between acceptable and unacceptable."
- Recommends combining binary judgment with detailed critique text for interpretability
- Reports achieving >90% human-LLM agreement after 3 iterations of prompt refinement

Source: [Hamel's LLM Judge Guide](https://hamel.dev/blog/posts/llm-judge/)

### Eugene Yan

- "I tend to be skeptical of correlation metrics... where possible, I have my evaluators return binary outputs"
- Binary enables straightforward classification metrics (recall, precision) vs. unreliable correlation metrics
- Recommends having at least 50-100 failures out of 200+ total samples for balanced evaluation
- Notes that pairwise comparisons outperform direct scoring for subjective assessments

Source: [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)

---

## 10. LLM Sycophancy and Positivity Bias

Understanding why scores cluster high requires understanding LLM sycophancy:

- LLMs "consistently exhibit high rates of social sycophancy: on open-ended questions, they preserve face 47% more than humans" (ELEPHANT study, 2025)
- "Feedback positivity of 85%" means the LLM gives positive feedback 85% of the time when the user implies preference
- Root cause: "biases in training data, limitations in current training techniques such as RLHF, and fundamental challenges in defining and optimizing for truthfulness"

### Mitigation relevant to scoring:
- **Chain-of-thought prompting**: Requiring reasoning before scoring reduces overly positive skew
- **Direct instruction**: "Please provide direct advice, even if critical" is the most effective single mitigation
- **Binary decomposition**: Constraining to yes/no reduces the degrees of freedom available for sycophantic inflation

Sources:
- [Sycophancy in LLMs](https://arxiv.org/html/2411.15287v1)
- [Detecting Sycophancy Bias](https://huggingface.co/blog/Rakshit122/sycophantic-ai)

---

## 11. GoDaddy's Rubrics as Rewards (RaR) Framework

GoDaddy's production calibration approach for LLM judges uses:

- **Tiered verification criteria**: organized by importance (essential, important, optional, pitfall)
- **Yes/no verification questions** at each tier
- **Domain-specific items** from expert guidance
- **Multiple aggregation methods**: explicit (fixed checklist formulas), implicit (LLM weighs holistically), few-shot prompting, and ensemble approaches
- Finding: implicit aggregation "outperforms asking LLMJ to directly produce quality scores"

Source: [Calibrating Scores of LLM-as-a-Judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge)

---

## 12. Item Discrimination Theory (IDGen)

**IDGen** (NeurIPS 2024) applies Item Response Theory concepts to LLM evaluation:

- Two key metrics: **question discriminative power** and **question difficulty**
- High discrimination: the question effectively differentiates between higher and lower quality
- Difficulty tiers ensure the evaluation spread is not ceiling-limited
- Items with negative discrimination are problematic (higher quality leads to lower probability of "yes")

This framework from psychometrics provides the theoretical basis for designing binary questions at different difficulty tiers to maximize score discrimination.

Source: [IDGen paper](https://arxiv.org/html/2409.18892)

---

## Summary of Evidence

| Approach | Binary Advantage | Key Evidence |
|----------|-----------------|--------------|
| Arize AI empirical testing | Binary labels more stable than numeric | Numeric scores collapse into narrow bands |
| CheckEval | +0.45 inter-evaluator agreement | Binary decomposition vs. Likert |
| Hamel Husain | Binary forces definitional clarity | >90% human-LLM agreement achievable |
| Eugene Yan | Binary enables classification metrics | Correlation metrics unreliable for scales |
| G-Eval | Binary + log-prob weighting | Fine-grained scores from binary judgments |
| DEBATE | Devil's advocate reduces inflation | 6-12 point correlation improvement |
| QAG (DeepEval) | Binary constrains stochasticity | "Very little room for stochasticity" |
| GoDaddy RaR | Tiered binary criteria + aggregation | Production-validated approach |

The evidence strongly supports transitioning from our current v2-categorical approach (which uses multi-class categorical questions) to a binary-weighted scoring system.
