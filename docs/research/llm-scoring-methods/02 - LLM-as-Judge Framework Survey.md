# LLM-as-Judge Framework Survey

**Date:** 2026-02-18
**Purpose:** Survey major LLM-as-judge evaluation frameworks to inform whether Reader Triage should move from categorical scoring (0-100 on four dimensions) to binary-weighted scoring for content quality assessment.

---

## Executive Summary

This survey examines seven major LLM-as-judge frameworks and several 2025-2026 developments. The central finding is that **the field is converging on two recommendations directly relevant to our scoring redesign:**

1. **Decompose complex criteria into independent binary or low-precision sub-judgments.** Frameworks like G-Eval, DeCE, and FineSurE all demonstrate that breaking evaluation into smaller, focused questions produces higher human correlation than monolithic scale-based scoring. G-Eval's chain-of-thought decomposition achieves Spearman r=0.514 on summarization (best in class at publication), while DeCE's fully decomposed approach reaches r=0.78 on legal QA.

2. **Binary and pairwise judgments are more reliable than Likert-scale scoring.** AlpacaEval, PandaLM, and JudgeLM all use pairwise/binary outputs, and practitioners consistently report that binary evaluations produce more stable inter-rater agreement (Cohen's kappa 0.72-0.95 between LLM and human coders). LLMs are not naturally calibrated for high-precision numeric scoring.

**Recommendation for Reader Triage:** Transition from four 0-100 continuous scores to a set of binary sub-criteria per dimension (e.g., 6-8 yes/no questions per dimension), then aggregate into dimension scores. This mirrors the approach taken by the most reliable frameworks (DeCE, G-Eval with CoT decomposition) and aligns with the 2025-2026 consensus that decomposed binary judgments outperform monolithic scale scoring.

---

## Framework-by-Framework Analysis

### 1. G-Eval (Liu et al., 2023)

**Paper:** "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (EMNLP 2023)

**Scoring Mechanism:** Scale-based (1-5 or custom ranges), with probability-weighted continuous scoring.

**Prompt Structure:**
G-Eval operates in two phases:
1. **Auto-CoT Generation:** Given a task introduction and evaluation criteria, the LLM generates a chain-of-thought producing detailed evaluation steps. This was the first framework to apply CoT specifically for evaluation, forcing the judge to reason through sub-criteria before scoring.
2. **Form-Filling Evaluation:** The generated CoT steps are appended to the evaluation prompt. The LLM then scores the output using a "form-filling paradigm" -- filling in scores for each defined criterion.

**Calibration Method:**
G-Eval's key innovation is **probability-weighted scoring**. Rather than taking the raw integer output, it:
- Extracts the log-probabilities of each possible score token (1, 2, 3, 4, 5)
- Computes a weighted sum: `score = sum(p(i) * i)` for each score value i
- This produces a continuous float (e.g., 3.72) instead of an integer, reducing ties and increasing score variance
- For APIs that do not expose token probabilities, G-Eval samples 20 times (temperature=1, top_p=1) to estimate the distribution

**Reliability Metrics:**
- Spearman correlation with human judgments on SummEval: **0.514** (summarization, best-in-class at time of publication)
- GPT-4 backbone substantially outperforms GPT-3.5 on both Spearman and Kendall-Tau
- Evaluated on coherence, consistency, fluency, and relevance dimensions

**Decomposed Criteria:** Yes -- the Auto-CoT step naturally decomposes the evaluation criterion into sub-steps. The framework explicitly generates evaluation steps before scoring, enabling the judge to reason through multiple sub-criteria.

**Relevance to Reader Triage:** G-Eval's probability-weighted scoring addresses a problem we likely face: LLMs clustering scores around certain integers. However, the approach requires token probability access, which is not universally available across API providers. The CoT decomposition strategy is directly applicable -- we could have the LLM generate evaluation steps for each of our four dimensions before scoring.

**Sources:**
- [G-Eval Paper (arXiv)](https://arxiv.org/abs/2303.16634)
- [G-Eval Definitive Guide - Confident AI](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)
- [G-Eval Implementation - DeepEval](https://deepeval.com/docs/metrics-llm-evals)
- [G-Eval - Microsoft Learn](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/g-eval-metric-for-summarization)

---

### 2. ARES (Stanford, 2023)

**Paper:** "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems" (NAACL 2024)

**Scoring Mechanism:** Binary classification per dimension, with statistical confidence intervals.

**Prompt Structure:**
ARES does not use a single evaluation prompt. Instead, it follows a three-stage pipeline:
1. **Synthetic Data Generation:** Uses few-shot prompting (5+ examples) to generate synthetic query-passage-answer triples from a corpus
2. **Classifier Fine-tuning:** Fine-tunes a lightweight LM with a binary classification head for each of three dimensions: context relevance, answer faithfulness, and answer relevance
3. **Prediction-Powered Inference (PPI):** Uses ~150 human-annotated datapoints to calibrate confidence intervals around the classifier predictions

**Calibration Method:**
ARES's calibration is uniquely statistical. **Prediction-Powered Inference (PPI)** is a technique from Angelopoulos et al. (2023) that:
- Takes predictions from a large unlabeled sample to estimate performance
- Uses a smaller labeled subset (150+ annotations) to explicitly correct for prediction error
- Produces confidence intervals guaranteed to contain the true value at the desired confidence level
- More accurate predictions yield tighter intervals
- Example output: `Prediction: 0.606, Confidence Interval: [0.547, 0.664]`

**Reliability Metrics:**
- Provides statistical confidence intervals (not just point estimates)
- Confidence intervals are valid regardless of the underlying classifier quality
- Requires minimal human annotation (~150 datapoints) compared to full manual evaluation

**Decomposed Criteria:** Yes -- evaluates three independent binary dimensions (context relevance, answer faithfulness, answer relevance), each with its own fine-tuned classifier.

**Relevance to Reader Triage:** ARES demonstrates that binary classification per dimension, combined with statistical calibration, can be more reliable than continuous scoring. The PPI approach of using a small human-annotated set to calibrate automated scores is directly relevant -- we could maintain a small "gold standard" set of scored articles to calibrate our LLM scorer.

**Sources:**
- [ARES Paper (arXiv)](https://arxiv.org/abs/2311.09476)
- [ARES GitHub](https://github.com/stanford-futuredata/ARES)
- [ARES Documentation](https://ares-ai.vercel.app/)
- [ARES - NAACL 2024](https://aclanthology.org/2024.naacl-long.20/)

---

### 3. AlpacaEval / AlpacaEval 2.0 (Tatsu Lab, Stanford)

**Paper:** "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators" (2024)

**Scoring Mechanism:** Pairwise comparison (win/loss), producing win-rate percentages.

**Prompt Structure:**
For each evaluation:
1. A prompt/instruction is presented
2. Two model-generated responses are provided (one baseline, one candidate)
3. A judge LLM (typically GPT-4) determines which response is superior
4. The prompt is simplified to output a single preference token
5. Win rates are aggregated across the full evaluation set

**Calibration Method:**
AlpacaEval 2.0 introduced **Length-Controlled Win Rate (LC WR)** to address severe length bias:
- Fits a generalized linear model (GLM) to predict annotator preferences based on length differences and other features
- Predicts counterfactual preferences when conditioned on zero length difference
- This asks: "What would the preference be if both responses had the same length?"
- Reduced normalized standard deviation of win rates across verbosity prompts from **25% to 10%**

**Reliability Metrics:**
- Spearman correlation with LMSYS Chatbot Arena: **0.94** (raw), **0.98** (length-controlled)
- The length-controlled version is one of the highest-correlated automated benchmarks with human preference rankings

**Decomposed Criteria:** No -- AlpacaEval uses holistic preference judgments ("which response is better overall?") rather than decomposed criteria. This is both a strength (simplicity, high reliability) and a limitation (no diagnostic insight into *why* one response is preferred).

**Relevance to Reader Triage:** AlpacaEval's pairwise approach is less directly applicable to our use case (we need absolute quality assessment, not model comparison). However, the length-bias debiasing technique is relevant -- our articles vary greatly in length, and our scorer may be biased toward longer articles. The GLM-based debiasing approach could be adapted.

**Sources:**
- [AlpacaEval GitHub](https://github.com/tatsu-lab/alpaca_eval)
- [Length-Controlled AlpacaEval Paper (arXiv)](https://arxiv.org/abs/2404.04475)
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [AlpacaEval 2.0 - Emergent Mind](https://www.emergentmind.com/topics/alpacaeval-2-0)

---

### 4. Prometheus / Prometheus 2 (KAIST, 2023-2024)

**Paper:** "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models" (EMNLP 2024)

**Scoring Mechanism:** Dual-mode -- supports both direct assessment (1-5 scale) and pairwise ranking.

**Prompt Structure:**
Prometheus uses a structured prompt with four components:
1. **Instruction:** The original task description
2. **Response:** The output to evaluate
3. **Reference Answer:** (optional) A gold-standard response
4. **Score Rubric:** A custom rubric including:
   - Criteria description
   - Score descriptions for each level (1-5)
   - The model generates verbal feedback before outputting a score

The system prompt switches between direct assessment and pairwise ranking modes.

**Calibration Method:**
- **Weight merging:** Prometheus 2 is trained by merging weights of models separately trained for direct assessment and pairwise ranking, which surprisingly improves performance on both tasks
- **Preference Collection:** Trained on 1,000+ instance-wise evaluation criteria beyond basic helpfulness/harmlessness
- **Cross-format consistency:** The selected response in pairwise mode usually also receives a higher score in direct assessment mode, demonstrating internal calibration

**Reliability Metrics:**
- Direct assessment: Pearson correlation surpasses baselines by **0.2 units** across Vicuna Bench, MT Bench, FLASK, Feedback Bench
- Pairwise ranking: **72-85% agreement** with human judgments across HHH Alignment, MT Bench Human Judgment, Auto-J Eval
- Available in 7B and 8x7B parameter variants (7B requires only 16GB VRAM)

**Decomposed Criteria:** Yes -- the score rubric system explicitly supports custom, fine-grained criteria with per-level descriptions. The framework is designed for "fine-grained evaluation on a customized score rubric."

**Relevance to Reader Triage:** Prometheus is the most directly applicable framework for our use case. Its custom rubric system maps naturally to our four scoring dimensions. Being open-source, it could replace our Anthropic API calls if we wanted to reduce costs. The rubric structure (with explicit score-level descriptions) is a pattern we should adopt regardless of which LLM we use.

**Sources:**
- [Prometheus 2 Paper (arXiv)](https://arxiv.org/abs/2405.01535)
- [Prometheus GitHub](https://github.com/prometheus-eval/prometheus-eval)
- [Prometheus 1 Paper (arXiv)](https://arxiv.org/abs/2310.08491)
- [M-Prometheus (Multilingual, 2025)](https://arxiv.org/abs/2504.04953)

---

### 5. JudgeLM (BAAI, 2023)

**Paper:** "JudgeLM: Fine-tuned Large Language Models are Scalable Judges" (ICLR 2025 Spotlight)

**Scoring Mechanism:** Pairwise comparison with numeric scores and detailed reasoning.

**Prompt Structure:**
JudgeLM takes as input:
1. A task/question
2. A pair of candidate answers (from different LLMs)
3. Optionally, a reference answer
4. The judge outputs: scores for each answer + detailed reasoning

The training data pipeline:
- 105K seed tasks as questions
- Answers extracted from 11 different LLMs
- Random sampling of answer pairs
- GPT-4 generates scores and detailed reasons as "teacher" judgments

**Calibration Method:**
JudgeLM addresses three systematic biases through data augmentation:
1. **Position Bias (Swap Augmentation):** Swaps the order of answer pairs during training, forcing the model to attend to content rather than position. Simple but effective.
2. **Knowledge Bias (Reference Support):** Includes reference answers to ground the judge's knowledge
3. **Format Bias (Reference Drop):** Randomly drops reference answers during training to prevent over-reliance on format matching

**Reliability Metrics:**
- Agreement with human judgments exceeding **90%** (surpasses human-to-human agreement)
- Available at 7B, 13B, and 33B parameter scales
- JudgeLM-7B evaluates 5K samples in 3 minutes on 8 A100 GPUs

**Decomposed Criteria:** Partially -- the model generates detailed reasoning before scoring, but the evaluation criteria are not explicitly decomposed in the prompt structure. The reasoning serves as implicit decomposition.

**Relevance to Reader Triage:** JudgeLM's bias mitigation techniques are directly applicable. Position bias is less relevant for our single-article scoring, but the reference support / reference drop technique is interesting -- we could sometimes include example high-scoring articles as reference points, and sometimes omit them, to train robustness. The swap augmentation concept could apply if we move to pairwise article comparison.

**Sources:**
- [JudgeLM Paper (arXiv)](https://arxiv.org/abs/2310.17631)
- [JudgeLM GitHub](https://github.com/baaivision/JudgeLM)
- [JudgeLM - Hugging Face](https://huggingface.co/papers/2310.17631)

---

### 6. PandaLM (WeOpenML, 2023)

**Paper:** "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization" (ICLR 2024)

**Scoring Mechanism:** Ternary pairwise comparison (Win / Lose / Tie).

**Prompt Structure:**
PandaLM receives:
1. An instruction/context
2. Two model responses to compare
3. The model outputs one of three labels: Win, Lose, or Tie
4. Additionally generates: reasoning for the decision + a reference answer

Trained via supervised fine-tuning (SFT) on GPT-3.5 distilled data with heuristic filtering. No specialized classification head -- the model generates the verdict as text.

**Calibration Method:**
- **Heuristic data filtering:** Filters noisy examples from the GPT-3.5 distilled training data
- **Multi-factor evaluation:** Considers conciseness, clarity, adherence to instructions, comprehensiveness, and formality (not just correctness)
- No explicit statistical calibration mechanism

**Reliability Metrics:**
- PandaLM-7B achieves **93.75% of GPT-3.5's** evaluation ability (F1-score)
- PandaLM-7B achieves **88.28% of GPT-4's** evaluation ability (F1-score)
- PandaLM-70B surpasses both GPT-3.5 and GPT-4
- Results reported as (#win, #lose, #tie) tuples, e.g., (45, 26, 99) out of 170 instructions

**Decomposed Criteria:** Partially -- the model considers multiple factors (conciseness, clarity, etc.) but produces a single ternary judgment rather than per-criterion scores.

**Relevance to Reader Triage:** PandaLM's simplicity is instructive. The ternary output (Win/Lose/Tie) is highly reliable because it reduces the decision space. For Reader Triage, a similar simplification could work: instead of scoring 0-100, score each sub-criterion as Present/Absent/Partial, then aggregate.

**Sources:**
- [PandaLM Paper (arXiv)](https://arxiv.org/abs/2306.05087)
- [PandaLM GitHub](https://github.com/WeOpenML/PandaLM)
- [PandaLM - ICLR 2024](https://openreview.net/forum?id=5Nn2BLV7SB)

---

### 7. Fine-Grained Evaluation Approaches (DeCE, FineSurE)

#### DeCE -- Decomposed Criteria-Based Evaluation (2025)

**Paper:** "Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses" (EMNLP 2025 Industry Track)

**Scoring Mechanism:** Decomposed binary sub-criteria (precision + recall), aggregated into composite scores.

**Prompt Structure:**
DeCE decomposes evaluation into two orthogonal dimensions:
1. **Precision:** Factual accuracy and relevance of generated content
2. **Recall:** Coverage of required concepts from a gold answer
- Criteria are automatically extracted from gold answer requirements (no predefined taxonomies needed)
- Each sub-criterion is evaluated independently as present/absent

**Calibration:** Model-agnostic, domain-general. Only 11.95% of LLM-generated criteria required expert revision, indicating high auto-calibration.

**Reliability Metrics:**
| Method | Correlation with Expert Judgments |
|--------|----------------------------------|
| Traditional metrics (ROUGE, etc.) | r = 0.12 |
| Pointwise LLM scoring | r = 0.35 |
| Multidimensional LLM evaluators | r = 0.48 |
| **DeCE** | **r = 0.78** |

**Decomposed Criteria:** This is the defining feature. DeCE explicitly decomposes into binary sub-criteria.

#### FineSurE -- Fine-Grained Summarization Evaluation (ACL 2024)

**Paper:** "FineSurE: Fine-grained Summarization Evaluation using LLMs" (ACL 2024)

**Scoring Mechanism:** Per-sentence binary classification across three dimensions.

**Approach:**
- Evaluates summaries at the sentence/keyfact level (not holistic)
- Three dimensions: Faithfulness (fact-checking), Completeness (keyfact alignment), Conciseness (no extraneous information)
- Each summary sentence is independently checked against the source

**Relevance to Reader Triage:** DeCE is the strongest validation for moving to binary-weighted scoring. Its r=0.78 correlation (vs. r=0.35 for pointwise LLM scoring) is the most compelling evidence that decomposed binary criteria outperform scale-based scoring. FineSurE's per-sentence approach could inform how we evaluate article sections.

**Sources:**
- [DeCE Paper (arXiv)](https://arxiv.org/abs/2509.16093)
- [DeCE - EMNLP 2025](https://aclanthology.org/2025.emnlp-industry.136/)
- [FineSurE Paper (arXiv)](https://arxiv.org/abs/2407.00908)
- [FineSurE - ACL 2024](https://aclanthology.org/2024.acl-long.51/)

---

## 2025-2026 Developments

### Mixture of Prompts (MoPs)

A 2025 framework that dynamically selects specialized prompt modules based on input characteristics. Rather than using a single evaluation prompt, MoPs routes different types of content to different evaluation prompts optimized for that content type. This improves adaptability across heterogeneous tasks.

### Agent-as-a-Judge

Emerging in 2025-2026, this paradigm uses multiple LLM agents in ensemble to produce evaluations. Multi-agent approaches aggregate independent judgments from several models or prompt variants, reducing variance and bias. Approaches include:
- Ensemble voting across multiple judge models
- Contextual evaluation prompt routing
- Multi-agent debate before final scoring

### Preference Leakage (Li et al., 2025)

A critical 2025 finding that exposes **preference leakage** -- a contamination problem where LLM judges are biased toward outputs from related models. Three types of relatedness cause bias:
1. Same model (self-preference)
2. Inheritance relationship (fine-tuned variants)
3. Same model family

The severity correlates with the degree of relatedness and proportion of synthetic data. This is harder to detect than previously identified biases and has substantial impact on leaderboard rankings.

**Relevance:** Since we use Claude for both content processing and scoring, preference leakage is a real concern. If we ever score AI-generated or AI-summarized content, the scorer may be systematically biased.

### M-Prometheus (2025)

Extension of Prometheus to multilingual evaluation. A suite of open-weight judges (3B-14B parameters) supporting 20+ languages for both direct assessment and pairwise comparison. Demonstrates that backbone model selection and synthetic multilingual feedback data (rather than translated data) are key factors.

### Comprehensive Survey (Zheng et al., 2025/2026)

A major survey published in early 2026 formally defines the LLM-as-a-Judge paradigm and classifies approaches across five dimensions: Functionality, Methodology, Applications, Meta-evaluation, and Limitations. Key recommendations include:
- Reproducible scoring templates
- Documented chain-of-thought reasoning
- Inter-judge reliability metrics (Cohen's Kappa, Krippendorff's Alpha)

**Sources:**
- [LLM-as-a-Judge Survey (arXiv)](https://arxiv.org/abs/2411.15594)
- [LLMs-as-Judges Comprehensive Survey (arXiv)](https://arxiv.org/abs/2412.05579)
- [Preference Leakage Paper (arXiv)](https://arxiv.org/abs/2502.01534)
- [M-Prometheus (arXiv)](https://arxiv.org/abs/2504.04953)
- [Agent-as-a-Judge (arXiv)](https://arxiv.org/html/2508.02994v1)

---

## Comparison Table

| Framework | Scoring Type | Scale | Decomposed Criteria | Calibration Method | Human Correlation | Open Source |
|-----------|-------------|-------|---------------------|--------------------|-------------------|-------------|
| **G-Eval** | Pointwise (continuous) | 1-5 (probability-weighted) | Yes (Auto-CoT steps) | Token probability weighting | Spearman 0.514 (summarization) | Prompt-based (any LLM) |
| **ARES** | Binary classification | Binary per dimension | Yes (3 dimensions) | PPI with 150 human annotations | Confidence intervals | Yes (GitHub) |
| **AlpacaEval 2.0** | Pairwise preference | Win/Loss | No (holistic) | Length-controlled GLM debiasing | Spearman 0.98 with Arena | Prompt-based (any LLM) |
| **Prometheus 2** | Dual (pointwise + pairwise) | 1-5 direct; pairwise | Yes (custom rubrics) | Weight merging; cross-format consistency | Pearson +0.2 over baselines; 72-85% pairwise agreement | Yes (7B, 8x7B models) |
| **JudgeLM** | Pairwise with scores | Numeric + reasoning | Partial (implicit via reasoning) | Swap augmentation, reference support/drop | >90% agreement (exceeds human-human) | Yes (7B-33B models) |
| **PandaLM** | Ternary pairwise | Win/Lose/Tie | Partial (multi-factor) | Heuristic data filtering | 88-94% of GPT-3.5/GPT-4 F1 | Yes (7B, 70B models) |
| **DeCE** | Decomposed binary | Binary sub-criteria | Yes (precision + recall) | Auto-generated criteria (11.95% revision rate) | **r = 0.78** (vs. 0.35 pointwise) | Paper only |
| **FineSurE** | Per-sentence binary | Binary per sentence | Yes (3 dimensions) | Keyfact alignment | Improved on completeness/conciseness | Yes (GitHub) |

---

## Key Findings for Binary vs. Scale Scoring

### Evidence Favoring Binary/Decomposed Scoring

1. **DeCE achieves r=0.78 vs. r=0.35 for pointwise scoring** -- the single most compelling datapoint. Decomposing into binary sub-criteria more than doubles correlation with expert judgments.

2. **Practitioner consensus:** Multiple surveys and guides (Evidently AI, Monte Carlo Data, Confident AI) recommend binary or low-precision scores, split across separate evaluators. "Binary outputs tend to produce more stable and reliable evaluations than subtle numeric scoring."

3. **LLMs are not calibrated for numeric precision.** LLMs generate text and are not naturally calibrated for high-precision scoring. They cluster around certain integers, producing low variance even when prompted for fine-grained scores. G-Eval's probability weighting is a workaround, not a solution.

4. **Inter-rater reliability is higher for binary tasks.** Cohen's kappa between LLM and human coders ranges from 0.72 to 0.95 for binary/categorical tasks, significantly higher than for numeric scales.

5. **Forced decomposition reduces bias.** G-Eval showed that more sub-criteria leads to greater robustness and less randomness, while simpler sub-criteria reduce bias and improve accuracy.

### Evidence Favoring Scale Scoring

1. **G-Eval's probability weighting** can produce meaningful continuous scores when token probabilities are available.

2. **Prometheus's rubric system** with explicit per-level descriptions achieves reasonable correlation with scale-based scoring.

3. **Information loss:** Binary scoring loses gradation. A "barely quotable" passage and a "profoundly quotable" passage would both score 1.

### Recommended Hybrid Approach

The evidence strongly suggests a **decomposed binary approach with weighted aggregation:**

1. For each of our four dimensions (Quotability, Surprise, Argument Quality, Applicable Insight), define 5-8 binary sub-criteria (yes/no questions)
2. Each sub-criterion gets a weight reflecting its importance to the dimension
3. Dimension score = weighted sum of binary sub-criterion answers
4. Total score = sum of dimension scores

This mirrors DeCE's approach (highest human correlation) while preserving the multi-dimensional structure we need for user-facing display.

---

## Recommendations for Reader Triage

### Immediate Actions

1. **Define binary sub-criteria for each dimension.** For example, Quotability might decompose into:
   - Does the article contain specific data points or statistics? (yes/no)
   - Does it include memorable phrasing or turns of phrase? (yes/no)
   - Are there passages that stand alone as insights? (yes/no)
   - Does it contain expert quotes worth preserving? (yes/no)
   - Are there concrete examples or case studies? (yes/no)

2. **Adopt G-Eval's CoT pattern.** Before answering each binary question, require the LLM to briefly reason about it. This improves accuracy per the G-Eval findings.

3. **Build a calibration set.** Following ARES's PPI approach, maintain ~100-150 articles with human-validated scores. Use these to periodically verify that automated scoring remains calibrated.

### Medium-Term Improvements

4. **Test pairwise comparison for edge cases.** For articles near scoring thresholds (around 30 and 60), use AlpacaEval-style pairwise comparison against reference articles at each threshold to validate placement.

5. **Monitor for length bias.** Implement AlpacaEval 2.0's length-debiasing approach -- track whether longer articles systematically receive higher scores and correct if so.

6. **Evaluate Prometheus 2 as an alternative judge.** The 7B model runs on 16GB VRAM and could serve as a cost-free alternative to API-based scoring, especially for bulk re-scoring operations.

### Architecture Considerations

7. **Preference leakage.** Since we use Claude for scoring, be aware that if articles contain AI-generated content (increasingly common), Claude may be biased toward that content. Consider using a different model family for scoring than for any content generation.

8. **Separate evaluators per dimension.** Following the decomposed criteria best practice, run each dimension as a separate LLM call rather than scoring all four dimensions in a single prompt. This reduces cross-contamination between criteria and improves individual dimension accuracy.

---

## References

### Primary Papers

- Liu, Y., et al. (2023). "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." EMNLP 2023. [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
- Saad-Falcon, J., et al. (2023). "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." NAACL 2024. [arXiv:2311.09476](https://arxiv.org/abs/2311.09476)
- Dubois, Y., et al. (2024). "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)
- Kim, S., et al. (2024). "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." EMNLP 2024. [arXiv:2405.01535](https://arxiv.org/abs/2405.01535)
- Zhu, L., et al. (2023). "JudgeLM: Fine-tuned Large Language Models are Scalable Judges." ICLR 2025 Spotlight. [arXiv:2310.17631](https://arxiv.org/abs/2310.17631)
- Wang, Y., et al. (2023). "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." ICLR 2024. [arXiv:2306.05087](https://arxiv.org/abs/2306.05087)
- Yu, F., et al. (2025). "Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses." EMNLP 2025 Industry. [arXiv:2509.16093](https://arxiv.org/abs/2509.16093)
- Song, H., et al. (2024). "FineSurE: Fine-grained Summarization Evaluation using LLMs." ACL 2024. [arXiv:2407.00908](https://arxiv.org/abs/2407.00908)

### Surveys and Meta-Analyses

- Zheng, S., et al. (2025). "A Survey on LLM-as-a-Judge." [arXiv:2411.15594](https://arxiv.org/abs/2411.15594)
- Li, D., et al. (2025). "Preference Leakage: A Contamination Problem in LLM-as-a-judge." [arXiv:2502.01534](https://arxiv.org/abs/2502.01534)
- Gu, S., et al. (2024). "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods." [arXiv:2412.05579](https://arxiv.org/abs/2412.05579)

### Practitioner Guides

- [Confident AI: LLM-as-a-Judge Complete Guide](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [Evidently AI: LLM-as-a-Judge Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Cameron Wolfe: Using LLMs for Evaluation](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- [Monte Carlo Data: LLM-as-Judge Best Practices](https://www.montecarlodata.com/blog-llm-as-judge/)
- [Eugene Yan: Evaluating LLM Evaluators](https://eugeneyan.com/writing/llm-evaluators/)
