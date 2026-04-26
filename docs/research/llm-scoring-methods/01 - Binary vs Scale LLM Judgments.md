# Binary vs. Scale LLM Judgments: A Research Report

## Executive Summary

LLMs produce systematically less reliable judgments on numeric and Likert scales than on binary (yes/no) decisions. The evidence is consistent across academic papers, framework evaluations, and practitioner reports: numeric scores cluster in narrow ranges (typically the top 20-30% of the scale), exhibit round-number bias, suffer from anchoring effects when multiple attributes are scored together, and show poor inter-rater reliability across runs. Binary decomposition -- breaking a complex quality judgment into multiple independent yes/no questions -- addresses these failures by constraining each decision to an unambiguous boundary, producing scores that are more stable across runs, more discriminating across content quality levels, and more aligned with human expert judgments. For our content scoring system where articles cluster at 80-100, the research strongly supports replacing the current multi-point categorical rubric with a larger set of binary questions at varying difficulty levels.

---

## 1. The Score Compression Problem

### 1.1 Narrow-Band Clustering in Numeric Scales

When LLMs are asked to assign scores on 1-100 scales, they systematically underutilize the full range. Stureborg, Alikaniotis, and Suhara (2024) found that LLM evaluators disproportionately assign scores that are multiples of 5 and 10: "Frequencies of some scores, such as 90 and 95, are far higher than 'odd' scores such as 92 or 19," with much of the lower range (1-60) largely ignored ([Stureborg et al., 2024](https://arxiv.org/html/2405.01724v1)). This round-number bias mirrors a well-documented human cognitive bias, but LLMs exhibit it more severely.

The compression effect is not limited to wide scales. Even on a 1-10 scale, Stureborg et al. found that models "struggle to differentiate individual predictions despite acceptable system-level correlations." Their inter-sample agreement for GPT-4 measured Krippendorff's alpha of 0.587, below human inter-annotator agreement of 0.659 -- meaning the models produce substantially varying evaluations across similar conditions while compressing the range they actually use.

Practitioner reports confirm this at scale. Nayeem Islam, writing about production LLM judge systems, observed that "LLMs tend to compress everything into a narrow 7-9 range, which masks useful signal" -- ratings "collapsed into a meaningless 7-9 range, with random drift and little real discrimination" ([Islam, 2024](https://medium.com/@nomannayeem/scaling-evaluation-with-llm-judges-our-approach-and-findings-0a046e8344c4)).

### 1.2 Score Range Bias: The Scale Itself Changes the Output

Fujinuma et al. (2025) demonstrated that the numerical outputs of LLM judges "depend sharply on the range and labeling of the scoring scale (e.g., 0-4, 1-5, 2-6, 3-7), with observed preferences toward specific values, independent of content quality." This score range bias was observed across different model sizes and families (Llama-3 and Qwen-2.5), indicating it is a systematic property of LLM scoring rather than a model-specific artifact ([Fujinuma et al., 2025](https://arxiv.org/abs/2510.18196)).

Their mitigation -- contrastive decoding using the formula `log p_main - lambda * log p_asst` -- achieved up to 11.3% relative improvement in Spearman correlation with human judgments, but this requires access to logits from models of the same family, limiting practical applicability.

### 1.3 Variance Compression Is a General LLM Property

The compression problem is not just a scoring-prompt issue. Research on LLM-based measurement pipelines shows that "any LLM-based measurement pipeline will exhibit compressed error distributions relative to embedding-based measurement, regardless of whether the generation and rating models are the same." Cross-model controls show nearly identical compression, indicating this is a general LLM property rather than a within-model artifact ([Arxiv, 2602.13862](https://arxiv.org/html/2602.13862)).

### 1.4 Implications for Our System

Our content scoring system exhibits exactly this pattern: most articles score 80-100 on a 0-100 scale. The research indicates this is an inherent property of asking LLMs to produce numeric scores, not a deficiency in our rubric or prompt design. The models lack a "relatively consistent internal scoring mechanism" for absolute scoring (Wolfe, 2024), so scores naturally gravitate toward a comfort zone in the upper range.

---

## 2. Position Bias in Numeric Scoring

### 2.1 Anchoring Effects Across Attributes

When LLMs score multiple attributes in a single generation, prior scores contaminate subsequent ones. Stureborg et al. (2024) measured this directly: for human raters, the Pearson correlation between coherence and consistency scores was 0.315, while GPT-4 showed a correlation of 0.979 -- indicating near-total anchoring where the first score effectively determines all subsequent scores ([Stureborg et al., 2024](https://arxiv.org/html/2405.01724v1)).

This finding has direct implications for multi-dimensional scoring rubrics (like our 4-dimension system). If an LLM scores "quotability" first and assigns an 85, the scores for surprise factor, argument quality, and applicable insight will cluster near 85 regardless of the actual content characteristics.

**Mitigation**: Stureborg et al. recommend predicting "single attributes per generation" -- that is, making separate API calls for each scoring dimension rather than asking for all dimensions in a single prompt.

### 2.2 Score Rubric Order and ID Bias

Li et al. (2025) identified three categories of scoring bias: score rubrics order bias (how criteria are sequenced), score IDs bias (Arabic numerals vs. letters vs. Roman numerals), and reference answer score bias. GPT-4o maintained correlation fluctuations within 0.03 across most perturbations, but smaller models like Qwen3-8B showed fluctuations exceeding 0.05 for rubric order and ID bias, and reference answer bias caused correlation changes reaching nearly 0.2 ([Li et al., 2025](https://arxiv.org/html/2506.22316v1)).

### 2.3 Position Bias in Pairwise Comparisons

The seminal Zheng et al. (2023) MT-Bench paper documented that all LLM judges exhibit strong position bias in pairwise comparisons, with most favoring the first position. Eugene Yan's analysis quantified this: GPT-3.5 showed first-position bias 50% of the time, Claude-v1 approximately 70% of the time ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685); [Yan, 2024](https://eugeneyan.com/writing/llm-evaluators/)).

---

## 3. Calibration Quality Across Response Formats

### 3.1 Binary Achieves Higher Agreement Rates

Across the literature, binary scoring consistently produces higher agreement between LLM judges and human experts:

- **Binary scoring**: 89-93% agreement rates with human experts
- **Likert/ordinal scoring**: 60-70% agreement rates with human experts
- **Pairwise comparison**: ~80% agreement (Zheng et al., 2023)

The Evidently AI guide summarizes the mechanism: "It's easier to get accurate results with two simple choices rather than trying to decide if a specific response scores 73 vs. 82 for 'politeness.' LLMs generate text and aren't naturally calibrated for high-precision scoring" ([Evidently AI, 2024](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)).

### 3.2 Google's Adaptive Precise Boolean Rubrics

Metwally and McDuff (Google Research, 2025) developed a framework for converting complex evaluation criteria into granular yes/no questions for health LLM evaluation. Their Adaptive Precise Boolean rubrics achieved "significantly higher inter-rater reliability" (measured by ICC) compared to traditional Likert rubrics. Key findings:

- **Evaluation time**: Boolean rubrics reduced completion time by over 50% compared to Likert scales
- **Sensitivity**: Boolean rubrics reliably detected quality drops when critical health data was omitted, while "Likert scores showed inconsistent, smaller magnitude discrepancies"
- **Consistency**: Despite having more individual questions, the boolean format produced more consistent evaluations

The authors note the framework is domain-agnostic and applicable beyond healthcare ([Metwally & McDuff, 2025](https://research.google/blog/a-scalable-framework-for-evaluating-health-language-models/); [arXiv 2503.23339](https://arxiv.org/abs/2503.23339)).

### 3.3 Optimal Scale Granularity

Stureborg et al. (2024) tested across scale widths and found the 1-10 scale showed optimal performance (0.428 average Kendall's tau), outperforming both 1-5 (coarser) and 1-100 (too wide) scales. However, this was for system-level correlation -- at the individual item level, all numeric scales showed poor discrimination.

Godfrey et al. (2025) found that 11-point ordinal scales closed the gap with listwise ranking methods (NDCG@10 of 0.737-0.851 vs. 0.700-0.852 for listwise), but only 9 of 40 model-dataset combinations showed statistically significant listwise superiority at 95% confidence ([Godfrey et al., 2025](https://arxiv.org/html/2505.19334v1)). Their key conclusion: "prior work on LLM ranking has underestimated the performance of pointwise scoring" when sufficiently granular scales are used.

### 3.4 G-Eval Probability Weighting as a Calibration Mechanism

Liu et al. (2023, Microsoft) identified that direct integer scoring from LLMs suffers from a dominant-score problem: "One digit usually dominates the distribution of the scores, such as 3 for a 1-5 scale." Their solution was probability-weighted scoring: `score = sum(p(s_i) * s_i)`, where `p(s_i)` represents the token probability for each score option.

On the SummEval benchmark, G-Eval-4 achieved average Spearman correlation of 0.514 with human ratings -- a substantial improvement over prior methods. The probability weighting "significantly reduces bias and enables the model to better differentiate between outputs of similar quality" by producing continuous scores rather than integer jumps ([Liu et al., 2023](https://ar5iv.labs.arxiv.org/html/2303.16634)).

**Limitation for our use case**: Probability-weighted scoring requires access to token-level log probabilities, which adds API complexity. The Anthropic API supports this, but the approach is more complex to implement than binary decomposition and still relies on the LLM having a meaningful probability distribution over score tokens.

---

## 4. Inter-Rater Reliability and Run-to-Run Consistency

### 4.1 Dramatic Variability Across Runs

Schroeder and Wood-Doughty (2024, Northwestern University) conducted the most rigorous study of LLM judge reliability to date. Testing three LLM evaluators on BBH questions across 100 random seed variations, they found inter-rater reliability (IRR) "ranged drastically from 0.167 to 1.00," proving "highly sensitive to random seed variation" ([Schroeder & Wood-Doughty, 2024](https://arxiv.org/html/2412.12509v2)).

Their proposed alternative metric, McDonald's omega (measuring internal consistency reliability), revealed concerning patterns across benchmarks:

| Model | BBH | SQuAD | MT-Bench |
|-------|-----|-------|----------|
| Starling-LM-7B | 0.677-0.713 (questionable) | 0.598-0.639 (poor-questionable) | 0.462-0.640 (poor-questionable) |
| Meta-Llama-3-8B | 0.661-0.712 (questionable) | 0.533-0.655 (poor-questionable) | 0.421-0.602 (poor-questionable) |
| Gemma-1.1-7b | 0.723-0.803 (acceptable-good) | 0.640-0.770 (questionable-good) | 0.585-0.732 (questionable-good) |

A crucial finding: "models optimized for performance may sacrifice reliability in their evaluations" -- Chatbot Arena top performers showed lower judgment consistency than lower-ranked models.

### 4.2 Binary Judgments Show Dramatically Higher Reliability

The same study found that when applied to simplified binary judgments with ground truth provided, reliability increased substantially to omega of 0.98-0.99, demonstrating that "with careful design and clear criteria, LLMs can be dependable judges" ([Schroeder & Wood-Doughty, 2024](https://arxiv.org/html/2412.12509v2)).

This is arguably the single most important finding for our use case: the same models that show poor-to-questionable reliability on numeric scales achieve near-perfect reliability on binary judgments.

### 4.3 Temperature Effects

Research consistently recommends low temperature settings (0.0-0.2) for evaluation tasks to maximize determinism. However, Schroeder and Wood-Doughty found the optimal temperature varies by model and task -- Gemma improved at lower temperatures (0.25), while other models showed inconsistent patterns. The blanket assumption that temperature 0 guarantees reliability is incorrect.

### 4.4 Pointwise Scoring Is More Stable Than Pairwise

Tripathi et al. (2025, COLM) found that pairwise preference comparisons "reversed in approximately 35% of test cases, compared to only 9% for absolute scores." Absolute scoring proved more resilient to manipulation through distractor features, producing "judgments that better reflect response quality and are less influenced by distractor features" ([Tripathi et al., 2025](https://arxiv.org/abs/2504.14716)). This suggests that while pairwise comparisons are popular, pointwise scoring (including binary pointwise scoring) is more stable for production systems.

---

## 5. The Binary Decomposition Approach

### 5.1 CheckEval: The Definitive Binary Framework

Lee et al. (2024, EMNLP 2025) developed CheckEval, which explicitly replaces Likert-scale evaluation with decomposed binary questions. Their three-stage process:

1. **Define dimensions**: Select evaluation dimensions and sub-dimensions
2. **Generate checklist**: Create Boolean yes/no questions through seed writing, diversification, elaboration, and filtering
3. **Evaluate**: Model answers binary questions; final score = proportion of "yes" responses

Results were substantial:
- Improved average inter-evaluator agreement by **0.45** compared to G-Eval's Likert approach
- Achieved Krippendorff's alpha of **0.67** on SummEval and Topical-Chat, comparable to human rater agreement (~0.7)
- Dramatically reduced score variance across different evaluator models

The authors note that existing Likert methods "achieved decent human correlation but showed low inter-evaluator agreement and high variance" -- binary decomposition fixes this by making each judgment "atomic and unambiguous" ([Lee et al., 2024](https://arxiv.org/abs/2403.18771)).

### 5.2 DeCE: Decomposed Criteria-Based Evaluation

Yu et al. (2025, Thomson Reuters) developed DeCE, which decomposes evaluation into precision and recall workflows. Each factual element is verified individually (a binary check), then aggregated. On legal QA evaluation with 224 expert-curated pairs:

| Method | Correlation with Human Experts |
|--------|-------------------------------|
| ROUGE-L | r = 0.11 |
| Pointwise LLM Judge | r = 0.35 |
| G-Eval F2 | r = 0.42 |
| **DeCE F2** | **r = 0.78** |

DeCE achieved "substantially stronger correlation with expert judgments" than all baselines, with only 11.95% of auto-generated criteria requiring expert revision ([Yu et al., 2025](https://arxiv.org/html/2509.16093)).

### 5.3 QAG (Question Answer Generation) Scoring

The QAG approach, productionized by Confident AI's DeepEval framework, uses "confined answers (usually either a 'yes' or 'no') to close-ended questions to compute a final metric score." The key advantage: "it is reliable because it does NOT use LLMs to directly generate scores. Instead, QAG constrains verdicts to a binary 'yes' or 'no' for close-ended questions, with very little room for stochasticity" ([Confident AI, 2024](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)).

QAG transforms arbitrary LLM scores into mathematically grounded metrics. Instead of asking "rate the relevance from 1-10," it extracts individual claims, asks "is this claim relevant? yes/no" for each, and computes the proportion. This produces a score that is both interpretable and reproducible.

### 5.4 Practitioner Consensus on Binary Evaluation

**Hamel Husain** (practitioner, widely cited in ML engineering):
- "A common mistake is straying from binary pass/fail judgments. If your evaluations consist of metrics that LLMs score on a 1-5 scale or any other scale, you're doing it wrong."
- "The distinction between a '3' and a '4' is often subjective and inconsistent across different reviewers, leading to noisy, unreliable data."
- Reports achieving >90% human-LLM agreement after 3 iterations of prompt refinement with binary judgments
- Source: [Hamel's LLM Judge Guide](https://hamel.dev/blog/posts/llm-judge/)

**Eugene Yan** (Applied Scientist):
- "Where possible, I have my evaluators return binary outputs. This improves model performance while making it easier to apply classification metrics."
- "I tend to be skeptical of correlation metrics. They don't account for chance agreement."
- Binary enables straightforward precision/recall measurement vs. unreliable correlation metrics on ordinal data
- Source: [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)

---

## 6. Systematic Bias: Binary vs. Continuous Formats Produce Different Results

### 6.1 The Lu et al. Negativity Bias Finding

Lu, Zhang, and Wang (2025) discovered that LLMs exhibit a consistent negative bias when using binary formats compared to continuous scales. In their experiments:

**Experiment 1 (Value Judgment, 210 statements)**:
- Support for statements decreased from 74.5% (continuous 0-10 scale) to 60.7% (binary yes/no)
- Mean bias: delta P(Support) = -0.138

**Experiment 2 (Sentiment Analysis, 213 headlines)**:
- Positive judgments decreased from 39.9% (continuous 1-6 Likert) to 24.6% (binary)
- Strong preference for "No" responses observed (theta_Yes = -1.320)

Models tested: Llama-3.3-70b, Qwen-2.5-72b, DeepSeek-v3, GPT-4o-mini, GPT-4o (temperature = 0). The negative bias persisted across all models and control conditions ([Lu et al., 2025](https://arxiv.org/html/2504.19445)).

### 6.2 Implications for Content Scoring

This finding is critically important for our use case. If binary questions naturally produce a negativity bias, they will counteract the positivity/sycophancy bias that causes score inflation on numeric scales. An article that would score 85/100 on a numeric scale might only pass 60% of well-designed binary quality criteria -- which is actually more discriminating and useful for triage.

However, this also means binary questions and numeric scales are not measuring the same construct in the same way. Transitioning from numeric to binary scoring will shift the absolute score distribution, and calibration against historical data will require re-establishing thresholds.

---

## 7. When Scale Scoring Is Preferable

Not all evidence favors binary. Several findings suggest contexts where scale scoring works better:

### 7.1 Information-Rich Contexts

Godfrey et al. (2025) found that 11-point ordinal scales performed competitively with listwise ranking when the evaluation domain is well-defined (information retrieval relevance). In domains with established, widely-understood scales, LLMs can apply them more consistently.

### 7.2 Subjective Quality Dimensions

The Freeplay evaluation guide notes: "A range (e.g., 1 to 5) yields higher-quality and more useful data especially for subjective evaluations" where the difference between pass and fail is genuinely continuous rather than categorical ([Freeplay, 2024](https://freeplay.ai/blog/defining-the-right-evaluation-criteria-for-your-llm-project-a-practical-guide)).

### 7.3 Probability-Weighted Scoring

When token probability access is available, G-Eval-style probability weighting can produce continuous scores from integer scales, partially mitigating the clustering problem. This works best on smaller scales (1-5) where the probability distribution over tokens is more meaningful.

### 7.4 The Tradeoff

Scale scoring preserves more information per question but introduces more noise per question. Binary scoring loses information per question but can be compensated by asking more questions, each with higher reliability. The net effect depends on the number of binary questions relative to the information content of the scale points.

---

## 8. Actionable Recommendations for Our Content Scoring System

Based on the full body of evidence reviewed above, the following recommendations apply to our article/podcast scoring system that currently clusters scores at 80-100.

### 8.1 Replace Multi-Point Categories with Binary Questions (High Confidence)

The evidence overwhelmingly supports this change:
- Binary judgments achieve omega reliability of 0.98-0.99 vs. 0.42-0.80 for numeric scales (Schroeder & Wood-Doughty, 2024)
- Inter-evaluator agreement improves by 0.45 with binary decomposition (CheckEval, Lee et al., 2024)
- Production systems (Confident AI QAG, Criteria-Eval, Google Health) validate binary at scale
- Binary naturally counteracts the positivity bias causing our score inflation (Lu et al., 2025)

### 8.2 Design Questions at Multiple Difficulty Levels (High Confidence)

To produce a discriminating score distribution (not just binary pass/fail), design questions at varying stringency levels:

- **Easy questions** (most articles pass): "Does the article present at least one concrete fact, statistic, or specific example?" -- This separates content from pure opinion.
- **Medium questions** (roughly half pass): "Does the article challenge a commonly held assumption or present a finding that would surprise a knowledgeable reader?" -- This requires genuine novelty.
- **Hard questions** (few articles pass): "Does the article contain a specific passage that is worth saving verbatim for future reference?" -- This identifies truly exceptional content.

This approach is supported by IDGen (NeurIPS 2024), which applies Item Response Theory to LLM evaluation: questions need both discriminative power and varying difficulty levels to spread scores across the full range.

### 8.3 Score Each Dimension Independently (High Confidence)

Do not score multiple dimensions in a single prompt. Stureborg et al. (2024) found GPT-4 shows near-total anchoring (r = 0.979) when scoring multiple attributes together, vs. human anchoring of r = 0.315. Either:
- Make separate API calls for each dimension, or
- Ask binary questions one at a time within a single prompt but with no prior score context

### 8.4 Use Sufficient Question Count per Dimension (Medium Confidence)

With binary questions, the score is the proportion of "yes" answers. For meaningful gradations:
- 4 questions per dimension gives 5 possible scores (0%, 25%, 50%, 75%, 100%) -- minimum viable
- 8 questions per dimension gives 9 possible scores -- recommended for our 4-dimension system (32 total binary questions)
- More questions improve granularity but increase cost and latency

### 8.5 Include Adversarial/Negative Questions (Medium Confidence)

The DEBATE framework (Kim et al., 2024) showed that devil's advocate questioning improves scoring by 6-12 Spearman correlation points. Include questions designed to identify weaknesses:
- "Is the article primarily summarizing well-known information without adding new perspective?"
- "Does the article rely on vague generalities rather than specific evidence?"

These questions are scored in reverse (a "yes" means lower quality) and directly combat sycophantic inflation.

### 8.6 Set Temperature to 0 and Run Multiple Times for Critical Decisions (Medium Confidence)

For production scoring where cost permits, run each evaluation 3 times and use majority vote. At temperature 0, most binary judgments will be deterministic, but the ~9% flip rate (Tripathi et al., 2025) means occasional instability that majority voting resolves.

### 8.7 Consider Not Using 0-100 Final Scores (Low-Medium Confidence)

If the final consumer of scores is a triage system (high/medium/low), consider whether the intermediate 0-100 representation adds value. A system that reports "passed 6/8 quotability criteria, 3/8 surprise criteria, 7/8 argument criteria, 2/8 insight criteria" may be more interpretable and actionable than "quotability: 75, surprise: 38, argument: 88, insight: 25."

---

## Sources

### Primary Papers

- Stureborg, R., Alikaniotis, D., & Suhara, Y. (2024). "Large Language Models are Inconsistent and Biased Evaluators." [arXiv:2405.01724](https://arxiv.org/html/2405.01724v1)
- Schroeder, K. & Wood-Doughty, Z. (2024). "Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge." [arXiv:2412.12509](https://arxiv.org/html/2412.12509v2)
- Lu, Y-L., Zhang, C., & Wang, W. (2025). "Systematic Bias in Large Language Models: Discrepant Response Patterns in Binary vs. Continuous Judgment Tasks." [arXiv:2504.19445](https://arxiv.org/html/2504.19445)
- Godfrey, C. et al. (2025). "Likert or Not: LLM Absolute Relevance Judgments on Fine-Grained Ordinal Scales." [arXiv:2505.19334](https://arxiv.org/html/2505.19334v1)
- Li, Q. et al. (2025). "Evaluating Scoring Bias in LLM-as-a-Judge." [arXiv:2506.22316](https://arxiv.org/html/2506.22316v1)
- Fujinuma et al. (2025). "Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge." [arXiv:2510.18196](https://arxiv.org/abs/2510.18196)
- Liu, Y. et al. (2023). "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." [arXiv:2303.16634](https://ar5iv.labs.arxiv.org/html/2303.16634)
- Zheng, L. et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Lee et al. (2024). "CheckEval: Robust LLM Evaluation via Checklist." [arXiv:2403.18771](https://arxiv.org/abs/2403.18771)
- Yu, F. et al. (2025). "Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses." [arXiv:2509.16093](https://arxiv.org/html/2509.16093)
- Tripathi, T. et al. (2025). "Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation." COLM 2025. [arXiv:2504.14716](https://arxiv.org/abs/2504.14716)
- Metwally, A. A. & McDuff, D. (2025). "A Scalable Framework for Evaluating Health Language Models." [arXiv:2503.23339](https://arxiv.org/abs/2503.23339)

### Practitioner Reports and Guides

- Yan, E. (2024). "Evaluating the Effectiveness of LLM-Evaluators (aka LLM-as-Judge)." [eugeneyan.com](https://eugeneyan.com/writing/llm-evaluators/)
- Husain, H. (2024). "LLM Judge Guide." [hamel.dev](https://hamel.dev/blog/posts/llm-judge/)
- Wolfe, C. R. (2024). "Using LLMs for Evaluation." [cameronrwolfe.substack.com](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- Evidently AI. (2024). "LLM-as-a-judge: A Complete Guide." [evidentlyai.com](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- Confident AI. (2024). "LLM Evaluation Metrics: The Ultimate Guide." [confident-ai.com](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- Islam, N. (2024). "Scaling Evaluation with LLM Judges." [medium.com](https://medium.com/@nomannayeem/scaling-evaluation-with-llm-judges-our-approach-and-findings-0a046e8344c4)
- Freeplay. (2024). "Defining the Right Evaluation Criteria for Your LLM Project." [freeplay.ai](https://freeplay.ai/blog/defining-the-right-evaluation-criteria-for-your-llm-project-a-practical-guide)
- Google Research. (2025). "A Scalable Framework for Evaluating Health Language Models." [research.google](https://research.google/blog/a-scalable-framework-for-evaluating-health-language-models/)
