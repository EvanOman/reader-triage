# Binary-Weighted LLM Scoring: Research Findings

## Executive Summary

This research investigates whether replacing the current v2-categorical scoring approach (8 multi-class questions mapped to point buckets) with a binary yes/no question system using calibrated weights would improve score discrimination and reduce ceiling effects.

**Conclusion: Yes, binary-weighted scoring is strongly supported by both academic research and practitioner experience.** The evidence is consistent across multiple independent sources that binary (yes/no) LLM judgments are more reliable, more consistent, and more interpretable than multi-class categorical or numeric judgments.

---

## Key Findings

### 1. Binary judgments are more reliable than categorical or numeric judgments

Arize AI's empirical testing (2025) across four frontier models found that numeric scores "yielded plateaus, discontinuous jumps, and long stretches where passages with very different error rates landed on the same number." Binary labels were the most stable format, with low variance across runs. Hamel Husain states flatly: "If your evaluations consist of metrics that LLMs score on a 1-5 scale or any other scale, you're doing it wrong."

### 2. CheckEval demonstrates that binary decomposition dramatically improves evaluator agreement

The CheckEval framework (EMNLP 2025) improved inter-evaluator agreement by 0.45 (Krippendorff's alpha) compared to G-Eval's Likert-scale approach, achieving agreement levels comparable to human raters (~0.7). The approach: decompose evaluation criteria into Boolean yes/no checklists, then score = percentage of "yes" answers.

### 3. Penalty questions are essential for avoiding ceiling effects

The current v2-categorical system has no mechanism for actively pulling scores *down*. Every question's minimum contribution is 0. The DEBATE framework (ACL 2024) shows that adversarial/critical evaluation signals improve scoring accuracy by 6-12 percentage points. Our proposed system includes 4 penalty questions (where "yes" reduces the score) to actively differentiate commodity content from quality content.

### 4. Weights can be calibrated from engagement data

Starting with expert-assigned weights, we can use logistic regression on historical highlight data to learn which questions best predict actual engagement. The existing calibration toolkit (`just cal-*`) provides the infrastructure for this analysis. Approximately 200 scored articles with engagement data are needed for stable weight estimation.

### 5. The approach is production-validated

Multiple production systems use binary checklist scoring: Criteria-Eval (Samaya AI, financial analysis), DeepEval's QAG metrics, GoDaddy's Rubrics as Rewards framework, and ARES (Stanford, RAG evaluation). This is not a novel research technique -- it is established practice.

---

## Projected Impact on Score Distribution

| Metric | v2-categorical (current) | v3-binary (projected) |
|--------|-------------------------|----------------------|
| Score range used | 50-100 (narrow) | 15-90 (wide) |
| Median score | ~80 | ~45-50 |
| Score where "high value" starts | 60 (most pass) | 60 (selective) |
| False positive rate (high score, no highlights) | High | Lower (penalties + hard questions) |
| Score resolution | ~10 distinct values common | ~30+ distinct values |

---

## Research Documents

| Document | Contents |
|----------|----------|
| [literature-review.md](literature-review.md) | Detailed review of 12 papers and practitioner sources on binary LLM evaluation |
| [question-design.md](question-design.md) | Principles for designing discriminating binary questions, anti-ceiling strategies, concrete question examples per dimension |
| [weighting-strategies.md](weighting-strategies.md) | Five weighting approaches (equal, expert, data-calibrated, hierarchical, Bayesian), phased recommendation, dimension sub-score computation |
| [proposed-approach.md](proposed-approach.md) | Complete implementation proposal: 20 questions with weights, prompt template, score computation, podcast variants, calibration plan, implementation roadmap |

---

## Implementation Recommendation

Implement v3-binary as a new `ScoringStrategy` class alongside the existing scorer. Run in shadow mode for ~1 week to collect parallel scores, then calibrate against engagement data using the existing calibration toolkit before switching over. The full rollout plan is detailed in [proposed-approach.md](proposed-approach.md).

---

## Key Sources

### Papers
- CheckEval: [arxiv.org/abs/2403.18771](https://arxiv.org/abs/2403.18771) -- Checklist-based binary evaluation, EMNLP 2025
- DEBATE: [arxiv.org/abs/2405.09935](https://arxiv.org/abs/2405.09935) -- Devil's advocate scoring, ACL 2024
- G-Eval: [confident-ai.com/blog/g-eval-the-definitive-guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide) -- Chain-of-thought decomposed evaluation
- ARES: [arxiv.org/abs/2311.09476](https://arxiv.org/abs/2311.09476) -- Binary classification for RAG evaluation
- IDGen: [arxiv.org/html/2409.18892](https://arxiv.org/html/2409.18892) -- Item discrimination theory for LLM evaluation
- Likert or Not: [arxiv.org/abs/2505.19334](https://arxiv.org/abs/2505.19334) -- Ordinal scales for relevance judgments
- Sycophancy in LLMs: [arxiv.org/html/2411.15287v1](https://arxiv.org/html/2411.15287v1) -- Causes and mitigations

### Practitioner Guides
- Hamel Husain: [hamel.dev/blog/posts/llm-judge/](https://hamel.dev/blog/posts/llm-judge/) -- Binary pass/fail advocacy
- Eugene Yan: [eugeneyan.com/writing/llm-evaluators/](https://eugeneyan.com/writing/llm-evaluators/) -- LLM evaluator effectiveness
- Arize AI: [arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/) -- Binary vs score empirical comparison
- Evidently AI: [evidentlyai.com/llm-guide/llm-as-a-judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) -- LLM judge comprehensive guide
- GoDaddy: [godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge) -- Production calibration techniques
- Criteria-Eval: [samaya.ai/blog/criteria-eval](https://samaya.ai/blog/criteria-eval) -- Production binary checklist for financial analysis
- Cameron Wolfe: [cameronrwolfe.substack.com/p/llm-as-a-judge](https://cameronrwolfe.substack.com/p/llm-as-a-judge) -- Rubric-based evaluation best practices

### Frameworks
- DeepEval: [deepeval.com/docs/metrics-introduction](https://deepeval.com/docs/metrics-introduction) -- QAG binary scoring implementation
- AlpacaEval: [github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) -- Binary preference with weighted aggregation
