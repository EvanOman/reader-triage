# Decomposed and Atomic Evaluation

## Executive Summary

Decomposed evaluation -- breaking complex quality assessments into smaller, atomic sub-questions -- is one of the most empirically validated techniques for improving LLM-based scoring reliability. Research consistently shows that binary or low-arity questions outperform Likert-scale holistic scoring on inter-rater agreement, sensitivity to quality differences, and correlation with human judgment.

**Key findings for our system:**

- **Binary decomposition works.** CheckEval improved average inter-rater agreement by 0.45 across 12 evaluator models by replacing Likert scales with binary checklists ([Kim et al., 2024](https://arxiv.org/abs/2403.18771)). Google's Adaptive Precise Boolean rubrics achieved higher inter-rater reliability than Likert scales while cutting evaluation time by over 50% ([Pfohl et al., 2025](https://arxiv.org/abs/2503.23339)).
- **Separate criteria evaluated independently beat bundled scoring.** LLM-Rubric found that evaluating 9 criteria in separate calls, then calibrating with a neural network, explained ~75% of variance in human satisfaction -- roughly 2x the correlation of single-score baselines ([Hashemi et al., 2025](https://arxiv.org/html/2501.00274v1)).
- **The sweet spot for question count is 8-20 for our use case.** Below 8, discrimination suffers. Above 20, diminishing returns set in and computational costs escalate. The TICK framework generates variable-length checklists dynamically, but practitioners report that 3-5 independent evaluation dimensions (each potentially containing 2-5 sub-questions) provide the best tradeoff.
- **Our current 8-question single-call approach is a reasonable baseline** but has known limitations: attention dilution on later questions, halo effects between correlated criteria, and limited discrimination in the middle range.

**Recommendation:** Expand to 15-20 binary questions organized into 4-5 independent scoring dimensions, evaluated in separate LLM calls per dimension (4-5 calls total, not 15-20). This balances reliability gains against computational cost while mitigating the attention dilution and halo effects documented in single-call multi-criteria evaluation.

---

## 1. Foundations: Why Decompose?

### 1.1 The Problem with Holistic Scoring

Holistic LLM scoring -- asking a model to produce a single quality score on a numeric scale -- suffers from well-documented problems:

**Poor calibration.** LLMs generate text, not calibrated probabilities. Asking for a score of 73 vs. 82 on a 0-100 scale exceeds the natural precision of text generation ([Evidently AI, 2025](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)). Research on ordinal scales found that performance improvements plateau around 11 label points, suggesting that finer-grained numeric scales provide no additional signal ([Gupta et al., 2025](https://arxiv.org/abs/2505.19334)).

**Halo effects.** When multiple criteria are evaluated simultaneously, positive impressions in one dimension contaminate others. Research found that Claude 3 Opus exhibited significant halo effect bias in multi-dimensional evaluations, and that a criterion like "Clarity" scored a full point lower (4.0 vs. 5.0 on a 5-point scale) when evaluated holistically alongside other criteria compared to isolation ([Koo et al., 2025](https://arxiv.org/html/2412.05579v2)).

**Low discriminative power.** FActScore demonstrated that binary sentence-level judgment assigns 0.0 to both a mostly-accurate and a mostly-inaccurate output, while atomic-level decomposition differentiates them at 66.7% vs. 10% accuracy ([Min et al., 2023](https://arxiv.org/abs/2305.14251)). Coarse scoring hides meaningful quality differences.

### 1.2 The Decomposition Hypothesis

The core insight across the literature is that **evaluation tasks can be decomposed into simpler sub-tasks that LLMs handle more reliably**. This mirrors findings in reasoning research (chain-of-thought prompting) and task decomposition more broadly: models produce better outputs when they solve one simple problem at a time.

The TICK framework articulates this directly: "the decomposed task of answering a single, targeted question is much simpler than coming up with a holistic score or preference ranking" ([Cook et al., 2024](https://arxiv.org/abs/2410.03608)).

---

## 2. Key Approaches in the Literature

### 2.1 Atomic Fact Checking: FActScore

**Paper:** Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation" (EMNLP 2023).

FActScore decomposes generated text into "atomic facts" -- short sentences each conveying one piece of information -- then verifies each against a knowledge source.

**Key metrics:**
- ChatGPT generates ~4.4 atomic facts per sentence, with 40% of sentences containing a mix of supported and unsupported facts
- Generated biographies contain 26-41 atomic facts per response
- Automated estimation achieves <2% error rate vs. human judgment
- Atomic-level evaluation reveals quality differences invisible at sentence level

**Relevance to our system:** FActScore targets factual accuracy, not subjective quality assessment. However, its core insight -- that decomposition into atomic units dramatically improves discrimination -- applies directly. Our scoring questions should be as atomic as possible: each question should probe exactly one attribute.

### 2.2 Checklist-Based Evaluation: TICK and CheckEval

**TICK (Targeted Instruct-evaluation with Checklists)**
Paper: Cook et al., "TICKing All the Boxes" (2024).

TICK uses an LLM to generate checklists of YES/NO questions for evaluating instruction-following quality.

**Key findings:**
- Exact agreement with human preferences increased from 46.4% to 52.2% (a 12.5% relative improvement)
- Checklist length does not correlate with task difficulty -- it varies by instruction complexity
- Generated checklists matched or exceeded human-written checklist quality
- The approach also improved generation quality: +7.8% on LiveBench reasoning tasks via self-refinement

**CheckEval**
Paper: Kim et al., "CheckEval: A Reliable LLM-as-a-Judge Framework" (EMNLP 2025).

CheckEval translates evaluation criteria into binary (yes/no) checklists, tested across 12 evaluator models.

**Key findings:**
- Average inter-rater agreement improved by 0.45 over Likert-scale baselines
- Score variance reduced substantially
- Results strongly correlate with human judgments
- Traceable binary decisions allow analysis of specific attributes driving quality judgments

**Relevance:** Both frameworks validate our proposed direction of expanding to binary questions. The CheckEval result (0.45 improvement in agreement) is particularly compelling -- it suggests that simply reformulating our existing questions as binary checks would improve consistency.

### 2.3 Decomposed Criteria-Based Evaluation: DeCE

**Paper:** Thomson Reuters Labs, "Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses" (EMNLP 2025 Industry).

DeCE decomposes evaluation into precision (factual accuracy of claims) and recall (coverage of required information), with automatically extracted per-instance criteria.

**Key findings:**
- 979 criteria across 224 questions (~4-5 criteria per question on average)
- Pearson correlation with expert judgments: r=0.78, vs. pointwise LLM scoring r=0.35, G-Eval r=0.42
- Only 11.95% of auto-generated criteria needed expert revision
- Recall (r=0.80) and precision (r=0.69) each independently correlated with human judgment

**Relevance:** DeCE shows that even automatic criteria extraction works well. For our system, this suggests that criteria can be generated or refined programmatically rather than requiring hand-crafting of every sub-question.

### 2.4 Multi-Dimensional Calibrated Rubrics: LLM-Rubric

**Paper:** Hashemi et al., "LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation" (2025).

LLM-Rubric evaluates 9 criteria (8 specific + 1 overall quality) independently, then feeds all probability distributions into a calibration neural network.

**Key findings:**
- Uncalibrated single-score LLM: 0.85-0.90 RMSE
- LLM-Rubric: 0.40 RMSE (more than 2x improvement)
- Explains ~75% of variance in human satisfaction judgments
- **Each criterion is evaluated in a separate call** to avoid confounding
- Single-dimension calibration provided minimal improvement -- the multi-dimensional structure was essential

**Relevance:** This is the most directly applicable framework for our system. It demonstrates that (a) separate calls per criterion are worth the cost, (b) ~9 dimensions is a productive range, and (c) learned aggregation outperforms simple averaging. Our 4-dimension structure with 8 sub-questions maps cleanly to this approach.

### 2.5 Boolean Rubrics in Healthcare: Google's Adaptive Framework

**Paper:** Pfohl et al., "A Scalable Framework for Evaluating Health Language Models" (2025).

Google developed Adaptive Precise Boolean rubrics for healthcare LLM evaluation, converting complex assessments into dynamically filtered yes/no questions.

**Key findings:**
- Higher inter-rater reliability (ICC) than traditional Likert rubrics
- **Evaluation time reduced by over 50%** -- faster than even Likert-scale evaluation
- Boolean scores showed "clear, positive correlation" with input quality, while Likert scores "showed limited sensitivity" to quality differences
- Automated rubric question classifier (Gemini zero-shot): accuracy 0.77, F1 0.83
- Adaptive filtering retains only relevant criteria per instance, reducing total evaluations needed

**Relevance:** The adaptive filtering concept is particularly interesting for our system. Not all scoring questions are relevant to all articles (e.g., "named framework" is irrelevant for news roundups). Dynamically selecting applicable questions per content type could improve both accuracy and efficiency.

### 2.6 Rubric-Based Fine-Grained Evaluation: Prometheus

**Paper:** Kim et al., "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models" (ICLR 2024).

Prometheus is an open-source evaluator LLM trained specifically for customized rubric-based scoring.

**Key findings:**
- Pearson correlation with human evaluators: 0.897 (on par with GPT-4 at 0.882)
- Dramatically outperforms ChatGPT (0.392) as an evaluator
- Trained on 1K fine-grained score rubrics, demonstrating the value of rubric diversity
- Takes customized score rubric as input, enabling per-task evaluation criteria

**Relevance:** Demonstrates that purpose-built evaluation models can match frontier model performance at lower cost. If we move to 15-20 binary questions, using a fine-tuned smaller model as the evaluator could reduce per-article scoring costs significantly.

### 2.7 Claim Decomposition in RAG: RAGAS and ARES

**RAGAS** computes faithfulness as the ratio of supported claims to total claims, using LLM-based claim extraction and individual verification ([RAGAS Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)).

**ARES** fine-tunes lightweight LMs with binary classifier heads to evaluate RAG components, with separate classifiers for context relevance, answer faithfulness, and answer relevance ([Saad-Falcon et al., 2024](https://arxiv.org/abs/2311.09476)).

Both demonstrate that decomposing evaluation into claim-level binary judgments is the dominant paradigm in retrieval-augmented generation evaluation.

---

## 3. Critical Design Questions

### 3.1 How Many Sub-Questions Are Optimal?

The literature does not converge on a single number, but patterns emerge:

| Range | Evidence | Tradeoffs |
|-------|----------|-----------|
| **3-5 criteria** | Monte Carlo, Evidently AI recommend max 5 metrics for LLM judges; LLM-Rubric uses 9 but evaluates separately | Low cost, high focus, may miss important dimensions |
| **5-10 criteria** | Our current system (8 questions); DeCE averages ~4-5 per instance; Prometheus trained on rubrics of varying size | Good balance for single-call evaluation; approaching attention dilution limits |
| **10-20 criteria** | Google's Precise Boolean rubrics; TICK generates variable-length checklists; CheckEval uses binary decomposition | Requires separate calls or grouping; strong discrimination; manageable cost with batching |
| **20-50 criteria** | Google's full (non-adaptive) Boolean rubric; FActScore generates 26-41 atomic facts per response | High discrimination but diminishing returns; requires adaptive filtering to be practical |

**Practical guidance:**
- The "5 metrics max" guidance applies to **independent evaluation dimensions** in a single call, not to the total number of binary sub-questions
- Binary questions can be batched in groups of 3-5 within a single call without significant attention dilution
- 15-20 total binary questions, organized into 4-5 dimensions of 3-5 questions each, appears to be the sweet spot based on the convergence of LLM-Rubric, Google's approach, and CheckEval

### 3.2 Independent Calls vs. Single Call with All Questions

**Evidence strongly favors separate calls per dimension:**

| Approach | Pros | Cons | Evidence |
|----------|------|------|----------|
| **One call, all questions** | Minimal cost; simple implementation | Halo effects; attention dilution on later questions; order bias | Score drops of ~3.5% for criteria evaluated last vs. first ([Koo et al., 2025](https://arxiv.org/html/2412.05579v2)) |
| **Separate call per question** | Maximum independence; no cross-contamination | High cost (N calls per article); potentially slow | ARES uses separate classifiers per metric |
| **Grouped calls by dimension** | Good independence between dimensions; manageable cost; questions within a group are related so mild contamination is acceptable | Moderate complexity | LLM-Rubric evaluates 9 criteria in separate calls; Google uses dimension-grouped boolean rubrics |

**Recommendation:** Group 3-5 related binary questions per call, with 4-5 calls per article. This gives us 15-20 total questions in 4-5 API calls -- a 4-5x cost increase over our current single call, but with substantially better reliability and discrimination.

For our current 4-dimension structure (Quotability, Surprise Factor, Argument Quality, Applicable Insight), each dimension would get its own call with 3-5 binary sub-questions, eliminating cross-dimension halo effects.

### 3.3 Does Decomposition Actually Improve Reliability and Discrimination?

**Yes, with strong evidence:**

| Study | Metric | Improvement |
|-------|--------|-------------|
| CheckEval | Inter-rater agreement | +0.45 over Likert baselines |
| TICK | Human preference agreement | 46.4% -> 52.2% exact match |
| DeCE | Correlation with experts | 0.35 -> 0.78 (Pearson r) |
| LLM-Rubric | RMSE | 0.85 -> 0.40 (2x improvement) |
| Google Boolean | Sensitivity to quality | Boolean detected quality differences Likert missed |
| FActScore | Discrimination | Binary sentence scores: 0.0 vs. 0.0; atomic: 66.7% vs. 10% |

The evidence is consistent across domains (legal QA, instruction following, healthcare, factual accuracy, dialogue quality) and across evaluation paradigms (human correlation, inter-rater agreement, discrimination sensitivity).

**However, there are caveats:**
- Decomposition works best when sub-questions are truly independent -- correlated questions reintroduce halo effects
- The quality of the decomposition matters more than the quantity -- poorly designed binary questions can be worse than a well-designed holistic rubric
- Gains are largest when moving from holistic to decomposed; further decomposition shows diminishing returns

### 3.4 Computational Cost Tradeoffs

Expanding from 1 call to 4-5 calls per article has concrete cost implications:

**Token economics (estimated for our system):**
- Current: 1 call with ~15K input tokens (article + prompt) + ~200 output tokens = ~15.2K tokens
- Proposed: 4-5 calls with ~15K input tokens each (article must be included in each) + ~100 output tokens each = ~60-75K tokens
- **Cost multiplier: ~4-5x per article**

**Latency considerations:**
- Calls per dimension can be parallelized (they are independent)
- With parallel execution, wall-clock latency increases only marginally over the single-call approach
- Batch API pricing (where available) can reduce per-token costs by 50%

**Mitigation strategies:**
1. **Parallel execution:** Fire all 4-5 dimension calls simultaneously
2. **Smaller models for binary questions:** Binary yes/no evaluation may work with cheaper models (Haiku-class) since the task is simpler
3. **Adaptive question selection:** Skip irrelevant questions based on content type (e.g., skip "named framework" for news roundups)
4. **Caching article embeddings:** If the article text is the dominant token cost, explore whether a summary or key excerpt can replace full text for some dimensions
5. **Progressive scoring:** Score the most discriminative dimension first; if the article clearly scores very high or very low, skip remaining dimensions

---

## 4. Analysis of Our Current System

Our current system asks 8 questions in a single LLM call, grouped into 4 scoring dimensions:

| Dimension | Questions | Type |
|-----------|-----------|------|
| Quotability (0-25) | Q1: Standalone passages | 4-option categorical (none/a_few/several/many) |
| Surprise (0-25) | Q2: Novel framing | Binary (true/false) |
| | Q3: Content type | 5-option categorical |
| Argument (0-25) | Q4: Author conviction | Binary (true/false) |
| | Q5: Practitioner voice | Binary (true/false) |
| | Q6: Content completeness | 3-option categorical |
| Insight (0-25) | Q7: Named framework | Binary (true/false) |
| | Q8: Applicable ideas | 3-option categorical (broadly/narrowly/not_really) |

**Strengths of the current approach:**
- Questions are already relatively atomic -- each probes one specific attribute
- Mix of binary and categorical matches research recommendations
- Categorical-to-numeric mapping avoids asking the LLM for raw numbers
- Single call is cost-efficient

**Weaknesses identified by the literature:**
1. **Cross-dimension contamination:** All 8 questions in one call allows halo effects between Quotability and Argument Quality, for example
2. **Limited discrimination in the middle range:** With only 2-3 questions per dimension, the score space per dimension has few possible values, creating clustering
3. **Order effects:** Later questions (Q7, Q8 for Insight) may receive less careful evaluation due to attention dilution
4. **Fixed question set:** All questions are asked regardless of content type (aside from Q3/Q6 variants), even when some are irrelevant

---

## 5. Concrete Recommendations

### 5.1 Proposed Architecture: Dimension-Grouped Binary Evaluation

Expand to **4 separate LLM calls**, one per scoring dimension, each containing **4-5 binary questions**. Total: 16-20 binary questions.

**Quotability Dimension (5 questions):**
1. Does the content contain at least one passage that could stand alone as a memorable quote?
2. Does the content include specific data points, statistics, or quantified claims?
3. Does the content use vivid metaphors, analogies, or memorable phrasings?
4. Are there concrete examples or case studies that illustrate abstract points?
5. Could you extract a "key takeaway" sentence that would be worth saving?

**Surprise Dimension (4 questions):**
1. Does the content challenge a common assumption or conventional wisdom?
2. Does the content present a genuinely unexpected finding or counterintuitive result?
3. Does the content reframe a familiar topic through an unusual lens or perspective?
4. Would a knowledgeable reader in this domain find new information here?

**Argument Quality Dimension (4 questions):**
1. Does the author argue for a clear, identifiable thesis or position?
2. Are claims supported with evidence (data, citations, examples)?
3. Is this written from first-person practitioner experience rather than secondhand reporting?
4. Does the piece feel like a complete argument rather than a truncated excerpt?

**Applicable Insight Dimension (4 questions):**
1. Does the content introduce or describe a named framework, mental model, or methodology?
2. Could a reader directly apply a technique or approach described here?
3. Does the content provide actionable recommendations or decision criteria?
4. Does the content offer a perspective shift that would change how someone thinks about a problem?

### 5.2 Scoring Aggregation

For each dimension (0-25 points):
- Each binary "yes" = base points (25 / number_of_questions_in_dimension)
- Sum binary scores within each dimension
- Total = sum of 4 dimensions (0-100)

Alternatively, follow the LLM-Rubric approach: collect all binary answers, then use a simple learned calibration (even logistic regression) to predict overall quality, trained on a small set of human-labeled examples.

### 5.3 Implementation Phases

**Phase 1: Validate with parallel calls (low risk)**
- Keep current 8 questions but split into 4 calls (one per dimension)
- Compare reliability metrics (score variance across re-evaluations of the same article) vs. single-call baseline
- Estimated cost increase: ~4x per article

**Phase 2: Expand to binary questions (medium effort)**
- Replace current mixed questions with 16-20 binary questions
- Run both old and new scoring on a sample of 50-100 articles
- Measure discrimination improvement (score distribution spread, rank correlation with human preference)

**Phase 3: Optimize costs (after validation)**
- Test smaller/cheaper models for binary evaluation (the task is simpler)
- Implement adaptive question selection based on content type
- Explore progressive scoring (early termination for clearly high/low-scoring articles)

### 5.4 What to Monitor

- **Score distribution:** Current system likely shows clustering around certain values due to limited score space. Expanded binary questions should produce a more continuous distribution.
- **Re-score stability:** Score the same article multiple times and measure variance. Binary decomposition should reduce variance (per CheckEval findings).
- **Rank correlation:** Compare article rankings between old and new systems against human preference. Decomposed scoring should better separate articles in the middle range.
- **Dimension independence:** Measure correlation between dimension scores. High correlation suggests halo effects or redundant questions.
- **Cost per article:** Track actual API costs and latency under the new system.

---

## 6. Summary of Literature

| Framework | Year | Approach | Key Result | Citation |
|-----------|------|----------|------------|----------|
| FActScore | 2023 | Atomic fact decomposition + binary verification | <2% error vs. human; reveals quality differences hidden by sentence-level scoring | [Min et al.](https://arxiv.org/abs/2305.14251) |
| Prometheus | 2024 | Rubric-based fine-grained evaluation LLM | r=0.897 with humans (GPT-4 level) | [Kim et al.](https://arxiv.org/abs/2310.08491) |
| TICK | 2024 | Generated YES/NO checklists | 46.4% -> 52.2% human agreement | [Cook et al.](https://arxiv.org/abs/2410.03608) |
| ARES | 2024 | Binary classifiers per RAG dimension | Accurate with ~100s of human annotations | [Saad-Falcon et al.](https://arxiv.org/abs/2311.09476) |
| DeCE | 2025 | Precision-recall criteria decomposition | r=0.78 vs. pointwise r=0.35 | [Thomson Reuters Labs](https://arxiv.org/abs/2509.16093) |
| LLM-Rubric | 2025 | 9 independent criteria + calibration | RMSE 0.85 -> 0.40; explains ~75% variance | [Hashemi et al.](https://arxiv.org/html/2501.00274v1) |
| CheckEval | 2025 | Binary checklist decomposition | +0.45 inter-rater agreement | [Kim et al.](https://arxiv.org/abs/2403.18771) |
| Google Boolean | 2025 | Adaptive boolean rubrics (healthcare) | Higher ICC than Likert; 50% faster evaluation | [Pfohl et al.](https://arxiv.org/abs/2503.23339) |
| Likert or Not | 2025 | Ordinal scale size comparison | 11-point scale matches listwise ranking | [Gupta et al.](https://arxiv.org/abs/2505.19334) |

---

## 7. Additional References

- [Evidently AI: LLM-as-a-Judge Complete Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) -- Practical recommendations on binary vs. scaled scoring and separate evaluator design.
- [Confident AI: G-Eval Definitive Guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide) -- Chain-of-thought evaluation and sub-criteria decomposition.
- [Eugene Yan: Evaluating LLM Evaluators](https://eugeneyan.com/writing/llm-evaluators/) -- Meta-analysis of evaluator effectiveness; fine-grained criteria outperform general criteria.
- [Monte Carlo: LLM-as-Judge Best Practices](https://www.montecarlodata.com/blog-llm-as-judge/) -- Operational guidance; max 5 metrics per evaluator call.
- [Snorkel AI: The Right Tool for the Job (Rubrics A-Z)](https://snorkel.ai/blog/the-right-tool-for-the-job-an-a-z-of-rubrics/) -- Rubric granularity/specificity design tradeoffs; recommends 5-7 point scales.
- [Koo et al.: LLMs-as-Judges Comprehensive Survey](https://arxiv.org/html/2412.05579v2) -- Documents order bias and halo effects in multi-criteria single-call evaluation.
- [RAGAS Faithfulness Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) -- Claim-level binary verification for RAG evaluation.
- [CLEVER Framework](https://ai.jmir.org/2025/1/e72153) -- Clinical evaluation with decomposed dimensions (factuality, clinical relevance, conciseness).
- [Multi-Trait Essay Scoring with LLMs](https://arxiv.org/html/2410.14202v2) -- Trait-specific rubrics and rationale generation for multi-dimensional writing assessment.
