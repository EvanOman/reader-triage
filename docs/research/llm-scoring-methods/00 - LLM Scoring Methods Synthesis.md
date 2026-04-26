# LLM Scoring Methods: Research Synthesis

**Purpose:** Inform the redesign of Reader Triage's article scoring system, with emphasis on binary scoring approaches and calibration against reader highlight behavior.
**Provenance:** AI-generated synthesis of 12 research reports. Treat as a well-researched starting point, not a final decision.

---

## The One-Paragraph Summary

The current v2-categorical scoring system's ceiling effect (most articles scoring 80-100) is not a rubric bug -- it is an inherent property of asking LLMs to produce numeric or fine-grained categorical scores. The evidence overwhelmingly supports decomposing evaluation into 15-20 binary (yes/no) questions at varying difficulty levels, organized into 4-5 dimensions. This binary decomposition produces more stable, discriminating, and human-aligned scores across every framework and study examined. The questions should be designed using Item Response Theory principles with three tiers (baseline questions most articles pass, quality questions ~40% pass, exceptional questions ~15% pass) plus devil's advocate questions that deflate scores for mediocre content. Weights should start with expert priors, then be calibrated against highlight counts using logistic regression with bootstrap confidence intervals. Three immediate prompt engineering changes -- setting temperature=0, moving the rubric to a system prompt, and restructuring JSON to put reasoning before answers -- will improve the current system's consistency with zero architectural change.

---

## Key Findings

### 1. LLM Score Compression Is Systematic and Unsolvable at the Scale Level

Our ceiling effect is not a prompt engineering failure. Research consistently shows that LLMs systematically compress scores into a narrow band at the top of any numeric scale. Stureborg et al. (2024) found GPT-4's inter-sample agreement at Krippendorff's alpha of 0.587 -- below human inter-annotator agreement. Production practitioners report scores "collapsed into a meaningless 7-9 range" (Islam, 2024). Fujinuma et al. (2025) showed that merely changing the scale labels (0-4 vs 1-5 vs 2-6) shifts the output distribution, proving the scores are not calibrated to any objective standard.

Most critically, this is a general LLM property: variance compression occurs regardless of model family, prompt engineering, or scale design. The only reliable mitigation is to avoid asking for numeric scores entirely.

**Convergence across reports:** [[01]], [[02]], [[04]], [[07]], [[09]], [[12]] all independently confirm this finding from different angles.

### 2. Binary Decomposition Is the Consensus Solution

The field has converged on decomposed binary evaluation as the most reliable LLM-as-judge approach:

- **CheckEval** improved inter-rater agreement by +0.45 over Likert scales across 12 evaluator models (Kim et al., 2024)
- **DeCE** achieved Spearman r=0.78 with human judgments using full decomposition (vs. r=0.514 for G-Eval's scale-based approach)
- **Google's Adaptive Precise Boolean** rubrics achieved higher inter-rater reliability while cutting evaluation time by 50%+ (Pfohl et al., 2025)
- **LLM-Rubric** explained ~75% of variance in human satisfaction using 9 independently-evaluated binary criteria (Hashemi et al., 2025)
- **Hamel Husain** (after consulting 30+ companies): "If your evaluations consist of metrics on a 1-5 scale, you're doing it wrong"
- **Arize AI** (2025 empirical testing): Binary labels "consistently separate clean from corrupted passages, with low variance across runs"

The sweet spot is 15-20 binary questions in 4-5 dimensions. Below 8, discrimination suffers. Above 20, diminishing returns set in. Evaluating dimensions in separate API calls is ideal but expensive; our single-call approach is defensible when questions are well-structured.

**Convergence across reports:** [[01]], [[02]], [[03]], [[06]], [[07]]

### 3. Anti-Ceiling-Effect Techniques Require Tiered Difficulty + Devil's Advocate Questions

Even with binary scoring, naive question design will reproduce the ceiling effect if every question is easy enough that most articles pass. Two complementary techniques break the ceiling:

**Tiered difficulty** (from Item Response Theory):
- **Baseline questions** (target pass rate 70-85%): things most decent articles have. Score points for fundamental quality.
- **Quality questions** (target pass rate 30-50%): discriminate good from average. These are where the scoring work happens.
- **Exceptional questions** (target pass rate 10-20%): only the best articles pass. These prevent ceiling clustering.

**Devil's advocate questions** (from DEBATE framework): questions where "yes" *reduces* the score. LLMs have a well-documented agreeableness bias (True Positive Rate >96%, True Negative Rate <25%), so phrasing questions adversarially exploits this bias for score deflation rather than inflation. Example: "Does the article primarily restate well-known positions without adding new evidence or framing?" -- a "yes" here should subtract points.

The combination of tiered difficulty and adversarial questions produced +6-12 percentage points improvement in human correlation in the DEBATE framework.

**Primary sources:** [[04]], [[06]]

### 4. Highlight Count Is the Right Dependent Variable, With Caveats

For calibrating scores against engagement, `num_highlights` (raw count) is the best primary metric:

- Each highlight represents a deliberate reader decision to save a passage -- the closest proxy to what our scoring system tries to predict
- It is simple, available from the Readwise API, and already implemented in our calibration toolkit
- Highlighted word count is too sensitive to highlighting style (sentence vs. paragraph highlighters)
- Highlight density (per 1k words) is a valuable secondary metric for length-controlled confirmation

**Critical caveats:**
- **Reading progress must be controlled for.** Unread articles have zero highlights by definition, not because they lack value. Default minimum reading progress should be 0.5 (currently 0.0).
- **30-day settlement period** needed for recent articles -- highlights are often added days/weeks after reading.
- **Zero-inflation** is substantial: many articles have 0 highlights. Consider a hurdle model (separate "did they engage at all?" from "how much?") for advanced analysis.
- **Category confounding**: technical tutorials naturally get more highlights than opinion essays. Always report per-category correlations; check for Simpson's paradox.

**Correlation metrics:** Spearman rank correlation (primary), Kendall tau-b (secondary), Pearson (diagnostic). With 500+ articles, we have adequate power for overall and per-category analysis.

**Primary sources:** [[05]], [[08]]

### 5. Weight Calibration Should Use a Phased Approach

The research supports a three-phase calibration strategy:

**Phase 1 (Cold Start):** Use expert-assigned weights as priors. This is where we are now. The current point mappings in `scoring_strategy.py` represent domain knowledge about what predicts highlight-worthy content.

**Phase 2 (Initial Calibration, 100+ articles):** Run logistic regression with binary question responses as features and highlight count as the dependent variable. This directly produces question weights. Use bootstrap analysis (1000 resamples) to get confidence intervals on each weight. Questions whose CI includes zero are not contributing signal -- consider dropping or replacing them.

**Phase 3 (Ongoing Monitoring):** Track Spearman correlation monthly using temporal holdout validation (calibrate on oldest 70%, validate on newest 30%). Alert if overall rho drops below 0.20 or tier gradient becomes non-monotonic.

The key insight from [[05]] is that weight calibration and LLM question design are complementary, not alternatives. The LLM decides what's true about the article (binary answers); the regression decides how much each truth matters for predicting engagement.

**Primary sources:** [[05]], [[08]]

### 6. Three Prompt Engineering Changes Will Improve the Current System Immediately

Before any architectural changes, three modifications to the existing v2-categorical scorer will improve consistency:

| Change | Impact | Effort |
|--------|--------|--------|
| **Set `temperature=0`** | High -- eliminates random variance across runs | One line of code |
| **Move rubric to system prompt** | Medium -- better rubric adherence, enables prompt caching | Moderate refactor |
| **Restructure JSON: reasoning before answers** | Medium -- genuine CoT instead of post-hoc rationalization | Moderate refactor |

Additional medium-priority changes: add quantitative definitions for categorical options (what does "a_few" mean?), add an independence instruction ("evaluate each question independently"), and add 2-3 few-shot calibration examples spanning the score range.

Our current single-call, 8-question approach is defensible. The literature recommends separate calls per dimension, but the cost savings of one call outweigh the marginal accuracy gain for our volume. The current `max_tokens=700` and "brief reason" fields are well-calibrated.

**Primary source:** [[10]], with supporting evidence from [[02]], [[09]]

### 7. Production Systems Validate Binary Scoring at Scale

Real-world deployments confirm the research findings:

- **Pinterest** uses binary pass/fail with golden datasets for calibration, achieving 30x cost savings over human review
- **Ramp** achieved 99% accuracy on financial document classification using binary LLM scoring, deployed with a shadow mode validation period
- **Karpathy's HN Capsule** scored 930 articles for $58 using LLM evaluation with binary-style relevance filtering
- **Honeycomb** reached >90% agreement with human evaluators in 3 iterations of binary query quality assessment
- **Feedly Leo** uses binary priority classification (priority/not) for news filtering -- the closest production analog to our use case
- **Readwise Reader** notably avoids pre-read scoring entirely, filtering at the selection/subscription level instead

The consistent pattern: systems that ship successfully use binary or coarse scales, calibrate against human ground truth, and monitor for drift.

**Primary source:** [[09]]

---

## Recommendations

### Immediate (This Week)

1. **Set `temperature=0`** in `scoring_strategy.py`. Single highest-ROI change.
2. **Move rubric to system prompt.** Enables prompt caching and cleaner separation.
3. **Raise calibration defaults:** `--min-progress 0.5`, `MIN_CORR_N = 25`, add 30-day settlement period.

### Short-Term (Next Sprint)

4. **Design v3-binary question set.** 15-20 binary questions across 4 dimensions with three difficulty tiers (baseline/quality/exceptional) plus 3-4 devil's advocate questions. Use the question design framework from [[06]].
5. **Implement as a new `BinaryScoringStrategy`.** The existing strategy pattern infrastructure supports running v2 and v3 side-by-side.
6. **Add Kendall tau-b and bootstrap CIs** to calibration reports.

### Medium-Term (Next Month)

7. **Run parallel scoring.** Score new articles with both v2-categorical and v3-binary for 2-4 weeks. Compare Spearman correlations with highlight counts.
8. **Run initial weight calibration** using logistic regression on v3 results once 100+ articles are scored.
9. **Add few-shot calibration examples** to the scoring prompt (2-3 examples spanning the score range).

### Long-Term

10. **Implement temporal holdout validation** (`cal-validate` command).
11. **Set up monthly automated calibration monitoring** with alert thresholds.
12. **Consider category-specific scoring weights** if per-category correlations diverge significantly.

---

## Risks and Concerns

1. **Binary sycophancy.** LLMs have a well-documented affirmative bias. If binary questions are poorly designed (too easy, positively framed), 20 "yes" answers produce the same ceiling effect with more steps. Mitigation: tiered difficulty + devil's advocate questions + empirical pass-rate monitoring.

2. **Highlight count is an imperfect proxy.** Some high-value articles (opinion essays, narrative pieces) don't get highlighted because their value is in the arc, not extractable passages. The scoring system may learn to optimize for highlightable content rather than genuinely valuable content. Mitigation: per-category analysis, partial correlations.

3. **Overfitting weights to historical data.** With only 500 articles and 15-20 features, there is real risk of overfitting the weight calibration. Mitigation: temporal holdout validation, bootstrap CIs, regularization (Bayesian Ridge or L2).

4. **Temporal drift.** Both reading habits and model behavior change over time. A well-calibrated system today may drift. Mitigation: monthly monitoring with alert thresholds, scoring version tracking.

5. **Cost multiplication.** Moving from 8 questions in one call to 15-20 binary questions (even in one call) increases token count. If split across separate dimension calls, cost multiplies 4-5x. Our current volume is low enough that this is manageable.

---

## Cost and Effort Estimates

| Change | API Cost Impact | Dev Effort |
|--------|----------------|------------|
| Temperature=0 | None | 1 line |
| System prompt split | Slight reduction (caching) | 2-3 hours |
| v3-binary question design | None (design work) | 4-6 hours |
| BinaryScoringStrategy | ~20-40% more input tokens | 4-6 hours |
| Weight calibration script | None | 3-4 hours |
| Calibration toolkit improvements | None | 2-3 hours |
| Parallel scoring (v2 + v3) | 2x scoring cost for 2-4 weeks | 1-2 hours (infra exists) |

At current Sonnet pricing ($3/M input, $15/M output) and our ~50-100 articles/day volume, even doubling the per-article token count adds <$1/day.

---

## Dead Ends and Low-Value Paths

**Pairwise comparison scoring.** Several reports ([[01]], [[12]]) noted that pairwise comparison (comparing two articles head-to-head) is theoretically superior to pointwise scoring. However, pairwise requires O(n^2) comparisons, is impractical for our daily volume, and doesn't produce absolute scores needed for threshold-based filtering. Dead end for our use case.

**Self-consistency (multi-run voting) in production.** Running each evaluation 3-5 times and taking majority vote improves reliability, but multiplies cost proportionally. With temperature=0 and well-designed binary questions, single-run consistency should be adequate. Valuable only during calibration/development, not production.

**Embedding-based scoring.** Report [[12]] noted that embedding similarity can predict engagement. However, this requires a separate ML pipeline, training data, and infrastructure -- the separate regression model research already covers this territory. Not pursued here to avoid overlap.

**Fine-tuning a judge model.** Training a smaller model specifically for our scoring task would reduce costs and improve consistency. However, with only 500 labeled articles and an evolving rubric, we lack the data volume and stability for fine-tuning to be practical. Revisit when we have 5,000+ scored articles with stable highlight data.

**Complex multi-agent debate architectures.** The DEBATE framework's multi-agent architecture (Commander + Scorer + Critic) improves accuracy but adds 3x+ cost and latency per article. The key insight (devil's advocate questions) can be embedded directly in the scoring prompt without the multi-agent overhead.

---

## Report Index

| # | Report | Summary |
|---|--------|---------|
| 01 | [[01 - Binary vs Scale LLM Judgments]] | Evidence that LLM numeric scores are systematically unreliable; binary decomposition solves this |
| 02 | [[02 - LLM-as-Judge Framework Survey]] | Survey of 7 frameworks (G-Eval, DeCE, AlpacaEval, etc.); field converging on decomposed binary |
| 03 | [[03 - Decomposed and Atomic Evaluation]] | CheckEval, LLM-Rubric, TICK; sweet spot is 15-20 binary questions in 4-5 dimensions |
| 04 | [[04 - Anti-Ceiling-Effect Techniques]] | Devil's advocate questions, IRT difficulty tiers, forced distribution, penalty questions |
| 05 | [[05 - Weight Calibration from Engagement Data]] | OLS, logistic regression, Bayesian Ridge, bootstrap, online learning for weight optimization |
| 06 | [[06 - Designing Discriminative Binary Questions]] | Item discrimination theory, three-tier question design, target pass rates |
| 07 | [[07 - Mixed Scoring Patterns]] | Analysis of mixed binary + ordinal formats; practitioner consensus favors pure binary |
| 08 | [[08 - Calibrating Scores Against Highlight Behavior]] | Dependent variable selection, correlation metrics, sample size, temporal effects, evaluation protocol |
| 09 | [[09 - Production LLM Scoring Systems]] | 15 production case studies (Pinterest, Ramp, Honeycomb, Feedly, Karpathy, etc.) |
| 10 | [[10 - Prompt Engineering for LLM Evaluation]] | Temperature, CoT, few-shot, system prompts, model selection, verbosity, ordering/anchoring effects |
| 11 | [[11 - Current System Analysis]] | Analysis of the v2-categorical scoring system's design, score distribution, and structural properties |
| 12 | [[12 - Contrarian Views and Limitations]] | 12 arguments challenging the proposed approach; honest assessment of limitations |
