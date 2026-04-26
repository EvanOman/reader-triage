# Anti-Ceiling-Effect Techniques for LLM-as-Judge Scoring

## The Problem

When using LLMs to score content quality, scores reliably cluster at the top of the scale. In our system, most articles land between 80-100 despite meaningful quality differences within that range. This "ceiling effect" destroys the discrimination we need: the ability to distinguish a truly exceptional article from a merely good one.

The problem is compounded by our use case. The articles being scored are already pre-filtered by user subscriptions -- most content is at least decent. We are not trying to separate garbage from gold; we are trying to rank-order a pool of competent content by how likely a reader is to highlight passages. This makes fine discrimination at the top end critical and garden-variety "rate from 1 to 10" prompts nearly useless.

This report surveys techniques for combating score clustering, drawn from academic research, production systems, and psychometric theory. Each technique is evaluated for its applicability to content quality scoring where the dependent variable is highlight count.

---

## 1. Devil's Advocate Questions

### Concept

Rather than only asking questions that identify strengths (where "yes" adds points), include adversarial questions designed to find weaknesses. These questions are phrased so that "yes" reduces the score. The principle comes from the DEBATE framework (Kim et al., 2024), which demonstrated that adversarial critique improves LLM evaluation accuracy by 6-12 percentage points in correlation with human judgments.

### Why It Works

LLMs exhibit a well-documented "agreeableness bias" -- they reliably confirm positive assessments but fail to reject weak content. Research by Doosterlinck et al. (2025) quantified this: LLM judges achieve True Positive Rates above 96% but True Negative Rates below 25%. In other words, they almost always say "yes, this is good" and almost never say "no, this is bad." Devil's advocate questions exploit the high TPR in reverse: by asking "is this bad in a specific way?", the LLM's tendency to agree with the question now works in favor of score deflation rather than inflation.

### The DEBATE Framework

The DEBATE framework (Devil's Advocate-Based Assessment and Textual Evaluation) uses a multi-agent architecture with three roles:

- **Commander**: Manages the evaluation process and memory
- **Scorer**: Produces initial quality assessments
- **Critic**: Plays devil's advocate, instructed to "criticize the score as much as possible"

The Critic challenges foundational assumptions in the Scorer's reasoning. In one documented example evaluating "engagingness," the Critic pushed back on an overly harsh assessment by arguing that a response effectively "sparked curiosity" -- demonstrating that the adversarial role corrects both inflation and deflation.

Results on standard benchmarks:
- SummEval: +6.4 percentage points Spearman correlation vs. G-Eval
- Topical-Chat: +11.9 percentage points Pearson correlation vs. G-Eval

The full multi-agent debate architecture adds cost and latency that is impractical for per-article scoring. However, the key insight can be embedded directly into scoring questions: design questions where the honest answer for mediocre content is "yes" and that answer reduces the score.

### Concrete Devil's Advocate Questions for Content Scoring

These questions are designed so that a "yes" answer is a negative signal. Each targets a specific failure mode that inflates scores on mediocre content.

**Targeting conventional wisdom masquerading as insight:**
- "Is the author primarily restating ideas that are well-known among practitioners in this field?"
- "Could a knowledgeable reader predict the article's main conclusions from the headline alone?"

**Targeting shallow depth:**
- "Could the article's core argument be adequately summarized in a single tweet (280 characters)?"
- "Does the article rely on unsupported assertions rather than evidence, examples, or data?"

**Targeting commodity content:**
- "Is this content substantially a news report, product announcement, or event recap?"
- "Would a reader find equivalent coverage of this topic in the first page of search results?"

**Targeting false sophistication:**
- "Does the article use jargon or complex framing to dress up a straightforward point?"
- "Would a domain expert find the treatment superficial relative to the topic's complexity?"

### Implementation Notes

Devil's advocate questions should make up roughly 20-25% of total questions (4-5 out of 20). More than this risks overcorrecting into negative bias. Each negative question should target a specific, observable textual feature rather than abstract quality -- "Is this a summary of known ideas?" is testable; "Is this mediocre?" is not.

### Sources

- [DEBATE: Devil's Advocate-Based Assessment and Text Evaluation](https://arxiv.org/abs/2405.09935) (Kim et al., ACL 2024 Findings)
- [Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations](https://arxiv.org/abs/2510.11822) (Doosterlinck et al., 2025)

---

## 2. Comparative/Relative Framing

### Concept

Instead of asking "Is this article good?" (absolute judgment), ask "Is this article better than typical content in its category?" (relative judgment). Comparative framing forces the LLM to think about a reference class, which anchors its judgment against something concrete rather than an abstract quality scale.

### Why It Works

Absolute scoring suffers from scale compression: LLMs default to the upper portion of whatever scale they are given, regardless of content quality. Research on score range bias (2025) found that "the numerical outputs of LLMs acting as scoring judges depend sharply on the range and labeling of the scoring scale, with observed preferences toward specific values, independent of content quality." Relative framing partially bypasses this because the LLM does not need to produce a number -- it needs to make a comparison, which is a fundamentally different cognitive operation.

Pairwise comparison research consistently shows stronger alignment with human judgment. Studies demonstrate that "pairwise comparisons lead to more stable results and smaller differences between LLM judgments and human annotations relative to direct scoring." The Chatbot Arena methodology (LMSYS, 2023) demonstrated this at scale: by having users compare two outputs side-by-side rather than rate them independently, the platform achieves reliable Elo-based rankings that correlate highly with human preference.

### Approaches

**Pairwise comparison (strongest but most expensive):**
Present two articles and ask which one a reader would be more likely to highlight. This produces reliable ordinal rankings but requires O(n^2) comparisons, making it impractical for scoring individual articles at ingestion time.

**Category-relative questions (practical for single-article scoring):**
- "Among articles about [topic], would this rank in the top 20% for highlight-worthiness?"
- "Does this article offer insights beyond what's available in standard reference material on this topic?"
- "Is this more substantive than a typical newsletter entry on this subject?"

**Historical anchoring (requires infrastructure):**
Provide the LLM with 2-3 previously scored articles from similar categories as reference points. "Here is an article that scored 85 (with 12 highlights) and one that scored 45 (with 1 highlight). Where does this article fall?"

### Limitations

Comparative questions have a fundamental reliability problem for LLM scoring: the LLM does not have a stable, consistent reference set. When asked "is this in the top 20%?", different prompts and different contexts will produce different implicit reference distributions. Research shows that LLMs have "narrow proficiency distributions" compared to humans, making their relative judgments less well-calibrated than their absolute ones in certain domains.

Recommendation: Use comparative framing for 1-2 questions maximum, weighted lower than observable-feature questions. Comparative questions are best used as tiebreakers or calibration checks rather than primary scoring signals.

### Sources

- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) (Chiang et al., 2024)
- [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://openreview.net/forum?id=9gdZI7c6yr)
- [Evaluating Scoring Bias in LLM-as-a-Judge](https://arxiv.org/abs/2506.22316) (2025)

---

## 3. Difficulty-Tiered Criteria (IRT-Inspired)

### Concept

Design scoring questions at multiple difficulty levels, borrowing from Item Response Theory (IRT) in psychometrics. Easy questions that most content passes establish a baseline; medium questions separate good from average; hard questions that few articles pass identify truly exceptional content. This creates score spread across the full range rather than clustering at the top.

### Item Response Theory Background

IRT models the probability of a correct response as a function of both the respondent's ability (theta) and the item's characteristics. The two most relevant parameters are:

- **Difficulty (b)**: The ability level at which a respondent has a 50% probability of endorsing the item. Higher b means harder questions.
- **Discrimination (a)**: How sharply the item differentiates between respondents near its difficulty level. Higher a means the question is better at separating similar-ability respondents. The discrimination parameter is visualized as the steepness of the item characteristic curve.

A critical finding from IRT research applied to LLM benchmarking: "item discriminability is weakened by excessively high or low item difficulty." Items achieve optimal discrimination when difficulty is moderate for the population being assessed. A benchmark consisting of only easy and hard items fails to differentiate entities with medium proficiency (PATCH framework, Qi et al., 2024).

### Application to Content Scoring

Our "population" is articles from user subscriptions. Most are at least competent (they passed the user's subscription filter). The difficulty tiers should be calibrated to this population:

**Tier 1 -- Baseline (expected pass rate: 70-80%)**

These establish that the content meets a minimum quality bar. Most subscribed content should pass. Failing these is a strong negative signal.

| Question | What it tests | Expected behavior |
|----------|---------------|-------------------|
| "Does the article present at least one claim supported by evidence or examples?" | Basic substance | Most articles pass; news briefs and link roundups fail |
| "Is the content substantive enough to discuss with a colleague?" | Minimum depth | Filters out trivial content |
| "Does the author present original writing rather than just quoting or aggregating?" | Originality floor | Filters pure aggregation |

**Tier 2 -- Quality (expected pass rate: 30-50%)**

These separate genuinely good content from the competent baseline. This is where most of the score discrimination should happen.

| Question | What it tests | Expected behavior |
|----------|---------------|-------------------|
| "Does the article contain at least one passage (2-3 sentences) worth extracting as a standalone highlight?" | Highlightability | Core discriminator for our use case |
| "Does the author argue a clear position with conviction rather than neutral reporting?" | Author conviction | Separates opinion leaders from summarizers |
| "Does the author support claims with specific evidence (data, case studies, named examples)?" | Evidence quality | Separates rigorous from hand-wavy |
| "Does the article present a perspective that challenges a common assumption?" | Novelty | Separates fresh thinking from conventional wisdom |

**Tier 3 -- Exceptional (expected pass rate: 10-20%)**

These identify content that rises above "good" into "truly worth prioritizing." Only exceptional articles should pass these.

| Question | What it tests | Expected behavior |
|----------|---------------|-------------------|
| "Does this introduce a named framework, mental model, or methodology that could change how the reader approaches a class of problems?" | Conceptual contribution | Very few articles introduce genuinely new frameworks |
| "Would a knowledgeable reader in this domain encounter at least one idea they haven't seen before?" | Expert-level novelty | High bar: assumes the reader is well-read |
| "Does this article contain an insight that would still be relevant and valuable in 6 months?" | Durability | Filters time-sensitive content from lasting value |
| "Could the reader extract three or more independent, highlightable passages?" | Deep quotability | Only the richest content passes |

### Weighting by Tier

The scoring impact should be proportional to difficulty, reflecting the principle that harder-to-achieve qualities are rarer and more valuable:

```
Tier 1 (Baseline):    3-4 points each
Tier 2 (Quality):     5-7 points each
Tier 3 (Exceptional): 8-10 points each
```

This weighting means an article that passes all baselines but no quality or exceptional questions scores around 20-30 out of 100. Passing most quality questions pushes to 50-65. Only articles that also pass exceptional questions reach 75+. This directly addresses the ceiling effect by making the top of the scale hard to reach.

### Calibrating Difficulty Empirically

After deployment, use the calibration toolkit to verify actual pass rates match targets:

```bash
# After scoring ~100 articles, check pass rates per question
just cal-dimensions
```

If a "hard" question has a 60% pass rate, it is not actually hard and needs to be made more demanding. If a "baseline" question has a 30% pass rate, it is filtering too aggressively. The IRT framework suggests items with pass rates between 0.2 and 0.8 provide the most information; items outside this range are diagnostically weak.

### Sources

- [PATCH: Psychometrics-AssisTed benCHmarking of Large Language Models](https://arxiv.org/abs/2404.01799) (Qi et al., 2024)
- [Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](https://arxiv.org/abs/2505.15055) (2025)
- [Item Response Theory](https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory) (Columbia University)
- [The IRT Item Discrimination Parameter](https://assess.com/irt-item-discrimination-parameter/) (Assessment Systems)

---

## 4. Forced Distribution / Ranking

### Concept

Rather than asking the LLM to rate content on an absolute scale, force it to rank or distribute content across quality categories. This directly prevents clustering by construction: if the LLM must place content into bins with constrained quotas, the resulting distribution will have spread.

### Approaches

**Batch ranking (most natural but batch-dependent):**

Score multiple articles simultaneously and ask the LLM to rank-order them. This produces reliable ordinal rankings but has a fatal practical limitation: the ranking of any individual article depends on what else is in the batch. Adding or removing a single article can change the rank of all others.

```
Given the following 10 articles, rank them from most to least likely
to generate reader highlights. You must assign each article a unique
rank from 1-10. No ties.

Article A: [title + summary]
Article B: [title + summary]
...
```

**Calibrated bin assignment:**

Define quality bins with explicit quotas and ask the LLM to assign articles to bins. This approximates forced distribution while allowing single-article scoring:

```
Classify this article into one of these quality tiers. Across all
articles you evaluate, roughly 15% should fall in Tier 1, 35% in
Tier 2, 35% in Tier 3, and 15% in Tier 4:

Tier 1 (Exceptional): Rich in highlightable passages, offers novel
  frameworks, would surprise domain experts
Tier 2 (Strong): Contains several good passages, argues a clear
  position with evidence
Tier 3 (Competent): Readable and informative but few passages
  worth extracting
Tier 4 (Low Value): Commodity content, news recap, or shallow
  treatment
```

**Elo-based scoring (strongest but most expensive):**

Inspired by Chatbot Arena's methodology (LMSYS, 2023), use pairwise comparisons to build Elo ratings over time. Each new article is compared against a sample of previously scored articles. The resulting Elo rating naturally distributes scores across the full range because it is relative by construction.

The Bradley-Terry model underlying Elo ratings models the probability that article A is preferred over article B as a function of their rating difference. With enough comparisons, this produces well-calibrated scores. However, it requires multiple LLM calls per article (one per comparison) and a maintained reference set.

### Practical Considerations for Content Scoring

Forced distribution works best when:
- You are scoring batches of content simultaneously (e.g., a daily digest)
- You can maintain a reference set of previously scored articles for comparison
- Relative ranking matters more than absolute score

It works poorly when:
- Articles are scored individually at ingestion time (our primary use case)
- The LLM has no context for what other content exists in the pool
- Scores need to be stable (not change when new content is added)

### Hybrid Approach

A practical compromise: score articles individually using binary questions (for stable absolute scores), then apply a batch normalization step periodically. Every N articles, re-rank the recent batch and adjust scores to enforce a target distribution:

```python
def normalize_to_distribution(scores, target_percentiles):
    """Adjust scores so they match target distribution."""
    ranks = scipy.stats.rankdata(scores)
    percentiles = ranks / len(ranks)
    # Map percentiles to target score distribution
    normalized = np.interp(percentiles, target_percentiles, target_scores)
    return normalized
```

This preserves stable per-article scoring while ensuring the overall distribution does not cluster.

### Sources

- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) (Chiang et al., 2024)
- [LLM Arena-as-a-Judge: LLM-Evals for Comparison-Based Regression Testing](https://www.confident-ai.com/blog/llm-arena-as-a-judge-llm-evals-for-comparison-based-testing)
- [A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth](https://arxiv.org/abs/2601.21817) (2026)

---

## 5. Negative Signals / Penalty Questions

### Concept

Include scoring criteria where the presence of a feature reduces the score. This is the complement of devil's advocate questions (Section 1) but focuses on structural scoring mechanics rather than question design. The key idea: a scoring system that only adds points will always cluster high for content that has any merit. A system that also subtracts points for specific weaknesses can spread scores across the full range.

### Reverse Coding from Psychometrics

In psychometric survey design, "reverse coding" includes negatively phrased items to combat acquiescence bias (the tendency to agree with statements regardless of content). The standard formula is: New Score = (Max value + 1) - Original Score. This forces respondents to engage with each item independently rather than pattern-matching.

Applied to LLM scoring, reverse-coded items serve the same purpose: they disrupt the LLM's tendency to assign uniformly positive assessments. Since LLMs exhibit an agreeableness bias analogous to human acquiescence bias, reverse-coded questions exploit this tendency productively.

### Penalty Categories

**Content-type penalties (filter commodity content):**

| Signal | Penalty | Rationale |
|--------|---------|-----------|
| News report or event recap | -8 pts | Ephemeral content with low highlight potential |
| Product announcement or review | -6 pts | Commercial content rarely produces lasting insights |
| Link roundup or aggregation | -8 pts | No original analysis to highlight |
| Primarily AI-generated boilerplate | -6 pts | Lacks the personal voice that drives highlighting |

**Depth penalties (filter shallow treatment):**

| Signal | Penalty | Rationale |
|--------|---------|-----------|
| Main argument fits in a tweet | -6 pts | Insufficient depth for multiple highlights |
| No specific evidence cited | -4 pts | Claims without support lack quotability |
| Restates conventional wisdom | -6 pts | No surprise factor to drive engagement |
| Topic covered without new angle | -4 pts | Redundant with existing knowledge |

**Structural penalties (filter incomplete content):**

| Signal | Penalty | Rationale |
|--------|---------|-----------|
| Content appears truncated | Flag for rescore | Cannot evaluate partial content |
| Very narrow audience | -4 pts | Reduces broadly applicable value |

### Implementation: Integrated vs. Separate Penalty Tracks

**Approach A -- Integrated weights (recommended):**

Negative questions have negative weights in the same scoring vector as positive questions. The final score is a single weighted sum. This is what logistic regression would naturally produce when trained on engagement data.

```python
weights = {
    "standalone_passages": +8,    # positive signal
    "novel_framing": +7,          # positive signal
    "restates_known_ideas": -6,   # negative signal
    "fits_in_tweet": -6,          # negative signal
    "named_framework": +8,        # positive signal
}

raw_score = sum(weights[q] * responses[q] for q in all_questions)
final_score = max(0, min(100, raw_score))
```

**Approach B -- Multiplicative penalties:**

Negative signals act as multipliers that scale down the positive score. This prevents a single devastating penalty from zeroing out an otherwise good article.

```python
base_score = sum(positive_weights[q] * responses[q] for q in positive_qs)
penalty_factor = 1.0
for q in negative_qs:
    if responses[q]:
        penalty_factor *= 0.85  # each penalty reduces by 15%
final_score = base_score * penalty_factor
```

**Approach C -- Gatekeeper penalties:**

Certain negative signals hard-cap the maximum achievable score:

```python
if responses["is_news_recap"] and responses["restates_known_ideas"]:
    final_score = min(20, calculated_score)  # commodity content cap
elif responses["fits_in_tweet"]:
    final_score = min(50, calculated_score)  # shallow content cap
```

### Effect on Score Distribution

With 4 penalty questions carrying -4 to -8 points each, the maximum possible penalty is approximately -24 points. For an article that passes all positive questions (theoretical max ~108 before scaling), the effective ceiling becomes ~84-108 depending on penalties triggered. More importantly, a "typical decent article" that passes baseline and quality questions but also triggers 1-2 penalties would score 45-60 rather than 80-95, dramatically improving discrimination in the range that matters.

### Sources

- [Reverse Scoring Explained: What It Is and How to Do It](https://agolix.com/blog/tutorials/reverse-scoring-scale-questions/)
- [Calibrating Scores of LLM-as-a-Judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge) (GoDaddy, 2025)
- [DEBATE: Devil's Advocate-Based Assessment and Text Evaluation](https://arxiv.org/abs/2405.09935)

---

## 6. Calibration Anchoring

### Concept

Provide the LLM with concrete examples of previously scored content at known quality levels. This grounds the LLM's judgment in specific reference points rather than its own abstract notion of "good" or "bad." The technique is borrowed from few-shot prompting but applied specifically to calibrate the score scale.

### Why Anchoring Is Necessary

Research demonstrates that LLM scores are highly sensitive to scale framing. Score range bias studies (2025) found that "LLMs show observed preferences toward specific values, independent of content quality" -- meaning the model is not purely responding to quality but also to scale conventions baked into its training data. Anchoring with real examples forces the model to calibrate against actual quality variation rather than default to its preferred region of the scale.

The LLM-Rubric framework (Microsoft, ACL 2024) formalized this approach. Their system prompts the LLM with rubric questions independently, captures probability distributions over possible responses, and then trains a small calibration network to map these distributions to human judgments. The calibration step alone produced a 2x improvement in RMS error for predicting human satisfaction scores.

### Anchor Selection Criteria

Effective anchors should:

1. **Span the full score range**: Include a low-scoring article (15-25), a mid-scoring article (45-55), and a high-scoring article (80+)
2. **Come from the same content domain**: Anchors from tech blogs are useless for calibrating scores on academic papers
3. **Have known engagement data**: Ideally, the anchor articles have established highlight counts that validate their scores
4. **Be concise enough to fit in the prompt**: Use the title, a 2-3 sentence summary, and the score, not the full article text

### Anchor Example Template

```
CALIBRATION EXAMPLES (for reference only -- do not score these):

LOW (Score: 22, 0 highlights): "Weekly AI News Roundup #47"
A collection of links to AI announcements from the past week with
brief summaries. No original analysis or commentary. Readers rarely
highlight aggregation posts.

MEDIUM (Score: 52, 3 highlights): "Why Most A/B Tests Are Underpowered"
Explains a common statistical mistake in experiment design with
examples. Clear position, some evidence, but the core insight is
well-known among experienced practitioners. Readers highlight the
specific examples but not the thesis.

HIGH (Score: 87, 14 highlights): "The Bitter Lesson of Scaling Laws"
Introduces a novel framework for thinking about when to scale
compute vs. improve algorithms, drawing on first-hand experience
running large training runs. Multiple memorable passages with
specific data. Challenges conventional wisdom about efficiency
research. Readers highlight extensively.

Now evaluate the following article:
```

### Implementation Considerations

**Static vs. dynamic anchors:**

Static anchors (hardcoded in the prompt) are simplest but risk staleness. Dynamic anchors (selected from the database based on the article's category) provide better calibration but add database queries and prompt complexity.

A practical middle ground: maintain a curated set of 9-12 anchor articles (3-4 per score tier, covering common categories). Select the 3 most relevant anchors for each scoring call based on content category.

**Token budget impact:**

Each anchor adds roughly 100-150 tokens to the prompt. With 3 anchors, this is ~400 tokens -- modest relative to the article content itself (which can be 4000-15000 tokens). The calibration benefit justifies this overhead.

**Anchors for binary questions:**

With a binary-question scoring system, anchors can be even more compact. Instead of describing overall quality, show the binary response pattern:

```
CALIBRATION: An article scoring 85 typically passes 15-17 of 20
questions, including most Tier 3 (exceptional) questions and
triggering 0-1 penalty questions. An article scoring 45 typically
passes 10-12 questions, mostly Tier 1-2, triggering 1-2 penalties.
```

### The LLM-Rubric Calibration Network

For maximum rigor, the LLM-Rubric approach (Microsoft) goes beyond prompt-level anchoring. It:

1. Prompts the LLM with each rubric question independently, collecting probability distributions over responses
2. Trains a small feed-forward neural network with both shared and judge-specific parameters
3. The network maps raw LLM probability distributions to calibrated scores that match individual human judges' preferences

This achieves RMS error below 0.5 on a 1-4 scale for dialogue quality assessment. The approach requires a training set of human-scored examples (the authors used existing annotation datasets), but the calibration network itself is tiny and cheap to run.

### Sources

- [LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts](https://arxiv.org/abs/2501.00274) (Microsoft, ACL 2024)
- [Calibrating Scores of LLM-as-a-Judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge) (GoDaddy, 2025)
- [RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation](https://arxiv.org/abs/2601.08654) (2026)
- [Evaluating Scoring Bias in LLM-as-a-Judge](https://arxiv.org/abs/2506.22316)

---

## 7. IRT-Inspired Approaches

### Concept

Apply Item Response Theory (IRT) not just to design questions (Section 3), but as a full analytical framework for understanding and improving the scoring system. IRT provides mathematical tools for measuring how much information each question contributes, how well questions discriminate between different quality levels, and where the scoring system is strongest and weakest.

### The Four-Parameter Model

The most comprehensive IRT model (4PL) characterizes each item with four parameters:

```
P(correct | theta) = c + (d - c) / (1 + exp(-a * (theta - b)))
```

Where:
- **a (discrimination)**: How sharply the item differentiates near its difficulty level. High a means the question is a strong quality separator; low a means it is noise.
- **b (difficulty)**: The quality level at which an article has a 50% chance of passing the question. This should span the full range of content quality.
- **c (guessing)**: The probability that a low-quality article passes the question by chance. High c means the question is too easy or poorly worded.
- **d (feasibility)**: The maximum probability that even the highest-quality article passes. If d < 1.0, the question may be ambiguous or underspecified.

### Fisher Information for Question Selection

The PSN-IRT framework (2025) demonstrated that selecting benchmark items based on Fisher information -- a measure of how much statistical information an item provides about the respondent's ability -- produces superior rankings with fewer items. A subset of just 1,000 items chosen via Fisher information achieved Kendall's tau of 0.90, substantially outperforming rankings using all available items.

Applied to content scoring: not all questions contribute equally to discrimination. Some questions are nearly always answered the same way (either too easy or too hard for the content pool) and provide no information. Fisher information analysis can identify which questions to keep, which to revise, and which to discard.

### Practical IRT Analysis for Content Scoring

After scoring approximately 100+ articles with binary questions, you can run IRT analysis to evaluate question quality:

**Step 1: Compute classical item statistics**

```python
# For each binary question, compute:
pass_rate = sum(responses[q]) / len(responses[q])
# Target: 0.2-0.8 for informative items
# < 0.1 or > 0.9: question is too hard or too easy

# Point-biserial correlation with total score:
discrimination = scipy.stats.pointbiserialr(
    responses[q], total_scores
).statistic
# Target: > 0.3 for useful items
# < 0.1: question is noise
# < 0: question is anti-correlated (possible scoring error)
```

**Step 2: Map questions to the IRT difficulty scale**

```python
from scipy.special import logit

# Convert pass rates to IRT difficulty (logit scale)
difficulty = -logit(pass_rate)
# Negative = easy (most articles pass)
# Positive = hard (few articles pass)
# 0 = 50% pass rate
```

**Step 3: Identify problem questions**

| Diagnostic | Symptom | Action |
|------------|---------|--------|
| Pass rate > 0.9 | Too easy, no discrimination | Make harder or remove |
| Pass rate < 0.1 | Too hard, no discrimination | Make easier or remove |
| Discrimination < 0.1 | Question is noise | Rewrite or remove |
| Discrimination < 0 | Anti-correlated | Check if it should be a penalty question, or it is broken |
| High difficulty + low discrimination | Hard but uninformative | Question may be ambiguous |

**Step 4: Evaluate coverage of the difficulty spectrum**

Plot question difficulties against the score distribution of articles. If all questions cluster at one difficulty level, scores will cluster at the corresponding region of the scale. For good discrimination, questions should be distributed across the difficulty range, with more questions in the region where the most articles fall (the middle of the distribution).

### Adaptive Item Selection (Advanced)

IRT enables computerized adaptive testing (CAT), where the next question is selected based on previous answers to maximize information gain. For content scoring, a simplified version:

1. Ask 4-5 baseline questions first
2. Based on how many pass, select the next set of questions:
   - If 0-1 baseline questions pass: skip hard questions (the article is clearly low value), ask diagnostic questions about content type
   - If 4-5 baseline questions pass: skip easy questions, focus on quality and exceptional tiers
   - If 2-3 pass: ask the full set for maximum discrimination

This reduces token cost (fewer questions per article) while maintaining discrimination where it matters most. However, it requires conditional prompt logic and is significantly more complex to implement and validate.

### Connection to Highlight Count as the Latent Trait

In classical IRT, theta represents a latent ability. In our system, the latent trait is "highlight-worthiness" -- the propensity of content to generate reader highlights. Highlight count is an observed proxy for this latent trait. IRT's framework lets us estimate how well each question measures this trait and how much uncertainty remains in our score estimate for any given article.

### Sources

- [Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](https://arxiv.org/abs/2505.15055) (2025)
- [PATCH: Psychometrics-AssisTed benCHmarking of Large Language Models](https://arxiv.org/abs/2404.01799) (Qi et al., 2024)
- [Leveraging LLM Respondents for Item Evaluation: A Psychometric Analysis](https://arxiv.org/abs/2407.10899) (Liu et al., 2025)
- [Learning Compact Representations of LLM Abilities via Item Response Theory](https://arxiv.org/abs/2510.00844) (2025)
- [Item Response Theory](https://en.wikipedia.org/wiki/Item_response_theory) (Wikipedia)

---

## 8. Additional Techniques

### 8.1. Chain-of-Thought Before Scoring

Requiring the LLM to articulate reasoning before producing a score reduces the positivity skew. Research shows that "conclusions generated afterward lack adequate support" and that explanation-first ordering is essential for alignment with human judgment.

Implementation: For each binary question, require a brief reason alongside the yes/no answer. This is already present in the current v2-categorical system (the `*_reason` fields) and should be preserved in any redesign.

### 8.2. Contrastive Decoding for Score Range Bias

Research on score range bias (2025) found that contrastive decoding -- adjusting output logits using a secondary model from the same family -- "removes shared bias directions between models and yields up to 11.3% improvement in correlation with human judgments across varied score ranges." This is a model-level technique that requires access to logits and is not directly applicable to API-based scoring, but it validates the principle that score range bias is a systematic, correctable artifact.

### 8.3. Evidence-Anchored Scoring (RULERS)

The RULERS framework (2026) compiles natural language rubrics into executable specifications and enforces "structured decoding with deterministic evidence verification." Each score must be grounded in specific textual passages. This prevents the LLM from assigning high scores based on abstract impression rather than concrete textual features.

For our binary questions, this translates to: when the LLM answers "yes" to "Does the article contain a memorable phrase worth highlighting?", require it to quote the specific phrase. This grounding makes the judgment verifiable and reduces hallucinated positive assessments.

### 8.4. Ensemble Judging

Using multiple LLM judges and aggregating their scores reduces individual model biases. Research shows that a panel of smaller models can outperform a single large model while being cheaper. For our use case, this is likely overkill for per-article scoring but could be valuable for periodic calibration: score a sample of articles with multiple models and flag articles where models disagree strongly.

### 8.5. Minority-Veto for Rejection

The agreeableness bias research (Doosterlinck et al., 2025) proposes a "minority-veto" strategy where a content piece is flagged as low quality only when a minority threshold of validators agree. This achieved 95.5% TPR and 30.9% TNR -- substantially improving the True Negative Rate over standard majority voting's 19.2%. This technique is most applicable to binary quality gates (read/skip decisions) rather than continuous scoring.

### Sources

- [RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation](https://arxiv.org/abs/2601.08654) (2026)
- [Using LLMs for Evaluation](https://cameronrwolfe.substack.com/p/llm-as-a-judge) (Cameron Wolfe, 2025)
- [Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations](https://arxiv.org/abs/2510.11822)
- [Score Range Bias: Analysis & Mitigation](https://www.emergentmind.com/topics/score-range-bias)

---

## Summary: Recommended Technique Stack

The following combination of techniques is recommended for our content scoring system, ordered by expected impact and implementation feasibility:

### High Impact, Implement Now

| Technique | Expected Effect | Implementation Cost |
|-----------|----------------|---------------------|
| **Difficulty-tiered questions** (Section 3) | Spreads scores across full range by making high scores hard to achieve | Medium -- requires question redesign |
| **Devil's advocate / penalty questions** (Sections 1, 5) | Pulls inflated scores down for mediocre content | Low -- add 4-5 negative questions |
| **Evidence grounding** (Section 8.3) | Prevents hallucinated positive assessments by requiring specific textual citations | Low -- add "quote the passage" requirement to key questions |

### Medium Impact, Implement After Baseline

| Technique | Expected Effect | Implementation Cost |
|-----------|----------------|---------------------|
| **Calibration anchoring** (Section 6) | Grounds the score scale in concrete examples | Low -- add 3 anchor examples to prompt |
| **IRT-based question analysis** (Section 7) | Identifies and replaces questions that do not discriminate | Medium -- requires 100+ scored articles and analysis tooling |
| **Batch normalization** (Section 4) | Enforces target distribution post-hoc | Medium -- periodic batch processing step |

### Lower Priority / Advanced

| Technique | Expected Effect | Implementation Cost |
|-----------|----------------|---------------------|
| **Comparative questions** (Section 2) | Adds relative framing for 1-2 questions | Low cost but uncertain reliability |
| **LLM-Rubric calibration network** (Section 6) | Mathematically optimal calibration | High -- requires probability distributions and training infrastructure |
| **Pairwise ranking / Elo** (Section 4) | Strongest ordinal discrimination | High -- multiple LLM calls per article, reference set maintenance |

### Target Score Distribution

With the recommended technique stack, the target score distribution for subscribed content is:

| Score Range | Target Percentage | Interpretation |
|-------------|-------------------|----------------|
| 0-19 | 10-15% | Commodity content (news, roundups, stubs) |
| 20-39 | 25-30% | Competent but unremarkable |
| 40-59 | 30-35% | Good content with some highlight potential |
| 60-79 | 15-20% | Strong content likely to generate highlights |
| 80-100 | 5-10% | Exceptional content with high highlight density |

This distribution should be validated against actual highlight counts using the calibration toolkit after deployment.
