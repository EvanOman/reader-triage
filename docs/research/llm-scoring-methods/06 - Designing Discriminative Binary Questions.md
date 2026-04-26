# Designing Discriminative Binary Questions for LLM-Based Content Evaluation

## The Problem

Our current v2-categorical scoring system uses 8 questions (a mix of binary true/false and categorical multi-option) to produce scores across four dimensions: Quotability, Surprise Factor, Argument Quality, and Applicable Insight. Each dimension is scored 0-25, totaling 0-100.

The problem: **scores cluster at the top of the range (80-100)**, producing a ceiling effect that fails to discriminate among the articles a reader would actually want to compare. When everything scores "high," the scoring system provides no useful triage signal in precisely the tier where it matters most.

The dependent variable we are predicting is **highlight count** -- how many passages a reader bookmarks while reading. A well-calibrated scoring system should spread articles across its range in a way that correlates with this engagement signal.

This document synthesizes research from psychometrics (Item Response Theory), survey methodology, and LLM evaluation literature to propose a set of binary questions designed to maximize discrimination, particularly in the upper tier.

---

## Part 1: What Makes a Binary Question Discriminative?

### The Item Discrimination Index

In classical test theory, the **item discrimination index** (D) measures how well a single test item separates high performers from low performers. It is calculated by comparing the pass rate of the top 27% of test-takers to the pass rate of the bottom 27%:

```
D = P_upper - P_lower
```

The index ranges from -1.0 to +1.0:

| D Value | Interpretation |
|---------|---------------|
| >= 0.40 | Excellent discrimination |
| 0.30 - 0.39 | Good discrimination |
| 0.20 - 0.29 | Acceptable, consider revision |
| < 0.20 | Poor -- revise or discard |
| Negative | Problematic -- item is anti-discriminative |

A question that 95% of articles pass (or 5% pass) is essentially a constant -- it contributes no information about relative quality. Maximum discrimination potential occurs when the item difficulty (pass rate) is around **0.50**, creating the widest possible separation between groups.

Source: [Evaluating Item Difficulty and Discrimination in Knowledge Tests](https://foodsafety.institute/research-methodology/evaluating-item-difficulty-discrimination-tests/)

### The Two-Parameter IRT Model

Item Response Theory formalizes this with two parameters per item:

1. **Difficulty (b)**: The ability level at which 50% of test-takers endorse the item. In our context, this is the quality level at which 50% of articles would receive a "yes."
2. **Discrimination (a)**: How steeply the probability curve transitions from "no" to "yes" around the difficulty point. Higher discrimination means the question creates a sharper boundary.

The key design insight: **we need items spread across the difficulty spectrum**, with most items concentrated in the region where we want the finest discrimination (the upper tier of article quality, since that is where our current system fails).

Source: [Item Response Theory - Columbia University](https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory)

### What Makes a Question Non-Discriminative

A question fails to discriminate when any of the following hold:

1. **Near-universal pass rate (p > 0.85)**. "Does this article discuss a real topic?" -- everything passes. The question contributes a constant to every score, adding zero information.

2. **Vague criteria with no decision boundary**. "Is this article interesting?" -- the LLM has no concrete standard by which to say "no." It defaults to "yes" for anything that is not obviously terrible.

3. **Tests for surface features rather than quality**. "Does the article contain a quote?" -- the presence of quotes is weakly correlated with the quality signal we care about (highlight-worthy content). A listicle full of pull-quotes passes; a tightly argued essay without attributed quotes may fail.

4. **Double-barreled questions**. "Does this article present novel insights supported by strong evidence?" -- this conflates two dimensions. The LLM may say "yes" if either half is true, inflating the pass rate.

5. **Questions the LLM cannot reliably answer**. "Would a domain expert find this surprising?" -- the LLM must simulate an expert persona it cannot fully represent. Reliability drops, and the LLM defaults to the positive.

### What Makes a Question Highly Discriminative

The inverse principles:

1. **Target pass rate in the 30-60% range** for the article population you care about. This maximizes the mathematical potential for discrimination.

2. **Point to a concrete, observable textual feature**. "Does the author describe a specific failure from their own professional experience?" -- the LLM can scan the text for a specific pattern (first-person narration + failure/mistake + professional context).

3. **Atomic scope**. Each question tests exactly one thing. CheckEval's entire methodology depends on this: "Questions within the checklist are formatted in a Boolean QA style, allowing for binary responses (Yes/No), which improves the precision and clarity of evaluation." (Source: [CheckEval, EMNLP 2025](https://arxiv.org/abs/2403.18771))

4. **Clear threshold**. The question implicitly defines what "enough" means. "Does the article contain at least three distinct supporting examples for its central claim?" has a numeric threshold. "Does the article support its claims?" does not.

5. **Predictive of the target variable**. In our case, the question should ask about a feature that, when present, correlates with the reader actually highlighting passages. The best discriminators are proxies for the engagement behavior we want to predict.

---

## Part 2: Countering LLM Positivity Bias

### The Scope of the Problem

LLMs exhibit systematic **acquiescence bias** -- a tendency to agree with presented statements and to evaluate positively by default. This is the LLM analogue of the well-documented "yea-saying" bias in human survey respondents.

Key findings:

- LLMs "consistently exhibit high rates of social sycophancy: on average, they preserve user's face 45 percentage points above baseline" (ELEPHANT study, 2025). Source: [ELEPHANT paper](https://arxiv.org/html/2505.13995v2)
- LLM judges show "average inconsistency rates indicating framing has non-negligible effects" across all models tested, including the most robust ones. Source: [When Wording Steers the Evaluation](https://arxiv.org/html/2601.13537)
- Binary question formats (yes/no) are "particularly prone to acquiescence bias" because they provide maximum opportunity for the model to agree. Source: [Acquiescence Bias in Large Language Models](https://arxiv.org/html/2509.08480v1)

### Mitigation Strategy 1: Concrete Observable Criteria

The single most effective mitigation is specificity. Hamel Husain's principle: "Skip generic eval criteria to evaluate specific product problems instead." The more concrete the question, the less room the LLM has to default to "yes."

| Vague (inflated) | Concrete (discriminative) |
|-------------------|---------------------------|
| "Does this article have good insights?" | "Does the author introduce a named framework, model, or methodology that the reader could apply?" |
| "Is the writing compelling?" | "Does the article contain a sentence or phrase that encapsulates a key idea in a memorable, quotable way -- something you would highlight specifically for its phrasing?" |
| "Does this contribute to the field?" | "Does the article present a finding, data point, or conclusion that directly contradicts a commonly held assumption in its domain?" |

Source: [Hamel Husain - Using LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/)

### Mitigation Strategy 2: Negative Signal Questions (Reverse-Scored Items)

Including questions where "yes" **lowers** the score (penalty questions) serves two purposes:

1. It disrupts the acquiescence pattern. The LLM cannot maximize the score by saying "yes" to everything.
2. It tests for the absence of quality signals rather than only their presence, which is often easier for the LLM to evaluate reliably.

From psychometric research on reverse-scored items: mixing positively and negatively worded items helps identify response inconsistency and counteracts acquiescence bias. However, the negative items should be clearly phrased -- double negatives and confusing wording can impair accuracy rather than helping.

Effective pattern: "Is this primarily [low-value category]?" where a "yes" answer is a clear negative signal.

**Important caveat**: Research shows that reverse-worded items can confuse respondents (and LLMs) if poorly constructed. Best practice is to phrase negative-signal questions as **positive assertions about a negative characteristic**, not as negations of a positive characteristic.

- Good: "Is this primarily a summary or aggregation of others' ideas, rather than original analysis?" (positive assertion of a negative quality)
- Bad: "Does this article NOT contain original analysis?" (negation -- confusing)
- Bad: "Does this article lack depth?" (vague + subjective negative)

Source: [Examining the Effect of Reverse Worded Items](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0157795)

### Mitigation Strategy 3: Difficulty Tiering

If all questions are at the same difficulty level, the LLM can develop a "set" -- an implicit expectation of how many "yes" answers a typical article should get. By mixing easy baseline questions (expected ~75% yes) with hard elite questions (expected ~15% yes), we force the LLM to genuinely evaluate each item rather than pattern-matching to an expected distribution.

### Mitigation Strategy 4: Explicit Calibration Instructions

In the scoring prompt, include calibration language that sets expectations:

```
Most articles will pass 8-12 of these questions. An article passing more than
18 is exceptional and should be rare. If you find yourself answering "yes" to
nearly every question, reconsider whether you are applying the criteria
strictly enough.
```

This leverages the finding that "direct instruction" is the single most effective prompt-level intervention against sycophancy.

### Mitigation Strategy 5: Require Brief Evidence

For each "yes" answer, require the LLM to cite specific evidence from the text. This forces grounding and makes "lazy yes" responses harder to produce. A question answered "yes" without a concrete passage reference is more likely to be an acquiescence artifact.

The CheckEval framework and QAG (Question Answer Generation) approach both support this: "confined answers to close-ended questions with very little room for stochasticity." Requiring evidence text alongside the binary judgment further constrains the output.

Source: [DeepEval LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

---

## Part 3: Question Framing -- Negative vs. Positive

### The Framing Dilemma

Survey methodology and psychometric research reveal a tension in question framing:

**Positive framing** ("Does this have X?"):
- Clearer and easier to process cognitively
- More reliable responses from both humans and LLMs
- Susceptible to acquiescence bias -- respondents tend to say "yes"

**Negative framing** ("Does this lack X?"):
- Can counteract acquiescence bias
- But: "Negative phrasing makes people's brains work harder to understand what you're actually asking" (survey design research)
- LLMs show higher error rates on negatively framed questions
- Risk of double-negative confusion

Source: [The Psychology of Survey Responses](https://insight7.io/the-psychology-of-survey-responses-how-wording-impacts-data-quality/)

### Our Approach: Primarily Positive, with Structural Negatives

The recommended strategy for our binary questions:

1. **Most questions (16-18) use positive framing** targeting the presence of specific quality indicators. These are the "does this have..." questions.

2. **A minority of questions (3-4) are structural negatives** -- positive assertions about negative characteristics. These serve as penalty items where "yes" lowers the score. They are not phrased as negations ("Does this lack...") but as affirmations of a negative property ("Is this primarily a news summary?").

3. **Never use double negatives** or "Does this fail to..." phrasings. These increase LLM error rates without improving discrimination.

4. **Never negate a positive question to create a "hard" question**. "Does this NOT have memorable passages?" is worse than creating a genuinely harder positive question: "Does the article contain a passage so distinctive that a reader encountering it out of context would want to find and read the full article?"

### Framing Examples by Effectiveness

| Framing | Example | Effectiveness |
|---------|---------|---------------|
| Positive, concrete | "Does the author describe a specific decision they made and its outcome?" | High -- observable, clear threshold |
| Positive, vague | "Does the author share experience?" | Low -- too easy to say "yes" |
| Structural negative | "Is this primarily a restatement of ideas available on the topic's Wikipedia page?" | Medium-high -- clear category, penalty signal |
| Direct negation | "Does this article lack originality?" | Low -- subjective, confusing for LLM |
| Double negative | "Is it not the case that this article fails to provide evidence?" | Terrible -- do not use |

---

## Part 4: Specificity Levels

### The Specificity Gradient

Questions exist on a spectrum from abstract to hyperspecific. Discrimination improves as you move toward specificity, but at the far end you risk questions that are so narrow they only apply to certain genres of content.

**Level 1: Abstract (poor discrimination, p ~ 0.90)**
> "Is this article well-written?"

The LLM has no concrete test to apply. Almost everything that isn't obviously broken passes.

**Level 2: Dimensional (mediocre discrimination, p ~ 0.75)**
> "Does this article contain surprising information?"

Points toward a dimension but doesn't specify what observable feature constitutes "surprising."

**Level 3: Observable Feature (good discrimination, p ~ 0.40-0.60)**
> "Does the article present a finding or conclusion that directly contradicts a commonly held assumption in its domain?"

Specifies a concrete pattern: there is a claim in the article, and that claim runs counter to conventional wisdom. The LLM can search for this pattern.

**Level 4: Quantified Observable (excellent discrimination, p ~ 0.25-0.45)**
> "Does the article provide three or more distinct pieces of evidence (data points, case studies, or documented examples) supporting its central argument?"

Adds a numeric threshold. The LLM must count instances, which produces more reliable binary splits than vague "does it have enough" judgments.

**Level 5: Behavior-Anchored (excellent but narrow, p ~ 0.15-0.30)**
> "Does the article contain a passage that provides a reusable decision framework -- a set of criteria, steps, or conditions that the reader could apply to make a specific type of decision in their own work?"

Describes the exact reader behavior the passage would support. Very discriminative but only applicable to certain article types.

### The Sweet Spot

For our purpose, **Levels 3 and 4** provide the best balance of discrimination and generalizability. Level 5 is appropriate for 2-3 "elite" questions that identify truly exceptional content, understanding that most articles will (correctly) fail them.

---

## Part 5: Base Rate Considerations

### Why Base Rate Matters

The **base rate** (proportion of "yes" answers across the article population) determines how much information each question contributes to the total score.

From IRT: maximum discrimination occurs at **p = 0.50**. But this assumes we want to discriminate equally across the entire ability range. We don't -- we need to discriminate specifically in the upper tier, where our current system fails.

### Optimal Base Rate for Our Use Case

Our population is pre-filtered: articles in the Readwise inbox are already things the user chose to save. This is not a random sample of all web content -- it is already a quality-filtered set. Within this set, we need to spread scores to identify which saved articles are worth deep reading.

**Target base rate distribution across questions:**

| Difficulty Tier | Target Base Rate (p) | Purpose | Question Count |
|----------------|---------------------|---------|---------------|
| Easy / Baseline | 0.65 - 0.80 | Establish floor; catch genuinely poor content | 5-6 |
| Medium | 0.35 - 0.55 | Core discrimination; separate good from great | 8-10 |
| Hard / Elite | 0.10 - 0.25 | Identify exceptional content; break ceiling ties | 5-6 |
| Negative Signal | 0.15 - 0.35 (yes = penalty) | Pull down inflated scores; counteract acquiescence | 3-4 |

### Why Not All Questions at p = 0.50?

If every question has a 50% base rate, the total score distribution would approximate a binomial with mean 50 and relatively tight variance. This would fix the ceiling problem but create a new one: insufficient spread in the tails.

By including easy questions (high base rate), we ensure that genuinely poor content scores low. By including hard questions (low base rate), we create separation among the top tier. The medium questions provide the core discriminative power in the middle range.

This mirrors the psychometric principle that **an optimal test includes items spanning the difficulty range**, with concentration at the region of interest.

### Expected Score Distribution

With the proposed question distribution (assuming 22 scored questions, each worth equal points):

- **Weak articles** (news summaries, link roundups): pass ~6-8 easy questions, 1-2 medium = score ~30-40
- **Average articles** (decent analysis, some substance): pass ~5 easy, 4-5 medium, 0-1 hard = score ~45-55
- **Good articles** (clear argument, practitioner voice, useful): pass ~5 easy, 7-8 medium, 1-2 hard = score ~60-70
- **Exceptional articles** (original framework, surprising findings, memorable writing): pass ~5 easy, 8-9 medium, 4-5 hard = score ~75-90

This spreads scores across the full range and -- critically -- provides 20+ points of discrimination within the "good" tier (60-90) where the current system compresses everything into 80-100.

---

## Part 6: Lessons from Psychometric Test Design

### Classical Test Theory Principles Applied

**1. Item-Total Correlation**

In psychometrics, each item's correlation with the total test score measures how well it contributes to the overall construct. For our binary questions, we should track: for each question, what is the correlation between "yes" on that question and the total highlight count? Questions with low or negative item-total correlation should be revised or dropped.

**2. Ferguson's Delta (Score Spread)**

Ferguson's Delta measures the discrimination of a test as a whole -- how well the total score distribution uses the available range. A test where all scores cluster at 80-100 has poor Ferguson's Delta. The goal of our question redesign is to maximize this metric.

Source: [Ceiling and Floor Effects in Psychometric Testing](https://www.cogn-iq.org/learn/theory/ceiling-floor-effects/)

**3. Reliability Through Redundancy**

Having multiple questions per dimension (5-6 each) provides internal consistency. If one question misfires on a particular article, the other questions in that dimension can compensate. This is analogous to Cronbach's alpha in test theory -- more items per subscale (up to a point) improve reliability.

**4. The Difficulty-Discrimination Relationship**

Empirical studies consistently find an inverted-U relationship: items at extreme difficulty levels (p < 0.10 or p > 0.90) almost always have low discrimination. The practical implication: even our "hard" questions should not be so hard that virtually nothing passes. A question with p = 0.05 carries almost no information.

Source: [Professional Testing, Inc - Item Analysis](https://proftesting.com/test_topics/steps_9.php)

### The IDGen Framework (NeurIPS 2024)

IDGen applies IRT principles specifically to LLM evaluation, training models to measure both **question discriminative power** and **question difficulty**. Their key insight for our work:

- Items with high discriminative power "effectively differentiate between higher and lower quality"
- Items with negative discrimination are "problematic" -- they signal the opposite of what we intend
- A good evaluation set should have items "spread over a range" of difficulty levels

We can apply this by treating our initial question set as a hypothesis and empirically measuring each question's discrimination against our dependent variable (highlight count) after deployment.

Source: [IDGen paper, NeurIPS 2024](https://arxiv.org/abs/2409.18892)

### Practical Calibration Loop

The psychometric approach to item development is iterative:

1. **Draft questions** based on theory and domain knowledge (this document)
2. **Pilot** the questions on a sample of articles with known highlight counts
3. **Compute item statistics**: difficulty (pass rate), discrimination (correlation with highlight count), and inter-item correlations
4. **Revise**: drop or rewrite items with poor discrimination, adjust difficulty of items that are too easy or too hard, add items if coverage gaps appear
5. **Repeat** until the question set produces scores that correlate well with highlights and spread across the full range

---

## Part 7: Designing Questions at Different Difficulty Levels

### Easy Questions (Target p = 0.65-0.80)

**Purpose**: Establish a quality floor. These questions should reliably flag genuinely low-value content (news briefs, link roundups, trivial announcements) while passing anything with real substance.

**Design principles**:
- Test for the *presence* of basic structural features
- Use low thresholds ("at least one...")
- Ask about features that are necessary but not sufficient for quality

**Example difficulty calibration**:
- "Does the content present at least one specific claim supported by evidence?" -- most substantive articles pass this; news briefs and pure opinion without evidence fail.
- Estimated pass rate on our population: ~75% (our inbox skews toward substantive content, but some newsletters, announcements, and thin posts will fail)

### Medium Questions (Target p = 0.35-0.55)

**Purpose**: Core discrimination. These separate "worth reading closely" from "skim and archive." They test for features that correlate with the reader actually highlighting passages.

**Design principles**:
- Test for *quality* of features, not just presence
- Use moderate thresholds ("multiple," "clearly," "specific")
- Ask about features that require craft, effort, or expertise to produce
- Each question should be independently informative -- avoid testing the same thing twice

**Example difficulty calibration**:
- "Does the author argue for a clear, specific position that a reasonable person could disagree with?" -- requires genuine conviction and a non-obvious stance, not just reporting facts. Maybe 40-50% of articles do this well.

### Hard Questions (Target p = 0.10-0.25)

**Purpose**: Break the ceiling. These identify the truly exceptional content that merits priority reading. Only the best articles in the inbox should pass these.

**Design principles**:
- Test for rare, high-value features
- Use high thresholds or compound criteria (presence of feature + quality of feature)
- Ask about features that only appear in content produced by domain experts with original thinking
- Accept that most articles -- even good ones -- will correctly fail these

**Example difficulty calibration**:
- "Does the article introduce a named concept or framework that provides a new way to categorize, evaluate, or decide something in its domain?" -- this is rare even among good articles. Maybe 10-15% of our inbox has this.

### Negative Signal Questions (Target p = 0.15-0.35, yes = penalty)

**Purpose**: Pull down inflated scores and counteract acquiescence bias. These identify content whose surface features might trigger positive responses on other questions but whose core value is low.

**Design principles**:
- Frame as positive assertions about negative characteristics
- Test for features that are anti-correlated with engagement
- Focus on content *types* that reliably produce low highlights regardless of apparent quality

**Example difficulty calibration**:
- "Is this primarily a summary or aggregation of existing information, rather than original analysis or argument?" -- about 25% of inbox content fits this pattern, and such content consistently gets fewer highlights.

---

## Part 8: Proposed Binary Questions

The following 24 questions are organized by dimension and difficulty tier. Each question is designed to maximize discrimination within its tier and to target the observable textual features that predict highlight behavior.

### Scoring Mechanics

- **Positive questions**: yes = +1 point toward the dimension
- **Negative signal questions**: yes = -1 point toward the dimension (marked with [PENALTY])
- **Each dimension has 6 questions** (5 positive + 0-1 negative, or 4 positive + 1 negative, etc.)
- **Total scored items**: 24 questions, producing a raw score that maps to 0-100

The raw score per dimension = (positive yes count - negative yes count) / max positive count * 25, clamped to [0, 25].

---

### Dimension 1: Quotability (Memorable Passages Worth Saving)

These questions test whether the article contains passages a reader would want to highlight, bookmark, or excerpt as standalone notes.

| ID | Question | Difficulty | Notes |
|----|----------|-----------|-------|
| Q1 | Does the article contain at least one specific claim, data point, or concrete example that goes beyond abstract generalization? | Easy (p~0.75) | Baseline: tests for any substance at all. News briefs with no specifics fail. |
| Q2 | Does the article include a sentence or short passage that encapsulates a key idea in a way that is memorable for its phrasing, not just its content -- a striking metaphor, a crisp formulation, or a vivid example? | Medium (p~0.40) | Tests for *craft* in expression, not just information. Requires the LLM to evaluate language quality in a specific, bounded way. |
| Q3 | Does the article cite a specific quantified finding (a number, percentage, measurement, or research result) that would be worth remembering on its own? | Medium (p~0.45) | Tests for citable data. Many opinion pieces and frameworks articles fail this; research-backed pieces and data-driven analysis pass. |
| Q4 | Could you extract a self-contained passage of 2-4 sentences from this article that would be valuable as a standalone note -- making sense without the surrounding context? | Hard (p~0.25) | The core "highlightability" test. Requires not just good content but modular, extractable content. This is the strongest proxy for actual highlight behavior. |
| Q5 | Does the article contain a direct, attributed statement from a named individual (the author speaking from experience, or a quoted expert/practitioner) that adds credibility or insight beyond what unsourced claims would provide? | Medium (p~0.50) | Tests for concrete, authoritative voice. Generic advice fails; specific practitioners sharing specific lessons pass. |
| Q6 | Is the article's value primarily in linking to or summarizing other sources, rather than in its own prose? | Negative (p~0.20) [PENALTY] | Penalty for aggregation content (link roundups, "best of" lists, news digests). Yes = the article's passages are not themselves worth saving. |

### Dimension 2: Surprise Factor (Challenges Assumptions, Unexpected Findings)

These questions test whether the content offers genuinely novel perspectives or information the reader has not encountered before.

| ID | Question | Difficulty | Notes |
|----|----------|-----------|-------|
| Q7 | Does the article address a real topic with enough depth that a reader could discuss it substantively with a colleague? | Easy (p~0.75) | Baseline: filters out trivial announcements and thin content. Most inbox articles pass this. |
| Q8 | Does the article present a specific finding, conclusion, or perspective that directly contradicts or complicates a widely-held assumption in its domain? | Medium (p~0.35) | Tests for genuine intellectual friction. Most articles that "add to the conversation" don't actually challenge anything -- they confirm. |
| Q9 | Does the article reframe a familiar topic by connecting it to an unexpected domain, analogy, or historical parallel that changes how the reader might think about it? | Hard (p~0.15) | Tests for creative reframing. This is rare and high-signal for engagement when present. |
| Q10 | Does the article contain a case study, example, or piece of evidence that the reader is unlikely to have encountered in prior reading on this topic? | Medium (p~0.40) | Tests for informational novelty at the evidence level. Not "is the thesis new" but "is the supporting material new." |
| Q11 | Does the article develop an idea through multiple stages or layers, reaching a conclusion that is not obvious from its opening premise? | Medium (p~0.35) | Tests for argumentative development. Articles that state their conclusion upfront and then repeat it fail. Articles that *build* toward a non-obvious destination pass. |
| Q12 | Could the article's central point be accurately conveyed in a single sentence without significant loss of meaning? | Negative (p~0.30) [PENALTY] | Penalty for thin content dressed up in length. If the idea is simple and the article adds no real complexity, it is low surprise regardless of word count. |

### Dimension 3: Argument Quality (Well-Supported Claims, Strong Opinions)

These questions test whether the author constructs a genuine argument with conviction, evidence, and reasoning, rather than reporting, hedging, or summarizing.

| ID | Question | Difficulty | Notes |
|----|----------|-----------|-------|
| Q13 | Does the author take a clear position on a debatable topic -- one where a reasonable, informed person could argue the opposite? | Easy (p~0.65) | Baseline: filters out pure reporting and neutral explainers. |
| Q14 | Does the author support their central claim with at least two distinct types of evidence (e.g., personal experience AND external data; case study AND logical argument; historical example AND current research)? | Medium (p~0.40) | Tests for evidence breadth. Single-type arguments (pure opinion, pure data) score lower than mixed-evidence arguments. |
| Q15 | Does the author write from direct, first-person professional experience -- describing specific situations they personally faced, decisions they made, or results they observed? | Medium (p~0.45) | Tests for practitioner voice. Separates genuine practitioners from commentators and summarizers. |
| Q16 | Does the author engage with and respond to the strongest counterargument to their position, rather than ignoring or strawmanning opposing views? | Hard (p~0.20) | Tests for intellectual rigor. Most persuasive articles argue *for* their position but don't genuinely engage with the best opposing case. |
| Q17 | Does the argument contain an element of genuine intellectual risk -- where the author commits to a specific, falsifiable prediction, recommends against a popular approach, or admits to a significant failure? | Hard (p~0.15) | Tests for skin in the game. The highest-engagement articles often feature an author who took a risk with their reputation by making a concrete, checkable claim. |
| Q18 | Is the article primarily reporting or explaining what others have said or done, rather than advancing the author's own analysis or argument? | Negative (p~0.25) [PENALTY] | Penalty for passive/reporting content. Journalism and news coverage is valuable but generates fewer highlights than original argument. |

### Dimension 4: Applicable Insight (Frameworks, Mental Models, Techniques)

These questions test whether the reader could extract something usable from the article -- a framework, technique, decision criterion, or mental model that changes how they approach problems.

| ID | Question | Difficulty | Notes |
|----|----------|-----------|-------|
| Q19 | Does the article contain at least one idea, recommendation, or observation that could influence how a reader thinks about or approaches a problem in their own work? | Easy (p~0.70) | Baseline: tests for any degree of applicability. Pure entertainment, pure news, and pure historical narrative fail. |
| Q20 | Does the article introduce, name, or clearly articulate a framework, mental model, or structured approach that organizes thinking about a class of problems? | Hard (p~0.15) | Tests for conceptual tools. Named frameworks (e.g., "the OODA loop," "Wardley mapping") are rare and high-signal for engagement. Unnamed but clearly structured approaches also count. |
| Q21 | Does the article provide enough concrete detail (specific steps, criteria, examples, or conditions) that a reader could attempt to apply its core idea without needing to seek additional sources? | Medium (p~0.35) | Tests for completeness of insight. Many articles gesture at a good idea but don't operationalize it. |
| Q22 | Does the article describe a specific technique, practice, or decision process that the reader could try in a concrete situation within the next month? | Medium (p~0.40) | Tests for near-term applicability. Theoretical or long-horizon ideas fail; immediately usable techniques pass. |
| Q23 | Does the article offer a perspective shift -- a way of seeing a familiar situation that, once understood, would persistently change the reader's default interpretation or approach? | Hard (p~0.20) | Tests for lasting cognitive impact. The highest form of applicable insight: not a thing to do, but a new lens that changes everything you do. |
| Q24 | Is the article's content so narrowly domain-specific that it would be useful only to practitioners of a single specialized profession or technology? | Negative (p~0.20) [PENALTY] | Mild penalty for extreme narrowness. Articles about "how to configure Kubernetes ingress controllers" are valuable to that audience but don't produce highlights from a general reader. Scored as a mild negative, not a hard penalty. |

---

## Part 9: Question Difficulty Summary

### By Difficulty Tier

**Easy (5 questions, target p = 0.65-0.80):**
| ID | Dimension | Question (abbreviated) |
|----|-----------|----------------------|
| Q1 | Quotability | Contains specific claim, data, or concrete example |
| Q7 | Surprise | Enough depth to discuss substantively |
| Q13 | Argument | Author takes a clear position on a debatable topic |
| Q19 | Insight | Contains at least one applicable idea |
| Q5 | Quotability | Contains direct attributed statement from named individual |

**Medium (9 questions, target p = 0.35-0.55):**
| ID | Dimension | Question (abbreviated) |
|----|-----------|----------------------|
| Q2 | Quotability | Memorable phrasing, striking metaphor, crisp formulation |
| Q3 | Quotability | Cites specific quantified finding |
| Q8 | Surprise | Contradicts widely-held assumption |
| Q10 | Surprise | Contains unfamiliar case study or evidence |
| Q11 | Surprise | Develops idea through multiple stages to non-obvious conclusion |
| Q14 | Argument | Supports claim with 2+ distinct evidence types |
| Q15 | Argument | Writes from first-person professional experience |
| Q21 | Insight | Provides enough detail to apply without additional sources |
| Q22 | Insight | Describes technique applicable in next month |

**Hard (6 questions, target p = 0.10-0.25):**
| ID | Dimension | Question (abbreviated) |
|----|-----------|----------------------|
| Q4 | Quotability | Contains extractable 2-4 sentence standalone passage |
| Q9 | Surprise | Reframes through unexpected domain, analogy, or parallel |
| Q16 | Argument | Engages with strongest counterargument |
| Q17 | Argument | Contains intellectual risk: falsifiable prediction, anti-popular recommendation, or admitted failure |
| Q20 | Insight | Introduces a named framework or mental model |
| Q23 | Insight | Offers a persistent perspective shift |

**Negative Signal (4 questions, yes = penalty):**
| ID | Dimension | Question (abbreviated) |
|----|-----------|----------------------|
| Q6 | Quotability | Value is primarily in linking/summarizing other sources |
| Q12 | Surprise | Central point conveyable in a single sentence |
| Q18 | Argument | Primarily reporting/explaining what others said |
| Q24 | Insight | Content so narrow it's useful only to one specialized profession |

### By Dimension

| Dimension | Easy | Medium | Hard | Negative | Total |
|-----------|------|--------|------|----------|-------|
| Quotability | Q1, Q5 | Q2, Q3 | Q4 | Q6 | 6 |
| Surprise | Q7 | Q8, Q10, Q11 | Q9 | Q12 | 6 |
| Argument | Q13 | Q14, Q15 | Q16, Q17 | Q18 | 6 |
| Insight | Q19 | Q21, Q22 | Q20, Q23 | Q24 | 6 |

---

## Part 10: Implementation Considerations

### Expected Score Distribution Improvement

**Current system (v2-categorical):**
- 8 questions, mix of binary and categorical
- Score range 0-100, but practical range is 55-100
- 70%+ of articles score 75-100
- No discrimination in the top tier

**Proposed system (24 binary questions):**
- Expected practical range: 20-90
- Bell centered around 50-55 for the inbox population
- Top-tier articles (80+) represent ~10-15% of population
- Bottom-tier articles (<35) represent ~10-15% of population
- Core discrimination zone (40-75) captures ~70% of articles with meaningful spread

### Prompt Structure

The 24 questions should be presented in a single prompt, grouped by dimension. Each group should be labeled. The response format should be JSON with boolean values and brief evidence citations:

```json
{
  "q1": {"answer": true, "evidence": "The author cites a specific study..."},
  "q2": {"answer": false, "evidence": ""},
  ...
}
```

Requiring evidence for "yes" answers (even brief) grounds the LLM's responses and reduces acquiescence bias.

### Token Cost

The current system uses ~700 max output tokens for 8 questions with reasons. The proposed 24-question system with brief evidence will require approximately 1200-1500 output tokens. Input tokens remain dominated by article content (unchanged). Net cost increase: ~30-50% on output tokens, which is modest relative to the substantial improvement in discrimination.

### Calibration After Deployment

After deploying the new questions, the iterative calibration process should:

1. Score 100+ articles with the new system
2. Correlate each question's yes/no with actual highlight count
3. Compute item discrimination indices
4. Identify questions with D < 0.20 and revise or replace them
5. Check for questions with unexpectedly high or low base rates and adjust framing
6. Verify that the total score distribution has the expected spread
7. Compare Spearman correlation of new total scores vs. highlight counts against the v2-categorical baseline

The existing calibration toolkit (`just cal-report`, `just cal-dimensions`, `just cal-misses`) can be adapted to evaluate the new scoring system against the same highlight-count dependent variable.

---

## References

- [CheckEval: A reliable LLM-as-a-Judge framework (EMNLP 2025)](https://arxiv.org/abs/2403.18771)
- [IDGen: Item Discrimination Induced Prompt Generation (NeurIPS 2024)](https://arxiv.org/abs/2409.18892)
- [ELEPHANT: Measuring social sycophancy in LLMs (2025)](https://arxiv.org/html/2505.13995v2)
- [Acquiescence Bias in Large Language Models (2025)](https://arxiv.org/html/2509.08480v1)
- [When Wording Steers the Evaluation: Framing Bias in LLM Judges (2026)](https://arxiv.org/html/2601.13537)
- [Sycophancy in Large Language Models: Causes and Mitigations (2024)](https://arxiv.org/abs/2411.15287)
- [DEBATE: Devil's Advocate for LLM evaluation (ACL Findings 2024)](https://arxiv.org/abs/2405.09935)
- [Hamel Husain: Using LLM-as-a-Judge for Evaluation](https://hamel.dev/blog/posts/llm-judge/)
- [Hamel Husain: LLM Evals FAQ (2026)](https://hamel.dev/blog/posts/evals-faq/)
- [Eugene Yan: Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/)
- [Arize AI: Testing Binary vs Score Evals on Latest Models (2025)](https://arize.com/blog/testing-binary-vs-score-llm-evals-on-the-latest-models/)
- [Evidently AI: LLM-as-a-Judge Complete Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [DeepEval: LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [GoDaddy: Calibrating Scores of LLM-as-a-Judge](https://www.godaddy.com/resources/news/calibrating-scores-of-llm-as-a-judge)
- [Item Response Theory - Columbia University](https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory)
- [Item Discrimination in Psychometrics](https://www.cogn-iq.org/learn/theory/item-discrimination/)
- [Ceiling and Floor Effects in Psychometric Testing](https://www.cogn-iq.org/learn/theory/ceiling-floor-effects/)
- [Evaluating Item Difficulty and Discrimination](https://foodsafety.institute/research-methodology/evaluating-item-difficulty-discrimination-tests/)
- [Professional Testing: Steps in Test Development](https://proftesting.com/test_topics/steps_9.php)
- [Examining the Effect of Reverse Worded Items on Factor Structure](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0157795)
- [The Psychology of Survey Responses: How Wording Impacts Data Quality](https://insight7.io/the-psychology-of-survey-responses-how-wording-impacts-data-quality/)
