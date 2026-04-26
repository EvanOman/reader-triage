# Current Scoring System Analysis

Analysis of the v2-categorical scoring system as implemented in `app/services/scorer.py` and `app/services/scoring_strategy.py`.

## 1. Current Scoring Design

### Overview

The system asks Claude 8 categorical questions about an article, then maps the responses to numeric scores via lookup tables. The 8 questions feed into 4 scoring dimensions (25 points each, 100 total), with each dimension composed of 1-3 component questions.

### The 8 Questions

| # | Question | Response Type | Feeds Into |
|---|----------|--------------|------------|
| Q1 | STANDALONE PASSAGES: How many passages could stand alone as a saved note? | `none / a_few / several / many` | Quotability |
| Q2 | NOVEL FRAMING: Does it reframe a familiar topic or present a surprising lens? | `true / false` | Surprise |
| Q3 | CONTENT TYPE: What best describes this content? | 5 categorical options | Surprise |
| Q4 | AUTHOR CONVICTION: Does the author argue for a clear position with conviction? | `true / false` | Argument |
| Q5 | PRACTITIONER VOICE: Is this written from first-person practitioner experience? | `true / false` | Argument |
| Q6 | CONTENT COMPLETENESS: Does the available text appear complete? | `complete / appears_truncated / summary_or_excerpt` | Argument |
| Q7 | NAMED FRAMEWORK: Does it introduce or organize around a named concept/framework? | `true / false` | Insight |
| Q8 | APPLICABLE IDEAS: Could a reader apply ideas from this? | `broadly / narrowly / not_really` | Insight |

### Point Mapping Tables

**Quotability (0-25) -- single question:**

| Q1: standalone_passages | Points |
|------------------------|--------|
| `none`                 | 0      |
| `a_few`                | 9      |
| `several`              | 17     |
| `many`                 | 25     |

**Surprise Factor (0-25) -- two questions summed:**

| Q2: novel_framing | Points |
|-------------------|--------|
| `true`            | 15     |
| `false`           | 0      |

| Q3: content_type (article) | Points |
|---------------------------|--------|
| `original_analysis`       | 10     |
| `opinion_with_evidence`   | 8      |
| `informational_summary`   | 3      |
| `product_review`          | 2      |
| `news_or_roundup`         | 0      |

Maximum: 15 + 10 = 25. Minimum: 0 + 0 = 0.

**Argument Quality (0-25) -- three questions summed:**

| Q4: author_conviction | Points |
|----------------------|--------|
| `true`               | 12     |
| `false`              | 0      |

| Q5: practitioner_voice | Points |
|-----------------------|--------|
| `true`                | 8      |
| `false`               | 0      |

| Q6: content_completeness | Points |
|-------------------------|--------|
| `complete`              | 5      |
| `appears_truncated`     | 2      |
| `summary_or_excerpt`    | 0      |

Maximum: 12 + 8 + 5 = 25. Minimum: 0 + 0 + 0 = 0.

**Applicable Insight (0-25) -- two questions summed:**

| Q7: named_framework | Points |
|--------------------|--------|
| `true`             | 12     |
| `false`            | 0      |

| Q8: applicable_ideas | Points |
|---------------------|--------|
| `broadly`           | 13     |
| `narrowly`          | 7      |
| `not_really`        | 0      |

Maximum: 12 + 13 = 25. Minimum: 0 + 0 = 0.

### Theoretical Score Distribution

To understand the structural properties of this scoring system, we can enumerate the possible score combinations per dimension.

**Quotability** has 4 distinct values: {0, 9, 17, 25}. This is the dimension with the most granularity.

**Surprise Factor** has 6 distinct values (2 x 5 matrix, but capped at 25):
- novel_framing=false: {0, 2, 3, 8, 10}
- novel_framing=true: {15, 17, 18, 23, 25}

All 10 combinations produce distinct scores: {0, 2, 3, 8, 10, 15, 17, 18, 23, 25}.

**Argument Quality** has 8 distinct values (2 x 2 x 3 matrix):
- (F, F, excerpt): 0
- (F, F, truncated): 2
- (F, F, complete): 5
- (F, T, excerpt): 8
- (F, T, truncated): 10
- (T, F, excerpt): 12
- (F, T, complete): 13
- (T, F, truncated): 14
- (T, F, complete): 17
- (T, T, excerpt): 20
- (T, T, truncated): 22
- (T, T, complete): 25

12 distinct values: {0, 2, 5, 8, 10, 12, 13, 14, 17, 20, 22, 25}.

**Applicable Insight** has 4 distinct values (2 x 3 matrix, but some overlap):
- (F, not_really): 0
- (F, narrowly): 7
- (T, not_really): 12
- (F, broadly): 13
- (T, narrowly): 19
- (T, broadly): 25

6 distinct values: {0, 7, 12, 13, 19, 25}.

**Total possible score combinations:** 4 x 10 x 12 x 6 = 2,880 distinct response vectors. But many map to the same total score. The total score ranges from 0 to 100.

### Score Range Reachability

Here are the score ranges and what it takes to reach them:

**Score >= 60 (High tier):**
A score of 60 requires averaging 15/25 per dimension. Here is one of many paths to exactly 60:
- Q1 = several (17), Q2 = true + Q3 = opinion_with_evidence (23), Q4 = true + Q5 = false + Q6 = truncated (14), Q7 = false + Q8 = narrowly (7) = 61

Even more telling: consider an article where a reader would answer "moderately positive" to most questions:
- Q1 = a_few (9)
- Q2 = true, Q3 = opinion_with_evidence (8) --> 23
- Q4 = true (12), Q5 = false (0), Q6 = complete (5) --> 17
- Q7 = false (0), Q8 = broadly (13) --> 13

Total: 9 + 23 + 17 + 13 = **62**. This is a "mildly interesting" article that passes the High threshold.

**Score < 30 (Low tier):**
To score under 30, an article must average under 7.5 per dimension. This requires substantial failure across all four dimensions. Consider what it takes:
- If any single dimension scores 25, the other three must average under 1.67 -- essentially all zeroes.
- If Surprise = 15 (novel_framing=true, worst content type), the remaining three must sum to under 15. That means Quotability = none (0), Argument near zero, and Insight near zero. This requires the article to have zero quotable passages, no author conviction, no practitioner voice, no framework, and no applicable ideas.

This means the Low tier is structurally reserved for articles that fail on nearly every dimension.

**Score 30-59 (Medium tier):**
This is the "zone of ambiguity." An article needs to pass on some questions but fail on others. For example:
- Q1 = a_few (9), Q2 = false + Q3 = informational_summary (3), Q4 = false + Q5 = false + Q6 = complete (5), Q7 = false + Q8 = narrowly (7) = **24** (Low)
- Q1 = a_few (9), Q2 = false + Q3 = original_analysis (10), Q4 = true + Q5 = false + Q6 = complete (17), Q7 = false + Q8 = not_really (0) = **36** (Medium)

The Medium tier requires a mix of some passed and some failed questions.

## 2. Ceiling Effect Analysis

### The "Agreeable LLM" Baseline

The critical structural issue is: what score does an article get if the LLM answers positively to "most" questions? Let's define a "typical positive response" for each question:

| Question | "Typical positive" answer | Points |
|----------|--------------------------|--------|
| Q1 | `a_few` (not `many`) | 9 |
| Q2 | `true` | 15 |
| Q3 | `opinion_with_evidence` | 8 |
| Q4 | `true` | 12 |
| Q5 | `false` (most articles aren't practitioner-authored) | 0 |
| Q6 | `complete` | 5 |
| Q7 | `false` (most articles don't name a framework) | 0 |
| Q8 | `narrowly` | 7 |

Total: 9 + 23 + 17 + 7 = **56** (Medium, close to High boundary).

Now make Q5 true (practitioner) and Q7 true (framework), which are generous but not absurd for a decent blog post:

Total: 9 + 23 + 25 + 19 = **76** (solidly High).

### Minimum Score for "Mostly Positive"

If an LLM says "true" to all boolean questions and picks moderate categorical options:

| Q | Answer | Points |
|---|--------|--------|
| Q1 | `a_few` | 9 |
| Q2 | `true` | 15 |
| Q3 | `opinion_with_evidence` | 8 |
| Q4 | `true` | 12 |
| Q5 | `true` | 8 |
| Q6 | `complete` | 5 |
| Q7 | `true` | 12 |
| Q8 | `narrowly` | 7 |

Total: 9 + 23 + 25 + 19 = **76**.

With Q1 = `several` instead: 17 + 23 + 25 + 19 = **84**.
With Q8 = `broadly` instead of `narrowly`: 9 + 23 + 25 + 25 = **82**.
With both: **92**.

### How Many Questions Must Fail to Score Under 60?

Starting from the "all positive, moderate categorical" baseline of 76, each question's failure contributes:

| Question flipped negative | Points lost | New total |
|--------------------------|-------------|-----------|
| Q2: novel_framing = false | -15 | 61 |
| Q4: author_conviction = false | -12 | 64 |
| Q7: named_framework = false | -12 | 64 |
| Q5: practitioner_voice = false | -8 | 68 |
| Q8: broadly -> not_really | -7 (from 7 to 0) | 69 |
| Q1: a_few -> none | -9 | 67 |
| Q3: opinion -> news_or_roundup | -8 | 68 |
| Q6: complete -> excerpt | -5 | 71 |

From the 76-point baseline, flipping *any single* boolean question negative keeps the score above 60 (except Q2, which barely stays at 61). **You need to flip at least 2 of the high-value boolean questions negative** to drop below 60.

To reach under 60 from the 76 baseline, some example failure paths:
- Q2=false AND Q4=false: 76 - 15 - 12 = **49**
- Q2=false AND Q7=false: 76 - 15 - 12 = **49**
- Q4=false AND Q7=false: 76 - 12 - 12 = **52**
- Q5=false AND Q1=none: 76 - 8 - 9 = **59** (just barely under)

### How Many Questions Must Fail to Score Under 30?

From the 76 baseline, you need to lose at least 47 points. The total available to lose from "all positive" is 76 itself (down to 0). The biggest contributors:

- Q2=false: -15
- Q4=false: -12
- Q7=false: -12
- Q1=none: -9
- Q5=false: -8
- Q3=news_or_roundup: -8
- Q8=not_really: -7
- Q6=excerpt: -5

Sum of all: 76. To lose 47+, you need to fail on essentially *every* question: at minimum, all 4 boolean questions (15+12+12+8=47 lost) AND at least one categorical downgrade. **An article must fail on 5-6 of the 8 questions to score under 30.**

### Why Clustering at the Top Is Structurally Likely

1. **Binary questions with high point values dominate the scoring.** Four of eight questions are true/false, and they collectively control 47 of 100 possible points (Q2=15, Q4=12, Q5=8, Q7=12). An LLM that defaults to "true" for these creates a 47-point floor before even considering categorical questions.

2. **The categorical questions have high "default" scores.** Q1's `a_few` (9/25) is the lowest non-zero option. Q3's `opinion_with_evidence` (8/10) is common for any article with a point of view. Q8's `narrowly` (7/13) is the moderate answer. These add another ~24 points for "sort of positive" answers.

3. **Content completeness (Q6) almost always scores 5.** Unless the content was actually truncated during fetch, the LLM will answer `complete`, contributing 5 free points.

4. **The Low tier (<30) is nearly unreachable for real articles.** It requires failure across nearly every question. For any article that has *any* redeeming quality -- a memorable phrase, a clear opinion, applicable ideas -- the score will land in Medium or High.

5. **The effective scoring range is compressed.** With typical articles scoring 40-85, the 0-100 scale is used inefficiently. The 0-30 range is dead space for all but the most vacuous content.

## 3. Question Quality Assessment

### Q1: STANDALONE PASSAGES

**Definition clarity: Good.** "Passages that could stand alone as a saved note" is concrete and directly tied to the use case (Readwise highlights). The examples ("memorable phrasing, crisp claim, striking example") help calibrate.

**Likely LLM base rates:**
- `none`: ~5-10% (only for truly generic content)
- `a_few`: ~55-65% (the safe default -- almost any article has "a few" notable sentences)
- `several`: ~20-30% (well-written articles with strong examples)
- `many`: ~5-10% (exceptional, dense-with-insight articles)

**Point value match:** The 0/9/17/25 spread is good. The jump from `none` to `a_few` (0 to 9) provides meaningful differentiation. The gap between `a_few` (9) and `several` (17) is 8 points, which is appropriate for the difficulty of that distinction. This question has the best calibration in the system.

**Discrimination power: Moderate-high.** This is probably the most discriminating question because the LLM genuinely varies its answers, and the answer correlates with article quality. However, `a_few` is over-assigned, compressing most articles into the 9-point bucket.

**Redundancy: Low.** This measures something distinct from the other questions -- the density of highlight-worthy passages rather than structural or topical qualities.

### Q2: NOVEL FRAMING

**Definition clarity: Moderate.** "Reframe a familiar topic or present a surprising lens" is somewhat subjective. What counts as "surprising" depends heavily on the evaluator's baseline knowledge, which an LLM doesn't truly have. The LLM has to guess whether the *reader* would find it surprising.

**Likely LLM base rates:**
- `true`: ~55-70% for curated reading lists (these articles were saved by the user, implying some novelty)
- `false`: ~30-45%

**Point value match: Overweighted.** At 15 points, this is the single highest-value question in the system. A single true/false judgment controls 15% of the total score. Given the subjectivity and the LLM's tendency toward "true" (it's hard to know what the reader already knows), this creates score inflation.

**Discrimination power: Low-moderate.** The binary nature limits granularity. An article that offers one mildly novel reframe scores the same 15 as one with a genuinely paradigm-shifting perspective.

**Redundancy: Moderate overlap with Q3.** `original_analysis` (10 pts) in the content type question already captures much of what "novel framing" measures. An original analysis almost always involves some novel framing. These two questions are in the same dimension (Surprise), amplifying the issue.

### Q3: CONTENT TYPE

**Definition clarity: Good.** The categories are well-defined and mutually exclusive. An LLM can reliably classify content into these buckets.

**Likely LLM base rates (for tech/knowledge content):**
- `original_analysis`: ~25-35%
- `opinion_with_evidence`: ~30-40%
- `informational_summary`: ~15-20%
- `product_review`: ~3-5%
- `news_or_roundup`: ~10-15%

**Point value match: Reasonable.** The 0-10 scale is proportionate within the dimension. The gap between `original_analysis` (10) and `opinion_with_evidence` (8) may be too small -- it amounts to only 2 points of total score difference, which is noise.

**Discrimination power: Moderate.** The 5-way categorical split provides genuine differentiation. The main issue is that the top two categories (`original_analysis` and `opinion_with_evidence`) together capture 55-75% of articles and differ by only 2 points.

**Redundancy: Moderate overlap with Q2 and Q4.** Original analysis implies novel framing (Q2) and author conviction (Q4). An `opinion_with_evidence` piece almost always has author conviction. The content type is partially redundant with the qualities measured elsewhere.

### Q4: AUTHOR CONVICTION

**Definition clarity: Good.** "Argue for a clear position with conviction, rather than just reporting or summarizing" draws a clear line between opinion pieces and informational content.

**Likely LLM base rates:**
- `true`: ~60-75% (most articles saved to a reading list express some viewpoint)
- `false`: ~25-40%

**Point value match: Slightly overweighted.** At 12 points, this is the largest contributor to Argument Quality. Combined with Q5 (practitioner_voice=true, 8 pts), a single boolean pair controls 20 points of the total score. The issue: "conviction" is a low bar. Even a mildly opinionated article clears it.

**Discrimination power: Low-moderate.** Binary response limits nuance. An article with tepid opinions and one with a passionate, well-argued thesis both score 12.

**Redundancy: High overlap with Q3.** An article classified as `opinion_with_evidence` or `original_analysis` will almost always have author conviction. This creates double-counting: the article gets 8-10 points from Q3 AND 12 points from Q4 for essentially the same quality (being opinionated).

### Q5: PRACTITIONER VOICE

**Definition clarity: Good.** "First-person practitioner experience sharing hard-won opinions" is specific and well-bounded.

**Likely LLM base rates:**
- `true`: ~30-45% (many tech blogs are from practitioners, but news and analysis pieces are not)
- `false`: ~55-70%

**Point value match: Reasonable.** 8 points feels proportionate -- it's a meaningful but not dominant signal. Practitioner content tends to be more actionable, which aligns with the scoring goals.

**Discrimination power: Moderate.** This is one of the better-discriminating questions because it genuinely separates practitioners from commentators/journalists. The false rate is high enough to create differentiation.

**Redundancy: Low-moderate.** Some overlap with Q4 (practitioners usually have conviction), but the overlap is one-directional -- conviction doesn't imply practitioner experience.

### Q6: CONTENT COMPLETENESS

**Definition clarity: Good.** The three options are well-defined and the LLM can assess them mechanically.

**Likely LLM base rates:**
- `complete`: ~80-90% (most fetched articles are complete)
- `appears_truncated`: ~5-10%
- `summary_or_excerpt`: ~5-10%

**Point value match: Appropriate but low-impact.** At 5 maximum points, this is the lowest-weighted question. The near-universal `complete` answer means it contributes 5 points to almost every article, functioning as a constant rather than a discriminator.

**Discrimination power: Very low.** This question almost never varies. It exists to handle an edge case (bad content fetching) rather than to differentiate article quality. When content is genuinely truncated, the system already has a separate `content_fetch_failed` flag and rescore mechanism.

**Redundancy: N/A.** Not redundant with other questions, but also not discriminating.

### Q7: NAMED FRAMEWORK

**Definition clarity: Good.** "Named concept, framework, or mental model" is specific and verifiable.

**Likely LLM base rates:**
- `true`: ~25-40% (depends on content type; strategy/methodology articles often have frameworks, news rarely does)
- `false`: ~60-75%

**Point value match: Slightly overweighted.** At 12 points (same as author_conviction), this binary question heavily influences the Insight dimension. An article with a named framework gets 12/25 of the Insight dimension automatically. The issue: having a named framework is a proxy for actionability but not a guarantee of it. An article mentioning the "Eisenhower Matrix" in passing scores the same 12 as one that deeply explains and teaches a new framework.

**Discrimination power: Moderate.** The false rate is high enough (~60-75%) to create real differentiation. This is one of the more useful binary questions.

**Redundancy: Low.** Frameworks are distinct from the qualities measured by other questions.

### Q8: APPLICABLE IDEAS

**Definition clarity: Moderate.** "Apply ideas in their own work or thinking" is somewhat vague. What counts as "applying in thinking"? Almost any interesting article changes how you think about something. The `broadly/narrowly/not_really` distinction is also subjective.

**Likely LLM base rates:**
- `broadly`: ~25-35%
- `narrowly`: ~45-55%
- `not_really`: ~15-25%

**Point value match: The spread is awkward.** The jump from `not_really` (0) to `narrowly` (7) is steep -- it means any article with even narrowly applicable ideas gets 7 free points. The jump from `narrowly` (7) to `broadly` (13) adds only 6 more. The middle option captures the majority of articles and gives them a significant score.

**Discrimination power: Low-moderate.** `narrowly` is the default answer for most content, which means 45-55% of articles cluster at exactly 7 points. The question differentiates poorly between "sort of useful" and "transformatively useful."

**Redundancy: Moderate overlap with Q7.** Articles with named frameworks are almost always "applicable" -- the framework IS the applicable idea. This creates double-counting in the Insight dimension.

## 4. Point Table Analysis

### Effective Budget Per Question

The 100-point total is distributed across 8 questions with very different weights:

| Question | Max Points | % of Total | Type |
|----------|-----------|------------|------|
| Q1: Standalone passages | 25 | 25.0% | 4-level categorical |
| Q2: Novel framing | 15 | 15.0% | Binary |
| Q4: Author conviction | 12 | 12.0% | Binary |
| Q7: Named framework | 12 | 12.0% | Binary |
| Q3: Content type | 10 | 10.0% | 5-level categorical |
| Q1: Standalone passages | 9 (floor for `a_few`) | -- | (effective minimum for non-zero) |
| Q8: Applicable ideas | 13 | 13.0% | 3-level categorical |
| Q5: Practitioner voice | 8 | 8.0% | Binary |
| Q6: Content completeness | 5 | 5.0% | 3-level categorical |

**Key observation:** The four binary questions (Q2, Q4, Q5, Q7) collectively control **47 points** of the total 100. Binary questions are the least granular response format and the most susceptible to LLM "yes-bias." Nearly half the score is determined by four yes/no answers.

### Dimension Weight Imbalance

Although each dimension has a 25-point cap, the internal composition varies dramatically:

- **Quotability (Q1 only):** Single question controls the entire dimension. This is actually the best-designed dimension because the 4-level categorical provides real granularity across the full 0-25 range.

- **Surprise (Q2 + Q3):** Two questions, but Q2 (15 pts) dominates Q3 (10 pts). The novel_framing boolean is 60% of the dimension. A single yes/no answer is worth more than the 5-category content type classification.

- **Argument (Q4 + Q5 + Q6):** Three questions, but Q4 (12 pts) dominates. Q6 (5 pts) is nearly a constant. Effective range is really driven by Q4 and Q5 -- two booleans controlling 20 of 25 points.

- **Insight (Q7 + Q8):** Two questions with relatively balanced weights (12 + 13), but Q7 is binary. The dimension is well-designed in terms of weight balance but suffers from the binary granularity of Q7.

### Wasted Points: The Completeness Tax

Q6 (content_completeness) contributes 5 points to virtually every article (since ~85% are complete). This means Argument Quality's effective range for most articles is 0-20 from Q4+Q5, plus a constant 5. The dimension effectively runs on a 0-20 scale, not 0-25.

### Score Compression in Practice

Given typical LLM response patterns, the expected score distribution looks approximately like:

- **Quotability:** Most articles land at 9 (`a_few`), with some at 17 and 25. Effective range: 9-25 for "good" articles.
- **Surprise:** Most articles get 8-23. The floor for `opinion_with_evidence + novel_framing=false` is 8. The floor for anything with novel_framing=true is 15+.
- **Argument:** Most articles get 5-25. `complete` (5) is near-universal. With conviction=true, the floor is 17.
- **Insight:** Most articles get 7-25. `narrowly` applicable (7) is the default.

Expected total for a "typical decent article": 9 + 18 + 17 + 7 = **51** (Medium). Expected total for a "good article": 17 + 23 + 25 + 19 = **84** (High). The gap between "decent" and "good" is 33 points, which seems like discrimination, but the issue is that *most* articles cluster in the 50-80 range because the LLM rarely assigns the most negative options.

## 5. Structural Recommendations

### 5.1. Reduce Binary Question Dominance

**Problem:** 47% of the score comes from four yes/no questions, which are the least discriminating format and the most prone to LLM acquiescence bias.

**Recommendation:** Convert the high-value binary questions (Q2, Q4, Q7) to 3-level or 4-level categoricals. For example:

- Q2 (novel_framing): Instead of true/false, use `genuinely_surprising / somewhat_fresh / familiar_territory`. Map to 15/7/0 instead of 15/0.
- Q4 (author_conviction): Instead of true/false, use `passionate_thesis / mild_opinion / neutral_reporting`. Map to 12/5/0.
- Q7 (named_framework): Instead of true/false, use `teaches_framework / mentions_concept / no_framework`. Map to 12/5/0.

This preserves the point ceilings but introduces middle tiers that the LLM can use for "sort of" answers, reducing the all-or-nothing dynamic.

### 5.2. Remove or Redesign Q6 (Content Completeness)

**Problem:** This question contributes 5 points to nearly every article as a constant. It measures content fetch quality, not article quality. The system already has dedicated mechanisms for handling truncated content (`content_fetch_failed`, stub detection, rescore logic).

**Recommendation:** Either remove Q6 entirely and redistribute its 5 points across other questions, or change it to measure something about the article's depth of treatment:
- Replace with "DEPTH OF TREATMENT: Does the piece develop its ideas with evidence, examples, and reasoning?" Options: `thorough / adequate / superficial`. This would actually discriminate between articles.

### 5.3. Address Q2 + Q3 Redundancy in Surprise Dimension

**Problem:** Q2 (novel_framing) and Q3 (content_type) overlap significantly. Original analysis implies novel framing. This double-counts the "originality" signal within a single dimension.

**Recommendation:** Either:
- (a) Merge them: Replace both with a single 5-level question combining novelty and content type. E.g., "How original and surprising is this content?" with options mapping to 0/6/12/18/25.
- (b) Move Q3 to a different dimension or make it a modifier: Content type is more of a category label than a quality signal. Use it as a scoring context (weight other answers differently by type) rather than as a direct point contributor.

### 5.4. Address Q4 + Q3 Cross-Dimension Redundancy

**Problem:** Q3's `opinion_with_evidence` (8 pts in Surprise) and Q4's `author_conviction` (12 pts in Argument) measure overlapping things. An opinionated article scores on both, getting 20 "free" points for having an opinion.

**Recommendation:** Make Q4 more specific about argument *quality* rather than just presence. Instead of "does the author have conviction," ask about the strength of evidence or reasoning. E.g., "How well does the author support their claims?" Options: `rigorous_evidence / anecdotal_support / assertions_only / not_applicable`. This separates "has an opinion" (Q3) from "argues well" (Q4).

### 5.5. Tighten Q8 (Applicable Ideas) Scoring

**Problem:** The `narrowly` option (7 points) is the default bucket for most articles, and 7/13 of the maximum is too generous for "sort of applicable." The gap between `narrowly` and `not_really` (7 vs 0) is larger than the gap between `narrowly` and `broadly` (7 vs 13).

**Recommendation:** Rebalance to `broadly: 13 / narrowly: 4 / not_really: 0`. This makes the middle option less generous and creates a bigger gap between "narrowly applicable" and "broadly applicable," which is where the meaningful distinction lies.

### 5.6. Consider Adding a "Density" or "Signal-to-Noise" Question

**Problem:** The current system has no question that captures whether the article is concise and information-dense versus padded and repetitive. A 5,000-word article with 500 words of insight scores the same as a 500-word article with 500 words of insight.

**Recommendation:** Add a question like: "INFORMATION DENSITY: What is the ratio of novel insight to filler, context-setting, and repetition?" Options: `very_dense / moderate / diluted / mostly_filler`. This would help differentiate the truly valuable articles from those that bury a few good ideas in excessive padding.

### 5.7. Consider Adding a "Personal Relevance" Negative Signal

**Problem:** The system has no mechanism for articles that score well on quality metrics but are on topics irrelevant to the reader. A brilliantly written deep dive into crochet patterns would score High if it has conviction, frameworks, and quotable passages.

**Observation:** This is partially addressed by the tagging system and author boost mechanism, but the base score itself has no relevance signal. Since the LLM doesn't know the reader's interests, this may be intentionally omitted. However, the content type classification (Q3) could be extended to include a "relevance to reader's typical interests" dimension if historical data about the reader's engagement patterns were injected into the prompt.

### 5.8. Flatten the "Easy Pass" Combinations

**Problem:** Several common article archetypes automatically score High:
- Blog post with opinions + practitioner = 9 + 23 + 25 + 7 = 64 (High) with just Q1=a_few and Q8=narrowly.
- Any article with novel_framing=true and author_conviction=true starts at 27 points before Q1, Q3, Q5, Q6, Q7, Q8 are even counted.

**Recommendation:** Reduce the point values of the most commonly "passed" questions and increase the values of discriminating ones. Specifically:
- Reduce Q2 (novel_framing) from 15 to 10.
- Reduce Q4 (author_conviction) from 12 to 8.
- Increase Q1 (standalone_passages) granularity: use {0, 5, 12, 20, 25} for {none, one_or_two, a_few, several, many}.
- Increase Q8 (applicable_ideas) top value: use {0, 5, 15, 25} for {not_really, narrowly, broadly, transformative}.

This shifts scoring power away from easy-pass booleans toward the questions that genuinely differentiate quality.

### Summary of Priority Changes

Ranked by expected impact on score discrimination:

1. **Convert binary questions to categoricals** (Q2, Q4, Q7) -- highest impact, directly addresses ceiling clustering.
2. **Rebalance Q8 scoring** (`narrowly` from 7 to 4) -- quick fix, reduces the generous middle ground.
3. **Remove or replace Q6** -- frees 5 points for more useful questions.
4. **Reduce Q2 point value** (15 to 10) -- directly addresses the single most inflationary question.
5. **Add middle tiers to create granularity** -- enables more nuanced scoring across the board.
6. **Address Q3/Q4 redundancy** -- prevents double-counting of "has opinions."
