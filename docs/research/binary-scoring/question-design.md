# Question Design for Binary Scoring

## Principles of Effective Binary Question Design

### 1. Atomic and Unambiguous

Each question must test exactly one thing. CheckEval's success comes from decomposing complex evaluation dimensions into granular, Boolean-answerable sub-questions. A question like "Does this article have novel insights and strong arguments?" conflates two dimensions; split it into two separate questions.

**Good**: "Does the author explicitly name or define a framework, model, or methodology?"
**Bad**: "Does the article contain useful frameworks and novel ideas?"

### 2. Observable, Not Inferential

Questions should ask about concrete, observable features of the text rather than requiring the LLM to make abstract quality judgments. The more the question points to a specific textual feature, the more reliable the binary judgment.

**Good**: "Does the article contain a specific data point, statistic, or research finding that challenges a common assumption?"
**Bad**: "Is this article surprising?"

### 3. Falsifiable with Clear Decision Boundary

The question must have a clear threshold for "yes" vs. "no." If you can't articulate what would make the answer "no," the question is too vague. Hamel Husain's core insight: "Binary decisions force clarity and compel a domain expert to define a clear line."

**Good**: "Does the author describe a personal experience or anecdote from their own professional practice?"
**Bad**: "Does the author seem experienced?"

### 4. Designed for Discrimination, Not Agreement

Most content will pass easy questions (e.g., "Is this article about a real topic?"). The value comes from questions that differentiate between good and excellent content. Drawing from Item Response Theory's discrimination parameter, each question should have a target "pass rate" in mind.

---

## Anti-Ceiling-Effect Strategies

The central problem: most articles score 80-100 with the current system. The following question design strategies combat this.

### Strategy 1: Difficulty-Tiered Questions

Borrow from Item Response Theory: design questions at three difficulty tiers.

**Tier 1: Baseline (expected pass rate ~70-80%)**
These establish that the content meets basic quality thresholds. Most reasonable articles pass these.
- "Does the article present at least one specific claim supported by evidence, data, or concrete examples?"
- "Is the content substantive enough to discuss with a colleague (not just a news brief or link roundup)?"

**Tier 2: Quality (expected pass rate ~30-50%)**
These separate good from mediocre content. Only well-written, substantive pieces pass these.
- "Does the article contain at least one passage (2-3 sentences) that you could extract as a standalone highlight worth revisiting later?"
- "Does the author present an argument or position that goes against prevailing wisdom or challenges a common assumption?"

**Tier 3: Exceptional (expected pass rate ~10-20%)**
These identify truly outstanding content. Only the best pieces pass these.
- "Does this article introduce a named concept, framework, or mental model that could change how the reader approaches a class of problems?"
- "Would a knowledgeable reader in this domain likely encounter at least one idea they haven't seen before?"

### Strategy 2: Devil's Advocate / Negative Signal Questions

Inspired by the DEBATE framework, include questions where "yes" *reduces* the score. These act as penalty signals, pulling scores down from inflated baselines.

- "Is this primarily a summary or roundup of other people's ideas, rather than original analysis?" (yes = penalty)
- "Could this article's main points be adequately captured in a single tweet?" (yes = penalty)
- "Is the content mostly generic advice that applies to everyone equally?" (yes = penalty)

### Strategy 3: Comparative/Relative Anchoring

Rather than absolute judgments ("Is this good?"), anchor to a reference class:

- "Among articles on this topic you've encountered, would this rank in the top 20%?"
- "Does this article offer something a reader couldn't easily find in the first page of Google results on this topic?"

Note: Comparative questions are harder for LLMs to calibrate reliably (they don't have a stable reference set). Use sparingly and weight lower.

### Strategy 4: "So What?" Questions

Test whether the content has lasting value beyond initial consumption:

- "Could a reader apply a specific technique or framework from this article in their own work within the next month?"
- "Does this article contain an insight that would still be relevant and valuable in 6 months?"

---

## Question Design for the Four Scoring Dimensions

Mapping binary questions to the existing dimension structure (Quotability, Surprise Factor, Argument Quality, Applicable Insight):

### Quotability (Memorable Passages)

Questions should test for the presence of specific, extractable passages rather than general quality.

| # | Question | Difficulty | Notes |
|---|----------|-----------|-------|
| Q1 | Does the article contain a memorable phrase, vivid metaphor, or striking sentence that encapsulates a key idea? | Medium | Tests for "highlightable" language |
| Q2 | Does the article include a specific data point, statistic, or quantified claim worth remembering? | Medium | Tests for concrete, citable facts |
| Q3 | Could you extract at least one passage (2-3 sentences) that would make sense as a standalone saved note? | Hard | The core "highlightability" test |
| Q4 | Does the article contain a direct quote from a practitioner or expert that adds credibility or color? | Medium | Tests for quotable attributed claims |

### Surprise Factor (Novel Perspectives)

Questions should test whether the content challenges assumptions or presents unexpected framings.

| # | Question | Difficulty | Notes |
|---|----------|-----------|-------|
| Q5 | Does the article present a perspective or conclusion that contradicts common industry assumptions? | Hard | Tests for genuine novelty |
| Q6 | Does the article reframe a familiar topic through an unexpected lens or analogy? | Hard | Tests for creative reframing |
| Q7 | Does the article contain a finding, case study, or example that the reader likely hasn't encountered before? | Medium | Tests for informational novelty |
| Q8 | Is this primarily a summary or restatement of widely-known ideas? | Easy (negative) | Penalty signal -- "yes" reduces score |

### Argument Quality (Reasoning and Evidence)

Questions should test the strength of reasoning, not just the presence of opinions.

| # | Question | Difficulty | Notes |
|---|----------|-----------|-------|
| Q9 | Does the author argue for a clear, specific position rather than presenting "both sides" neutrally? | Medium | Tests for conviction |
| Q10 | Does the author support their claims with concrete evidence (data, case studies, specific examples)? | Medium | Tests for evidence quality |
| Q11 | Does the author write from personal professional experience, sharing lessons learned from practice? | Medium | Tests for practitioner voice |
| Q12 | Could this article's main argument be adequately captured in a single tweet? | Easy (negative) | Penalty for shallow content |

### Applicable Insight (Actionable Value)

Questions should test whether the reader can actually *use* what they read.

| # | Question | Difficulty | Notes |
|---|----------|-----------|-------|
| Q13 | Does the article introduce or build upon a named framework, mental model, or methodology? | Hard | Tests for structured knowledge |
| Q14 | Could a reader apply a specific technique or idea from this article in their own work? | Medium | Tests for practical applicability |
| Q15 | Does the article provide enough context and detail for a reader to act on its recommendations? | Medium | Tests for completeness of advice |
| Q16 | Is the content so domain-specific that it would only be useful to a very narrow audience? | Medium (negative) | Moderate penalty for narrow applicability |

### Cross-Cutting Quality Signals

Additional questions that don't map to a single dimension but indicate overall quality.

| # | Question | Difficulty | Notes |
|---|----------|-----------|-------|
| Q17 | Is the available text complete enough to evaluate (not truncated, paywalled, or a stub)? | Baseline | Gatekeeper -- if "no," flag for rescore |
| Q18 | Does this article offer something a reader couldn't easily find in the first page of search results on this topic? | Hard | Tests for added value over commodity content |
| Q19 | Would a knowledgeable reader in this domain likely learn something new from this article? | Hard | Expert novelty test |
| Q20 | Is this content substantially a news article, product announcement, or event recap? | Easy (negative) | Penalty for ephemeral content |

---

## How Many Questions? Practical Considerations

### Arguments for More Questions (15-20)
- Better coverage of quality dimensions
- Each question carries less weight, reducing impact of any single misjudgment
- More granular score resolution (20 binary questions = 21 possible scores from 0 to 100)
- Difficult-tiered questions need enough items at each tier

### Arguments for Fewer Questions (8-12)
- Lower token cost per evaluation
- Less prompt complexity, fewer chances for LLM confusion
- Easier to validate and calibrate
- Diminishing returns: CheckEval found that checklist quality matters more than quantity

### Recommendation: 16-20 Questions

For our use case, 16-20 questions in a single prompt is the sweet spot:
- Provides 4-5 questions per dimension, enabling meaningful sub-scores
- Includes room for 3-4 negative/penalty questions
- 20 questions produce a natural 0-100 scale (each question worth 5 points before weighting)
- Single LLM call keeps latency manageable
- Fits comfortably within context window alongside article content

---

## Podcast-Specific Adaptations

Some questions need podcast-specific variants:

| Article Version | Podcast Version |
|----------------|-----------------|
| "Does the author write from personal professional experience?" | "Does a guest or host share specific professional experiences or lessons learned from practice?" |
| "Does the article contain a specific data point worth remembering?" | "Does the episode cite specific data, research findings, or quantified claims?" |
| "Is this primarily a summary of widely-known ideas?" | "Is this primarily a casual conversation without substantive depth?" |
| "Is the available text complete enough to evaluate?" | "Is the transcript quality sufficient to evaluate the episode's content?" |

---

## Example Question Prompt Structure

```
For each question below, answer ONLY "yes" or "no" based on the content provided.

QUOTABILITY
Q1: Does the article contain a memorable phrase, vivid metaphor, or striking sentence?
Q2: Does the article include a specific data point, statistic, or quantified claim worth remembering?
Q3: Could you extract at least one 2-3 sentence passage that works as a standalone saved note?
Q4: Does the article contain a direct quote from a practitioner or expert?

SURPRISE
Q5: Does the article contradict common industry assumptions or conventional wisdom?
Q6: Does the article reframe a familiar topic through an unexpected lens?
Q7: Does the article contain a case study or example the reader likely hasn't encountered?
Q8: [NEGATIVE] Is this primarily a summary or restatement of widely-known ideas?

ARGUMENT
Q9: Does the author argue for a clear, specific position rather than neutral reporting?
Q10: Does the author support claims with concrete evidence (data, case studies, examples)?
Q11: Does the author write from personal professional experience?
Q12: [NEGATIVE] Could this article's main argument fit in a single tweet?

INSIGHT
Q13: Does the article introduce a named framework, mental model, or methodology?
Q14: Could a reader apply a specific technique from this article in their own work?
Q15: Does the article provide enough detail for a reader to act on its ideas?
Q16: [NEGATIVE] Is the content so domain-specific it's useful only to a very narrow audience?

OVERALL QUALITY
Q17: Is the available text complete enough to properly evaluate?
Q18: Does this offer something beyond what's easily found via search?
Q19: Would a domain expert likely learn something new from this?
Q20: [NEGATIVE] Is this primarily news reporting, a product announcement, or event recap?

Respond with ONLY a JSON object mapping question IDs to boolean values and brief reasons:
{"q1": true, "q1_reason": "...", "q2": false, "q2_reason": "...", ...}
```
