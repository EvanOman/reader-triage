# Alternative Scoring Frameworks: Predicting Highlighting Behavior

Analysis based on the backtest of 205 read articles against the current 4-dimension scoring system.

## Diagnosis: Why the Current System Fails

The current prompt asks Claude to evaluate "information content quality" across four dimensions:

| Dimension | rho vs highlights | p-value | Verdict |
|-----------|:-:|:-:|---|
| Specificity | -0.02 | 0.82 | Dead zero. No signal. |
| Novelty | 0.01 | 0.90 | Dead zero. No signal. |
| Depth | 0.10 | 0.14 | Weak trend, not significant. |
| Actionability | 0.20 | 0.005 | Only significant predictor. |

**The core mismatch:** The scoring system evaluates articles as if it were a peer reviewer judging a paper ("is this rigorous? novel? thorough?"). But the user highlights like a practitioner building a personal knowledge base ("can I use this? does this change how I think? would I quote this to someone?").

Three specific failures:

1. **Specificity measures the wrong thing.** An article packed with data tables and specific numbers (high specificity) may contain nothing worth highlighting if none of those details are personally relevant. Conversely, a single well-phrased conceptual insight (low specificity) might be the most highlighted passage in a reader's library.

2. **Novelty is ungrounded.** The LLM evaluates novelty against general knowledge, but the reader evaluates novelty against *their own* knowledge. An article that is "common knowledge" to the LLM might contain a framing the reader has never encountered.

3. **Depth rewards length and thoroughness**, but highlighting behavior is about density of insight, not breadth of coverage. A 500-word blog post with 3 sharp observations often gets more highlights per word than a 5000-word comprehensive review.

**Why actionability works (partially):** Actionable content is closest to "will I want to capture this for later use?" -- the actual motivation behind highlighting. But actionability alone is too narrow; people also highlight memorable framings, surprising facts, and resonant observations that aren't "actionable" in a direct sense.

---

## Theoretical Grounding

### What Research Says About Highlighting Behavior

Highlighting is a **capture behavior** driven by the reader's assessment that a passage has future value to them specifically. Research on Kindle popular highlights shows that the most-highlighted passages tend to be "digestible adages that reveal simple truths" -- not the most complex or data-rich content, but the most **quotable and personally resonant** content.

Key findings from reading science and information behavior research:

- **Highlights reflect the reader's mental model**, not objective text quality. Passages are highlighted when they connect to existing knowledge or current goals (Winchell 2020, Cognitive Science).
- **Emotional resonance drives engagement.** Content that evokes emotional responses (surprise, recognition, validation) gets more engagement than emotionally neutral content, even if the neutral content is "higher quality" by objective measures.
- **Information foraging theory** (Pirolli & Card 1999) shows that people assess "information scent" -- their estimate of value a source will deliver. This is inherently subjective and goal-dependent.
- **The SUCCESs framework** (Heath & Heath, "Made to Stick") identifies what makes ideas memorable: **S**imple, **U**nexpected, **C**oncrete, **C**redible, **E**motional, **S**tories. Note the overlap with highlighting behavior and the near-total disconnect with the current scoring dimensions.
- **PKM (Personal Knowledge Management) research** shows people save/highlight based on: actionability, connection potential to existing knowledge, and anticipated future retrieval value.

### The Fundamental Reframing

The scoring question should shift from:

> "Is this article high-quality information?" (current)

to:

> "Will this article contain passages the reader wants to save for later?" (proposed)

This is the difference between evaluating the **article as an artifact** vs. predicting the **reader's response to the article**.

---

## Proposed Alternative Scoring Frameworks

### Alternative A: "Capture Likelihood" Framework

**Premise:** Score articles on dimensions that predict whether a reader will want to capture (highlight/save) passages. Directly models the highlighting decision.

**Dimensions (0-25 each, total 0-100):**

1. **Quotability** (0-25): Does the article contain well-phrased, standalone statements that a reader might want to save verbatim? Look for: crisp formulations, memorable analogies, pithy summaries of complex ideas, sentences that could stand alone as a tweet or note. High quotability = many passages that "feel highlightable."

2. **Applicable Insight** (0-25): Does the article provide ideas, techniques, or frameworks the reader could apply to their own work or thinking? This goes beyond "actionability" to include mental models, reusable frameworks, and transferable principles -- not just step-by-step instructions.

3. **Surprise Density** (0-25): How many moments of "I didn't know that" or "I never thought of it that way" does the article contain per unit of text? Measures not whether the topic is novel in general, but whether the *specific claims and framings* are likely to challenge or expand a knowledgeable reader's existing understanding.

4. **Argument Strength** (0-25): Does the article make claims that are well-supported and convincing? Does it present evidence or reasoning that would make a reader confident enough to adopt and share the ideas? Weak arguments don't get highlighted even if the topic is interesting.

**Why this should work better:**
- **Quotability** directly targets the mechanical act of highlighting: "would I drag my finger over this sentence?" It replaces Specificity, which measured data density (irrelevant to highlighting).
- **Applicable Insight** is a broader version of Actionability (the one dimension that worked), extending it to include conceptual tools and frameworks.
- **Surprise Density** replaces Novelty with a reader-centric version. Instead of "is this novel in general?", it asks "would a smart reader be surprised by this?" -- and it measures density, not just presence.
- **Argument Strength** replaces Depth. A shallow but well-argued article is more highlightable than a thorough but poorly argued one.

**Expected improvement:** This framework should improve correlation with highlights because every dimension is defined in terms of reader response rather than text properties. The weakest dimension will likely be Argument Strength (the least directly tied to highlighting), but it serves as a quality floor that prevents highly quotable but substanceless content from scoring high.

---

### Alternative B: "Reader Response" Framework

**Premise:** Instead of evaluating text properties, ask Claude to simulate the reader's experience and predict specific responses. This is the most radical departure from the current approach.

**Dimensions (0-25 each, total 0-100):**

1. **Personal Relevance Potential** (0-25): How likely is this article to connect to a knowledge worker's active concerns? Consider: does it address recurring professional challenges, current technology trends, decision-making under uncertainty, productivity, writing, thinking, or learning? Score higher when the topic intersects with things a curious generalist would be actively thinking about.

2. **Insight Density** (0-25): Count the distinct, non-obvious insights in the article and consider their concentration. An article with 1 insight in 5000 words scores low. An article with 5 insights in 1000 words scores high. An "insight" is a claim, observation, or reframing that most readers would not have arrived at independently.

3. **Emotional Resonance** (0-25): Does the article evoke a feeling of recognition ("yes, exactly!"), surprise ("wait, really?"), or motivation ("I need to try this")? Articles that are purely informational with flat affect score low. Articles that create moments of intellectual excitement or validation score high. This is about *reading experience*, not sentiment.

4. **Retention Value** (0-25): If this article disappeared from the internet tomorrow, would the reader wish they had saved excerpts? Would they want to reference it in a conversation, share a passage with a colleague, or return to it in 6 months? This measures anticipated future utility. Score low if the content is time-sensitive (news), redundant (available elsewhere), or forgettable. Score high if it contains unique formulations or frameworks with long shelf life.

**Why this should work better:**
- Every dimension is defined as a **predicted reader response**, not a text property. This forces the LLM to model the reader, not the text.
- **Personal Relevance Potential** addresses the biggest gap in the current system: the complete absence of reader-topic matching. The scoring prompt would need to include a brief reader profile (e.g., "The reader is a software engineer interested in AI, productivity, writing, and decision-making.") to ground this dimension.
- **Insight Density** replaces both Specificity and Novelty with a single, engagement-aligned metric. It measures what actually triggers highlighting: "I just learned something I didn't know."
- **Emotional Resonance** has no equivalent in the current system. Research on content engagement consistently shows emotion as a top predictor, yet the current scoring is entirely cognitive.
- **Retention Value** directly asks "would the reader want to highlight this?" without asking it literally, by focusing on the downstream behavior that motivates highlighting.

**Trade-off:** This framework requires a reader profile in the prompt. Without it, "personal relevance" is undefined. The reader profile need not be long -- 2-3 sentences about professional role, key interests, and what they typically read.

---

### Alternative C: "Highlight Predictor" Framework (Minimal Change)

**Premise:** Keep the 4-dimension structure but replace the failed dimensions with ones aligned to highlighting behavior. This is the lowest-risk change that addresses the backtest findings directly.

**Dimensions (0-25 each, total 0-100):**

1. **Actionability** (0-25): *[KEEP -- the only dimension that works]* Does it provide practical takeaways, techniques, frameworks, or mental models you can apply? Score higher for concrete advice than abstract observations.

2. **Quotability** (0-25): *[REPLACE Specificity]* Does the article contain passages that are well-phrased enough to be worth saving verbatim? Look for memorable formulations, clean analogies, and ideas expressed with unusual clarity. An article can be rich in data but low in quotability, or sparse in data but full of quotable insights.

3. **Surprise Factor** (0-25): *[REPLACE Novelty]* Does this article present ideas, data, or framings that would make a well-read person say "I didn't know that" or "I never thought of it that way"? This is not about whether the topic is niche, but whether the specific claims challenge assumptions or present familiar topics in genuinely new light.

4. **Argument Quality** (0-25): *[REPLACE Depth]* Are the claims in this article well-supported? Does it present convincing evidence or reasoning? An article with strong arguments gives the reader confidence to adopt its ideas and share them -- which drives both highlighting and engagement.

**Why this should work better:**
- Preserves the one working dimension (Actionability).
- Replaces Specificity (rho = -0.02) with Quotability, which directly measures what triggers the highlighting reflex.
- Replaces Novelty (rho = 0.01) with Surprise Factor, which is reader-centric rather than knowledge-centric.
- Replaces Depth (rho = 0.10) with Argument Quality, which better predicts whether someone will trust content enough to highlight and share it.
- Minimal structural change: same 4x25 format, same JSON output, same database schema.

**Expected improvement:** Even modest improvement should be measurable. If Quotability achieves rho >= 0.15 with highlights (better than Specificity's -0.02), and Surprise Factor achieves rho >= 0.10 (better than Novelty's 0.01), the composite score should move from rho = 0.11 to approximately 0.20-0.25 -- a meaningful improvement from ~1% to ~5% variance explained.

---

## Comparison Matrix

| Property | Current | Alt A (Capture) | Alt B (Reader Response) | Alt C (Minimal) |
|---|---|---|---|---|
| Framing | "Is this quality info?" | "Will reader capture passages?" | "How will reader respond?" | "Is this worth saving?" |
| Reader model | None | Implicit | Explicit (needs profile) | Implicit |
| Structural change | Baseline | New dimensions | New dimensions + profile | Swap 3 of 4 dims |
| Risk | Known bad | Medium | Higher (profile dependency) | Lowest |
| Expected rho improvement | 0.11 baseline | 0.20-0.30 | 0.25-0.35 | 0.18-0.25 |
| Implementation effort | N/A | Prompt rewrite | Prompt rewrite + profile | Prompt edit |

---

## Recommendation

**Start with Alternative C** (Minimal Change). It carries the lowest risk, requires no schema changes, and directly addresses each failed dimension with a highlighting-aligned replacement. Run the same backtest protocol on a subset of articles to measure improvement.

If Alternative C shows meaningful improvement (rho > 0.18 for the composite score), iterate from there. If it does not, escalate to **Alternative B** (Reader Response), which is more theoretically grounded but requires the addition of a reader profile to the scoring prompt.

Alternative A is a reasonable middle ground if Alternative C underperforms but the overhead of building a reader profile for Alternative B feels premature.

**Regardless of which framework is chosen, the single most impactful change is reframing the scoring prompt from evaluating text quality to predicting reader behavior.** The current prompt asks "Analyze this article for information content quality." The replacement should say something like "Predict how likely a reader is to highlight passages in this article" or "Evaluate this article's value as a source of passages worth saving."

---

## Implementation Notes

For any alternative, the scoring prompt should:

1. **Frame the task as prediction, not evaluation.** "Predict whether a reader will highlight passages" vs. "Analyze information content quality."
2. **Define dimensions in terms of reader behavior.** "How many 'I didn't know that' moments?" vs. "Does it provide unique insights?"
3. **Provide anchoring examples.** Include 2-3 brief examples of high-scoring and low-scoring articles per dimension to calibrate the LLM's scoring.
4. **Consider adding a reader profile** (even 1-2 sentences) to ground personal relevance judgments. E.g., "The reader is a software engineer who reads broadly about technology, AI, productivity, writing, and decision-making."

A/B testing approach: Score a batch of 50 already-archived articles with both the current and new prompt, then compare correlations with actual highlight counts. This gives a controlled comparison without waiting for new articles to accumulate.

---

## Sources

- [Winchell 2020 - Highlights as an Early Predictor of Student Comprehension and Interests](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12901)
- [Amazon Kindle Most Highlighted Passages Analysis](https://blogs.illinois.edu/view/25/29224)
- [Information Foraging Theory - Pirolli & Card](https://www.interaction-design.org/literature/book/the-glossary-of-human-computer-interaction/information-foraging-theory)
- [How Emotion Affects Information Communication](https://arxiv.org/html/2502.16038v1)
- [Resonance and the Experience of Relevance - Ruthven 2021](https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.24424)
- [LLM-Rubric: Multidimensional Calibrated Evaluation](https://arxiv.org/abs/2501.00274)
- [Information Scent - Nielsen Norman Group](https://www.nngroup.com/articles/information-scent/)
