# Contrarian Views and Limitations of LLM-Based Scoring

This report challenges the assumption that LLM-based scoring can be meaningfully improved by presenting twelve contrarian perspectives. Each argument is presented fairly with supporting evidence, followed by an acknowledgment of validity or a rebuttal. The goal is to ensure intellectual honesty about our system's limitations and avoid overconfidence in any single approach.

---

## 1. "Scoring Is Fundamentally Broken"

### The Argument

Critics argue that LLMs cannot reliably evaluate content quality at all. The evidence is mounting:

**Positional and labeling biases are pervasive.** When LLMs are tasked with choosing between "Response A" and "Response B" across many trials, they select "Response B" approximately 61% of the time — far above the expected 50% baseline. Minor changes in prompt phrasing or label formatting can swing preferences by 5-10 percentage points ([Collective Intelligence Project](https://www.cip.org/blog/llm-judges-are-unreliable)).

**Self-preference bias is measurable.** LLMs assign significantly higher evaluations to outputs with lower perplexity, and since their own outputs tend to have lower perplexity, this creates a systematic self-enhancement effect. GPT-4 inflated its own win rate by 10%, while Claude-v1 showed 25% self-preference elevation ([Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/html/2410.21819v2)).

**Verbosity bias distorts scoring.** Models preferred longer responses over shorter, higher-quality alternatives over 90% of the time, even when the extra length added no informational value ([Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/)).

**Fine-grained scoring is unreliable.** As the scoring scale becomes more detailed with finer intervals, LLMs are more likely to produce arbitrary scores, making their judgments less reliable and more prone to randomness. The LLM tends to be biased toward a single number within a grading scale (e.g., a score of 3 is the most common output within a 1-4 Likert scale) ([Evidently AI](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)).

**Expert domains are the worst case.** Research on expert knowledge tasks found that LLMs missed critical details identified by subject matter experts, including gaps in information and failures to assess broader implications of content. Among 2,860 item-level evaluations, agreement between the LLM and human consensus varied from 72.3% to 94.7%, with some checklist items achieving near-random kappa values of -0.07 ([IUI 2025 Proceedings](https://dl.acm.org/doi/10.1145/3708359.3712091)).

### Validity and Rebuttal

This argument has genuine weight. The biases are real and well-documented. However, the research also shows that LLMs achieve 85% agreement with human experts on MT-Bench, exceeding human-human agreement at 81%. The question is not whether LLM scoring is perfect but whether it is useful — and for triaging a reading inbox, even a noisy signal with 70% accuracy is substantially better than no signal at all. Our system does not need courtroom-grade reliability; it needs to rank-order a few dozen articles well enough to surface the best 5-10.

The specific biases identified (position, verbosity, self-preference) are largely irrelevant to our use case because we are scoring articles independently rather than comparing LLM-generated outputs against each other. We are asking "is this article quotable?" not "which of these two LLM responses is better?"

---

## 2. "Binary Won't Help"

### The Argument

The proposed v3-binary scoring system assumes that binary (yes/no) questions will produce better discrimination than the current categorical system. But there are several failure modes where binary questions could actually perform worse:

**Sycophancy still applies.** Binary questions are not immune to LLM sycophancy. If the model defaults to "yes" on ambiguous cases — which research suggests it will, since LLMs have a well-documented affirmative bias — then 20 binary questions all answered "yes" produce the same ceiling effect as the current system, just with more steps.

**Binary loses nuance.** The current system's "a_few / several / many" scale for standalone passages captures a real gradient. A binary "does it contain a memorable phrase?" collapses the difference between an article with one decent sentence and one packed with brilliant writing. Lower-precision scoring (binary, 0-1) can "largely retain precision compared to higher precision scales" — but this cuts both ways. If the information was already captured at the binary level, we were already paying for fine-grained scoring we did not need. If it was not, we are now losing it.

**20 binary questions may still cluster.** Research on pointwise scoring shows that fine-grained relevance scales with more than two labels "significantly outperform binary pointwise scoring" for discrimination tasks ([Likert or Not, arXiv 2505.19334](https://arxiv.org/html/2505.19334v1)). Our move from 8 multi-option questions to 20 binary ones is not obviously an improvement in information content per question.

**Penalty questions assume the model says "yes" when it should say "no."** The v3-binary design relies on negative-polarity questions (Q8, Q12, Q16, Q20) to pull scores down. But if the LLM is systematically generous on positive questions (answering "yes" too often), it is likely systematically generous on negative questions too (answering "no" too often — i.e., failing to apply penalties). The biases cancel each other, leaving the system no better than before.

### Validity and Rebuttal

These are legitimate concerns, particularly the point about sycophancy being format-agnostic. However, the v3-binary system mitigates this in three ways the current system does not: explicit anti-sycophancy prompting ("be critical and honest — most articles should NOT pass the harder questions"), required per-question reasoning (chain-of-thought), and penalty questions that actively subtract points rather than simply awarding zero. The key empirical question is whether pass rates actually distribute well. If average pass rate across articles is 40-60%, the system is working; if it is 80%+, the sycophancy concern is validated. This is measurable during the shadow-scoring phase.

---

## 3. "The Ceiling Effect Is Correct"

### The Argument

Maybe the current system's score clustering at 80-100 is not a bug — it is an accurate reflection of reality. Consider the pipeline:

1. A human reader curates their Readwise feed by subscribing to specific authors, newsletters, and RSS feeds
2. Readwise's own algorithms further filter and surface content
3. Articles that appear in the inbox have already passed two layers of quality selection

By the time an article reaches our scorer, it has survived significant pre-filtering. Research on algorithmic curation shows that iterative information filtering introduces selection bias, narrowing the space of items available and creating a more homogeneous pool ([Nature Scientific Reports](https://www.nature.com/articles/s41598-018-34203-2)). In a curated feed, the base rate of "good articles" may genuinely be 80%+.

If most articles in the feed truly are high quality, then a system that scores most of them highly is not suffering from poor discrimination — it is accurately reflecting the input distribution. Forcing a wider score spread through harder questions or penalty signals would introduce artificial variance into what is genuinely a narrow-quality-range population.

This is the classic base rate problem: if 85% of articles in the feed deserve a score above 70, then any system with good accuracy will produce a right-skewed distribution.

### Validity and Rebuttal

This is one of the strongest contrarian arguments. The pre-filtering argument is sound: a curated RSS feed is not a random sample of the internet. However, even within a curated feed, there is meaningful variance in "how much I want to read this right now." The user's own behavior confirms this — they highlight passages in some articles and not others, they archive some unread, they spend different amounts of time on different pieces. If the scoring system cannot predict this within-feed variance, it is not useful regardless of whether the absolute quality levels are high.

The real test is correlation with engagement, not score distribution shape. A system where every article scores 80-100 but the ordering within that range correctly predicts highlight behavior is still useful. The v2-categorical system's problem is not just the ceiling effect — it is that the within-ceiling ordering does not predict engagement well enough. The proposed v3-binary system is an attempt to stretch the scale where it matters, not to artificially penalize good content.

---

## 4. "Highlights Are a Bad Proxy"

### The Argument

Our entire calibration framework uses Readwise highlight counts as the ground truth for article quality. But highlighting behavior is a poor proxy for the value a reader gets from an article, for several reasons:

**Article length bias.** Research on text highlighting demonstrates that both text length and reading difficulty significantly affect overall frequency of highlighting, with effects being larger for text length. Word highlighting frequency decreases for longer texts ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950329321003748)). A 10,000-word essay has more opportunities for highlights than a 1,000-word piece, inflating its apparent quality. Conversely, long articles may see decreased per-passage highlighting as readers fatigue.

**Topic bias.** Some subjects lend themselves to highlighting more than others. Technical tutorials with concrete code snippets get highlighted; philosophical essays that shift your worldview may not contain any single passage worth excerpting. An article that changes how you think about a problem may produce zero highlights while a mediocre listicle with quotable statistics gets five.

**Mood and context effects.** Highlighting is a behavioral measure that depends on when, where, and how the reader engages. Reading on a phone on the subway produces fewer highlights than reading at a desk with a notebook. Morning focus sessions produce different engagement than evening browsing.

**Highlighting habit evolution.** The reader's highlighting behavior may change over time as they become more or less deliberate about their note-taking system. A period of heavy highlighting does not mean the articles were better — it may mean the reader was actively building a Zettelkasten.

**Zero-highlight articles can be valuable.** Some of the most valuable reading experiences — articles that change a mental model, expose a blind spot, or prompt a career decision — may not contain any specific passage worth highlighting. The value was in the gestalt, not the parts.

### Validity and Rebuttal

This is a serious concern. Highlights are a proxy, and proxies always have limitations. The article-length bias in particular is likely present in our data. However, highlights are the best behavioral signal we have that is both specific (the reader marked this particular passage) and available (Readwise exposes the data via API). Alternative signals — reading time, scroll depth, archival behavior — are either unavailable or even noisier.

The mitigation is not to abandon highlights as ground truth but to treat them probabilistically. When calibrating, we should control for article length (highlights per 1,000 words), filter to articles with reading_progress > 0.5, and supplement with other signals where available (e.g., whether the article was saved to a permanent location). A composite engagement score that normalizes for confounds is more defensible than raw highlight count.

---

## 5. "More Questions = More Noise"

### The Argument

The v3-binary proposal increases the question count from 8 to 20. The intuition is that more questions capture more signal. But there is a well-documented risk that adding criteria degrades rather than improves scoring — this is rubric overfitting.

**The overfitting trade-off is documented.** Research on LLM rubric calibration explicitly warns: "Using additional or multiple variants of evaluation criteria could potentially provide more useful evidence to the calibration network, but at the cost of slowing down evaluation and at the risk of overfitting" ([LLM-Rubric, arXiv 2501.00274](https://arxiv.org/html/2501.00274v1)). The recommended starting point is 2-4 criteria with a simple numeric scale.

**Context effects compound with more questions.** Research on LLM-based scoring found that the same item received a 5.0/5 clarity rating in isolation but dropped to 4.0/5 when scored alongside other criteria — a full-point decline from identical content ([CIP Blog](https://www.cip.org/blog/llm-judges-are-unreliable)). With 20 questions, the context effects between questions are multiplied. Early questions frame how the model interprets later questions, creating ordering dependencies.

**Correlation between questions wastes capacity.** If Q5 ("contradicts assumptions"), Q6 ("unexpected lens"), and Q7 ("novel example") are highly correlated — which they likely are, since they all measure surprise — then three questions are doing the work of one, but each adds noise. The marginal information from Q6 given Q5 may be near zero, while the marginal noise is not.

**The LLM has finite attention.** Asking 20 questions about a single article in one prompt taxes the model's ability to maintain consistent reasoning. By question 15, the model may be pattern-completing rather than genuinely evaluating, especially for longer articles that already consume most of the context window.

### Validity and Rebuttal

The overfitting concern is real, and the research recommendation of 2-4 criteria is notable. However, the v3-binary design mitigates this in several ways: questions are grouped by dimension (helping the model reason locally rather than globally), each question has a distinct and non-overlapping criterion, and the weight system means low-discrimination questions contribute minimally to the total score.

The strongest rebuttal is empirical: the shadow-scoring phase should include a correlation analysis between questions within each dimension. If Q5-Q7 have Pearson r > 0.85, one should be dropped. The design explicitly builds in this calibration step (Phase 3: Calibration), with the expectation that some questions will be revised or removed based on discrimination analysis.

---

## 6. "Pairwise > Pointwise"

### The Argument

Instead of scoring each article independently (pointwise), a fundamentally different approach compares articles head-to-head (pairwise). The argument is that humans naturally think in relative terms ("this article is better than that one") rather than absolute scales, and LLMs should too.

**Pairwise comparisons are more natural.** Research on pairwise comparison tasks found that responses were 2/3 quicker than in Likert tasks, indicating that participants — and by extension, LLMs — find pairwise comparisons to be easier and more natural ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190393)). Absolute scores require an internal calibration standard that may drift; relative comparisons do not.

**Elo systems scale efficiently.** The Elo rating system, as demonstrated by Chatbot Arena, can handle large numbers of items efficiently and does not require extensive data for every possible pair. New items can be evaluated with relatively few trials, allowing quick integration into the ranking system ([LMSYS Org](https://lmsys.org/blog/2023-05-03-arena/)).

**Pointwise scoring is less stable.** Pointwise scoring expects the judge to possess a relatively consistent internal scoring mechanism — absolute scores are much more likely to fluctuate compared to pairwise comparisons ([Cameron Wolfe](https://cameronrwolfe.substack.com/p/llm-as-a-judge)).

**However, pairwise has its own problems.** Recent research shows pairwise preferences flip in about 35% of cases, compared to only 9% for absolute scores. Pairwise protocols are more vulnerable to "distracted evaluation" where the model fixates on superficial attributes. Biases are actually exacerbated in pairwise evaluation because the model directly compares two outputs and can easily prioritize surface features ([Pairwise or Pointwise, arXiv 2504.14716](https://arxiv.org/abs/2504.14716)).

### Validity and Rebuttal

The theoretical argument for pairwise is compelling, but the practical challenges are significant for our use case. An Elo-style system requires O(n^2) comparisons in the worst case, though the Elo convergence properties mean far fewer are needed in practice. For a daily inbox of 20-50 new articles, this means 10-25 pairwise comparisons per article versus 1 pointwise call — a 10-25x increase in API cost and latency.

More importantly, the 35% flip rate in pairwise evaluations is alarming. If a third of head-to-head comparisons are essentially random, the resulting Elo rankings inherit and obscure this noise. Pointwise scoring with known biases (sycophancy, ceiling effects) may be more predictable and debuggable than pairwise scoring with hidden transitivity violations.

A practical compromise: use pointwise scoring for initial triage, then apply pairwise comparison only among borderline articles (e.g., scores 40-60) where the absolute score has the least resolution.

---

## 7. "Human Calibration Is Circular"

### The Argument

Our calibration methodology trains the scoring system to predict human behavior (highlights), then evaluates the system against human behavior (highlights). This is circular in a subtle but important way.

**The optimization target and evaluation metric are the same.** If we tune question weights to maximize correlation with highlight counts, and then report "our Spearman correlation with highlight counts improved from 0.3 to 0.5," we have not demonstrated that the system is better — we have demonstrated that we can overfit to a noisy signal.

**Human-AI feedback loops amplify bias.** Research demonstrates that human-AI interactions create feedback loops that amplify biases. When a model is trained on accurate labels with minor bias, the average bias remained at 3%, but when trained on random labels with the same 3% bias, amplification reached 50% ([Nature Human Behaviour](https://www.nature.com/articles/s41562-024-02077-2)). If the scoring system influences which articles the user reads (by surfacing high-scoring ones), it then influences which articles get highlighted, which then influences the calibration data — a self-reinforcing cycle.

**AI collapse is a documented phenomenon.** Progressive degradation of model performance over multiple generations, where the model eventually generates only reduced sets of similar outputs, has been observed when models train on their own outputs. Our calibration loop is a slower version of this: human behavior shapes model weights, model weights shape article surfacing, article surfacing shapes human behavior.

**The user's preferences evolve.** Calibrating against historical highlight data assumes preference stability. But reading interests change — new jobs, new hobbies, evolving intellectual interests. A system calibrated against 6 months of data may be optimizing for a reader who no longer exists.

### Validity and Rebuttal

The circularity concern is philosophically sound but practically manageable. The key mitigation is temporal separation: calibrate on historical data (months 1-6), evaluate on future data (months 7-12), never optimize and evaluate on the same dataset. This is standard practice in machine learning (train/test splits) and breaks the circularity concern if applied rigorously.

The feedback loop concern is more serious and harder to address. If the scoring system causes the user to read only high-scoring articles, the calibration data becomes biased toward the system's own preferences. The practical mitigation is to ensure the user still reads some low-scoring articles (via random exploration or diverse surfacing), providing counterfactual data for calibration. Alternatively, calibrate only on articles the user discovered independently (via manual browsing) rather than articles surfaced by the system.

---

## 8. "Just Use Embeddings"

### The Argument

Instead of asking an LLM to judge article quality (expensive, slow, unreliable), a simpler approach might work better: compute embeddings and use vector similarity.

**Profile-based scoring.** Build a vector profile from articles the user has highlighted heavily. For each new article, compute cosine similarity between its embedding and the "ideal article" profile. High similarity suggests the new article matches the user's demonstrated preferences. This requires one embedding call per article (fractions of a cent) instead of one LLM judge call (cents).

**Clustering for categorization.** Cluster historically-engaged articles to identify preference "archetypes" — perhaps the user likes practitioner war stories, quantitative social science, and programming language design. New articles close to these clusters get higher scores. This captures the multi-modal nature of preference without explicit rubric design.

**BERTScore and semantic similarity.** Embedding-based metrics like BERTScore leverage contextual embeddings and cosine similarity to assess content quality. While LLM-as-a-judge performs best overall, embedding-based metrics "can serve as cost-effective alternatives" that provide useful signal at a fraction of the cost ([HuggingFace Blog](https://huggingface.co/blog/g-ronimo/semscore)).

**The cost argument is stark.** An embedding costs approximately $0.0001 per article. An LLM scoring call costs approximately $0.01-0.03 per article. For 50 articles per day, that is $0.005/day versus $0.50-1.50/day — a 100-300x cost difference. Over a year, embeddings cost about $2 while LLM scoring costs $180-550.

### Validity and Rebuttal

Embedding-based scoring has genuine appeal for its simplicity, speed, and cost. However, it solves a fundamentally different problem. Cosine similarity answers "is this article topically similar to articles you've liked?" but cannot answer "is the argument in this article well-supported?" or "does this reframe a familiar topic?" Two articles about the same subject can have very similar embeddings but vastly different quality.

The embedding approach fails on exactly the articles our system needs to differentiate: two pieces about the same trending topic where one is a rewritten press release and the other is deep original analysis. Their embeddings will be nearly identical, but their scores should be very different.

The strongest version of this argument is as a complement, not a replacement: use embeddings for topical relevance (does this match my interests?) and LLM scoring for quality assessment (is this a good exemplar of the topic?). A combined system could use the embedding as a cheap first-pass filter, only sending "topically relevant" articles through the expensive LLM scoring pipeline.

---

## 9. "The Prompt Is the Bottleneck"

### The Argument

Maybe the issue is not the scoring format (binary vs. categorical vs. Likert) but the prompt itself. Prompt engineering represents "a critical bottleneck to harness the full potential of Large Language Models for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention" ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

**Small prompt changes produce large score changes.** Research on LLM-as-judge evaluations found that system prompt instructions intended to reduce bias often backfire — one de-biasing prompt increased second-option preference by over 5 percentage points. Classification models changed their outputs nearly every time the prompt template or category ordering shifted ([CIP Blog](https://www.cip.org/blog/llm-judges-are-unreliable)).

**Prompt optimization is under-explored.** Rather than redesigning the entire scoring architecture (v2 to v3), we might achieve equivalent or better improvements by systematically optimizing the existing v2-categorical prompt. Techniques like few-shot examples (showing the model what a score of 40 vs. 80 looks like), chain-of-thought preambles, or explicit calibration instructions ("most articles should score between 30-60") could address the ceiling effect without changing the question format.

**The framing matters more than the questions.** Our current prompt says "evaluate this article for capture value — how likely a reader is to want to save and highlight passages." This framing may prime the model toward generosity (most articles have at least some value) rather than discrimination (most articles are not worth your time). Reframing as "identify reasons this article might NOT be worth reading" or "what percentage of a curated feed would this article rank in?" could produce better spread without any structural changes.

**DEEVO and automated prompt optimization.** Recent research has proposed frameworks like DEEVO (DEbate-driven EVOlutionary prompt optimization) that systematically evolve prompts through structured debates and Elo-based selection ([arXiv 2506.00178](https://arxiv.org/html/2506.00178)). Rather than manually iterating on prompts, automated optimization might find better-performing prompts that no human would write.

### Validity and Rebuttal

This is a strong practical argument. The v2-to-v3 transition involves significant engineering effort (new strategy class, shadow scoring, calibration, cutover). If the same improvement could be achieved by optimizing the v2 prompt — adding few-shot examples, adjusting framing, or using automated prompt search — that would be faster and lower-risk.

However, the prompt and the format are not independent. Binary questions constrain the model's output space in a way that makes the prompt more robust to variation. A categorical question with 5 options is more sensitive to prompt wording because the model must decide between fuzzy categories. A binary question with a clear criterion has less room for prompt-induced drift.

The strongest version of this argument suggests a staged approach: first, optimize the v2-categorical prompt using few-shot calibration examples and explicit anti-sycophancy framing. Measure the improvement. Only proceed to v3-binary if prompt optimization alone is insufficient. This is a valid experimental design that we should consider.

---

## 10. "Cost/Benefit Doesn't Justify It"

### The Argument

The entire LLM scoring pipeline — prompt engineering, rubric design, calibration, A/B testing, ongoing maintenance — may be solving a problem that simple heuristics handle adequately.

**Simple features predict surprisingly well.** Research on data quality filtering found that simple heuristics like word count, language ratio, and duplicates "can clean up about 20% of low-quality data for free" ([BRICS](https://brics-econ.org/measuring-data-quality-for-llm-training-model-based-and-heuristic-filters)). For article triage specifically, a handful of features may capture most of the variance:

- **Word count**: Articles under 500 words are rarely worth deep reading; articles over 5,000 words required significant author effort
- **Author reputation**: Previous highlight count for this author (our existing author_boost signal)
- **Source reputation**: Newsletter vs. random blog vs. major publication
- **Recency**: Older articles that appear in the feed may be evergreen (good) or stale (bad)
- **Category**: "Article" vs. "tweet" vs. "video" already provides signal

**The author_boost signal may dominate.** Our existing system already includes an author boost of up to +15 priority points. If a user's highlight behavior is primarily driven by author preference (they always highlight Matt Levine, rarely highlight random Medium posts), the author signal may carry more predictive weight than the entire LLM scoring rubric combined.

**Cascaded approaches are the standard.** Research recommends "a mix of fast heuristic filters and smarter model-based systems," with cascaded approaches that remove low-quality items before applying expensive evaluation. We might be applying the expensive evaluation to items that could be triaged with cheap heuristics.

**Maintenance cost is non-trivial.** LLM scoring requires ongoing prompt maintenance, model version tracking, calibration runs, and monitoring. A heuristic system based on word count + author + source is maintenance-free and deterministic.

### Validity and Rebuttal

The cost/benefit argument is pragmatically the strongest in this document. It is entirely possible that a simple model — `priority = 0.4 * author_score + 0.3 * word_count_bucket + 0.2 * source_score + 0.1 * recency` — would achieve 80% of the LLM scoring system's discrimination at 0% of the API cost.

However, this argument underestimates the "last mile" of quality assessment. Heuristics can filter out obviously low-value content (short articles, unknown authors, news roundups) but cannot distinguish between two 3,000-word pieces by established authors where one is a rehash and the other is genuinely novel analysis. That distinction is exactly what the LLM scoring is designed to capture.

The right response is to measure: score articles with both the heuristic model and the LLM model, then compare Spearman correlation with highlight counts. If the heuristic model achieves r=0.25 and the LLM model achieves r=0.30, the marginal improvement may not justify the cost. If the gap is larger, LLM scoring earns its keep. Our existing calibration toolkit makes this comparison straightforward to run.

---

## 11. "Temporal Instability"

### The Argument

LLM scoring changes with model updates, making it an unreliable foundation for any system that needs consistent behavior over time.

**Models are not deterministic.** Even with temperature=0 and fixed seeds, LLMs rarely produce identical outputs. Research found that GPT-4o achieved only 3% raw output consistency across 5 identical runs, while other models ranged from 30-97%. Parsed answer consistency was better (88-99%) but still imperfect, with accuracy spreads reaching 10% between minimum and maximum runs for some models ([LLM Stability, arXiv 2408.04667](https://arxiv.org/html/2408.04667v1)).

**Model versions drift significantly.** Research comparing GPT-4 from March 2023 and June 2023 found dramatic behavioral changes: Chain of Thought accuracy gains went from 56.6% in March to 3.2% in June. The model's ability to follow instructions decreased over time, partially explaining corresponding performance drops ([Chen et al., Harvard Data Science Review](https://hdsr.mitpress.mit.edu/pub/y95zitmz)).

**Scoring evaluator ICC values decay.** Research on GPT-4 as an evaluator found that intraclass correlation coefficient values were lower when calculated with feedback generated after a timespan of several weeks, suggesting evaluation susceptibility to temporal drift ([Frontiers in Education](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2023.1272229/full)).

**Provider-initiated changes are invisible.** Model providers routinely update API-accessible models without explicit version changes. An article scored 75 in January might score 60 in March under the same model name, not because the article changed but because the model did. Our scoring_version field tracks our prompt version but not the underlying model behavior.

**Recalibration is perpetual.** If model behavior drifts, calibration weights derived from one model version may be wrong for the next. This creates a treadmill: calibrate, model changes, re-calibrate, model changes again. The system never reaches a stable state.

### Validity and Rebuttal

This is a genuine engineering concern. The temporal instability research is alarming, and the practical experience of GPT-4 behavior changing dramatically between March and June 2023 is well-documented.

However, several mitigations are available. First, we use Claude (Anthropic) rather than GPT-4, and pin to a specific model version (claude-sonnet-4-20250514) rather than a moving "latest" endpoint. This gives us control over when model changes occur. Second, our scoring_version field enables systematic re-scoring when we upgrade models. Third, the calibration toolkit detects drift: if Spearman correlation with highlights suddenly drops, we know the model changed and can recalibrate.

The deeper point is that temporal instability is a cost of using LLMs, not a reason to avoid them. All ML systems require monitoring and maintenance. The question is whether the value delivered justifies the maintenance burden — which brings us back to argument #10.

---

## 12. "Personal Taste Is Not a Function"

### The Argument

The philosophical argument: aesthetic and intellectual taste cannot be reduced to weighted binary questions, and any attempt to do so produces a simulacrum of judgment rather than judgment itself.

**Kant's paradox.** Aesthetic judgment is both subjective and claims universality. As Kant argued, it is "not possible to teach someone how to make a beautiful work of art even if that person masters all the techniques of a given art" ([Internet Encyclopedia of Philosophy](https://iep.utm.edu/aesthetic-taste/)). If a human cannot articulate the rules of taste, how can an LLM be programmed to follow them? The judgment of taste "is not a judgment of cognition" — its determining ground is subjective, not logical.

**Taste is contextual and holistic.** What makes an article valuable to a specific reader depends on what they read yesterday, what problem they are currently solving, what conversation they had that morning. A rubric that asks "does this contain a specific framework?" cannot capture the fact that the reader already knows 50 frameworks and is specifically looking for narrative, not structure.

**The irreducibility of aesthetic response.** Research on modeling consumer aesthetic perceptions found that "capturing taste during the design process remains challenging because taste is abstract and subjective, and preference data alone provides limited guidance" ([arXiv 2601.17134](https://arxiv.org/abs/2601.17134)). Whenever a situation calls for subjective assessment "that is difficult to quantify (taste), machines will have no basis for making that assessment."

**Decomposition destroys meaning.** Breaking "is this a great article?" into 20 sub-questions may destroy the holistic gestalt that makes the judgment meaningful. An article can pass every individual criterion (has a framework, has evidence, has novel framing, has quotable passages) and still feel lifeless. Conversely, an article can fail most criteria and still produce the electric sense of "I need to think about this differently." The whole is not the sum of the parts.

**Overfitting to articulable preferences.** A scoring rubric captures only the articulable dimensions of preference — the things you can point to and name. But much of taste is inarticulate: "I don't know why I love this, but I do." A system that scores only the articulable dimensions systematically undervalues articles whose appeal lies in the inarticulate dimensions.

### Validity and Rebuttal

This is the most philosophically interesting argument, and it is fundamentally correct: personal taste is not a function that can be fully specified by a finite set of weighted binary questions. The philosophical literature on aesthetic judgment is clear that taste involves irreducible subjectivity, contextual dependence, and holistic evaluation that resists decomposition.

However, the goal of our system is not to replicate human taste — it is to approximate it well enough to be useful for inbox triage. The question is not "can this system perfectly predict what I want to read?" but "can this system save me 15 minutes per day by filtering out the 30% of articles I definitely would not highlight?" A lossy approximation of taste is still valuable if it correctly identifies the extremes (definitely skip, definitely read) even if it struggles with the middle.

The strongest version of this argument is that the irreducible, inarticulate dimension of taste is exactly where the most valuable articles live — the ones that change your mind in ways you cannot predict. A scoring system optimized for articulable preferences will systematically surface predictably good articles while missing genuinely surprising ones. This is a real limitation, and the honest response is to acknowledge it: the scoring system is a filter for the predictable, not a curator of the exceptional.

---

## Summary of Validity Assessments

| # | Argument | Validity | Implication |
|---|----------|----------|-------------|
| 1 | Scoring is fundamentally broken | Partially valid | Biases are real but manageable; our use case (independent scoring, not comparative) avoids the worst failure modes |
| 2 | Binary won't help | Partially valid | Sycophancy is format-agnostic; must validate empirically via pass-rate monitoring during shadow scoring |
| 3 | The ceiling effect is correct | Possibly valid | Pre-filtering means the input distribution is skewed; measure within-ceiling discrimination, not just spread |
| 4 | Highlights are a bad proxy | Valid | Length bias, topic bias, and mood effects are real; normalize highlight counts and supplement with other signals |
| 5 | More questions = more noise | Partially valid | Overfitting risk is documented; mitigate with correlation analysis and question pruning during calibration |
| 6 | Pairwise > pointwise | Mixed | Theoretically appealing but practically problematic (cost, flip rates); consider hybrid for borderline articles |
| 7 | Human calibration is circular | Technically valid | Break circularity with temporal train/test splits and counterfactual exploration |
| 8 | Just use embeddings | Partially valid | Good for topical relevance, bad for quality assessment; valuable as a complement, not a replacement |
| 9 | The prompt is the bottleneck | Probably valid | Optimize the existing prompt before committing to an architectural change; staged approach is prudent |
| 10 | Cost/benefit doesn't justify it | Possibly valid | Measure the marginal improvement of LLM scoring over simple heuristics; if the gap is small, simplify |
| 11 | Temporal instability | Valid concern | Pin model versions, track scoring_version, use calibration toolkit to detect drift |
| 12 | Personal taste is not a function | Philosophically valid | Accept that the system captures articulable preferences and will miss inarticulate ones; this is a known limitation |

---

## Recommended Actions

Based on this analysis, the following actions would strengthen our approach:

1. **Before building v3-binary, optimize the v2 prompt.** Add few-shot calibration examples, explicit anti-sycophancy framing, and test reframing the evaluation question. Measure improvement. Only proceed to v3-binary if prompt optimization alone is insufficient (addresses #9).

2. **Build a simple heuristic baseline.** Implement `priority = f(author_boost, word_count, source, category)` and compare its Spearman correlation with highlights against the LLM scorer. This establishes the value of LLM scoring empirically (addresses #10).

3. **Normalize highlight counts in calibration.** Control for article length (highlights per 1,000 words), reading progress, and topic category. Consider a composite engagement score (addresses #4).

4. **Monitor pass rates during shadow scoring.** If average binary question pass rate exceeds 70%, the sycophancy concern is validated and questions need hardening (addresses #2, #5).

5. **Implement temporal train/test splits.** Never calibrate and evaluate on the same time period. Reserve the most recent month of data for evaluation only (addresses #7).

6. **Pin and version model endpoints.** Track both the prompt version and the underlying model version. Re-calibrate when either changes (addresses #11).

7. **Consider embedding pre-filtering.** Use cheap embedding similarity as a first-pass topical relevance filter, reserving LLM scoring for topically-relevant articles only (addresses #8, #10).

---

## Sources

- [LLM Judges Are Unreliable — Collective Intelligence Project](https://www.cip.org/blog/llm-judges-are-unreliable)
- [Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge (arXiv 2412.12509)](https://arxiv.org/abs/2412.12509)
- [Self-Preference Bias in LLM-as-a-Judge (arXiv 2410.21819)](https://arxiv.org/html/2410.21819v2)
- [Evaluating the Effectiveness of LLM-Evaluators — Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/)
- [LLM Stability: A Detailed Analysis (arXiv 2408.04667)](https://arxiv.org/html/2408.04667v1)
- [Pairwise or Pointwise? Evaluating Feedback Protocols (arXiv 2504.14716)](https://arxiv.org/abs/2504.14716)
- [How Is ChatGPT's Behavior Changing Over Time? — Harvard Data Science Review](https://hdsr.mitpress.mit.edu/pub/y95zitmz)
- [Is GPT-4 a Reliable Rater? — Frontiers in Education](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2023.1272229/full)
- [Limitations of LLM-as-a-Judge in Expert Knowledge Tasks — IUI 2025](https://dl.acm.org/doi/10.1145/3708359.3712091)
- [LLM-Rubric: Calibrated Evaluation (arXiv 2501.00274)](https://arxiv.org/html/2501.00274v1)
- [Text Highlighting and Text Length — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950329321003748)
- [How Algorithmic Popularity Bias Hinders or Promotes Quality — Nature](https://www.nature.com/articles/s41598-018-34203-2)
- [Human-AI Feedback Loops — Nature Human Behaviour](https://www.nature.com/articles/s41562-024-02077-2)
- [Deconstructing Taste: AI Framework for Aesthetic Perceptions (arXiv 2601.17134)](https://arxiv.org/abs/2601.17134)
- [Aesthetic Taste — Internet Encyclopedia of Philosophy](https://iep.utm.edu/aesthetic-taste/)
- [Using LLMs for Evaluation — Cameron Wolfe](https://cameronrwolfe.substack.com/p/llm-as-a-judge)
- [Likert or Not: LLM Absolute Relevance Judgments (arXiv 2505.19334)](https://arxiv.org/html/2505.19334v1)
- [SemScore: Evaluating LLMs with Semantic Similarity — HuggingFace](https://huggingface.co/blog/g-ronimo/semscore)
- [Measuring Data Quality for LLM Training — BRICS](https://brics-econ.org/measuring-data-quality-for-llm-training-model-based-and-heuristic-filters)
- [Chatbot Arena: Benchmarking LLMs with Elo Ratings — LMSYS](https://lmsys.org/blog/2023-05-03-arena/)
- [Demystifying Evals for AI Agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Why Rate When You Could Compare? EloChoice — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190393)
- [A Survey on LLM-as-a-Judge (arXiv 2411.15594)](https://arxiv.org/abs/2411.15594)
- [Opportunities and Challenges of LLM-as-a-Judge — EMNLP 2025](https://aclanthology.org/2025.emnlp-main.138.pdf)
