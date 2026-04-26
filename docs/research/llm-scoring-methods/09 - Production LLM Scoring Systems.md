# Production LLM Scoring Systems

**Date:** 2026-02-18
**Purpose:** Document real-world production systems that use LLMs to score or evaluate free-text content. Covers case studies, deployed architectures, calibration practices, cost structures, failure modes, and optimization techniques drawn from companies and researchers operating at scale.

---

## Executive Summary

LLM-based content scoring has moved from research novelty to production infrastructure across multiple domains: content moderation, news curation, academic review, SEO optimization, and feed personalization. This survey examines 15+ deployed systems and distills cross-cutting patterns.

**Key findings:**

1. **Binary and low-precision scoring dominates production.** Systems that survive contact with production almost universally favor binary pass/fail or coarse scales (3-5 levels) over fine-grained numeric scoring. Hamel Husain, after consulting with 30+ companies, reports that binary verdicts with detailed critiques outperform numeric scales in every production context he has encountered.

2. **Calibration requires human ground truth -- there is no shortcut.** Pinterest's Decision Quality Evaluation Framework, Stanford's Agentic Reviewer, and Honeycomb's query judge all anchor to expert-curated golden datasets. Teams that skip this step report score drift and unreliable metrics within weeks.

3. **Tiered model cascades cut costs 45-85% without quality loss.** The dominant cost optimization pattern routes easy items through cheap/small models and escalates uncertain cases to expensive models. RouteLLM achieves 85% cost reduction on MT-Bench while maintaining 95% of GPT-4 quality.

4. **LLM judges exhibit predictable, manageable biases.** Position bias, verbosity bias, self-enhancement bias, and central tendency bias are well-documented. Production systems mitigate these through response order shuffling, structured rubrics, and multi-judge consensus.

5. **Scale is feasible and affordable.** Google's Perspective API processes 500 million requests per day. OpenAI's Moderation API is free at any scale. Karpathy scored 930 Hacker News threads for $58. Marmelab's curator-ai generates daily newsletters for pennies per run.

---

## Part 1: Production Case Studies

### 1.1 Pinterest -- Decision Quality Evaluation Framework

**Domain:** Content moderation at scale
**Paper:** "Decision Quality Evaluation Framework at Pinterest" (arXiv, February 2026)

**Scoring Approach:**
Pinterest uses a Golden Dataset (GDS) system curated by subject matter experts (SMEs) as ground truth. Moderation decisions are evaluated on two axes:
- **Reliability:** Cohen's Kappa measuring consistency of judgments among different labelers
- **Correctness:** Accuracy, precision, recall, F1, false positive/negative rates, informedness, and markedness metrics against the GDS benchmark

**Calibration:**
- A high-trust Golden Set serves as the fixed evaluation benchmark for iterative prompt development
- Engineers modify a prompt, test against the set, and receive immediate quantitative feedback
- Continuous validation monitors content drift via new GDS items and system stability through re-evaluation on fixed benchmarks
- Propensity sampling using XGBoost predicts underrepresented content for efficient SME labeling budget allocation
- Semantic coverage measured using PinCLIP image embeddings quantized through RQ-VAE, tracking 256 possible first-layer semantic codes
- Distributional divergence tracked via Jensen-Shannon Divergence (JSD) comparing dataset distribution against production traffic

**Accuracy:**
- GPT-4 balanced: -4.1% accuracy vs. 3x human majority baseline, but with precision/recall trade-offs
- GPT-5 balanced: +0.9% accuracy improvement over baseline
- LLMs generally match a single non-expert human but lag behind SME-level quality

**Cost and Scale:**
- Transitioned from 3x human majority labels to LLM-based approach
- Achieved **30x cost savings** and **10x reduction in labeling turnaround time**
- Production system handles Pinterest's global content moderation pipeline

**Key Insight:** Pinterest treats prompt development as "quantitative science" -- the Golden Set transforms prompt engineering from art to engineering with measurable iteration cycles.

**Sources:**
- [Decision Quality Evaluation Framework at Pinterest (arXiv)](https://arxiv.org/html/2602.15809)
- [Content Moderation by LLM: From Accuracy to Legitimacy (Springer)](https://link.springer.com/article/10.1007/s10462-025-11328-1)

---

### 1.2 Ramp -- LLM Judge for Transaction Classification

**Domain:** Financial transaction classification
**Source:** Ramp engineering blog and ZenML LLMOps database

**Scoring Approach:**
Ramp built an AI agent using LLMs, embeddings, and RAG to automatically fix incorrect merchant classifications. An LLM judge evaluates whether the agent's reclassification actions are correct:
- For changes: determines whether the agent's action resulted in an improvement per the user's request
- For rejections: assesses whether the rejection was reasonable

**Calibration -- Shadow Mode:**
Before production deployment, Ramp ran the agent in "shadow mode" where it determined what actions it would take without executing them. This enabled safe validation against human decisions, significantly de-risking rollouts. The shadow mode evaluation is described as "a best practice for high-stakes LLM applications."

**Scale and Accuracy:**
- Processes user requests in under 10 seconds
- Handles ~100+ requests daily
- **99% accuracy** according to LLM-based evaluation
- Previous manual handling rate: 1.5-3% of requests; now handles nearly 100%

**Cost:**
- Reduced customer support costs from hundreds of dollars to cents per request

**Sources:**
- [How Ramp Fixes Merchant Matches with AI (Ramp Builders Blog)](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai)
- [Ramp AI Agent Case Study (ZenML)](https://www.zenml.io/llmops-database/ai-agent-for-automated-merchant-classification-and-transaction-matching)
- [Ramp Case Study (LangChain)](https://www.langchain.com/breakoutagents/ramp)

---

### 1.3 Stanford Agentic Reviewer (PaperReview.ai)

**Domain:** Academic paper review and scoring
**Source:** Stanford ML Group, launched 2025

**Scoring Approach:**
7-dimensional evaluation framework feeding into a unified score via linear regression:
1. Originality
2. Importance of research question addressed
3. Whether claims are well supported
4. Soundness of experiments
5. Clarity of writing
6. Value to the research community
7. Contextualization relative to prior work

**Scoring Scale:** Numeric, with 5.5 serving as a calibration threshold. A linear regression model combines dimension scores into a final score.

**Architecture:**
1. PDF Processing: Converts papers to Markdown via LandingAI's Agentic Document Extraction
2. Related Work Discovery: Generates multi-specificity search queries via Tavily API for arXiv papers
3. Relevance Filtering: Evaluates metadata to select top papers, choosing between abstract or detailed summaries
4. Review Generation: Synthesizes comprehensive feedback using original paper and related work summaries

**Calibration and Accuracy:**
- Tested on 297 ICLR 2025 submissions (150 training, 147 test)
- **Spearman correlation with human reviewers: 0.42** (compared to 0.41 between two human reviewers)
- **AUC for acceptance prediction: 0.75** (vs. 0.84 for human scores)
- Low AI scores (<=5.5) align proportionally with papers receiving low average human scores

**Broader Context:**
- AAAI launched a pilot for their AAAI-26 conference incorporating LLMs into peer review, notably deciding **not** to provide numerical scores, only qualitative feedback
- Research from NEJM AI found GPT-4 reviewer feedback overlap with human reviewers (30.85% for Nature journals, 39.23% for ICLR) is comparable to overlap between two human reviewers
- 57.4% of users found GPT-4 generated feedback helpful/very helpful
- Known bias: LLMs never give lowest scores and cluster around middle scores

**Sources:**
- [Stanford Agentic Reviewer Tech Overview](https://paperreview.ai/tech-overview)
- [Stanford Launches AI Agentic Paper Reviewer (HowAIWorks)](https://howaiworks.ai/blog/paperreview-ai-stanford-agentic-reviewer-2025)
- [AAAI Launches AI-Powered Peer Review Assessment System](https://aaai.org/aaai-launches-ai-powered-peer-review-assessment-system/)
- [Can LLMs Provide Useful Feedback on Research Papers? (NEJM AI)](https://ai.nejm.org/doi/abs/10.1056/AIoa2400196)

---

### 1.4 Google Perspective API (Jigsaw)

**Domain:** Toxicity and content moderation scoring
**Launched:** 2017, continuously updated

**Scoring Approach:**
- Returns a score between 0 and 1 indicating likelihood of perceived toxicity
- Categories include toxicity, severe toxicity, identity attack, insult, profanity, threat, sexually explicit content
- Not LLM-based (ML classifiers trained on human-labeled data), but an instructive production reference for any content scoring system

**Scale:**
- **500 million requests per day** as of 2021
- Free API, no cost per request
- Used by The New York Times, Wikipedia, and numerous other publishers

**Calibration:**
- Trained on human-labeled datasets where annotators rated perceived toxicity
- Continuous model updates based on new labeled data
- Known limitations: subject to biases in training data, recommendation is always to augment human moderators rather than replace them

**Production Integration:**
- The New York Times built "Moderator" on top of Perspective to prioritize comments for human review
- Enabled the Times to open significantly more articles to comments by auto-approving low-risk content

**Key Insight:** Even at 500M requests/day, the recommendation is human-in-the-loop. Pure automated scoring is positioned as a prioritization and routing tool, not a final arbiter.

**Sources:**
- [Perspective API (Google)](https://perspectiveapi.com/)
- [Google Jigsaw Announces Perspective Processing 500M Requests Daily (PR Newswire)](https://www.prnewswire.com/news-releases/googles-jigsaw-announces-toxicity-reducing-api-perspective-is-processing-500m-requests-daily-301223600.html)
- [Perspective API Documentation (Google Developers)](https://developers.perspectiveapi.com/s/about-the-api-score)

---

### 1.5 OpenAI Moderation API

**Domain:** Content safety scoring
**Scoring Approach:**
- Returns a JSON object with: `flagged` (boolean), `categories` (per-category true/false flags), and `category_scores` (confidence levels 0-1)
- Categories: sexual, hate, harassment, self-harm, violence, and sub-categories
- The `omni-moderation-latest` model supports multi-modal inputs (text + image)

**Cost:** Free for all OpenAI API users. Usage does not count toward monthly limits.

**Scale:** Rate limits depend on usage tier, but no per-request cost barrier.

**Accuracy:** The omni-moderation-latest model reports 42% improvement on multilingual test sets vs. previous versions.

**Sources:**
- [OpenAI Moderation API (Documentation)](https://platform.openai.com/docs/guides/moderation)
- [OpenAI Pricing](https://platform.openai.com/docs/pricing)

---

### 1.6 IBM Granite Guardian

**Domain:** LLM content moderation and safety scoring
**Scoring Approach:**
- Outputs `Yes` (unsafe) or `No` (safe) labels with class confidence scores (0-1)
- Risk score ranges: Low (0-0.3), Medium (0.4-0.7), High (0.8-1.0)
- Categories: general harm, social bias, profanity, violence, sexual content, unethical behavior, jailbreaking, groundedness/relevance for RAG, and function calling hallucinations for agentic workflows
- Hybrid thinking model: can operate in thinking or non-thinking mode, producing detailed reasoning traces through `<think>` and `<score>` tags in thinking mode

**Open Source:** Available on Hugging Face (ibm-granite/granite-guardian-3.3-8b), enabling self-hosted deployment at any scale.

**Sources:**
- [Granite Guardian (IBM)](https://www.ibm.com/granite/docs/models/guardian)
- [LLM Content Moderation with Granite Guardian (IBM Tutorial)](https://www.ibm.com/think/tutorials/llm-content-moderation-with-granite-guardian)
- [Granite Guardian on Hugging Face](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)

---

### 1.7 Karpathy's HN Time Capsule -- Auto-Grading at Scale

**Domain:** Content quality and prescience scoring
**Source:** Andrej Karpathy, personal project (2026)

**Scoring Approach:**
6-section analysis framework per article:
1. Article and discussion summary
2. Historical outcome research
3. "Most prescient" and "Most wrong" comment awards
4. Notable discussion aspects
5. Individual commenter letter grades (A+ through F)
6. Overall interestingness score (0-10)

Commenters receive letter grades accumulated across articles to identify patterns in prescience or inaccuracy over time.

**Model and Cost:**
- Model: GPT 5.1 Thinking (with Claude Opus 4.5 used for vibe-coding the project itself)
- 930 LLM queries (31 days x 30 articles/day)
- **Total cost: ~$58**
- **Runtime: ~1 hour** for all processing
- Development time: ~3 hours

**Key Insight:** Demonstrates that meaningful content quality scoring at moderate scale (hundreds to low thousands of items) is remarkably affordable with modern LLMs. The interestingness score (0-10) is a direct analog to article quality scoring in feed curation.

**Sources:**
- [Auto-grading decade-old Hacker News discussions with hindsight (Karpathy)](https://karpathy.bearblog.dev/auto-grade-hn/)
- [HN Time Capsule (Live Results)](https://karpathy.ai/hncapsule/)
- [GitHub: karpathy/hn-time-capsule](https://github.com/karpathy/hn-time-capsule)

---

### 1.8 Marmelab Curator-AI -- Newsletter Curation

**Domain:** AI-curated news digests
**Source:** Marmelab engineering blog (2024), open source

**Scoring Approach:**
- 0-100 relevancy scale based on user-defined interests
- Single-dimension scoring: "how much the article talks about [user interests]"
- Articles sorted by relevancy score, top N returned
- Output: JSON with title, author, summary (max 3 sentences, 80 words), relevancy_score, link

**Architecture:**
1. Fetch articles from RSS feeds, aggregators (Hacker News, dev.to), or direct URLs
2. Extract content using Mozilla's Readability library (strips ads/navigation)
3. Summarize and rate via OpenAI API
4. Sort by relevancy, return top results
5. URL deduplication across sources

**Model and Cost:**
- Default: GPT-3.5-turbo-1106 (GPT-4 available but significantly more expensive)
- **Cost per newsletter: a few cents** with GPT-3.5
- Processing: sequential, "may take several minutes if there are many articles"

**Accuracy Limitations:**
Author acknowledges: "The AI is not perfect...misses the point...returns non-JSON output." Requires prompt tweaking for reliable results.

**Sources:**
- [Cut Through The Noise: AI-Curated News Digests (Marmelab)](https://marmelab.com/blog/2024/03/21/ai-curator.html)
- [GitHub: marmelab/curator-ai](https://github.com/marmelab/curator-ai)

---

### 1.9 Honeycomb -- Query Assistant LLM Judge

**Domain:** Database query correctness evaluation
**Source:** Hamel Husain's LLM-as-Judge guide, referencing Honeycomb as case study

**Scoring Approach:**
- Binary pass/fail with detailed critiques
- Task: judge whether AI-generated database queries are correct

**Calibration:**
- Domain expert (Phillip Carter) provided initial pass/fail judgments with detailed critiques
- Achieved **>90% agreement between LLM judge and domain expert in just 3 iterations**
- Key discovery: reviewing LLM critiques helped the expert articulate his own evaluation criteria more clearly, revealing inconsistencies in edge cases

**Iteration Process:**
1. Collect domain expert pass/fail judgments with detailed critiques (minimum ~30 examples)
2. Create initial judge prompt using 3+ expert examples
3. Run judge on test set, compare with expert judgments
4. Calculate agreement rates (using precision and recall, not raw agreement, due to class imbalance)
5. Manually adjust prompt based on failure patterns
6. Repeat until >90% agreement

**Sources:**
- [Using LLM-as-a-Judge For Evaluation: A Complete Guide (Hamel Husain)](https://hamel.dev/blog/posts/llm-judge/)
- [AI Evaluations Crash Course (Hamel Husain)](https://creatoreconomy.so/p/ai-evaluations-crash-course-in-50-minutes-hamel-husain)

---

### 1.10 Feedly Leo -- AI Feed Prioritization

**Domain:** RSS feed curation and article prioritization
**Source:** Feedly (commercial product, Pro+/Business/Enterprise tiers)

**Scoring Approach:**
- Topic-based priority scoring using ML and NLP
- Not a simple numeric score displayed to users; instead surfaces "priority" articles in a dedicated view
- Features: topic prioritization, content deduplication, content muting, business event tracking

**Calibration:**
- Users train Leo by providing examples of relevant and irrelevant content
- Feedback loop: users click Dislike icon and choose an action, which Leo incorporates into its ML algorithm
- Continuous refinement through user interaction

**Key Insight:** Feedly's approach sidesteps the numeric scoring problem entirely by framing the output as priority/not-priority (effectively binary), then letting users refine the boundary through feedback.

**Sources:**
- [Feedly Leo (Feedly)](https://feedly.com/i/leo)
- [Feedly Leo Review (AI Product Reviews)](https://ai-productreviews.com/feedly-leo-review/)

---

### 1.11 Readwise Reader -- Ghostreader AI

**Domain:** Read-later article processing and knowledge management
**Source:** Readwise (commercial product)

**Approach:**
Readwise Reader does not score articles for quality in a ranked feed. Instead, its AI features focus on post-save processing:
- **Ghostreader:** Summarizes documents, generates thought-provoking questions, creates Q&A pairs from highlights, answers questions about documents
- **AI Chat:** Chat with your entire highlight library to ask questions and get context
- **Themed Reviews:** Natural language topic input (e.g., "caring for a newborn") generates a curated review from existing highlights

**Key Insight:** Readwise chose to augment the reading experience rather than gate-keep it. There is no "quality score" filtering what enters the reading queue -- the AI works on what the user has already chosen to save. This is a deliberate design choice that avoids the calibration and trust challenges inherent in pre-read scoring.

**Sources:**
- [AI in Readwise (Learning Aloud)](https://learningaloud.com/blog/2025/02/12/ai-in-readwise/)
- [Readwise Reader Review (TutorialsWithAI)](https://tutorialswithai.com/tools/readwise-reader/)
- [Readwise Changelog](https://readwise.io/changelog)

---

### 1.12 Artifact -- News Quality Signals

**Domain:** AI-powered news aggregation
**Source:** Artifact (Instagram co-founders, 2023, shut down 2024)

**Scoring Approach:**
- Multi-signal quality scoring beyond clicks: dwell time, read time, shares, DM shares
- Epsilon-Greedy algorithm: 10-20% of recommendations explore outside the core recommendation spectrum
- Editorial consideration baked into training data selection, objective function, and included data

**Quality Philosophy:**
Co-founder Kevin Systrom: "If you let your algorithm focus on clicks, it will end up serving clickbait. If you simply optimize for only what people have clicked on, you end up having tunnel vision."

**Key Insight:** Artifact represents the strongest public articulation of multi-signal quality scoring in a news product. The explicit avoidance of click-only optimization and the use of dwell time as a proxy for genuine engagement are directly relevant to "save-worthiness" scoring.

**Sources:**
- [The tech behind Artifact (TechCrunch)](https://techcrunch.com/2023/03/07/the-tech-behind-artifact-the-newly-launched-news-aggregator-from-instagrams-co-founders/)
- [Details on the algorithm running Artifact (Nieman Lab)](https://www.niemanlab.org/reading/details-on-the-algorithm-running-the-new-news-aggregator-app-artifact/)

---

### 1.13 SEO Content Scoring Tools (Clearscope, Surfer SEO, MarketMuse)

**Domain:** Content quality and optimization scoring for SEO

**Clearscope:**
- Letter grades A++ to F, assigned in real-time as content is written
- Evaluates keyword density, semantic relevance, content depth, and topical coverage
- Benchmarks against top 20 search results for target keyword
- Uses latent semantic indexing (LSI) keywords and NLP for content gap analysis
- Readability scoring via Flesch-Kincaid index
- Pricing: $150+/month

**Surfer SEO:**
- Reverse-engineers SERP results, analyzing multiple ranking factors
- Numeric content score based on data-backed optimization recommendations
- Research shows content with consistent heading levels is 40% more likely to be referenced by ChatGPT
- Pricing: $50-100/month

**MarketMuse:**
- Proprietary AI topic modeling (not traditional TF-IDF)
- Strategic content planning that automates entire content inventory analysis
- Pricing: $150+/month

**Key Insight:** These tools demonstrate that real-time, multi-criteria content scoring at scale is commercially viable. Their grading approach (letter grades in Clearscope's case) is notably coarser than numeric scores, reflecting the practical reality that users understand "A+" vs "B-" better than "82 vs 71."

**Sources:**
- [How does Clearscope grade your content? (Clearscope)](https://www.clearscope.io/support/how-does-clearscope-grade-your-content)
- [Clearscope vs MarketMuse vs Surfer (Clearscope)](https://www.clearscope.io/blog/clearscope-vs-marketmuse-vs-surfer)
- [Best LLM SEO Tools 2025 (Superframeworks)](https://superframeworks.com/blog/best-llm-seo-tools)

---

## Part 2: Common Failure Modes in Production LLM Scoring

### 2.1 Position Bias

LLM judges exhibit strong preference for responses based on placement order. Research from "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (2024) found that **48.4% of verdicts reversed under mirrored response order**, despite 100% evaluator agreement on individual presentations. Position bias increases with more answer candidates.

**Mitigation:** Shuffle response order in pairwise comparisons; use direct scoring instead of preference selection; average scores across multiple orderings.

**Sources:**
- [Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (arXiv)](https://arxiv.org/html/2410.02736v1)
- [Exploring LLM-as-a-Judge (Weights & Biases)](https://wandb.ai/site/articles/exploring-llm-as-a-judge/)

### 2.2 Verbosity Bias

LLMs favor longer outputs regardless of quality. Advanced models like GPT-4o still fail in nearly one-third of cases where verbosity bias is tested. Increasing response length without quality improvement led to measurable decline in model robustness.

**Production risk:** Content creators can exploit this by producing verbose, repetitive content that receives artificially high scores.

**Mitigation:** Explicit instructions to ignore length; structured rubrics focusing on specific quality criteria; penalty terms for unnecessary verbosity.

### 2.3 Central Tendency Bias (Score Inflation)

LLMs cluster scores around the middle of any scale. In academic review contexts, LLMs never give the lowest possible score and most frequently assign middle scores. G-Eval's probability-weighted scoring was specifically designed to address this, as the token probability for "3" on a 1-5 scale is inherently higher.

**Mitigation:** Binary scoring eliminates this entirely; probability-weighted scoring (G-Eval style) reduces it; anchor examples at scale extremes help; explicit instruction to use the full range.

### 2.4 Self-Enhancement Bias

LLMs rate their own outputs higher than outputs from other models. This poses a special risk in automated training loops where the scoring model and the generation model are the same or share architecture.

**Mitigation:** Use a different model family for scoring than for generation; calibrate against human expert labels.

### 2.5 Fallacy Oversight

LLM judges overlook logical errors in reasoning, particularly when the text is well-structured and fluent. Surface-level quality masks substantive problems.

**Mitigation:** Decomposed evaluation with explicit logic-checking criteria; chain-of-thought reasoning before scoring.

### 2.6 Criteria Drift

Over time, as the production data distribution shifts, the LLM judge's calibration degrades. Pinterest addresses this through continuous validation with fixed benchmarks and new GDS items to detect content drift.

**Mitigation:** Regular re-calibration against human expert judgments; monitoring score distributions over time; alerting on distribution shifts.

**Sources:**
- [LLM-as-a-Judge (Wikipedia)](https://en.wikipedia.org/wiki/LLM-as-a-Judge)
- [A Survey on LLM-as-a-Judge (arXiv)](https://arxiv.org/html/2411.15594v6)
- [LLM Evaluation Bias: Ensuring Objective Assessment (Statsig)](https://www.statsig.com/perspectives/llm-evaluation-bias)

---

## Part 3: Cost Optimization Techniques

### 3.1 Tiered Model Cascades

The dominant cost optimization pattern in production LLM scoring:

**Architecture:** Route easy items through cheap/small models; escalate uncertain cases to expensive models.

**RouteLLM** (LMSYS, 2024):
- Open-source framework for cost-effective LLM routing
- **85% cost reduction on MT-Bench** while maintaining 95% of GPT-4 quality
- 45% reduction on MMLU, 35% on GSM8K
- Uses a trained router to predict which model will perform adequately for each query

**Select-then-Route (StR):**
- Two-stage framework: taxonomy-guided selector maps queries to proficient model pools, then confidence-based cascade starts with cheapest model and escalates when multi-judge agreement signals low reliability
- Presented at EMNLP 2025 Industry Track

**Practical Implementation:**
Start with the cheapest capable model (GPT-4o-mini, Claude Haiku, Gemini Flash) and benchmark against expensive models. Only escalate when cheap model confidence is below threshold.

**Sources:**
- [RouteLLM: Cost-Effective LLM Routing (LMSYS)](https://lmsys.org/blog/2024-07-01-routellm/)
- [A Unified Approach to Routing and Cascading for LLMs (ETH Zurich)](https://files.sri.inf.ethz.ch/website/papers/dekoninck2024cascaderouting.pdf)
- [Select-then-Route (ACL Anthology)](https://aclanthology.org/2025.emnlp-industry.28/)

### 3.2 Knowledge Distillation for Scoring

Train a smaller, cheaper model to replicate the scoring behavior of a large model:

- **OpenPipe approach:** Fine-tune Mistral 7B to replicate GPT-4 behavior, achieving up to 85% cost reduction
- **LinkedIn skills extraction:** 80% model size reduction through distillation
- **Mercari dynamic attribute extraction:** 95% model size reduction and 14x cost reduction vs. GPT-3.5-turbo through quantization
- **LlamaIndex example:** Distilling GPT-3.5 into a fine-tuned judge for correctness evaluation

**Key principle:** A teacher model can provide a numerical score (not just binary labels) to the student, giving richer training signal that enables the student to handle borderline cases.

**Sources:**
- [Fine-tuning LLMs at Scale: Cost Optimization (Xenoss)](https://xenoss.io/blog/fine-tuning-llm-cost-optimization)
- [Distillation with Programmatic Data Curation (TensorZero)](https://www.tensorzero.com/blog/distillation-programmatic-data-curation-smarter-llms-5-30x-cheaper-inference)
- [Knowledge Distillation for Fine-Tuning a GPT-3.5 Judge (LlamaIndex)](https://docs.llamaindex.ai/en/stable/examples/finetuning/llm_judge/correctness/finetune_llm_judge_single_grading_correctness/)

### 3.3 Prompt and Token Optimization

- **RAG reduces context token usage by 70-85%** (DeepSeek case study) -- retrieve only the most relevant content before sending to LLM
- **Content extraction before scoring:** Marmelab uses Mozilla's Readability to strip ads and navigation before sending to the LLM, reducing token count significantly
- **Structured output formats:** JSON output with constrained fields reduces unnecessary generation tokens
- **Batch processing:** For non-real-time scoring, batch API calls reduce per-request overhead

### 3.4 Cost Reference Points

| System | Model | Scale | Cost |
|--------|-------|-------|------|
| Karpathy HN Capsule | GPT 5.1 Thinking | 930 articles | $58 total |
| Marmelab Curator-AI | GPT-3.5-turbo | ~50 articles/day | Few cents per newsletter |
| OpenAI Moderation API | omni-moderation | Unlimited | Free |
| Perspective API | Custom ML | 500M req/day | Free |
| Ramp Transaction Agent | LLM (unspecified) | 100+ req/day | Cents per request |

**Sources:**
- [How to Monitor Your LLM API Costs (Helicone)](https://www.helicone.ai/blog/monitor-and-optimize-llm-costs)
- [Optimizing LLM Performance and Cost (ZenML)](https://www.zenml.io/blog/optimizing-llm-performance-and-cost-squeezing-every-drop-of-value)
- [Top 10 Methods to Reduce LLM Costs (DataCamp)](https://www.datacamp.com/blog/ai-cost-optimization)

---

## Part 4: Latency-Quality Tradeoffs

### 4.1 Batch vs. Real-Time

Most production content scoring systems operate in **batch or async mode**, not real-time:
- Marmelab Curator-AI: "may take several minutes if there are many articles" (acceptable for daily newsletter)
- Karpathy HN Capsule: 1 hour for 930 articles (offline batch)
- SEO tools (Clearscope, Surfer): real-time scoring as user types (lightweight, pre-computed benchmarks)
- Content moderation (Perspective, OpenAI): near-real-time (<100ms) using optimized ML models, not full LLM calls

**Pattern:** For content quality scoring (not safety-critical moderation), batch/async processing is the norm. Real-time scoring is reserved for interactive editing tools and safety gates.

### 4.2 Model Size vs. Quality

Smaller models trade quality for speed and cost:
- **Interactive/real-time needs:** Use smaller, quantized models or pre-computed classifiers
- **Batch quality scoring:** Use the best available model; latency is irrelevant when processing queues
- **Hybrid:** Use fast classifier for screening, full LLM for detailed scoring on items that pass the screen

### 4.3 Practical Guidance

For a system like Reader Triage scoring ~100-500 articles per day:
- Batch processing with a strong model (Claude Sonnet/Opus, GPT-4o) is both affordable and practical
- At $0.003-0.015 per article (depending on article length and model), daily costs would be $0.30-$7.50
- Latency of 5-30 seconds per article is acceptable in an async scoring pipeline
- No need for model cascade at this scale -- the cost savings do not justify the complexity

---

## Part 5: How Teams Iterate on Scoring Prompts

### 5.1 The Evaluation-Driven Development Loop

Anthropic's recommended approach, validated across their customer base:

1. **Define:** Collect 20-50 task examples drawn from real failures
2. **Test:** Run LLM judge against examples
3. **Diagnose:** Manually review disagreements between LLM and expert
4. **Fix:** Adjust prompt based on failure patterns
5. Repeat

**Critical rule:** "A good task is one where two domain experts would independently reach the same pass/fail verdict." Ambiguous specifications become noise in metrics.

**Sources:**
- [Demystifying Evals for AI Agents (Anthropic)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

### 5.2 Hamel Husain's Practitioner Framework

After consulting with 30+ companies (including OpenAI, Anthropic, Google engineers):

1. **Start with error analysis, not metrics.** Understand what is actually going wrong before measuring it.
2. **Use binary pass/fail, not numeric scales.** "A binary decision forces everyone to consider what truly matters." A "fail" is a clear signal to fix a bug; a "3" is ambiguous.
3. **Include detailed critiques alongside verdicts.** These critiques serve as both documentation and few-shot examples for prompt refinement.
4. **Hand-craft prompt adjustments.** Prefer manual iteration over automated prompt optimizers (though ALIGN Eval is mentioned as promising).
5. **Target >90% agreement with domain experts.** The Honeycomb case study achieved this in 3 iterations.
6. **Build specialized judges only after error analysis reveals concentration.** Do not create targeted judges upfront.
7. **Use precision and recall, not raw agreement rates.** Raw agreement is misleading with imbalanced datasets.

**Sources:**
- [Using LLM-as-a-Judge For Evaluation: A Complete Guide (Hamel Husain)](https://hamel.dev/blog/posts/llm-judge/)
- [LLM Evals: Everything You Need to Know (Hamel Husain)](https://hamel.dev/blog/posts/evals-faq/)
- [A Field Guide to Rapidly Improving AI Products (Hamel Husain)](https://hamel.dev/blog/posts/field-guide/)

### 5.3 Calibration Techniques

**Anchor Examples:**
Include pre-graded samples in the judge's prompt. "By providing a clear example of what a '5-star' response and a '1-star' response look like, you anchor the judge's understanding of your scoring scale." This is the single most effective technique for reducing score variance.

**Chain-of-Thought Before Scoring:**
"Forces the model to articulate its rationale, which often leads to more accurate evaluations." This is the G-Eval insight applied in production.

**Repeated Judgments:**
For critical evaluations, run the same assessment 2-3 times with slightly different but semantically identical prompts and average the scores. Referenced as "Replacing Judges with Juries" (Verga et al., 2024).

**Temperature Control:**
Set model temperature to 0 or very low values for deterministic, reproducible scoring.

**Structured Output:**
Enforce JSON output format to prevent format-related scoring inconsistencies.

**Isolated Dimension Scoring:**
Score each dimension with an isolated LLM call rather than using one monolithic prompt to grade all dimensions. This prevents cross-contamination between criteria.

**Human Calibration Validation Target:**
Teams should target a correlation coefficient greater than 0.7 between LLM judge scores and human labels on a representative sample.

**Sources:**
- [LLM-as-Judge Done Right: Calibrating, Guarding & Debiasing (Kinde)](https://kinde.com/learn/ai-for-software-engineering/best-practice/llm-as-a-judge-done-right-calibrating-guarding-debiasing-your-evaluators/)
- [LLM-as-a-judge: Complete Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [5 Methods for Calibrating LLM Confidence Scores (Latitude)](https://latitude.so/blog/5-methods-for-calibrating-llm-confidence-scores)

---

## Part 6: Relevance to Reader Triage

### What Production Systems Teach Us

1. **Our scale is modest and affordable.** Scoring 100-500 articles per day at $0.01-0.03 per article costs $1-15/day. No cascade or distillation needed at this scale.

2. **Binary decomposition is the production consensus.** The most reliable production systems (Honeycomb, Ramp, Pinterest) use binary or very coarse verdicts. Our move toward decomposed binary sub-criteria per dimension is well-aligned with industry practice.

3. **Calibration must be anchored to human judgment.** Pinterest's Golden Dataset approach and Honeycomb's 3-iteration calibration cycle are directly applicable. We should maintain a labeled golden set of articles with known-good scores and validate prompt changes against it.

4. **Async batch scoring is the right architecture.** No production system doing quality scoring (as opposed to safety moderation) requires real-time latency. Our background scoring pipeline is the standard pattern.

5. **Score inflation is the primary failure mode to guard against.** Central tendency bias and the inability of LLMs to use the low end of scales means we should either (a) use binary sub-criteria that aggregate into scores, or (b) include strong anchor examples at both extremes of the scale.

6. **Feedly and Readwise chose to avoid scored feeds.** Feedly uses binary priority/not-priority. Readwise processes content after saving, not before. These design choices reflect the difficulty of making numeric quality scores trustworthy and interpretable to end users. Reader Triage's three-tier system (High/Medium/Low) is a reasonable middle ground.

7. **Prompt iteration is engineering, not art.** Pinterest, Anthropic, and Hamel Husain all describe a quantitative, iterative process: define golden set, run judge, measure agreement, adjust prompt, repeat. We should formalize this for our scoring prompts.
