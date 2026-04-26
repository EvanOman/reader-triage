# Case Studies in Content Recommendation and Engagement Prediction

## Executive Summary

Predicting whether a reader will deeply engage with a text article -- highlighting passages, saving it, sharing it -- is a problem that every major content platform has tackled, yet none have fully solved. Production systems from Pocket/Mozilla to Google News to Twitter/X converge on a multi-stage architecture (candidate generation, then ranking), but differ dramatically in what signals they prioritize: collaborative filtering on user behavior (Twitter/X, Reddit), content-based NLP features (Microsoft's NRMS, Feedly Leo), or hybrid editorial-algorithmic approaches (Pocket, Apple News). Academic research, particularly Microsoft's MIND benchmark and the POPROX live experiment, reveals that content understanding quality is the single largest driver of recommendation accuracy, yet most research fails to translate to production because it ignores preference elicitation, temporal dynamics, and the fundamental problem that engagement does not equal satisfaction (Kleinberg et al., 2022). For a single-user personal reading assistant, the evidence strongly favors content-based approaches augmented by lightweight user feedback, since collaborative filtering requires a user population that does not exist in this context.

---

## 1. Production Systems

### 1.1 Pocket / Mozilla

Pocket, acquired by Mozilla in 2017, evolved from a read-it-later app into a content recommendation engine serving over 40 million users monthly via Firefox New Tab. Mozilla described their approach as **"algotorial"** -- a portmanteau of "algorithmic" and "editorial" coined by Spotify's Daniel Ek ([Mozilla Blog](https://blog.mozilla.org/en/mozilla/reflecting-on-10-years-of-time-well-spent-with-pocket/)).

**How it worked:**
- Machine learning algorithms scoured the web for high-quality content appealing to Firefox's diverse user base
- A human editorial team provided oversight, ensuring recommendations were trustworthy and high-quality
- Human moderators worked directly with ML engineers to solve the "Garbage In, Garbage Out" problem, focusing on data quality over quantity ([Mozilla Careers Blog](https://blog.mozilla.org/careers/pockets-journey-to-africa/))

**Key lesson:** Pocket's team found that purely algorithmic approaches risk perpetuating bias and causing harm, particularly impacting vulnerable populations. The human-in-the-loop approach was not just a quality filter -- editors actively shaped the training data and feedback loops. Pocket shut down in 2025, but Mozilla continues the algotorial approach for Firefox content recommendations.

**Relevance to personal triage:** The algotorial insight maps directly to our scoring system. Our Claude-based scoring acts as the "algorithm," while the user's feedback (marking articles as read, highlighting, dismissing) acts as the "editor" shaping future scoring calibration.

### 1.2 Medium

Medium uses a **hybrid recommendation system** combining content-based and collaborative filtering ([Medium Blog Recommendation analysis](https://medium.com/almabetter/medium-blog-recommendation-8748285300ce)).

**Content-based component:** Articles are represented primarily through tags and topic keywords. If a user reads articles tagged "Machine Learning" and "Data Science," the system surfaces more articles with those tags.

**Collaborative filtering component:** Users with similar reading histories get cross-pollinated recommendations. The system identifies clusters of users with overlapping engagement patterns.

**Engagement signals tracked:**
- Read ratio (how much of the article was actually read)
- Claps (Medium's equivalent of likes, with intensity -- users can clap up to 50 times)
- Highlights (direct passage-level engagement)
- Follows triggered by article reads
- Reading time and scroll depth

**Key insight:** Medium's read ratio metric is particularly relevant. They found that a user reading 80%+ of an article is a much stronger signal of quality than a click or even a clap. This aligns with our goal of predicting highlight-worthy content -- articles that people actually read deeply tend to be the ones they highlight.

### 1.3 Substack

Substack's recommendation system is architecturally distinct from most platforms. As described by Head of Data Mike Cohen, Substack **deliberately avoids optimizing for time-on-platform or ad clicks** ([Substack Official Blog](https://on.substack.com/p/substacks-recommendations-network)).

**How it works:**
- Instead of algorithmic collaborative filtering, Substack asks writers to recommend other writers to their subscribers -- a **social graph approach** rather than a behavioral one
- This recommendation network drives 50% of all new subscriptions and 25% of new paid subscriptions
- The algorithm matches users with content based on profile and activity, without bias for content type
- As of 2025, Substack's Notes algorithm amplifies content that drives actual subscription behavior, not just likes or comments ([The Publishing Spectrum](https://thepublishingspectrum.substack.com/p/substacks-algorithm-just-changed))

**Key lesson:** Substack optimizes for **conversion** (subscriptions, paid upgrades) rather than engagement (clicks, time-on-site). This is a fundamentally different objective function. The result is that their recommendations tend to surface higher-quality, longer-form content because that is what drives subscription decisions.

**Relevance:** This maps to our scoring philosophy. We are not predicting clicks -- we are predicting whether a user will find an article valuable enough to highlight and save passages. Like Substack, we want to optimize for deeper engagement signals.

### 1.4 Google News

Google's news recommendation research is among the most thoroughly documented in academic literature. The foundational paper, "Personalized News Recommendation Based on Click Behavior" (Liu et al., 2010), described a system built on click logs from 16,848 Google News users over 12 months ([Google Research](https://research.google.com/pubs/archive/35599.pdf)).

**Architecture:**
- **User profiling:** Bayesian framework predicts current news interests from individual user click history combined with population-level news trends
- **Content modeling:** CNN-based architectures extract features from headlines, abstracts, and article bodies. Attention mechanisms select important features for building news representations
- **Hybrid approach:** Content-based recommendation using learned user profiles combined with collaborative filtering. The hybrid method improved click-through rates by 30.9% over pure collaborative filtering

**Key features used:**
- Topic categories and subcategories
- Named entities (people, organizations, locations)
- Recency and freshness signals
- User click history and dwell time
- Cross-user trending patterns

**Modern evolution:** Google News now incorporates deep learning models for news content understanding, with pre-trained language models (BERT-family) significantly improving performance. The system must handle extreme temporal dynamics -- news preferences change hourly, not weekly ([Wu et al., 2021, "Personalized News Recommendation: Methods and Challenges"](https://arxiv.org/pdf/2106.08934)).

### 1.5 Apple News

Apple News operates a **dual-curation system** ([Cult of Mac](https://www.cultofmac.com/news/apple-news-app-2)):

- **Top Stories:** Curated by a human editorial team
- **Trending Stories:** Algorithmically selected based on engagement signals
- **Personalized feed:** Adapts based on reading behavior, topic interests, and regional relevance

Research by Jack Bandy auditing 4,000 stories found that human editors chose sources more evenly and preferred more diverse sources compared to the trending algorithm, which skewed toward "soft news" ([Bandy, 2020, ICWSM](https://ojs.aaai.org/index.php/ICWSM/article/download/7277/7131/10507)). This echoes the Pocket finding: algorithmic-only approaches tend to converge on a narrow band of "engaging" content that may not represent what users actually value.

**User controls:** "Suggest More" and "Suggest Less" buttons provide explicit preference signals, supplementing implicit behavioral data.

### 1.6 Twitter/X Algorithmic Feed

Twitter open-sourced its recommendation algorithm in 2023, providing unprecedented transparency into a large-scale engagement prediction system ([X Engineering Blog](https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm), [GitHub](https://github.com/twitter/the-algorithm)).

**Three-phase architecture:**
1. **Candidate sourcing:** ~1,500 tweets selected from in-network (people you follow) and out-of-network (people you don't follow, surfaced via engagement graph analysis)
2. **Ranking:** A 48M-parameter neural network based on **MaskNet** analyzes thousands of features per tweet, outputting 10 probability labels:
   - P(like), P(retweet), P(reply), P(click), P(media engagement), P(relevance), P(dwell time), P(negative engagement), P(report), P(see less)
3. **Heuristics and filters:** Post-ranking rules ensure feed diversity, author diversity, and content-type mixing

**Key component -- Real Graph:** The single most important ranking signal for in-network tweets is **Real Graph**, a model predicting engagement likelihood between two specific users. Higher Real Graph scores between you and an author mean more of their tweets appear in your feed.

**Engagement weighting:** The open-sourced code confirmed that predicted likes and retweets contribute far more to ranking than replies. Content likely to spark back-and-forth conversation receives an outsized boost. X Blue (paid) subscribers receive a 2-4x ranking multiplier.

**Relevance to personal triage:** Twitter's multi-label approach is instructive. Rather than predicting a single "engagement" score, they predict the probability of specific behaviors. Our four-dimension scoring (quotability, surprise, argument quality, applicable insight) follows a similar philosophy -- decomposing "engagement" into specific, predictable facets.

### 1.7 Reddit Ranking

Reddit uses two distinct algorithms for different content types ([Medium/Hacking and Gonzo](https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9)):

**Posts -- Hot Algorithm:**
- Score = log10(upvotes - downvotes) + (timestamp - reference_time) / 45000
- Logarithmic scaling means the first 10 upvotes matter as much as the next 100
- Time decay: every 12.5 hours, a new post with zero votes gains 1 full ranking point from freshness alone

**Comments -- Wilson Score Interval (Confidence Sort):**
- Treats votes as a sample of how everyone would eventually vote
- Computes a lower confidence bound for the positive ratio
- At 85% confidence level, a comment with 10 upvotes and 0 downvotes ranks higher than one with 10 upvotes and 5 downvotes, even though both have 10 upvotes
- This is statistically principled: it rewards certainty over raw volume

**Key lesson:** Reddit's logarithmic vote scaling is a deliberate choice to prevent runaway popularity from dominating. The first signals of quality matter most. This is relevant to our system: early engagement signals (first highlight, first save) should carry disproportionate weight in calibrating future scores.

### 1.8 Feedly / Leo AI

Feedly's Leo AI assistant represents the closest production analog to our personal triage system ([Feedly Leo](https://www.futuretools.io/tools/feedly-leo)).

**How Leo works:**
- Continuously scans a user's customized RSS feed
- Uses ML and NLP to filter noise and prioritize relevant articles
- Supports **trainable skills**: Topic skill, Like Board skill, Business Event skill, Security Threat skill, Mute Filter skill, De-duplication skill
- Users train Leo by providing examples and feedback; the system refines recommendations over time

**Key insight:** Leo is a **single-user personalization system** operating on content features. It does not use collaborative filtering (there is no shared user base informing your recommendations). Instead, it learns your preferences through explicit training (marking articles as important/not important) and applies NLP-based content analysis.

**Relevance:** Feedly Leo validates the approach of using content-based NLP scoring for a personal reading assistant. The "train by example" paradigm is directly applicable -- our system could use a user's highlighting and saving behavior as implicit training signal for score calibration.

### 1.9 Readwise / Reader

Readwise Reader is the most directly relevant production system to our project ([Readwise](https://readwise.io/read)).

**Key features:**
- **Triage workflow:** Inspired by Superhuman's email triage, Reader provides a game-like interface for deciding what to read now, read later, or archive
- **Ghostreader AI:** An LLM-powered assistant (default: GPT-5 Mini, configurable to GPT-5, GPT-4.1, o3) that can summarize documents, look up terms, and answer questions about content ([Readwise Docs](https://docs.readwise.io/reader/guides/ghostreader/overview))
- **Highlight syncing:** Highlights from Reader automatically sync to Readwise for spaced repetition review via Daily Review
- **No algorithmic prioritization of reading queue:** Reader does not currently rank or score articles in the queue by predicted engagement

**Critical gap:** Despite having sophisticated AI features (Ghostreader) and rich engagement data (highlights, reading time, saves), Reader does not offer automated queue prioritization based on predicted engagement. This is precisely the gap our system fills.

### 1.10 Instapaper

Instapaper, the original read-it-later app, takes a minimalist approach. It provides no recommendation or prioritization features -- the queue is strictly chronological or manual. This simplicity is both its appeal and its limitation for power readers with large backlogs.

---

## 2. Academic Research

### 2.1 Microsoft MIND Benchmark

The **MIND dataset** (Microsoft News Dataset) is the most important benchmark for news recommendation research. Published at ACL 2020 by Wu et al. ([ACL Anthology](https://aclanthology.org/2020.acl-main.331/), [GitHub](https://github.com/msnews/MIND)):

- **Scale:** 1 million users, 160k+ English news articles, 15 million+ impression logs
- **Content richness:** Each article includes title, abstract, body, category, subcategory, and entities
- **Baseline models benchmarked:**
  - **DKN** (Deep Knowledge-Aware Network)
  - **LSTUR** (Long- and Short-term User Representations, ACL 2019)
  - **NAML** (Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019)
  - **NPA** (Neural News Recommendation with Personalized Attention)
  - **NRMS** (Neural News Recommendation with Multi-Head Self-Attention, EMNLP 2019)

**Key finding:** "The performance of news recommendation highly relies on the quality of news content understanding and user interest modeling." Pre-trained language models significantly improved performance across all architectures. NRMS, which uses multi-head self-attention to capture word-level relatedness and news-level interaction patterns, emerged as one of the strongest baselines.

**Implication:** For content-based scoring, the quality of text understanding matters more than the sophistication of the recommendation algorithm. Using a strong language model (like Claude) for content analysis is likely more impactful than implementing a complex recommendation architecture.

### 2.2 POPROX: Lessons from Building a Live News Recommender

Higley, Burke, Ekstrand, and Knijnenburg (2025) published "What News Recommendation Research Did (But Mostly Didn't) Teach Us About Building A News Recommender" ([arXiv:2509.12361](https://arxiv.org/abs/2509.12361)), documenting their experience building POPROX, a live news recommendation platform.

**Gaps they identified in existing research:**
1. **Data utilization:** Research typically uses only a subset of available article attributes, limiting real-world applicability
2. **Explicit preferences:** Almost no published work addresses incorporating user-declared topic interests during onboarding
3. **Preference controls:** Industrial systems commonly offer topic preference interfaces, yet research papers rarely incorporate declared user interests
4. **Longitudinal evaluation:** Little research examines user experience over time in recommender systems

**Implementation challenges:**
1. **Transfer learning was necessary:** Fresh platforms lack behavior data. They used NRMS pre-trained on MIND, but this model only uses headlines, preventing use of richer metadata
2. **Preference elicitation:** They built a 14-topic preference interface with separate pipelines for interest/disinterest, but found almost no research foundation for this
3. **Curation-personalization balance:** No guidance exists for combining editorial curation with algorithmic ranking
4. **User satisfaction measurement:** Weekly surveys had "very low engagement," making longitudinal evaluation extremely difficult

**Recommendations:**
- Prioritize data handling as a foundational concern
- Evaluate model affordances -- ensure models can accommodate real system functionality
- Recognize that recommendation extends far beyond modeling (filtering, re-ranking, user controls, multi-stage pipelines)
- Leverage live systems for diverse evaluation approaches

**Relevance:** This paper is the strongest evidence that academic news recommendation research has a significant gap between theory and practice. For our personal triage system, this means we should not expect off-the-shelf models to work well. Our approach of using an LLM for content scoring, calibrated against user behavior, is pragmatically sound precisely because it sidesteps the issues POPROX encountered.

### 2.3 The Language That Drives Engagement

Banerjee and Urminsky (2024) published a systematic analysis of headline experiments in *Marketing Science* ([DOI](https://pubsonline.informs.org/doi/10.1287/mksc.2021.0018)).

**Dataset:** 32,488 A/B tests from Upworthy.com (Jan 2013 - Apr 2015), where different headlines were tested on the same article.

**Key findings on what increases click-through:**
- **Forward reference** (creating curiosity gaps) increases engagement
- **Emotional intensity** words increase engagement
- **Surprise/shock references** ("you won't believe," "shocked") increase clicks
- **Discomfort/embarrassment** framing increases clicks
- **Numeric fluency** (specific numbers, data points) improves engagement
- **Concreteness** (specific, vivid language) improves engagement
- **Deliberative processing** cues increase engagement

**What decreases engagement:**
- Mentions of initiatives that positively impact a community
- Abstract, vague language

**Critical caveat:** The researchers note that prior literature "does not provide useful guidance as to the direction of the effects" -- many features that theory predicted would increase engagement actually decreased it, and vice versa. This highlights the importance of empirical calibration over theoretical assumptions.

**Relevance to our scoring:** Several of these findings directly inform our scoring dimensions. "Quotability" aligns with concreteness and numeric fluency. "Surprise Factor" maps to the shock/surprise signals. "Argument Quality" relates to deliberative processing cues. The finding that headline language predicts engagement supports using textual features for scoring.

### 2.4 Dwell Time as Engagement Predictor

Kim et al. (2019) proposed a **content-based dwell time engagement prediction model** using deep neural networks ([ACL Anthology, NAACL 2019](https://aclanthology.org/N19-2028/)).

**Approach:**
- Extracted emotion, event, and entity features from articles
- Learned interactions among these features with word-based features via a deep neural network
- Predicted expected article dwell time (time users spend reading)

**Key finding:** Article dwell time is one of the most important factors showing article engagement. The content-based approach outperformed baselines on a real newspaper dataset, demonstrating that textual features alone can predict reading engagement.

**Nuances from related work:**
- Dwell time is widely used as a proxy for satisfaction, but the relationship is complex (Epstein et al., 2022, [arXiv:2209.10464](https://arxiv.org/abs/2209.10464))
- Topic, article length, and readability level mediate the dwell-time-to-satisfaction relationship
- Short dwell time does not always indicate low quality -- it can reflect user-article mismatch rather than content quality ([Sato et al., 2020](https://arxiv.org/abs/2012.13992))
- Users dwell longer on sensationalized content but less on credible content -- dwell time measures attention ("trying"), while sharing/highlighting measures value ("buying")

**Relevance:** This distinction between attention and value is critical for our system. We should not optimize for predicted reading time. Instead, our scoring dimensions (quotability, surprise, argument quality, insight) target the "buying" signals -- would the reader highlight or save this? -- rather than the "trying" signals.

### 2.5 Inconsistent Preferences and Engagement Optimization

Kleinberg, Mullainathan, and Raghavan (2022) published "The Challenge of Understanding What Users Want" in *Management Science* ([arXiv:2202.11776](https://arxiv.org/abs/2202.11776)).

**Core contribution:** A formal model demonstrating that engagement optimization can diverge from user utility when users have **inconsistent preferences** -- impulsive in-the-moment choices that conflict with longer-term values.

**Key findings:**
- Users can have long platform sessions but derive very little utility
- Platform changes can steadily raise engagement before abruptly causing users to quit ("cold turkey" effect)
- For certain content directions, increasing engagement makes users less happy; for others, more happy
- The behavioral economics literature confirms: we choose mindlessly and myopically on platforms

**Implication:** This is the theoretical foundation for why our system should score articles on dimensions aligned with *reflective* value (quotability, insight, argument quality) rather than *impulsive* engagement (clickbait, sensationalism). A personal triage system has the advantage of serving one user's long-term interests rather than maximizing a platform's engagement metrics.

### 2.6 Personalized News Recommendation: Methods and Challenges

Wu et al. (2021) published a comprehensive survey ([arXiv:2106.08934](https://arxiv.org/pdf/2106.08934)) covering the state of personalized news recommendation.

**Architecture taxonomy:**
- **Content modeling:** CNN most frequently used for text representation in news systems; attention mechanisms widely used for selecting important features
- **User modeling:** Attention-based approaches for capturing user interest from click history; GRU/LSTM for modeling temporal interest evolution
- **Ranking:** Multi-objective optimization across clicks, dwell time, and user satisfaction

**Key challenges identified:**
- News preferences are highly dynamic (hourly, not weekly changes)
- Cold-start problem is severe for new users and new articles
- Diversity vs. accuracy tradeoff (echo chamber risk)
- Computational efficiency for real-time recommendation at scale

---

## 3. What Features Matter Most in Practice

Evidence across production systems and research converges on a hierarchy of feature importance:

### Tier 1: Content Understanding (Highest Impact)

- **Article text quality:** Full-text NLP features (entities, topics, sentiment, writing quality) consistently outperform metadata-only approaches. The MIND benchmark showed that pre-trained language models provided the largest performance gains.
- **Headline features:** Curiosity gaps, emotional intensity, concreteness, and numeric specificity predict engagement (Banerjee & Urminsky, 2024).
- **Content structure:** Article length, readability level, and structural features (lists, headers, images) correlate with engagement patterns.

### Tier 2: User History (High Impact, Requires Data)

- **Past engagement patterns:** Which topics, sources, and article types the user has previously engaged with deeply (not just clicked). Twitter's Real Graph model shows that user-to-author affinity is the strongest single predictor of in-network engagement.
- **Temporal patterns:** Interests change over time. Google News's Bayesian framework explicitly models interest evolution. Static profiles degrade quickly.
- **Explicit preferences:** Topic interests declared by users (Feedly Leo's training, Apple News's Suggest More/Less) are underused in research but valuable in practice (POPROX findings).

### Tier 3: Metadata and Context (Moderate Impact)

- **Source/author reputation:** Consistent quality signals from known authors/publications.
- **Recency:** Freshness matters enormously for news (Reddit's time decay, Google News's temporal dynamics) but less for evergreen content.
- **Social proof:** Collaborative signals (others' engagement) are powerful for platforms but unavailable for single-user systems.

### Tier 4: Presentation Context (Low but Real Impact)

- **Position in feed:** Strong position bias in engagement data (items shown first get more clicks regardless of quality).
- **Time of day:** Reading behavior varies by time; certain content types perform better at different times.
- **Device:** Mobile vs. desktop reading patterns differ significantly.

---

## 4. Single-User Personalization vs. Collaborative Filtering

This distinction is fundamental to our system's architecture.

### Collaborative Filtering: Strengths and Limitations

**Strengths:**
- Can discover surprising, serendipitous recommendations ("users like you also read...")
- Requires no content understanding -- operates purely on behavior patterns
- Netflix Prize (2009) demonstrated that hybrid approaches with collaborative filtering outperform pure content-based methods

**Limitations:**
- **Requires a user population:** With N=1 (a personal reading assistant), there is no collaborative signal. This eliminates collaborative filtering as a primary approach.
- **Cold-start problem:** New items with no engagement data cannot be recommended. For a reading queue with fresh articles, every item is cold-start from a collaborative perspective.
- **Popularity bias:** Collaborative filtering tends to recommend popular items, which may not align with a single user's niche interests.

### Content-Based Filtering: The Single-User Path

**Strengths for personal triage:**
- Works with N=1 -- no user population needed
- Every article can be scored immediately, solving the cold-start problem
- Provides explainable recommendations (scored on specific dimensions)
- Can leverage powerful pre-trained language models (Claude, GPT) for content understanding

**Limitations:**
- Cannot discover truly surprising content outside the user's known interests
- Quality depends entirely on the content analysis model
- May create filter bubbles without explicit diversity mechanisms

### Hybrid Approach for Personal Systems

The evidence suggests a practical hybrid for single-user systems:

1. **Content-based scoring** (our current approach) as the primary signal
2. **User feedback loop** (highlighting, saving, dismissing) to calibrate content scoring over time -- this is the Feedly Leo model
3. **Source-level trust signals** to learn which publications consistently deliver value for this specific user
4. **Diversity injection** to prevent filter bubble effects (surface some lower-scored articles from varied topics)

---

## 5. Lessons Learned: Common Failures and What Actually Moves the Needle

### What Doesn't Work

1. **Optimizing for clicks/engagement as a proxy for value.** Kleinberg et al. (2022) demonstrated formally that engagement and utility can diverge. Dwell time research shows users spend more time on sensationalized content but derive less value from it. Every platform that optimized purely for engagement has faced backlash (Facebook's News Feed controversies, YouTube's rabbit hole problem).

2. **Assuming static user preferences.** Google News research showed preferences shift hourly. POPROX found that models trained on historical data degrade without temporal adaptation. User profiles must be living documents.

3. **Ignoring the cold-start problem.** POPROX had to use transfer learning from MIND because fresh platforms lack data. For a personal system processing new articles daily, every article is effectively a cold-start item -- content-based scoring is the only viable initial approach.

4. **Over-indexing on headline/metadata features.** While headlines predict clicks (Banerjee & Urminsky), they do not predict deep engagement. Full-text content understanding is necessary for predicting highlights and saves.

5. **Treating recommendation as purely a modeling problem.** The POPROX team found that data handling, preference elicitation, user controls, and feed composition are as important as the ranking model itself.

6. **Academic benchmarks that don't transfer.** The POPROX paper explicitly documents that MIND-trained models failed to accommodate real-world requirements (explicit preferences, diverse metadata, longitudinal evaluation). Wilson score and other simple statistical approaches (Reddit) can outperform complex neural models when the problem is well-understood.

### What Actually Works

1. **High-quality content understanding.** The MIND benchmark's central finding: recommendation quality depends primarily on content understanding quality. Using a strong language model (Claude) for content scoring is well-supported by evidence.

2. **Multi-dimensional scoring.** Twitter/X predicts 10 separate engagement labels. Our four-dimension scoring (quotability, surprise, argument quality, insight) follows the same principle: decompose "engagement" into specific, predictable dimensions. This is more robust than a single score.

3. **Explicit user feedback.** Feedly Leo's "train by example" model, Apple News's Suggest More/Less, and Substack's writer-recommendation network all rely on explicit signals. Implicit signals (reading time, scroll depth) are noisy; explicit signals (highlight, save, dismiss) are gold.

4. **Human-in-the-loop calibration.** Pocket's algotorial approach, Apple News's editorial curation, and Substack's writer-recommendation system all demonstrate that human judgment significantly improves recommendation quality over pure algorithmic approaches.

5. **Logarithmic scaling of early signals.** Reddit's ranking algorithm gives disproportionate weight to early votes. For personal calibration, the first few explicit feedback signals from a user should carry outsized weight in adjusting scoring parameters.

6. **Content features over collaborative signals for niche/personal use.** Research comparing content-based vs. collaborative filtering consistently shows content-based approaches have higher accuracy with limited data (precisely our situation). Collaborative filtering requires rich behavioral data across a user population.

---

## 6. Open-Source Tools and Frameworks

### Recommendation Frameworks

| Tool | Language | Focus | Relevance |
|------|----------|-------|-----------|
| [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | Python | Full-stack recommendation with news specialization (NRMS, NAML, LSTUR). 16k+ GitHub stars. MIND dataset integration. | High -- includes news-specific models and notebooks |
| [RecBole](https://recbole.io/) | Python | 100+ recommendation algorithms, deep learning focus | Medium -- comprehensive but oriented toward multi-user systems |
| [Surprise](https://surpriselib.com/) | Python | Collaborative filtering, rating prediction | Low -- requires multi-user data |
| [LensKit](https://lenskit.org/) | Python | Research-oriented, PyData ecosystem integration | Medium -- good for experimentation |
| [Sentence Transformers](https://github.com/huggingface/sentence-transformers) | Python | Text embeddings for semantic similarity. 15k+ pre-trained models on Hugging Face | High -- directly applicable for content-based similarity scoring |

### NLP and Content Analysis

| Tool | Use Case | Relevance |
|------|----------|-----------|
| [Sentence Transformers / SBERT](https://sbert.net/) | Generate article embeddings for similarity, clustering, and content-based scoring | High |
| [Hugging Face Transformers](https://huggingface.co/transformers/) | Pre-trained models for sentiment, topic classification, named entity recognition | High |
| [spaCy](https://spacy.io/) | Entity extraction, readability analysis, linguistic features | Medium |
| Anthropic / OpenAI APIs | LLM-based content scoring and summarization (our current approach) | High -- already in use |

### Datasets for Benchmarking

| Dataset | Description | Access |
|---------|-------------|--------|
| [MIND](https://msnews.github.io/) | 1M users, 160k articles, 15M impressions. The gold standard for news recommendation research | [GitHub](https://github.com/msnews/MIND) |
| [Upworthy Archive](https://upworthy.natematias.com/) | 32k+ headline A/B tests. Used in Banerjee & Urminsky (2024) | Public |
| [PENS](https://msnews.github.io/pens.html) | Personalized News Headline Generation dataset (Microsoft) | Public |

---

## 7. Applicable Insights for a Personal Reading Assistant

Drawing from the evidence above, here are the most actionable insights for our article triage system:

### 7.1 Content-Based Scoring is the Right Foundation

For a single-user system with no collaborative signal, content-based scoring using a strong language model is the most evidence-supported approach. The MIND benchmark demonstrated that content understanding quality is the primary driver of recommendation accuracy. Our use of Claude for multi-dimensional scoring aligns with this finding.

### 7.2 Multi-Dimensional Scoring Outperforms Single Scores

Twitter/X's 10-label prediction system and the general trend in production systems toward multi-objective ranking validate our four-dimension approach (quotability, surprise, argument quality, applicable insight). Each dimension captures a different facet of "engagement-worthiness" and provides more robust predictions than a single composite score.

### 7.3 Optimize for Reflective Value, Not Impulsive Engagement

Kleinberg et al. (2022) proved that engagement and user utility can diverge. Our scoring dimensions are deliberately aligned with reflective value (would you highlight this? would you save this passage?) rather than impulsive engagement (would you click this?). The dwell time research confirms this distinction: attention and value are different constructs.

### 7.4 Calibrate Against User Behavior Over Time

Feedly Leo's "train by example" model and the POPROX finding that explicit preferences are underserved in research both point to the same conclusion: build a feedback loop. When users highlight passages, save articles, or dismiss items, these signals should adjust scoring weights. Even a simple approach (track which score dimensions correlate with user engagement and upweight them) would be valuable.

### 7.5 Full-Text Analysis Beats Headline/Metadata Approaches

While headline features predict clicks (Banerjee & Urminsky), predicting deeper engagement (highlighting, saving) requires full-text content understanding. Our system's use of article body text for scoring is well-supported by the literature. The dwell time prediction paper (Kim et al., 2019) showed that emotion, entity, and event features extracted from full text outperform surface-level features.

### 7.6 Handle Temporal Dynamics

Google News and POPROX both emphasize that preferences are dynamic. Our system should consider:
- Recency-weighting user feedback (recent highlights matter more than old ones)
- Detecting topic interest shifts over time
- Periodic recalibration of scoring weights

### 7.7 Inject Diversity Deliberately

Every production system struggles with filter bubbles. Apple News found that algorithmic-only approaches narrow source diversity. For a personal system, consider:
- Occasionally surfacing lower-scored articles from underrepresented topics
- Tracking source diversity in the reading queue
- Using the "surprise factor" scoring dimension as a natural diversity mechanism

### 7.8 Simple Statistical Methods Can Compete

Reddit's Wilson Score Interval (a 1927 statistical method) effectively ranks comments. Simple approaches like Bayesian updating on user preferences can be surprisingly effective. Do not over-engineer the recommendation layer -- the content understanding (via Claude) is where the value is. A simple weighted combination of the four scoring dimensions, calibrated against user behavior, may outperform a complex neural ranking model.

---

## Sources

### Production Systems
- [Mozilla Blog: Reflecting on 10 Years of Pocket](https://blog.mozilla.org/en/mozilla/reflecting-on-10-years-of-time-well-spent-with-pocket/)
- [Mozilla Careers: Pocket's Journey to Africa (Algotorial Approach)](https://blog.mozilla.org/careers/pockets-journey-to-africa/)
- [X Engineering Blog: Twitter's Recommendation Algorithm](https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm)
- [GitHub: twitter/the-algorithm](https://github.com/twitter/the-algorithm)
- [GitHub: twitter/the-algorithm-ml](https://github.com/twitter/the-algorithm-ml)
- [Substack Official: Upgrading the Recommendations Network](https://on.substack.com/p/substacks-recommendations-network)
- [The Publishing Spectrum: Substack's Algorithm Just Changed](https://thepublishingspectrum.substack.com/p/substacks-algorithm-just-changed)
- [Cult of Mac: Why Apple News Relies on Human Curation](https://www.cultofmac.com/news/apple-news-app-2)
- [Bandy (2020): Editors vs. Algorithms in Apple News (ICWSM)](https://ojs.aaai.org/index.php/ICWSM/article/download/7277/7131/10507)
- [Readwise Reader](https://readwise.io/read)
- [Readwise Docs: Ghostreader Overview](https://docs.readwise.io/reader/guides/ghostreader/overview)
- [Feedly Leo AI](https://www.futuretools.io/tools/feedly-leo)
- [Medium/Hacking and Gonzo: How Reddit Ranking Algorithms Work](https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9)

### Academic Papers
- [Wu et al. (2020): MIND: A Large-scale Dataset for News Recommendation (ACL 2020)](https://aclanthology.org/2020.acl-main.331/)
- [Wu et al. (2019): NRMS: Neural News Recommendation with Multi-Head Self-Attention (EMNLP 2019)](https://github.com/msnews/MIND)
- [Wu et al. (2021): Personalized News Recommendation: Methods and Challenges](https://arxiv.org/pdf/2106.08934)
- [Liu et al. (2010): Personalized News Recommendation Based on Click Behavior (Google Research)](https://research.google.com/pubs/archive/35599.pdf)
- [Higley et al. (2025): What News Recommendation Research Did (But Mostly Didn't) Teach Us (POPROX)](https://arxiv.org/abs/2509.12361)
- [Banerjee & Urminsky (2024): The Language That Drives Engagement (Marketing Science)](https://pubsonline.informs.org/doi/10.1287/mksc.2021.0018)
- [Kim et al. (2019): Content-based Dwell Time Engagement Prediction Model (NAACL 2019)](https://aclanthology.org/N19-2028/)
- [Kleinberg, Mullainathan & Raghavan (2022): The Challenge of Understanding What Users Want (Management Science)](https://arxiv.org/abs/2202.11776)
- [Epstein et al. (2022): Quantifying Attention via Dwell Time and Engagement](https://arxiv.org/abs/2209.10464)
- [Sato et al. (2020): Analysis of Short Dwell Time in Relation to User Interest](https://arxiv.org/abs/2012.13992)

### Open-Source Tools
- [Microsoft Recommenders (GitHub)](https://github.com/recommenders-team/recommenders)
- [RecBole](https://recbole.io/)
- [Surprise (scikit-surprise)](https://surpriselib.com/)
- [LensKit](https://lenskit.org/)
- [Sentence Transformers (Hugging Face)](https://github.com/huggingface/sentence-transformers)
- [MIND Dataset (GitHub)](https://github.com/msnews/MIND)

### Surveys and Overviews
- [Georgetown KGI (2025): Better Feeds: Algorithms That Put People First](https://kgi.georgetown.edu/wp-content/uploads/2025/02/Better-Feeds_-Algorithms-That-Put-People-First.pdf)
- [Knight First Amendment Institute: Understanding Social Media Recommendation Algorithms](https://knightcolumbia.org/content/understanding-social-media-recommendation-algorithms)
- [KDnuggets: Latest Innovations in Recommendation Systems with LLMs](https://www.kdnuggets.com/latest-innovations-in-recommendation-systems-with-llms)
- [Eugene Yan: Improving Recommendation Systems & Search in the Age of LLMs](https://eugeneyan.com/writing/recsys-llm/)
