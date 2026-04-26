# Feature Engineering Beyond Embeddings

## Executive Summary

While 768-dimensional article embeddings capture semantic content effectively, substantial predictive gains for reader engagement (highlight counts) come from non-embedding features that encode metadata, structural, and linguistic signals embeddings inherently miss. Research consistently shows that hybrid models combining embeddings with handcrafted features outperform pure embedding approaches -- a 2024 study on automated essay scoring found a hybrid XGBoost model (QWK 0.941) significantly outperformed pure RoBERTa embeddings (QWK 0.927) and BERT (QWK 0.918) ([Maalej et al., 2024](https://www.mdpi.com/2227-7390/12/21/3416)). For our highlight prediction task, the most promising non-embedding features fall into four categories: metadata features (word count, author engagement history, domain signals), text structure features (readability scores, heading/list density), linguistic features (vocabulary richness, named entity density, sentiment), and author features (historical highlight rate, topic consistency). A stacked ensemble architecture -- where a gradient-boosted meta-learner combines embedding-based predictions with handcrafted feature predictions -- offers the best balance of performance, interpretability, and maintainability.

---

## 1. Metadata Features

### 1.1 Word Count

Word count is one of the strongest individual predictors of reader engagement. Chartbeat's analysis of millions of articles (2019--2022) found that between 0 and 2,000 words, average engaged time increases linearly with word count; beyond 4,000 words, variability grows and the return on additional length diminishes ([Chartbeat, 2022](https://chartbeat.com/resources/articles/is-there-an-optimal-article-length-the-relationship-between-word-count-and-engagement/)). Pew Research Center's study of 117 million cellphone interactions confirmed that long-form articles (1,000+ words) display higher overall visitor engagement ([Pew Research, 2016](https://www.pewresearch.org/journalism/2016/05/05/2-analysis-suggests-some-readers-willingness-to-dig-into-long-form-news-on-cellphones/)).

For highlight prediction specifically, word count has a dual role: longer articles provide more surface area for highlights, but engagement per unit length may plateau. Recommended encoding:

- **Raw word count** (log-transformed to reduce skew)
- **Word count buckets**: short (<500), medium (500--1500), long (1500--4000), very long (4000+)
- **Estimated reading time**: `word_count / 238` (average adult reading speed in WPM)

### 1.2 Author Engagement History

Prior author engagement is a potent signal, particularly for cold-start mitigation in recommendation systems ([Wikipedia: Cold start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))). Content-based filtering uses item features like author to suggest similar content, and author reputation serves as a strong prior when no interaction data exists for a new article ([Vinija's Notes on Cold Start](https://vinija.ai/recsys/cold-start/)).

Given that we already track total highlights per author, recommended features:

- **Author highlight rate**: `total_author_highlights / total_author_articles`
- **Author highlight density**: `total_author_highlights / total_author_word_count`
- **Author article count**: raw count (log-transformed), indicating familiarity
- **Author recency**: days since last article from this author was read
- **Author engagement trend**: slope of highlight rate over the last N articles (is engagement increasing or decreasing with this author?)

### 1.3 Domain / Site Name

Domain encodes editorial quality, content type, and reader familiarity. A research blog vs. a marketing site vs. a newsletter each carry different engagement priors.

Recommended features:

- **Domain highlight rate**: historical average highlights per article from this domain
- **Domain frequency**: how often articles from this domain appear in the feed
- **Domain category**: binary flags or learned embeddings for domain clusters (tech blogs, news outlets, newsletters, personal blogs)
- **Is new domain**: binary flag for domains with fewer than N prior articles (cold-start indicator)

### 1.4 Article Category

The existing `category` field (article, email, rss, podcast) provides a coarse content-type signal:

- **One-hot encoding** of category
- **Category interaction features**: category x word_count, category x author_highlight_rate (engagement patterns differ by content type)

### 1.5 Publication Timing

Publication timing affects engagement through recency effects and reading context. Research shows that content freshness and alignment with reading windows significantly impact engagement ([Hashmeta, 2025](https://hashmeta.com/blog/unlock-peak-engagement-finding-the-absolute-best-times-to-post-on-social-media-in-2025/)).

Recommended features:

- **Article age at triage time**: hours/days since publication
- **Day of week published**: cyclical encoding (sin/cos of day_of_week / 7)
- **Hour of day published**: cyclical encoding (sin/cos of hour / 24)
- **Weekend flag**: binary
- **Recency bucket**: fresh (<24h), recent (1--3d), aging (3--7d), old (7d+)

### 1.6 Reading Progress

The existing `reading_progress` field (0.0--1.0) is a post-hoc signal but can be used for model evaluation and for features in a two-stage pipeline:

- **Completed flag**: reading_progress >= 0.95
- **Abandoned flag**: 0 < reading_progress < 0.2
- **Progress bucket**: unread, started, midway, mostly-read, completed

---

## 2. Text Structure Features

Text structure features capture how an article is organized, independent of semantic content. These are signals that embeddings typically miss because embedding models collapse structure into a single dense vector.

### 2.1 Readability Scores

Readability directly affects engagement: lower bounce rates and longer dwell times correlate with appropriate readability levels. Research combining BERT embeddings with readability metrics showed significant improvements (RMSE of 0.30 vs. 0.44 for BERT alone) ([Supervised and Unsupervised Neural Approaches to Text Readability, MIT Press](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to)). The `textstat` Python library provides easy extraction of multiple readability metrics ([textstat on PyPI](https://pypi.org/project/textstat/)):

- **Flesch Reading Ease** (0--100 scale; higher = easier)
- **Flesch-Kincaid Grade Level** (US grade level; score of 9.3 means ninth-grade reading level)
- **Dale-Chall Readability Score** (based on familiar word lists)
- **Automated Readability Index** (character-count based)
- **SMOG Index** (based on polysyllabic word count)
- **Coleman-Liau Index** (character-based alternative to syllable counting)
- **Gunning Fog Index** (sentence length + complex words)

Recommendation: Use Flesch Reading Ease and Flesch-Kincaid Grade Level as the primary pair (they are the most widely validated). Add Dale-Chall if the model benefits from an alternative word-list-based measure. Avoid using all seven -- they are highly correlated and add multicollinearity without proportional information gain.

### 2.2 Paragraph and Sentence Structure

- **Paragraph count**: total paragraphs
- **Average paragraph length**: words per paragraph
- **Paragraph length variance**: standard deviation of paragraph lengths (irregular paragraph lengths may indicate diverse content)
- **Sentence count**: total sentences
- **Average sentence length**: words per sentence
- **Sentence length variance**: standard deviation (varied sentence length correlates with engaging writing)

### 2.3 Header Density

Headers signal article organization and scannability:

- **Header count**: number of `<h1>`--`<h6>` tags (or markdown headers)
- **Header density**: headers per 1000 words
- **Header depth distribution**: fraction of h2 vs. h3 vs. h4 (deeper hierarchy suggests more detailed structure)
- **Has table of contents**: binary flag (detected from anchor-linked headers or explicit ToC)

### 2.4 List and Code Block Density

Lists and code blocks signal structured, reference-worthy content that may attract highlights:

- **List count**: number of `<ul>` / `<ol>` elements
- **List item density**: list items per 1000 words
- **Code block count**: number of `<pre>` / `<code>` blocks
- **Code block density**: code blocks per 1000 words
- **Has inline code**: binary flag
- **Image count**: number of images (visual richness)
- **Link density**: hyperlinks per 1000 words (high link density may indicate reference/resource articles)

### 2.5 Blockquote Density

Blockquotes often contain the kind of quotable passages readers highlight:

- **Blockquote count**: number of blockquote elements
- **Blockquote density**: blockquotes per 1000 words
- **Blockquote text ratio**: fraction of total text inside blockquotes

---

## 3. Linguistic Features

### 3.1 Sentiment and Subjectivity

Sentiment and subjectivity capture the emotional and opinion dimensions of text. Research shows a positive correlation between sentiment valence and engagement metrics like favorites and retweets ([Sentiment Analysis, Wikipedia](https://en.wikipedia.org/wiki/Sentiment_analysis)). Subjectivity detection -- distinguishing factual from opinion content -- is a critical subtask that affects downstream engagement predictions ([Liu, Sentiment Analysis and Subjectivity](https://www.cs.uic.edu/~liub/FBS/NLP-handbook-sentiment-analysis.pdf)).

Recommended features (extractable via TextBlob or VADER):

- **Overall sentiment polarity**: mean polarity across sentences (-1 to +1)
- **Sentiment variance**: standard deviation of sentence-level polarity (high variance may indicate provocative or balanced articles)
- **Overall subjectivity**: mean subjectivity (0 = objective, 1 = subjective)
- **Positive sentence ratio**: fraction of sentences with polarity > 0.1
- **Negative sentence ratio**: fraction of sentences with polarity < -0.1
- **Emotional intensity**: absolute mean polarity (strength regardless of direction)

### 3.2 Named Entity Density

Named entity density reflects the informational richness of an article. Pages with broader entity variety reflect richer informational context, which correlates with content depth ([ThatWare: Named Entity Recognition Enhanced Ranking](https://thatware.co/named-entity-recognition/)). spaCy's EntityRecognizer efficiently extracts non-overlapping labeled spans ([spaCy Linguistic Features](https://spacy.io/usage/linguistic-features)).

Recommended features:

- **Total entity count**
- **Entity density**: entities per 1000 words
- **Entity type distribution**: counts/fractions of PERSON, ORG, GPE, DATE, MONEY, etc.
- **Unique entity count**: number of distinct entities (topic breadth)
- **Entity diversity ratio**: unique entities / total entities
- **Has numeric data**: binary flag for articles with MONEY, QUANTITY, PERCENT entities (data-rich articles may attract more highlights)

### 3.3 Vocabulary Richness

Lexical diversity and vocabulary richness are established indicators of writing quality and proficiency ([Lexical Diversity, Wikipedia](https://en.wikipedia.org/wiki/Lexical_diversity)). The Type-Token Ratio (TTR) is the foundational measure, though it must be length-normalized since TTR naturally decreases with text length ([Sketch Engine: TTR](https://www.sketchengine.eu/glossary/type-token-ratio-ttr/)).

Recommended features:

- **Type-Token Ratio (TTR)**: unique words / total words (compute on a fixed-length window, e.g., first 500 tokens, to control for length effects)
- **Root TTR** (Guiraud's Index): unique words / sqrt(total words) -- partially controls for length
- **Hapax Legomena ratio**: words appearing exactly once / total words
- **Average word length**: characters per word
- **Proportion of long words**: fraction of words with 3+ syllables
- **Proportion of difficult words**: fraction not on Dale-Chall familiar word list (available via `textstat.difficult_words()`)

The `lexicalrichness` Python package ([PyPI](https://pypi.org/project/lexicalrichness/)) provides MTLD (Measure of Textual Lexical Diversity) and HD-D, which are more robust to text length than raw TTR.

### 3.4 Discourse and Argumentation Markers

Articles with strong arguments and structured reasoning may attract more highlights:

- **Discourse connective density**: frequency of connectives (however, therefore, moreover, although, nevertheless) per 1000 words
- **Causal language density**: frequency of causal markers (because, since, as a result, consequently)
- **Hedging language density**: frequency of hedges (perhaps, might, possibly, it seems)
- **First-person pronoun ratio**: I/we usage (personal writing style)
- **Question density**: questions per 1000 words (may indicate Socratic or exploratory style)

---

## 4. Author Features

### 4.1 Historical Engagement Rate

Author-level features provide strong priors, especially when the reader has previously encountered an author's work:

- **Author highlight rate**: highlights per article (mean and median)
- **Author highlight density**: highlights per 1000 words read
- **Author completion rate**: average reading_progress across their articles
- **Author engagement consistency**: standard deviation of highlights across articles (consistent vs. inconsistent quality)
- **Author best-article highlight count**: max highlights on any single article (ceiling signal)

### 4.2 Publication Frequency

- **Author publication frequency**: articles per month in the feed
- **Author recency**: days since last article from this author
- **Author tenure**: days since first article from this author appeared
- **Is prolific author**: binary flag for authors with > N articles in the corpus

### 4.3 Topic Consistency

Authors who consistently write on topics the reader engages with should score higher:

- **Author-reader topic similarity**: cosine similarity between the average embedding of the author's articles and the average embedding of the reader's highlighted articles
- **Author topic variance**: average pairwise cosine distance among the author's article embeddings (low variance = focused author, high variance = eclectic)
- **Author topic overlap with reader interests**: Jaccard similarity between extracted topics (via LDA or keyword extraction)

---

## 5. Feature Interaction with Embeddings

There are three primary architectures for combining embeddings with handcrafted features, each with distinct tradeoffs.

### 5.1 Concatenation (Early Fusion)

The simplest approach: concatenate the 768-d embedding vector with the N-dimensional handcrafted feature vector to create a single (768+N)-dimensional input.

```
[embedding_768d | metadata_features | structure_features | linguistic_features]
         ↓
    Single Model (e.g., XGBoost, MLP)
         ↓
    Highlight Prediction
```

**Pros:**
- Simple to implement
- Model can learn interactions between embeddings and handcrafted features
- Single training pipeline

**Cons:**
- Dimensionality imbalance: 768 embedding dimensions may dominate ~50 handcrafted features
- Gradient boosting models handle this reasonably well (feature selection is built in), but neural networks may need careful normalization
- Embeddings may need PCA/UMAP reduction to 50--100 dimensions to balance with handcrafted features ([Machine Learning Mastery: Combining XGBoost and Embeddings](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/))

**Recommended mitigation:** Apply PCA to reduce embeddings from 768 to 50--100 dimensions before concatenation, preserving 90--95% of variance. This brings the embedding and handcrafted feature spaces into comparable dimensionality.

### 5.2 Separate Models (Late Fusion)

Train two independent models -- one on embeddings, one on handcrafted features -- and combine their predictions.

```
[embedding_768d]                     [handcrafted_features]
       ↓                                     ↓
  Embedding Model                    Feature Model
  (e.g., MLP or cosine sim)         (e.g., XGBoost)
       ↓                                     ↓
  prediction_embed                    prediction_feat
              ↓           ↓
         Simple Combination
         (weighted average, learned weight)
              ↓
       Highlight Prediction
```

**Pros:**
- Each model can be optimized independently
- Interpretability: handcrafted feature model is fully explainable via SHAP
- Easier to debug which signal source is contributing

**Cons:**
- Cannot learn cross-domain interactions (e.g., "long articles from author X" behaving differently than "long articles from author Y")
- Requires tuning combination weights

### 5.3 Stacked Ensemble (Recommended)

Use a meta-learner that takes predictions from separate models plus raw handcrafted features as input.

```
[embedding_768d]                     [handcrafted_features]
       ↓                                     ↓
  Embedding Model                    Feature Model (Stage 1)
  (MLP or kNN-based)                (XGBoost / LightGBM)
       ↓                                     ↓
  prediction_embed                    prediction_feat
              ↓           ↓           ↓
         Meta-Learner (Stage 2)
         (LightGBM on [pred_embed, pred_feat, key_handcrafted_features])
              ↓
       Highlight Prediction
```

**Pros:**
- Captures complementary strengths of diverse base models ([scikit-learn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html))
- Meta-learner can learn when to trust embeddings vs. handcrafted features
- Stage 1 models can be swapped or upgraded independently
- Raw handcrafted features in stage 2 allow the meta-learner to learn interaction effects
- Best empirical results in analogous tasks: a stacking model between neural networks and XGBoost produced the best F1-score in essay evaluation ([AES Two-Stage Learning](https://arxiv.org/pdf/1901.07744))

**Cons:**
- More complex training pipeline
- Requires proper cross-validation at stage 1 to avoid target leakage (use out-of-fold predictions)

**Implementation detail:** Use scikit-learn's `StackingRegressor` or implement manually with K-fold out-of-fold predictions for the stage 1 models.

---

## 6. Evidence: Hybrid vs. Pure Embeddings

The evidence consistently favors hybrid approaches over pure embedding models:

### 6.1 Automated Essay Scoring (2024)

Maalej et al. (2024) compared pure embedding models against a hybrid combining RoBERTa embeddings with handcrafted linguistic features (grammar errors, readability, sentence length) via LightXGBoost ([MDPI Mathematics](https://www.mdpi.com/2227-7390/12/21/3416)):

| Model | QWK | MSE | RMSE |
|-------|-----|-----|------|
| SVM (handcrafted only) | 0.855 | -- | -- |
| BERT (embeddings only) | 0.918 | 0.115 | 0.195 |
| RoBERTa (embeddings only) | 0.927 | 0.101 | 0.179 |
| **LwXGBoost (hybrid)** | **0.941** | **0.081** | **0.159** |

The hybrid model's advantage: "LwXGBoost's ability to integrate both handcrafted linguistic features and RoBERTa embeddings allows it to capture complex feature interactions that are often missed by models that rely purely on embeddings or handcrafted features individually."

### 6.2 Text Readability Estimation

Research on readability estimation found that a BERT+Readability hybrid model achieved an RMSE of 0.30, compared to 0.44 for BERT alone -- a 32% improvement from adding handcrafted readability features ([Martinc et al., Computational Linguistics](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to)).

### 6.3 Text Similarity Detection

A hybrid combining TF-IDF features with SBERT embeddings achieved F1 = 0.903, outperforming TF-IDF alone (F1 = 0.871) and demonstrating that even mature embedding models benefit from complementary features ([Preprints.org, 2025](https://www.preprints.org/manuscript/202510.2427)).

### 6.4 Text Classification with Embeddings and Gradient Boosting

Keras documentation demonstrates that a Gradient Boosted Tree model with pretrained embeddings achieved 81.6% accuracy vs. 54.4% without embeddings, but the key insight is that adding structured features on top of embeddings provides additional gains ([Keras: Text Classification with Decision Forests](https://keras.io/examples/nlp/tweet-classification-using-tfdf/)).

### 6.5 Display Advertising Engagement

Chen, Mitra, and Swaminathan (2020) showed that metadata features capturing visual appearance and contextual fit significantly improved user engagement prediction beyond content features alone ([SIGIR 2020](https://dl.acm.org/doi/abs/10.1145/3397271.3401201)).

### 6.6 Why Hybrid Works

The fundamental reason hybrid outperforms pure embeddings: embeddings and handcrafted features encode complementary information.

- **Embeddings capture**: semantic meaning, topic, writing style (implicitly), discourse structure (partially)
- **Embeddings miss**: exact word count, publication metadata, author identity, readability level, structural formatting (headers, lists), historical engagement patterns, temporal signals

Handcrafted features fill precisely these gaps. The information is not redundant -- it is additive.

---

## 7. Feature Importance Analysis Techniques

Understanding which features drive predictions is critical for iterating on the feature set and building trust in the model.

### 7.1 SHAP (Shapley Additive Explanations)

SHAP provides both local (per-prediction) and global (dataset-wide) feature importance based on cooperative game theory. Global importance is computed as the mean of absolute Shapley values across all instances: `I_j = (1/n) * sum(|phi_j^(i)|)` ([Molnar, Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/shap.html)).

**TreeSHAP** is the optimized variant for gradient-boosted models (XGBoost, LightGBM, CatBoost). It reduces computational complexity from exponential to polynomial: O(TLD^2) where T = trees, L = leaves, D = depth. This makes it practical for models with hundreds of features.

Key SHAP outputs for our use case:

- **Global feature importance bar plot**: identifies top features overall
- **SHAP summary plot (beeswarm)**: shows feature value distributions and their directional impact on predictions
- **SHAP dependence plots**: reveal interaction effects (e.g., how word count's impact varies by author engagement rate)
- **SHAP interaction values**: explicitly quantify pairwise feature interactions

**Practical note for hybrid models:** When using PCA-reduced embeddings, SHAP values apply to the PCA components, which are not directly interpretable. To maintain interpretability, report SHAP values for the handcrafted features alongside an aggregate "embedding signal" importance (sum of absolute SHAP values across all PCA components).

### 7.2 Permutation Feature Importance

Permutation importance measures how much model error increases when a feature is shuffled, breaking its relationship with the target. It is model-agnostic and works with any trained model ([Molnar, Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/feature-importance.html)).

**Advantages over SHAP:**
- Directly measures impact on predictive performance (not just attribution)
- More robust to overfitting detection (SHAP may assign importance to spurious features)
- Faster for quick feature screening

**Disadvantages:**
- Does not capture directional effects (only magnitude)
- Correlated features inflate each other's importance (permuting one still leaves the correlated feature intact)
- Results vary across random seeds; use multiple repetitions (N >= 10)

**Recommendation for correlated features:** Group correlated features (e.g., all readability scores together, all author features together) and permute them as a block. This gives a truer picture of feature group importance.

### 7.3 Practical Feature Importance Workflow

1. **Initial screening**: Use permutation importance on a quick LightGBM model to identify clearly irrelevant features (importance near zero). Remove them.
2. **Detailed analysis**: Run TreeSHAP on the refined model to understand feature interactions and directional effects.
3. **Feature group analysis**: Compute group-level SHAP importance for feature categories (metadata, structure, linguistic, author, embedding) to understand which signal families matter most.
4. **Ablation studies**: Train the model with each feature group removed and compare holdout performance. This is the most reliable way to quantify each group's marginal contribution.

---

## 8. Practical Feature Engineering Pipeline

### 8.1 Recommended Python Libraries

| Task | Library | Notes |
|------|---------|-------|
| Readability scores | `textstat` | 7+ readability indices out of the box |
| NER, POS tagging, tokenization | `spacy` (en_core_web_sm or en_core_web_md) | Industrial-strength pipeline; use `nlp.pipe()` for batch processing |
| Sentiment/subjectivity | `textblob` or `vaderSentiment` | VADER preferred for short texts; TextBlob for subjectivity |
| Vocabulary richness | `lexicalrichness` | MTLD and HD-D (length-robust TTR alternatives) |
| HTML structure parsing | `beautifulsoup4` | Extract headers, lists, blockquotes, code blocks from article HTML |
| Feature importance | `shap` | TreeSHAP for gradient-boosted models |
| Dimensionality reduction | `scikit-learn` (PCA, UMAP via `umap-learn`) | Reduce 768d embeddings before concatenation |
| Gradient boosting | `lightgbm` or `xgboost` | LightGBM preferred for speed; CatBoost for native categorical handling |

### 8.2 Pipeline Architecture

```python
# Pseudocode for the feature engineering pipeline

class ArticleFeatureExtractor:
    """Extract non-embedding features from article content and metadata."""

    def extract_metadata_features(self, article) -> dict:
        """Word count, author history, domain stats, timing."""
        return {
            "log_word_count": log1p(article.word_count),
            "word_count_bucket": bucketize(article.word_count),
            "est_reading_minutes": article.word_count / 238,
            "author_highlight_rate": get_author_highlight_rate(article.author),
            "author_highlight_density": get_author_highlight_density(article.author),
            "author_article_count": log1p(get_author_article_count(article.author)),
            "domain_highlight_rate": get_domain_highlight_rate(article.domain),
            "domain_article_count": log1p(get_domain_article_count(article.domain)),
            "category_onehot": one_hot_encode(article.category),
            "article_age_hours": hours_since(article.published_date),
            "day_of_week_sin": sin(2 * pi * article.pub_dow / 7),
            "day_of_week_cos": cos(2 * pi * article.pub_dow / 7),
        }

    def extract_structure_features(self, html_content: str) -> dict:
        """Parse HTML for structural signals."""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        word_count = len(text.split())
        return {
            "paragraph_count": len(soup.find_all("p")),
            "avg_paragraph_length": mean_paragraph_words(soup),
            "header_count": len(soup.find_all(["h1","h2","h3","h4","h5","h6"])),
            "header_density_per_1k": header_count / max(word_count, 1) * 1000,
            "list_item_count": len(soup.find_all("li")),
            "list_density_per_1k": list_item_count / max(word_count, 1) * 1000,
            "code_block_count": len(soup.find_all(["pre", "code"])),
            "blockquote_count": len(soup.find_all("blockquote")),
            "image_count": len(soup.find_all("img")),
            "link_count": len(soup.find_all("a")),
        }

    def extract_readability_features(self, text: str) -> dict:
        """Compute readability scores via textstat."""
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "dale_chall_score": textstat.dale_chall_readability_score(text),
            "difficult_word_count": textstat.difficult_words(text),
            "avg_sentence_length": textstat.avg_sentence_length(text),
            "avg_syllables_per_word": textstat.avg_syllables_per_word(text),
        }

    def extract_linguistic_features(self, text: str) -> dict:
        """NER, sentiment, vocabulary richness via spaCy + TextBlob."""
        doc = nlp(text)
        blob = TextBlob(text)
        lex = LexicalRichness(text)
        entities = list(doc.ents)
        return {
            "entity_count": len(entities),
            "entity_density_per_1k": len(entities) / max(len(text.split()), 1) * 1000,
            "unique_entity_count": len(set(e.text for e in entities)),
            "person_entity_count": sum(1 for e in entities if e.label_ == "PERSON"),
            "org_entity_count": sum(1 for e in entities if e.label_ == "ORG"),
            "sentiment_polarity": blob.sentiment.polarity,
            "sentiment_subjectivity": blob.sentiment.subjectivity,
            "sentiment_variance": std([s.sentiment.polarity for s in blob.sentences]),
            "ttr_guiraud": lex.guiraud,  # length-normalized TTR
            "mtld": lex.mtld(threshold=0.72),
            "avg_word_length": mean(len(w) for w in text.split()),
        }
```

### 8.3 Feature Computation Strategy

1. **Compute at triage time**: Metadata features (word count, author stats, domain stats, timing) are cheap to compute and should be extracted when articles enter the pipeline.
2. **Compute in batch**: Structure and linguistic features require parsing article content. Run these in batch using `spacy.pipe()` for efficiency (buffer texts in batches rather than one-by-one, per [spaCy best practices](https://spacy.io/usage/processing-pipelines)).
3. **Cache aggressively**: Features derived from article content are immutable. Compute once, store in the database alongside the article record.
4. **Update rolling features**: Author and domain aggregate features (highlight rates, article counts) should be refreshed periodically (e.g., after each reading session).

### 8.4 Feature Preprocessing

- **Numeric features**: StandardScaler or RobustScaler (the latter is better for features with outliers like word count)
- **Categorical features**: One-hot encoding for low-cardinality (category), target encoding for high-cardinality (author, domain) -- use proper cross-validated target encoding to avoid leakage
- **Missing values**: For author/domain features on cold-start items, use global median imputation or a dedicated "unknown" indicator feature
- **Embedding reduction**: PCA to 50--100 components; fit PCA on training set only, transform at inference

### 8.5 Recommended Model Configuration

```python
# Stacked ensemble configuration
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# Stage 1: Base models
embedding_model = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    input_dim=100,  # PCA-reduced embeddings
)
feature_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    min_child_samples=20,
)

# Stage 2: Meta-learner
meta_learner = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=15,  # Deliberately simple to avoid overfitting
)

# Assemble stacking ensemble
ensemble = StackingRegressor(
    estimators=[
        ("embedding_mlp", embedding_model),
        ("feature_lgbm", feature_model),
    ],
    final_estimator=meta_learner,
    passthrough=True,  # Pass raw features to meta-learner
    cv=5,  # Out-of-fold predictions to avoid leakage
)
```

### 8.6 Iteration Roadmap

Given the current system already uses 768d embeddings with Claude-based scoring, a practical rollout order:

1. **Phase 1 -- Low-hanging fruit (days):** Add metadata features (log word count, author highlight rate, domain highlight rate, category one-hot). These require no content parsing and likely provide the largest initial lift.
2. **Phase 2 -- Content parsing (1 week):** Add readability scores (textstat) and structure features (BeautifulSoup). These require article HTML/text access but are computationally cheap.
3. **Phase 3 -- Linguistic depth (1--2 weeks):** Add NER density, sentiment, vocabulary richness. Requires spaCy pipeline integration.
4. **Phase 4 -- Stacking (1 week):** Implement the stacked ensemble architecture with proper cross-validation.
5. **Phase 5 -- Analysis and pruning (ongoing):** Run SHAP analysis, remove low-importance features, tune interaction features.

---

## 9. Sources

- [Maalej et al. (2024). Hybrid Approach to Automated Essay Scoring: Integrating Deep Learning Embeddings with Handcrafted Linguistic Features for Improved Accuracy. MDPI Mathematics.](https://www.mdpi.com/2227-7390/12/21/3416)
- [Martinc et al. (2021). Supervised and Unsupervised Neural Approaches to Text Readability. Computational Linguistics, MIT Press.](https://direct.mit.edu/coli/article/47/1/141/97334/Supervised-and-Unsupervised-Neural-Approaches-to)
- [Chen, Mitra, Swaminathan (2020). Metadata Matters in User Engagement Prediction. SIGIR 2020.](https://dl.acm.org/doi/abs/10.1145/3397271.3401201)
- [Chartbeat (2022). Is There an Optimal Article Length? The Relationship Between Word Count and Engagement.](https://chartbeat.com/resources/articles/is-there-an-optimal-article-length-the-relationship-between-word-count-and-engagement/)
- [Pew Research Center (2016). Long-Form Reading on Cellphones.](https://www.pewresearch.org/journalism/2016/05/05/2-analysis-suggests-some-readers-willingness-to-dig-into-long-form-news-on-cellphones/)
- [Molnar, C. (2022). Interpretable Machine Learning: SHAP.](https://christophm.github.io/interpretable-ml-book/shap.html)
- [Molnar, C. (2022). Interpretable Machine Learning: Permutation Feature Importance.](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
- [scikit-learn. Ensemble Methods: Stacking.](https://scikit-learn.org/stable/modules/ensemble.html)
- [Machine Learning Mastery. Combining XGBoost and Embeddings: Hybrid Semantic Boosted Trees.](https://machinelearningmastery.com/combining-xgboost-and-embeddings-hybrid-semantic-boosted-trees/)
- [Keras Documentation. Text Classification Using Decision Forests and Pretrained Embeddings.](https://keras.io/examples/nlp/tweet-classification-using-tfdf/)
- [textstat Python Library.](https://pypi.org/project/textstat/)
- [lexicalrichness Python Library.](https://pypi.org/project/lexicalrichness/)
- [spaCy Linguistic Features Documentation.](https://spacy.io/usage/linguistic-features)
- [Wikipedia. Lexical Diversity.](https://en.wikipedia.org/wiki/Lexical_diversity)
- [Wikipedia. Cold Start (Recommender Systems).](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))
- [Liu, B. Sentiment Analysis and Subjectivity. NLP Handbook.](https://www.cs.uic.edu/~liub/FBS/NLP-handbook-sentiment-analysis.pdf)
- [Hashmeta (2025). Finding the Best Times to Post on Social Media.](https://hashmeta.com/blog/unlock-peak-engagement-finding-the-absolute-best-times-to-post-on-social-media-in-2025/)
- [ThatWare. Named Entity Recognition Enhanced Ranking.](https://thatware.co/named-entity-recognition/)
- [Sketch Engine. Type-Token Ratio (TTR).](https://www.sketchengine.eu/glossary/type-token-ratio-ttr/)
- [Preprints.org (2025). A Hybrid TF-IDF and SBERT Approach for Enhanced Text Classification Performance.](https://www.preprints.org/manuscript/202510.2427)
- [Vinija's Notes. Recommendation Systems: Cold Start.](https://vinija.ai/recsys/cold-start/)
