# V4 Scoring Implementation Plan

## Why V4

V3 binary scoring (20 questions, weighted yes/no) failed to improve on V2 categorical scoring:

| Metric | V2 | V3 |
|--------|----|----|
| Spearman rho (highlighted_words) | 0.106 | 0.044 |
| Median score | 83 | 92 |
| P25 score | — | 86 |
| Positive Qs with pass rate >80% | N/A | 12 of 16 |
| Qs with significant engagement correlation | N/A | 1 (q4, r=0.205) |

**Root cause:** V3 questions are too easy. The LLM says "yes" to nearly everything. 12 of 16 positive questions have pass rates above 80%. The 4 negative questions fire at only 5-6% (target: 20-30%). Scores cluster at 86-100 with no useful discrimination.

## V4 Design

### Core changes from V3

1. **24 questions** (up from 20), 6 per dimension, difficulty-tiered
2. **Tiered difficulty** with weight proportional to difficulty:
   - Easy (5 Qs, target p=0.65-0.80): 3-4 pts each
   - Medium (9 Qs, target p=0.35-0.55): 5-7 pts each
   - Hard (6 Qs, target p=0.10-0.25): 8-10 pts each
   - Negative (4 Qs, target p=0.15-0.35 for "yes"): -5 to -8 pts each
3. **Broader negative questions** — target common patterns ("restates well-known ideas") not rare ones ("is a news article")
4. **Evidence grounding** — require the LLM to cite a specific passage for each "yes" answer
5. **Calibration instruction** in prompt — "Most articles should pass 10-14 of 24 questions"
6. **temperature=0** — eliminate random variance
7. **Reasoning before answer** in response JSON — genuine CoT not post-hoc rationalization

### Question set

From research doc `06 - Designing Discriminative Binary Questions.md`, Part 8.

#### Dimension 1: Quotability

| ID | Tier | Weight | Question |
|----|------|--------|----------|
| Q1 | Easy | 4 | Contains at least one specific claim, data point, or concrete example beyond abstract generalization? |
| Q2 | Medium | 6 | Contains a sentence/passage memorable for its *phrasing* — a striking metaphor, crisp formulation, or vivid example? |
| Q3 | Medium | 6 | Cites a specific quantified finding (number, percentage, research result) worth remembering on its own? |
| Q4 | Hard | 9 | Could you extract a self-contained 2-4 sentence passage that works as a standalone note without surrounding context? |
| Q5 | Medium | 5 | Contains a direct, attributed statement from a named individual that adds credibility or insight? |
| Q6 | Negative | -6 | Is the article's value primarily in linking to or summarizing other sources rather than its own prose? |

#### Dimension 2: Surprise Factor

| ID | Tier | Weight | Question |
|----|------|--------|----------|
| Q7 | Easy | 4 | Addresses a real topic with enough depth for substantive discussion with a colleague? |
| Q8 | Medium | 6 | Presents a finding/conclusion that directly contradicts or complicates a widely-held assumption? |
| Q9 | Hard | 10 | Reframes a familiar topic through an unexpected domain, analogy, or historical parallel? |
| Q10 | Medium | 6 | Contains a case study/example the reader is unlikely to have encountered in prior reading on this topic? |
| Q11 | Medium | 5 | Develops an idea through multiple stages, reaching a conclusion not obvious from its opening premise? |
| Q12 | Negative | -7 | Could the central point be accurately conveyed in a single sentence without significant loss of meaning? |

#### Dimension 3: Argument Quality

| ID | Tier | Weight | Question |
|----|------|--------|----------|
| Q13 | Easy | 3 | Author takes a clear position on a debatable topic where a reasonable person could argue the opposite? |
| Q14 | Medium | 6 | Supports central claim with at least two distinct types of evidence (e.g., personal experience AND data)? |
| Q15 | Medium | 6 | Author writes from direct, first-person professional experience — specific situations, decisions, results? |
| Q16 | Hard | 9 | Engages with and responds to the strongest counterargument rather than ignoring or strawmanning it? |
| Q17 | Hard | 10 | Contains intellectual risk — falsifiable prediction, recommends against popular approach, or admits significant failure? |
| Q18 | Negative | -6 | Is this primarily reporting/explaining what others said or did, rather than advancing the author's own analysis? |

#### Dimension 4: Applicable Insight

| ID | Tier | Weight | Question |
|----|------|--------|----------|
| Q19 | Easy | 4 | Contains at least one idea that could influence how a reader thinks about or approaches a problem? |
| Q20 | Hard | 10 | Introduces, names, or clearly articulates a framework/mental model that organizes thinking about a class of problems? |
| Q21 | Medium | 6 | Provides enough concrete detail that a reader could apply the core idea without seeking additional sources? |
| Q22 | Medium | 5 | Describes a specific technique/practice the reader could try in a concrete situation within the next month? |
| Q23 | Hard | 9 | Offers a perspective shift that would persistently change the reader's default interpretation or approach? |
| Q24 | Negative | -5 | Is the content so narrowly domain-specific that it's useful only to practitioners of one specialized profession? |

#### Weight summary

- Max positive: 4+6+6+9+5 + 4+6+10+6+5 + 3+6+6+9+10 + 4+10+6+5+9 = **129**
- Max negative: -6 + -7 + -6 + -5 = **-24**
- Raw range: [-24, 129]
- Scaled linearly to 0-100

### Expected score distribution

| Range | V3 Actual | V4 Target |
|-------|-----------|-----------|
| 0-19 | ~2% | 10-15% |
| 20-39 | ~3% | 25-30% |
| 40-59 | ~5% | 30-35% |
| 60-79 | ~15% | 15-20% |
| 80-100 | ~75% | 5-10% |

### Prompt structure

```
System prompt: Rubric definition + calibration instruction + dimension descriptions

User prompt: Article metadata + content

Response format: JSON with evidence-first structure:
{
  "q1_evidence": "The author states '67% of startups...'",
  "q1": true,
  "q2_evidence": "",
  "q2": false,
  ...
  "overall_assessment": "1-2 sentence summary"
}
```

Calibration instruction:
> Be critical and discriminating. Most articles should pass 10-14 of 24 questions. An article passing more than 20 is exceptional and rare. Easy questions test minimum quality — most articles pass. Hard questions test for excellence — most articles fail. If you find yourself answering "yes" to nearly every question, you are not applying the criteria strictly enough.

## Implementation Steps

### Step 1: Add V4ScoringStrategy class

Add `V4ScoringStrategy` to `app/services/scoring_strategy.py`:
- New prompt templates (article + podcast variants) with 24 questions
- System prompt with rubric, calibration instruction, difficulty-tier context
- `temperature=0` on the API call
- Evidence-first JSON response format
- New weight table: `V4_WEIGHTS` dict mapping q1-q24 to signed integers
- New dimension mapping: `V4_DIMENSION_QUESTIONS`
- `compute_v4_total()` and `compute_v4_dimension()` functions (same linear scaling pattern as V3)
- `max_tokens=2000` (24 Qs with evidence needs more output space)

### Step 2: Add v4 scoring version support

- Add `"v4-binary"` as a valid `scoring_version` in the article model
- Create `article_scores_v4` table (same schema as `article_scores_v3`) or reuse `article_scores_v3` with `scoring_version='v4-binary'`
- Wire up the new strategy in the scorer service so it can be selected

### Step 3: Add rescore tooling

- Add `just rescore-v4` recipe that scores existing articles with V4
- Start with a batch of ~100 articles that have highlight data for immediate calibration comparison

### Step 4: Validate

- Compare V4 Spearman rho against V2 (0.106) and V3 (0.044)
- Check actual pass rates per question — verify they hit target tiers
- Check score distribution — verify spread matches target
- Run IRT analysis: drop/revise questions with pass rate >80% or <5%

### Step 5: Make V4 default (if validated)

- Set V4 as the default scoring strategy for new articles
- Update analytics page calibration charts to include V4

## Dead ends (do NOT pursue)

- Pairwise comparison scoring (O(n^2), impractical)
- Multi-agent debate architecture (3x cost; embed insight as devil's advocate Qs)
- Fine-tuning a judge model (too few labeled articles, evolving rubric)
- Self-consistency voting (cost-prohibitive in production)
- Embedding-based ML scoring (requires separate pipeline, not enough training data)
