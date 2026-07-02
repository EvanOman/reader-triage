# Calibration Log

One bounded calibration pass (July 2026). Baseline first, at most two improvement
attempts, then stop per the stopping rules. After this pass, calibration is passive:
cal-report numbers ride along in the weekly roundup; reopen only if top-tier
precision drops below 50% over a rolling month, or after 90 days of 👍/👎 digest
feedback.

## Baseline — 2026-07-01 (v2-categorical, incumbent)

Data: `just cal-sync` (159/1810 articles matched highlights), then
`cal-report --min-progress 0.1` → 80 opened articles, Jan–Jun 2026.

### Headline numbers

| Metric | Value |
|---|---|
| Spearman ρ (score vs highlights, opened only) | **+0.35** (p = 0.001) |
| Top-tier precision (High ≥60, opened, ≥1 highlight) | **64.4%** (73 articles) |
| Medium tier engaged | 25.0% (n=4) |
| Low tier engaged | 0.0% (n=3) |
| Tier gradient | monotonic ✓ |
| Score distribution (opened) | mean 81.6, median 89 — **91% score High** |

### Dimension predictiveness (`cal-dimensions --min-progress 0.1`)

| Dimension | Spearman ρ | p | Verdict |
|---|---|---|---|
| Quotability | +0.42 | 0.000 | best predictor |
| Applicable Insight | +0.34 | 0.002 | moderate |
| Surprise Factor | +0.21 | 0.057 | weak |
| Argument Quality | **+0.06** | 0.571 | non-predictive |

No redundant pairs (all inter-dimension |r| < 0.70). OLS R² = 0.14; only
Quotability approaches significance (p = 0.068); Argument Quality has a
*negative* coefficient in the joint model.

### Miss themes (`cal-misses`)

**False positives** (opened — progress 15–100% verified in DB — but zero
highlights): AI-industry / era-of-AI thinkpieces with maxed Surprise/Argument/
Insight (S:25 A:25 I:25 pattern, several perfect 100s). The scorer rewards big
framing; the reader reads them but saves nothing. FP #1–2 scored 100 with 0
highlights.

**False negatives**: 9 of the top 10 still scored High (67–84) — these are
within-tier rank gaps, not tier misses. Only one true tier miss (Noah Smith
rebuttal, 46/Medium, 3 highlights). Recurring pattern: Quotability pinned at
its middle categorical value (Q:17, sometimes Q:9) on heavily-highlighted
articles, i.e. the best-predicting dimension is under-firing on true positives.

### Reading

The scorer is a decent *ranker* (ρ +0.35, monotonic tiers) but a poor
*discriminator at the top*: 91% of what the user opens scores ≥60, so the High
tier is barely a filter, and digest slots are rationed by `ORDER BY info_score
DESC` among inflated scores. FN tier-miss rate on heavily-highlighted articles
is already low (1/10). The leverage is therefore:

1. Rebalance dimension weights toward Quotability (+0.42) and away from
   Argument Quality (+0.06, negative joint coefficient) — pure post-hoc
   re-weighting of already-stored dimension scores, no rescoring cost.
2. The 25-point categorical quantization (9/17/25) compresses Quotability's
   range exactly where it matters.

### Bias note

All FP analysis conditioned on exposure: the 7 top FPs each had reading
progress ≥ 0.15, so they are genuine "read but didn't save" negatives.
Never-opened articles excluded throughout (min-progress 0.1). 143/348 backtest
articles were never opened and are treated as unlabeled, not negative.

## Attempt 1 — 2026-07-01: weight recalibration (v5-reweighted) — **WIN**

Per priority order: threshold/weight recalibration from miss analysis, using
the stored v2 dimension subscores (zero LLM cost). Evaluation script preserved
the session scratchpad; protocol summarized here.

### What was tried

- **Fitted models** (logistic on engaged, ridge on log1p(highlights),
  per-fold correlation weights), all LOOCV: **every fitted model
  underperformed the incumbent out-of-sample** (ρ +0.18 to +0.31 vs +0.35).
  n=80 is too small to fit even 4 coefficients — consistent with research
  docs 08–10, which put reliable ML recalibration at 100–200+ labels.
- **Fixed reweightings** (no fitting): drop-argument, ρ-ordered weights,
  Q-heavy, Q+I-only, Q-only. Winner: **Q+I only** — keep the two
  individually significant dimensions (Quotability ρ=+0.42 p<.001,
  Applicable Insight ρ=+0.34 p=.002), drop the two non-significant ones
  (Surprise p=.057, Argument p=.57). `total = (Q + I) * 2`.
- **Threshold-only recalibration** rejected: raising the High cutoff on v2
  scores buys precision only by excluding heavily-highlighted articles
  (t=85 → FN-heavy 33%).

### Candidate vs incumbent (80 opened articles, matched 91% coverage)

| Metric | incumbent v2 | Q+I (v5) | verdict |
|---|---|---|---|
| Spearman ρ vs highlights | +0.350 | **+0.421** | ✓ |
| Top-tier precision | 0.644 | **0.649** | ✓ (marginal) |
| FN rate, heavy (≥3 HL) | 0.033 | **0.000** | ✓ |

Selection-bias guard: temporal holdout (train-period selection, late-period
eval at 50/40/30% splits) — Q+I beats incumbent on Spearman in all three
splits (+0.504/+0.423/+0.430 vs +0.487/+0.386/+0.390), precision at parity.

At the production threshold (60, unmatched coverage): precision 0.672 vs
0.644, Spearman ✓, FN-heavy 2/30 vs 1/30 — still 2 of 3. High tier deflates
from 91% → 84% of opened articles (73% → 53% of the full corpus), which is
the intended fix for score inflation.

### Deployment

- `ReweightedCategoricalScoringStrategy` (v5-reweighted) added additively in
  `app/services/scoring_strategy.py`; v2 untouched. Same LLM call, same four
  stored subscores, only the total changes.
- `tools/backfill_v5.py` converted all 879 v2 rows arithmetically (no LLM
  calls); strategy `accepted_versions` prevents the sync loop from
  re-scoring the corpus.
- Post-deploy `cal-report --min-progress 0.1`: **ρ +0.42 (p<0.001),
  top-tier precision 67.2%**, tier gradient monotonic.

## Stop decision — 2026-07-01

**Calibration pass is CLOSED.** Attempt 1 beat the incumbent on 2 of 3
metrics (3 of 3 at matched coverage) and is deployed as v5-reweighted.
Attempt 2 is not used. Per the stopping rules, calibration is now passive:

- cal-report numbers ride in the weekly Telegram roundup.
- Reopen only if top-tier precision drops below 50% over a rolling month,
  or after ~90 days of 👍/👎 digest feedback (then one bounded
  active-learning pass — research doc 10 suggests a Bayesian calibration
  layer becomes viable at ~100–200 dense labels, which the digest feedback
  stream should reach around October 2026).
- Surprise/Argument subscores are still collected and stored, so a future
  pass has full raw material.
