# Scoring Calibration Toolkit

CLI tools for comparing predicted article scores against actual user engagement (Readwise highlights) to calibrate the scoring model over time.

## Commands

All commands are available via `just cal-*` or `python -m tools.calibrate <subcommand>`.

### Sync highlights

Fetches highlight counts from Readwise and caches locally (1hr TTL).

```bash
just cal-sync
```

Run this first before any analysis to ensure highlight data is fresh.

### Calibration report

Overall calibration health: Spearman correlation, tier accuracy, category/tag breakdown.

```bash
just cal-report
just cal-report --since 2025-06-01        # Date range filter
just cal-report --tag ai-agents           # Filter to one tag
just cal-report --category article        # Filter by category
just cal-report --min-progress 0.1        # Exclude unread articles
```

### Dimension analysis

Which of the 4 scoring dimensions best predict actual engagement, redundancy between dimensions, OLS regression.

```bash
just cal-dimensions
just cal-dimensions --min-progress 0.1
```

### Temporal trends

Rolling correlation over time, score drift, engagement drift, monthly summary table.

```bash
just cal-trends
just cal-trends --window 60               # 60-day rolling window (default 30)
```

### Miss analysis

Find articles where score diverges most from engagement. False positives (high score, no highlights) and false negatives (low score, many highlights).

```bash
just cal-misses
just cal-misses --type fp --top 10        # Just false positives
just cal-misses --type fn --top 5         # Just false negatives
just cal-misses --since 2025-06-01
```

### Article inspection

Deep dive on a single article: metadata, score breakdown with visual dimension bars, engagement, assessment text.

```bash
just cal-inspect <article_id>
just cal-inspect <article_id> --content   # Also fetch and display article text
```

## Typical workflow

1. `just cal-sync` — refresh highlight data
2. `just cal-report --min-progress 0.1` — check overall calibration health
3. `just cal-dimensions` — identify which dimensions predict well vs poorly
4. `just cal-misses` — find the biggest scoring errors
5. `just cal-inspect <id> --content` — review individual misses to understand root causes

## Files

| File | Purpose |
|------|---------|
| `tools/calibrate.py` | CLI entry point (argparse subcommands) |
| `tools/cal_data.py` | Data layer: DB scores + Readwise highlights, unified DataFrame |
| `tools/cal_report.py` | Statistical reports: correlations, dimensions, trends |
| `tools/cal_review.py` | Miss analysis: FP/FN finder, article inspector, theme patterns |

## Scoring dimensions reference

Scores are 0-100 across four dimensions (25 points each). DB column names retain legacy names:

| Dimension | DB Column | What it measures |
|-----------|-----------|-----------------|
| Quotability | `specificity_score` | Memorable passages worth saving |
| Surprise Factor | `novelty_score` | Challenges assumptions, unexpected findings |
| Argument Quality | `depth_score` | Well-supported claims, strong opinions |
| Applicable Insight | `actionability_score` | Frameworks, mental models, techniques |

Tiers: High >= 60, Medium 30-59, Low < 30.
