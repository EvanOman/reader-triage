"""Scoring calibration reports — statistical analysis of predicted scores vs actual engagement.

Compares AI-generated article scores against user highlight counts from Readwise
to assess how well the scoring system predicts real engagement.

Usage:
    uv run python -m tools.cal_report report [--since DATE] [--until DATE] [--min-progress 0.1]
    uv run python -m tools.cal_report dimensions [--since DATE] [--min-progress 0.1]
    uv run python -m tools.cal_report trends [--window 30] [--min-progress 0.1]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = ["specificity_score", "novelty_score", "depth_score", "actionability_score"]
DIM_LABELS = {
    "specificity_score": "Quotability",
    "novelty_score": "Surprise Factor",
    "depth_score": "Argument Quality",
    "actionability_score": "Applicable Insight",
}

# Tier thresholds
TIER_HIGH = 60
TIER_LOW = 30

MIN_CORR_N = 5  # minimum articles for a meaningful correlation

# ANSI color codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _header(title: str, width: int = 55) -> str:
    """Return a section header line."""
    pad = width - len(title) - 3
    return f"\n{C_CYAN}{C_BOLD}\u2500\u2500 {title} {'\u2500' * max(pad, 2)}{C_RESET}"


def _banner(lines: list[str], width: int = 55) -> str:
    """Return a top-level report banner."""
    bar = "\u2550" * width
    body = "\n".join(f"  {line}" for line in lines)
    return f"\n{C_BOLD}{bar}\n{body}\n{bar}{C_RESET}"


def _color_rho(rho: float) -> str:
    """Color a Spearman rho value based on strength."""
    abs_rho = abs(rho)
    if abs_rho >= 0.3:
        return f"{C_GREEN}{rho:+.2f}{C_RESET}"
    elif abs_rho >= 0.15:
        return f"{C_YELLOW}{rho:+.2f}{C_RESET}"
    else:
        return f"{C_RED}{rho:+.2f}{C_RESET}"


def _color_pval(p: float) -> str:
    """Color a p-value based on significance."""
    if p < 0.01:
        return f"{C_GREEN}{p:.3f}{C_RESET}"
    elif p < 0.05:
        return f"{C_YELLOW}{p:.3f}{C_RESET}"
    else:
        return f"{C_RED}{p:.3f}{C_RESET}"


def _interpret_rho(rho: float) -> str:
    """Return a plain-English interpretation of Spearman rho."""
    abs_rho = abs(rho)
    if abs_rho >= 0.5:
        return "STRONG"
    elif abs_rho >= 0.3:
        return "MODERATE"
    elif abs_rho >= 0.15:
        return "WEAK"
    else:
        return "NEGLIGIBLE"


def _color_verdict(verdict: str) -> str:
    """Color a calibration verdict."""
    if verdict == "GOOD":
        return f"{C_GREEN}{C_BOLD}{verdict}{C_RESET}"
    elif verdict == "FAIR":
        return f"{C_YELLOW}{C_BOLD}{verdict}{C_RESET}"
    else:
        return f"{C_RED}{C_BOLD}{verdict}{C_RESET}"


def _pct(n: int, total: int) -> str:
    """Format a percentage safely."""
    if total == 0:
        return "  n/a"
    return f"{100.0 * n / total:5.1f}%"


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    """Compute Spearman correlation, returning None if not enough data."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < MIN_CORR_N:
        return None
    # spearmanr needs variance in both series
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return None
    rho, p = spearmanr(valid["x"], valid["y"])
    return float(rho), float(p)


def _safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    """Compute Pearson correlation, returning None if not enough data."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < MIN_CORR_N:
        return None
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return None
    r, p = pearsonr(valid["x"], valid["y"])
    return float(r), float(p)


def _assign_tier(score: float) -> str:
    if score >= TIER_HIGH:
        return "High"
    elif score >= TIER_LOW:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(
    since: str | None = None,
    until: str | None = None,
    min_progress: float = 0.0,
    tag: str | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    """Load dataset from cal_data and apply optional filters."""
    from tools.cal_data import load_dataset

    df = load_dataset(
        since=since,
        until=until,
        min_reading_progress=min_progress,
        exclude_content_failed=True,
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Apply tag filter
    if tag:
        df = df[df["tags"].apply(lambda tags: tag in tags if isinstance(tags, list) else False)]

    # Apply category filter
    if category:
        df = df[df["category"] == category]

    # Add tier column
    if "info_score" in df.columns:
        df["tier"] = df["info_score"].apply(_assign_tier)

    return df.reset_index(drop=True)


def _date_range_str(df: pd.DataFrame) -> str:
    """Return a human-readable date range from added_at."""
    if "added_at" not in df.columns or df["added_at"].isna().all():
        return "unknown dates"
    dates = pd.to_datetime(df["added_at"], errors="coerce").dropna()
    if len(dates) == 0:
        return "unknown dates"
    mn = dates.min().strftime("%b %Y")
    mx = dates.max().strftime("%b %Y")
    if mn == mx:
        return mn
    return f"{mn} \u2013 {mx}"


# ---------------------------------------------------------------------------
# Function 1: run_report
# ---------------------------------------------------------------------------


def run_report(
    since: str | None = None,
    until: str | None = None,
    min_progress: float = 0.0,
    tag: str | None = None,
    category: str | None = None,
) -> None:
    """Print a comprehensive calibration report to stdout."""

    df = _load(since=since, until=until, min_progress=min_progress, tag=tag, category=category)

    if len(df) == 0:
        print(f"\n{C_RED}No articles found matching the given filters.{C_RESET}")
        return

    # ── Banner ──
    filters = []
    if tag:
        filters.append(f"tag={tag}")
    if category:
        filters.append(f"category={category}")
    if min_progress > 0:
        filters.append(f"progress>={min_progress:.0%}")
    filter_str = f"  Filters: {', '.join(filters)}" if filters else ""

    banner_lines = [
        "CALIBRATION REPORT",
        f"{len(df)} articles \u00b7 {_date_range_str(df)}",
    ]
    if filter_str:
        banner_lines.append(filter_str.strip())
    print(_banner(banner_lines))

    # ── Section 1: Dataset Summary ──
    print(_header("Dataset Summary"))

    total = len(df)
    has_highlights = int((df["num_highlights"] > 0).sum()) if "num_highlights" in df.columns else 0
    engagement_rate = 100.0 * has_highlights / total if total > 0 else 0.0

    print(f"  Total articles:      {C_BOLD}{total}{C_RESET}")
    print(f"  With highlights:     {has_highlights}  ({engagement_rate:.1f}%)")

    if "reading_progress" in df.columns:
        rp = df["reading_progress"].dropna()
        if len(rp) > 0:
            print(
                f"  Reading progress:    "
                f"mean {rp.mean():.0%}, median {rp.median():.0%}, "
                f"min {rp.min():.0%}, max {rp.max():.0%}"
            )

    if "info_score" in df.columns:
        sc = df["info_score"].dropna()
        if len(sc) > 0:
            print(
                f"  Score distribution:  "
                f"mean {sc.mean():.1f}, median {sc.median():.1f}, "
                f"std {sc.std():.1f}"
            )

    # Tier distribution
    if "tier" in df.columns:
        for tier_name in ["High", "Medium", "Low"]:
            cnt = int((df["tier"] == tier_name).sum())
            print(f"    {tier_name:>6s}: {cnt:>4d}  ({_pct(cnt, total)})")

    # ── Section 2: Overall Correlation ──
    print(_header("Overall Correlation"))

    if "info_score" not in df.columns or "num_highlights" not in df.columns:
        print(f"  {C_RED}Missing score or highlight data.{C_RESET}")
    else:
        result = _safe_spearman(df["info_score"], df["num_highlights"])
        if result is None:
            print(
                f"  {C_YELLOW}Not enough data for correlation (need {MIN_CORR_N}+ articles).{C_RESET}"
            )
        else:
            rho, p = result
            interp = _interpret_rho(rho)
            print(f"  Spearman \u03c1 = {_color_rho(rho)}  (p = {_color_pval(p)})")
            print(f"  Interpretation: {interp} correlation")

    # ── Section 3: Tier Accuracy ──
    print(_header("Tier Accuracy"))

    if "tier" in df.columns and "num_highlights" in df.columns:
        print(f"  {'Tier':<8s} {'Count':>6s} {'Engaged':>8s} {'Avg Highlights':>15s}")
        print(f"  {'\u2500' * 40}")

        tier_order = ["High", "Medium", "Low"]
        tier_engagement_rates = []

        for tier_name in tier_order:
            subset = df[df["tier"] == tier_name]
            cnt = len(subset)
            if cnt == 0:
                print(f"  {tier_name:<8s} {cnt:>6d} {'n/a':>8s} {'n/a':>15s}")
                continue
            engaged = int((subset["num_highlights"] > 0).sum())
            eng_rate = 100.0 * engaged / cnt
            avg_hl = subset["num_highlights"].mean()
            tier_engagement_rates.append(eng_rate)

            # Color the engagement rate
            if eng_rate >= 60:
                eng_str = f"{C_GREEN}{eng_rate:5.1f}%{C_RESET}"
            elif eng_rate >= 40:
                eng_str = f"{C_YELLOW}{eng_rate:5.1f}%{C_RESET}"
            else:
                eng_str = f"{C_RED}{eng_rate:5.1f}%{C_RESET}"

            print(f"  {tier_name:<8s} {cnt:>6d}  {eng_str:>18s}    {avg_hl:>10.1f}")

        # Check if tiers show the expected gradient (High > Med > Low)
        if len(tier_engagement_rates) >= 2:
            if tier_engagement_rates == sorted(tier_engagement_rates, reverse=True):
                print(
                    f"\n  {C_GREEN}Tier gradient is monotonic (good calibration signal).{C_RESET}"
                )
            else:
                print(
                    f"\n  {C_YELLOW}Tier gradient is NOT monotonic "
                    f"(engagement does not strictly decrease across tiers).{C_RESET}"
                )

    # ── Section 4: Category Breakdown ──
    print(_header("Category Breakdown"))

    if "category" in df.columns and "num_highlights" in df.columns:
        cats = df["category"].dropna().unique()
        if len(cats) == 0:
            print(f"  {C_DIM}No category data available.{C_RESET}")
        else:
            print(
                f"  {'Category':<15s} {'Count':>6s} {'Engaged':>8s} {'Avg HL':>7s} {'Spearman':>9s}"
            )
            print(f"  {'\u2500' * 48}")

            for cat in sorted(cats, key=lambda c: -len(df[df["category"] == c])):
                subset = df[df["category"] == cat]
                cnt = len(subset)
                if cnt == 0:
                    continue
                engaged = int((subset["num_highlights"] > 0).sum())
                eng_rate = 100.0 * engaged / cnt
                avg_hl = subset["num_highlights"].mean()

                corr_result = _safe_spearman(subset["info_score"], subset["num_highlights"])
                corr_str = _color_rho(corr_result[0]) if corr_result else f"{C_DIM}  n/a{C_RESET}"

                print(f"  {cat:<15s} {cnt:>6d}  {eng_rate:>6.1f}%  {avg_hl:>6.1f}  {corr_str:>19s}")
    else:
        print(f"  {C_DIM}No category data available.{C_RESET}")

    # ── Section 5: Tag Performance ──
    if not tag:
        print(_header("Tag Performance"))

        if "tags" in df.columns and "num_highlights" in df.columns:
            # Explode tags to get per-tag stats
            tag_rows = []
            for _, row in df.iterrows():
                tags = row.get("tags", [])
                if isinstance(tags, list):
                    for t in tags:
                        tag_rows.append(
                            {
                                "tag": t,
                                "info_score": row["info_score"],
                                "num_highlights": row["num_highlights"],
                            }
                        )
            if tag_rows:
                tag_df = pd.DataFrame(tag_rows)
                tag_counts = tag_df.groupby("tag").size()
                valid_tags = tag_counts[tag_counts >= 10].index.tolist()

                if valid_tags:
                    print(f"  {'Tag':<20s} {'Count':>6s} {'Avg HL':>7s} {'Spearman':>9s}")
                    print(f"  {'\u2500' * 45}")

                    tag_results = []
                    for t in valid_tags:
                        sub = tag_df[tag_df["tag"] == t]
                        avg_hl = sub["num_highlights"].mean()
                        corr_result = _safe_spearman(sub["info_score"], sub["num_highlights"])
                        rho_val = corr_result[0] if corr_result else None
                        tag_results.append((t, len(sub), avg_hl, rho_val))

                    # Sort by count descending
                    tag_results.sort(key=lambda x: -x[1])

                    for t, cnt, avg_hl, rho_val in tag_results:
                        corr_str = (
                            _color_rho(rho_val) if rho_val is not None else f"{C_DIM}  n/a{C_RESET}"
                        )
                        print(f"  {t:<20s} {cnt:>6d}  {avg_hl:>6.1f}  {corr_str:>19s}")
                else:
                    print(f"  {C_DIM}No tags with 10+ articles found.{C_RESET}")
            else:
                print(f"  {C_DIM}No tag data available.{C_RESET}")
        else:
            print(f"  {C_DIM}No tag data available.{C_RESET}")

    # ── Section 6: Quick Verdict ──
    print(_header("Quick Verdict"))

    if "info_score" in df.columns and "num_highlights" in df.columns:
        result = _safe_spearman(df["info_score"], df["num_highlights"])
        if result is None:
            print(f"  {C_YELLOW}Insufficient data for a verdict.{C_RESET}")
        else:
            rho, p = result
            abs_rho = abs(rho)
            if abs_rho >= 0.3 and p < 0.05:
                verdict = "GOOD"
            elif abs_rho >= 0.15 and p < 0.1:
                verdict = "FAIR"
            else:
                verdict = "POOR"
            print(
                f"  Calibration is {_color_verdict(verdict)} "
                f"based on Spearman \u03c1 = {rho:+.2f} (p = {p:.3f})"
            )
    else:
        print(f"  {C_YELLOW}Cannot compute verdict without score and highlight data.{C_RESET}")

    print()


# ---------------------------------------------------------------------------
# Function 2: run_dimensions
# ---------------------------------------------------------------------------


def run_dimensions(
    since: str | None = None,
    min_progress: float = 0.0,
) -> None:
    """Analyze which scoring dimensions best predict engagement."""

    df = _load(since=since, min_progress=min_progress)

    if len(df) == 0:
        print(f"\n{C_RED}No articles found matching the given filters.{C_RESET}")
        return

    banner_lines = [
        "DIMENSION ANALYSIS",
        f"{len(df)} articles \u00b7 {_date_range_str(df)}",
    ]
    print(_banner(banner_lines))

    # ── Section 1: Per-Dimension Correlations ──
    print(_header("Per-Dimension Correlations"))
    print(f"  {'Dimension':<20s} {'Spearman \u03c1':>12s} {'p-value':>10s} {'Strength':>12s}")
    print(f"  {'\u2500' * 56}")

    dim_correlations: dict[str, tuple[float, float] | None] = {}
    for dim in DIMENSIONS:
        if dim not in df.columns:
            dim_correlations[dim] = None
            continue
        result = _safe_spearman(df[dim], df["num_highlights"])
        dim_correlations[dim] = result
        label = DIM_LABELS.get(dim, dim)
        if result is None:
            print(f"  {label:<20s} {C_DIM}{'n/a':>12s} {'n/a':>10s} {'n/a':>12s}{C_RESET}")
        else:
            rho, p = result
            interp = _interpret_rho(rho)
            print(f"  {label:<20s} {_color_rho(rho):>22s} {_color_pval(p):>20s} {interp:>12s}")

    # Highlight best predictor
    valid_dims = {d: r for d, r in dim_correlations.items() if r is not None}
    if valid_dims:
        best_dim = max(valid_dims, key=lambda d: abs(valid_dims[d][0]))
        best_rho = valid_dims[best_dim][0]
        print(
            f"\n  {C_BOLD}Best predictor:{C_RESET} "
            f"{DIM_LABELS.get(best_dim, best_dim)} (\u03c1 = {_color_rho(best_rho)})"
        )

    # ── Section 2: Dimension Redundancy ──
    print(_header("Dimension Redundancy (Pearson r between dimensions)"))

    available_dims = [d for d in DIMENSIONS if d in df.columns]
    if len(available_dims) >= 2:
        # Header row
        short_labels = {
            "specificity_score": "Quot",
            "novelty_score": "Surp",
            "depth_score": "Arg",
            "actionability_score": "Ins",
        }
        header_row = f"  {'':>20s}"
        for d in available_dims:
            header_row += f"  {short_labels.get(d, d[:4]):>6s}"
        print(header_row)
        print(f"  {'\u2500' * (20 + 8 * len(available_dims))}")

        for i, d1 in enumerate(available_dims):
            row = f"  {DIM_LABELS.get(d1, d1):<20s}"
            for j, d2 in enumerate(available_dims):
                if i == j:
                    row += f"  {C_DIM}{'--':>6s}{C_RESET}"
                elif j < i:
                    # Already printed in upper triangle
                    result = _safe_pearson(df[d1], df[d2])
                    if result is None:
                        row += f"  {C_DIM}{'n/a':>6s}{C_RESET}"
                    else:
                        r, _ = result
                        if abs(r) >= 0.7:
                            row += f"  {C_RED}{r:>6.2f}{C_RESET}"
                        elif abs(r) >= 0.4:
                            row += f"  {C_YELLOW}{r:>6.2f}{C_RESET}"
                        else:
                            row += f"  {C_GREEN}{r:>6.2f}{C_RESET}"
                else:
                    result = _safe_pearson(df[d1], df[d2])
                    if result is None:
                        row += f"  {C_DIM}{'n/a':>6s}{C_RESET}"
                    else:
                        r, _ = result
                        if abs(r) >= 0.7:
                            row += f"  {C_RED}{r:>6.2f}{C_RESET}"
                        elif abs(r) >= 0.4:
                            row += f"  {C_YELLOW}{r:>6.2f}{C_RESET}"
                        else:
                            row += f"  {C_GREEN}{r:>6.2f}{C_RESET}"
            print(row)

        # Flag highly correlated pairs
        flagged = []
        for i, d1 in enumerate(available_dims):
            for j, d2 in enumerate(available_dims):
                if j <= i:
                    continue
                result = _safe_pearson(df[d1], df[d2])
                if result and abs(result[0]) >= 0.7:
                    flagged.append((DIM_LABELS.get(d1, d1), DIM_LABELS.get(d2, d2), result[0]))
        if flagged:
            print(f"\n  {C_RED}{C_BOLD}Redundancy warning:{C_RESET}")
            for d1_label, d2_label, r in flagged:
                print(f"    {d1_label} <-> {d2_label}: r = {r:.2f} (may be measuring same thing)")
        else:
            print(f"\n  {C_GREEN}No highly redundant dimension pairs (all |r| < 0.70).{C_RESET}")
    else:
        print(f"  {C_DIM}Need at least 2 dimensions with data.{C_RESET}")

    # ── Section 3: Multiple Regression ──
    print(_header("Multiple Regression"))

    valid_cols = [d for d in DIMENSIONS if d in df.columns]
    reg_df = df[valid_cols + ["num_highlights"]].dropna()

    if len(reg_df) < len(valid_cols) + 2:
        print(
            f"  {C_YELLOW}Not enough data for regression "
            f"(have {len(reg_df)}, need {len(valid_cols) + 2}+).{C_RESET}"
        )
    else:
        try:
            import statsmodels.api as sm

            X = sm.add_constant(reg_df[valid_cols])
            y = reg_df["num_highlights"]
            model = sm.OLS(y, X).fit()

            print(f"  OLS: num_highlights ~ {' + '.join(DIM_LABELS[d] for d in valid_cols)}")
            print(
                f"  R\u00b2 = {model.rsquared:.4f}   Adj R\u00b2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}"
            )
            print()
            print(f"  {'Variable':<20s} {'Coeff':>8s} {'Std Err':>8s} {'p-value':>10s} {'Sig':>4s}")
            print(f"  {'\u2500' * 52}")

            for i, name in enumerate(model.params.index):
                coef = model.params.iloc[i]
                stderr = model.bse.iloc[i]
                pval = model.pvalues.iloc[i]
                sig = "*" if pval < 0.05 else ""
                label = DIM_LABELS.get(name, name)
                if name == "const":
                    label = "(intercept)"

                pval_str = _color_pval(pval)
                print(f"  {label:<20s} {coef:>8.3f} {stderr:>8.3f} {pval_str:>20s} {sig:>4s}")

            if model.f_pvalue is not None:
                print(f"\n  F-stat p-value: {_color_pval(model.f_pvalue)}")

        except ImportError:
            print(f"  {C_RED}statsmodels not available. Install with: uv add statsmodels{C_RESET}")
        except Exception as e:
            print(f"  {C_RED}Regression failed: {e}{C_RESET}")

    # ── Section 4: Recommendations ──
    print(_header("Recommendations"))

    if valid_dims:
        # Sort dimensions by absolute correlation strength
        ranked = sorted(valid_dims.items(), key=lambda x: abs(x[1][0]), reverse=True)

        strong = [(d, r) for d, (r, p) in ranked if abs(r) >= 0.15 and p < 0.1]
        weak = [(d, r) for d, (r, p) in ranked if abs(r) < 0.10]

        if strong:
            print(f"  {C_GREEN}Weight MORE:{C_RESET}")
            for d, rho in strong:
                print(f"    {DIM_LABELS.get(d, d):<20s} (\u03c1 = {rho:+.2f})")

        if weak:
            print(f"  {C_YELLOW}Consider de-weighting:{C_RESET}")
            for d, rho in weak:
                print(f"    {DIM_LABELS.get(d, d):<20s} (\u03c1 = {rho:+.2f})")

        if not strong and not weak:
            print(f"  {C_DIM}All dimensions show similar predictive power.{C_RESET}")

        # Check for redundancy-based recommendation
        for i, d1 in enumerate(available_dims):
            for j, d2 in enumerate(available_dims):
                if j <= i:
                    continue
                result = _safe_pearson(df[d1], df[d2])
                if result and abs(result[0]) >= 0.7:
                    # Which is the weaker predictor?
                    r1 = dim_correlations.get(d1)
                    r2 = dim_correlations.get(d2)
                    if r1 and r2:
                        weaker = d1 if abs(r1[0]) < abs(r2[0]) else d2
                        print(
                            f"  {C_YELLOW}Consider merging/dropping {DIM_LABELS.get(weaker, weaker)} "
                            f"(redundant with {DIM_LABELS.get(d1 if weaker == d2 else d2, '')}).{C_RESET}"
                        )
    else:
        print(f"  {C_DIM}Insufficient data for recommendations.{C_RESET}")

    print()


# ---------------------------------------------------------------------------
# Function 3: run_trends
# ---------------------------------------------------------------------------


def run_trends(
    window: int = 30,
    min_progress: float = 0.0,
) -> None:
    """Temporal calibration analysis: how does scoring quality change over time?"""

    df = _load(min_progress=min_progress)

    if len(df) == 0:
        print(f"\n{C_RED}No articles found matching the given filters.{C_RESET}")
        return

    # Parse dates
    if "added_at" not in df.columns:
        print(f"\n{C_RED}No added_at column found — cannot do temporal analysis.{C_RESET}")
        return

    df["added_date"] = pd.to_datetime(df["added_at"], errors="coerce")
    df = df.dropna(subset=["added_date"]).sort_values("added_date").reset_index(drop=True)

    if len(df) == 0:
        print(f"\n{C_RED}No articles with valid dates found.{C_RESET}")
        return

    banner_lines = [
        "TEMPORAL TRENDS",
        f"{len(df)} articles \u00b7 {_date_range_str(df)}",
        f"Rolling window: {window} days",
    ]
    print(_banner(banner_lines))

    # ── Section 1: Rolling Correlation ──
    print(_header("Rolling Correlation"))

    if "info_score" not in df.columns or "num_highlights" not in df.columns:
        print(f"  {C_RED}Missing score or highlight data.{C_RESET}")
    else:
        # Compute rolling Spearman over time windows
        df_sorted = df.sort_values("added_date")
        window_td = pd.Timedelta(days=window)

        # Sample at regular intervals to show progression
        date_min = df_sorted["added_date"].min()
        date_max = df_sorted["added_date"].max()
        total_days = (date_max - date_min).days

        if total_days < window:
            print(
                f"  {C_YELLOW}Date range ({total_days} days) is shorter than window "
                f"({window} days). Showing single correlation.{C_RESET}"
            )
            result = _safe_spearman(df["info_score"], df["num_highlights"])
            if result:
                rho, p = result
                print(
                    f"  Overall: \u03c1 = {_color_rho(rho)} (p = {_color_pval(p)}, n = {len(df)})"
                )
        else:
            # Step through in increments of window/2 for overlap
            step_days = max(window // 2, 7)
            rolling_results = []

            current_start = date_min
            while current_start + window_td <= date_max:
                current_end = current_start + window_td
                mask = (df_sorted["added_date"] >= current_start) & (
                    df_sorted["added_date"] < current_end
                )
                subset = df_sorted[mask]
                if len(subset) >= MIN_CORR_N:
                    result = _safe_spearman(subset["info_score"], subset["num_highlights"])
                    if result:
                        rho, p = result
                        mid_date = current_start + window_td / 2
                        rolling_results.append(
                            {
                                "date": mid_date,
                                "rho": rho,
                                "p": p,
                                "n": len(subset),
                            }
                        )
                current_start += pd.Timedelta(days=step_days)

            if rolling_results:
                print(f"  {'Period Center':<16s} {'\u03c1':>6s} {'p':>7s} {'n':>5s}")
                print(f"  {'\u2500' * 36}")

                for r in rolling_results:
                    date_str = r["date"].strftime("%Y-%m-%d")
                    print(
                        f"  {date_str:<16s} {_color_rho(r['rho']):>16s} "
                        f"{_color_pval(r['p']):>17s} {r['n']:>5d}"
                    )

                # Trend in rho over time
                rhos = [r["rho"] for r in rolling_results]
                if len(rhos) >= 3:
                    trend_result = _safe_spearman(pd.Series(range(len(rhos))), pd.Series(rhos))
                    if trend_result:
                        trend_rho, trend_p = trend_result
                        if trend_rho > 0.3 and trend_p < 0.1:
                            print(
                                f"\n  {C_GREEN}Calibration is IMPROVING over time "
                                f"(trend \u03c1 = {trend_rho:+.2f}).{C_RESET}"
                            )
                        elif trend_rho < -0.3 and trend_p < 0.1:
                            print(
                                f"\n  {C_RED}Calibration is DEGRADING over time "
                                f"(trend \u03c1 = {trend_rho:+.2f}).{C_RESET}"
                            )
                        else:
                            print(
                                f"\n  {C_DIM}No significant trend in calibration quality "
                                f"(trend \u03c1 = {trend_rho:+.2f}, p = {trend_p:.2f}).{C_RESET}"
                            )
            else:
                print(
                    f"  {C_YELLOW}Not enough data in any window for rolling correlation.{C_RESET}"
                )

    # ── Section 2: Score Drift ──
    print(_header("Score Drift"))

    if "info_score" in df.columns and "added_date" in df.columns:
        df_sorted = df.sort_values("added_date")
        # Compute monthly means
        df_sorted["month"] = df_sorted["added_date"].dt.to_period("M")
        monthly_scores = df_sorted.groupby("month")["info_score"].mean()

        if len(monthly_scores) >= 2:
            first_half = monthly_scores.iloc[: len(monthly_scores) // 2].mean()
            second_half = monthly_scores.iloc[len(monthly_scores) // 2 :].mean()
            diff = second_half - first_half

            if abs(diff) > 5:
                direction = "UP" if diff > 0 else "DOWN"
                color = C_YELLOW
                print(
                    f"  {color}Scores are drifting {direction}: "
                    f"first half avg {first_half:.1f}, second half avg {second_half:.1f} "
                    f"({diff:+.1f}){C_RESET}"
                )
            else:
                print(
                    f"  {C_GREEN}Scores are stable: "
                    f"first half avg {first_half:.1f}, second half avg {second_half:.1f} "
                    f"({diff:+.1f}){C_RESET}"
                )

            # Linear trend
            ordinals = np.arange(len(monthly_scores))
            result = _safe_spearman(pd.Series(ordinals), pd.Series(monthly_scores.values))
            if result:
                rho, p = result
                print(f"  Trend: \u03c1 = {_color_rho(rho)} (p = {_color_pval(p)})")
        else:
            print(f"  {C_DIM}Not enough months for drift analysis.{C_RESET}")
    else:
        print(f"  {C_DIM}Missing data for score drift analysis.{C_RESET}")

    # ── Section 3: Engagement Drift ──
    print(_header("Engagement Drift"))

    if "num_highlights" in df.columns and "added_date" in df.columns:
        df_sorted = df.sort_values("added_date")
        df_sorted["month"] = df_sorted["added_date"].dt.to_period("M")
        monthly_hl = df_sorted.groupby("month")["num_highlights"].mean()

        if len(monthly_hl) >= 2:
            first_half = monthly_hl.iloc[: len(monthly_hl) // 2].mean()
            second_half = monthly_hl.iloc[len(monthly_hl) // 2 :].mean()
            diff = second_half - first_half

            if abs(diff) > 0.5:
                direction = "UP" if diff > 0 else "DOWN"
                color = C_YELLOW
                print(
                    f"  {color}Highlighting is trending {direction}: "
                    f"first half avg {first_half:.2f}, second half avg {second_half:.2f} "
                    f"({diff:+.2f} highlights/article){C_RESET}"
                )
            else:
                print(
                    f"  {C_GREEN}Highlighting behavior is stable: "
                    f"first half avg {first_half:.2f}, second half avg {second_half:.2f} "
                    f"({diff:+.2f}){C_RESET}"
                )

            ordinals = np.arange(len(monthly_hl))
            result = _safe_spearman(pd.Series(ordinals), pd.Series(monthly_hl.values))
            if result:
                rho, p = result
                print(f"  Trend: \u03c1 = {_color_rho(rho)} (p = {_color_pval(p)})")
        else:
            print(f"  {C_DIM}Not enough months for engagement drift analysis.{C_RESET}")
    else:
        print(f"  {C_DIM}Missing data for engagement drift analysis.{C_RESET}")

    # ── Section 4: Monthly Summary Table ──
    print(_header("Monthly Summary"))

    if "info_score" in df.columns and "num_highlights" in df.columns and "added_date" in df.columns:
        df_sorted = df.sort_values("added_date")
        df_sorted["month"] = df_sorted["added_date"].dt.to_period("M")

        months = sorted(df_sorted["month"].unique())

        if len(months) == 0:
            print(f"  {C_DIM}No monthly data available.{C_RESET}")
        else:
            print(
                f"  {'Month':<10s} {'Count':>6s} {'Mean Score':>11s} "
                f"{'Mean HL':>8s} {'Spearman':>9s}"
            )
            print(f"  {'\u2500' * 47}")

            for month in months:
                subset = df_sorted[df_sorted["month"] == month]
                cnt = len(subset)
                mean_score = subset["info_score"].mean()
                mean_hl = subset["num_highlights"].mean()
                corr_result = _safe_spearman(subset["info_score"], subset["num_highlights"])
                corr_str = _color_rho(corr_result[0]) if corr_result else f"{C_DIM}  n/a{C_RESET}"

                print(
                    f"  {str(month):<10s} {cnt:>6d} {mean_score:>11.1f} "
                    f"{mean_hl:>8.2f} {corr_str:>19s}"
                )
    else:
        print(f"  {C_DIM}Missing data for monthly summary.{C_RESET}")

    print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cal_report",
        description="Scoring calibration reports",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- report --
    p_report = subparsers.add_parser("report", help="Comprehensive calibration report")
    p_report.add_argument(
        "--since", type=str, default=None, help="ISO date filter (e.g. 2025-01-01)"
    )
    p_report.add_argument("--until", type=str, default=None, help="ISO date upper bound")
    p_report.add_argument(
        "--min-progress", type=float, default=0.0, help="Minimum reading progress (0.0-1.0)"
    )
    p_report.add_argument("--tag", type=str, default=None, help="Filter to articles with this tag")
    p_report.add_argument(
        "--category", type=str, default=None, help="Filter to this category (article, email, rss)"
    )

    # -- dimensions --
    p_dims = subparsers.add_parser("dimensions", help="Per-dimension predictive analysis")
    p_dims.add_argument("--since", type=str, default=None, help="ISO date filter")
    p_dims.add_argument("--min-progress", type=float, default=0.0, help="Minimum reading progress")

    # -- trends --
    p_trends = subparsers.add_parser("trends", help="Temporal calibration trends")
    p_trends.add_argument("--window", type=int, default=30, help="Rolling window in days")
    p_trends.add_argument(
        "--min-progress", type=float, default=0.0, help="Minimum reading progress"
    )

    args = parser.parse_args()

    if args.command == "report":
        run_report(
            since=args.since,
            until=args.until,
            min_progress=args.min_progress,
            tag=args.tag,
            category=args.category,
        )
    elif args.command == "dimensions":
        run_dimensions(since=args.since, min_progress=args.min_progress)
    elif args.command == "trends":
        run_trends(window=args.window, min_progress=args.min_progress)


if __name__ == "__main__":
    main()
