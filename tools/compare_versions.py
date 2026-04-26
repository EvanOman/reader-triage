"""Cross-version scoring comparison: V2 vs V3 vs V4 correlated with highlight engagement.

Syncs fresh highlights from Readwise, then computes Spearman/Pearson correlations
for each scoring version against:
  - num_highlights (raw count)
  - highlights_per_1000_words (length-normalized density)

Goal: identify which single scoring version best predicts user engagement
so we can drop the others and save API costs.
"""

import sqlite3

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from tools.cal_data import _PROJECT_ROOT, fetch_highlights

DB_PATH = _PROJECT_ROOT / "reader_triage.db"

DIMENSIONS = ["specificity_score", "novelty_score", "depth_score", "actionability_score"]
DIM_LABELS = {
    "specificity_score": "Quotability",
    "novelty_score": "Surprise",
    "depth_score": "Argument",
    "actionability_score": "Insight",
}

# ANSI
B = "\033[1m"
R = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"


def color_rho(rho: float) -> str:
    a = abs(rho)
    if a >= 0.3:
        return f"{GREEN}{rho:+.3f}{R}"
    elif a >= 0.15:
        return f"{YELLOW}{rho:+.3f}{R}"
    return f"{RED}{rho:+.3f}{R}"


def safe_corr(x: pd.Series, y: pd.Series) -> dict:
    """Compute both Spearman and Pearson, return dict with rho/r/p values."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    valid = valid[(valid["x"].notna()) & (valid["y"].notna())]
    n = len(valid)
    if n < 5 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return {"n": n, "spearman": None, "sp_p": None, "pearson": None, "pe_p": None}
    rho, sp = spearmanr(valid["x"], valid["y"])
    r, pp = pearsonr(valid["x"], valid["y"])
    return {
        "n": n,
        "spearman": float(rho),
        "sp_p": float(sp),
        "pearson": float(r),
        "pe_p": float(pp),
    }


def load_all_versions() -> pd.DataFrame:
    """Load articles joined with V2, V3, V4 scores into one DataFrame."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            a.id AS article_id,
            a.title,
            a.url,
            a.word_count,
            a.location,
            a.category,
            a.reading_progress,
            a.num_highlights AS db_num_highlights,
            -- V2 scores
            s2.info_score AS v2_total,
            s2.specificity_score AS v2_specificity_score,
            s2.novelty_score AS v2_novelty_score,
            s2.depth_score AS v2_depth_score,
            s2.actionability_score AS v2_actionability_score,
            -- V3 scores
            s3.info_score AS v3_total,
            s3.specificity_score AS v3_specificity_score,
            s3.novelty_score AS v3_novelty_score,
            s3.depth_score AS v3_depth_score,
            s3.actionability_score AS v3_actionability_score,
            -- V4 scores
            s4.info_score AS v4_total,
            s4.specificity_score AS v4_specificity_score,
            s4.novelty_score AS v4_novelty_score,
            s4.depth_score AS v4_depth_score,
            s4.actionability_score AS v4_actionability_score
        FROM articles a
        LEFT JOIN article_scores s2 ON s2.article_id = a.id
        LEFT JOIN article_scores_v3 s3 ON s3.article_id = a.id
        LEFT JOIN article_scores_v4 s4 ON s4.article_id = a.id
        WHERE s2.info_score IS NOT NULL
           OR s3.info_score IS NOT NULL
           OR s4.info_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def main() -> None:
    # Step 1: Sync highlights
    print(f"{B}Syncing highlights from Readwise...{R}")
    highlights = fetch_highlights(force=True)
    print()

    # Step 2: Load all versions
    df = load_all_versions()
    print(f"{B}Loaded {len(df)} articles with at least one score version{R}")

    # Merge highlights
    df["num_highlights"] = (
        df["article_id"]
        .map(lambda aid: highlights.get(aid, {}).get("count", 0))
        .fillna(0)
        .astype(int)
    )
    df["highlighted_words"] = (
        df["article_id"]
        .map(lambda aid: highlights.get(aid, {}).get("words", 0))
        .fillna(0)
        .astype(int)
    )
    df["hl_per_1k"] = df.apply(
        lambda r: (r["num_highlights"] / r["word_count"] * 1000)
        if r["word_count"] and r["word_count"] > 0
        else 0,
        axis=1,
    )
    df["hl_words_per_1k"] = df.apply(
        lambda r: (r["highlighted_words"] / r["word_count"] * 1000)
        if r["word_count"] and r["word_count"] > 0
        else 0,
        axis=1,
    )

    has_hl = (df["num_highlights"] > 0).sum()
    print(f"Articles with highlights: {has_hl} / {len(df)} ({100 * has_hl / len(df):.1f}%)")

    versions = {
        "V2 (categorical)": "v2",
        "V3 (binary)": "v3",
        "V4 (tiered)": "v4",
    }

    engagement_metrics = {
        "num_highlights": "# Highlights",
        "hl_per_1k": "Highlights/1k words",
        "highlighted_words": "Highlighted words",
        "hl_words_per_1k": "HL words/1k words",
    }

    # ── Overall total score correlations ──
    print(f"\n{CYAN}{B}{'═' * 70}")
    print("  TOTAL SCORE vs ENGAGEMENT — Cross-Version Comparison")
    print(f"{'═' * 70}{R}\n")

    header = f"  {'Metric':<22s}"
    for vname in versions:
        header += f"  {vname:>18s}"
    print(f"{B}{header}{R}")
    print(f"  {'─' * 22}" + f"  {'─' * 18}" * len(versions))

    for metric_col, metric_label in engagement_metrics.items():
        row = f"  {metric_label:<22s}"
        for _vname, prefix in versions.items():
            total_col = f"{prefix}_total"
            if total_col not in df.columns:
                row += f"  {'n/a':>18s}"
                continue
            subset = df[df[total_col].notna()]
            result = safe_corr(subset[total_col], subset[metric_col])
            if result["spearman"] is not None:
                rho = result["spearman"]
                row += f"  {color_rho(rho)} (n={result['n']:>3d})"
            else:
                row += f"  {'insufficient':>18s}"
        print(row)

    # ── Per-dimension breakdown for each version ──
    for metric_col, metric_label in [
        ("num_highlights", "# Highlights"),
        ("hl_per_1k", "Highlights/1k words"),
    ]:
        print(f"\n{CYAN}{B}{'─' * 70}")
        print(f"  PER-DIMENSION Spearman ρ vs {metric_label}")
        print(f"{'─' * 70}{R}\n")

        header = f"  {'Dimension':<16s}"
        for vname in versions:
            header += f"  {vname:>18s}"
        print(f"{B}{header}{R}")
        print(f"  {'─' * 16}" + f"  {'─' * 18}" * len(versions))

        for dim in DIMENSIONS:
            label = DIM_LABELS[dim]
            row = f"  {label:<16s}"
            for _vname, prefix in versions.items():
                dim_col = f"{prefix}_{dim}"
                if dim_col not in df.columns:
                    row += f"  {'n/a':>18s}"
                    continue
                subset = df[df[dim_col].notna()]
                result = safe_corr(subset[dim_col], subset[metric_col])
                if result["spearman"] is not None:
                    rho = result["spearman"]
                    row += f"  {color_rho(rho)} (n={result['n']:>3d})"
                else:
                    row += f"  {'insufficient':>18s}"
            print(row)

    # ── Score distribution comparison ──
    print(f"\n{CYAN}{B}{'─' * 70}")
    print("  SCORE DISTRIBUTIONS")
    print(f"{'─' * 70}{R}\n")

    for vname, prefix in versions.items():
        total_col = f"{prefix}_total"
        if total_col not in df.columns:
            continue
        s = df[total_col].dropna()
        if len(s) == 0:
            continue
        print(f"  {B}{vname}{R} (n={len(s)})")
        print(f"    Mean: {s.mean():.1f}  Median: {s.median():.1f}  Std: {s.std():.1f}")
        print(
            f"    Min: {s.min():.0f}  25th: {s.quantile(0.25):.0f}  75th: {s.quantile(0.75):.0f}  Max: {s.max():.0f}"
        )

        # Tier distribution
        high = (s >= 60).sum()
        med = ((s >= 30) & (s < 60)).sum()
        low = (s < 30).sum()
        n = len(s)
        print(
            f"    Tiers: High {high} ({100 * high / n:.0f}%) | Med {med} ({100 * med / n:.0f}%) | Low {low} ({100 * low / n:.0f}%)"
        )
        print()

    # ── Head-to-head: same articles, all 3 versions ──
    all3 = df[df["v2_total"].notna() & df["v3_total"].notna() & df["v4_total"].notna()]
    print(f"\n{CYAN}{B}{'─' * 70}")
    print(f"  HEAD-TO-HEAD (articles with all 3 versions: n={len(all3)})")
    print(f"{'─' * 70}{R}\n")

    if len(all3) >= 5:
        for metric_col, metric_label in engagement_metrics.items():
            results = []
            for vname, prefix in versions.items():
                total_col = f"{prefix}_total"
                result = safe_corr(all3[total_col], all3[metric_col])
                results.append((vname, result))

            print(f"  {B}{metric_label}{R}")
            best_rho = -999
            for _vname, result in results:
                if result["spearman"] is not None:
                    rho = result["spearman"]
                    if rho > best_rho:
                        best_rho = rho
            for vname, result in results:
                if result["spearman"] is not None:
                    rho = result["spearman"]
                    marker = " ★" if rho == best_rho else ""
                    p_str = f"p={result['sp_p']:.3f}" if result["sp_p"] is not None else ""
                    print(f"    {vname:<20s}  ρ = {color_rho(rho)}  {p_str}{marker}")
                else:
                    print(f"    {vname:<20s}  insufficient data")
            print()

    # ── Recommendation ──
    print(f"\n{CYAN}{B}{'═' * 70}")
    print("  RECOMMENDATION")
    print(f"{'═' * 70}{R}\n")

    # Find best version for num_highlights on the head-to-head set
    if len(all3) >= 5:
        best_version = None
        best_rho = -999.0
        for vname, prefix in versions.items():
            total_col = f"{prefix}_total"
            result = safe_corr(all3[total_col], all3["num_highlights"])
            if result["spearman"] is not None and result["spearman"] > best_rho:
                best_rho = result["spearman"]
                best_version = vname

        if best_version:
            print(
                f"  Best correlating version (num_highlights): {GREEN}{B}{best_version}{R} (ρ = {best_rho:+.3f})"
            )

        best_version_density = None
        best_rho_density = -999.0
        for vname, prefix in versions.items():
            total_col = f"{prefix}_total"
            result = safe_corr(all3[total_col], all3["hl_per_1k"])
            if result["spearman"] is not None and result["spearman"] > best_rho_density:
                best_rho_density = result["spearman"]
                best_version_density = vname

        if best_version_density:
            print(
                f"  Best correlating version (hl/1k words):    {GREEN}{B}{best_version_density}{R} (ρ = {best_rho_density:+.3f})"
            )

    print()


if __name__ == "__main__":
    main()
