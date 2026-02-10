"""Analyze how well article scoring predicts user engagement (highlights).

Consumes two data files:
- archived_articles.json: article metadata with highlight counts
- scores.json: AI-generated scores per article

Produces correlation analysis, regression models, calibration analysis,
per-dimension breakdowns, visualizations, and a JSON report.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr

matplotlib.use("Agg")  # Non-interactive backend for saving plots
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_DIR = Path(__file__).resolve().parent
ARTICLES_PATH = BACKTEST_DIR / "archived_articles.json"
SCORES_PATH = BACKTEST_DIR / "scores.json"
REPORT_PATH = BACKTEST_DIR / "analysis_report.json"

DIMENSIONS = ["specificity_score", "novelty_score", "depth_score", "actionability_score"]
ENGAGEMENT_METRICS = ["num_highlights", "total_highlighted_words", "highlights_per_1000_words"]

# Pretty labels for display
DIM_LABELS = {
    "specificity_score": "Specificity",
    "novelty_score": "Novelty",
    "depth_score": "Depth",
    "actionability_score": "Actionability",
}
METRIC_LABELS = {
    "num_highlights": "# Highlights",
    "total_highlighted_words": "Highlighted Words",
    "highlights_per_1000_words": "Highlights/1k Words",
}


def load_data() -> pd.DataFrame:
    """Load and merge articles and scores on doc_id."""
    if not ARTICLES_PATH.exists():
        print(f"ERROR: {ARTICLES_PATH} not found")
        sys.exit(1)
    if not SCORES_PATH.exists():
        print(f"ERROR: {SCORES_PATH} not found")
        sys.exit(1)

    with open(ARTICLES_PATH) as f:
        articles = json.load(f)
    with open(SCORES_PATH) as f:
        scores = json.load(f)

    df_articles = pd.DataFrame(articles)
    df_scores = pd.DataFrame(scores)

    print(f"Loaded {len(df_articles)} articles, {len(df_scores)} scores")

    # Merge on doc_id
    df = df_articles.merge(df_scores, on="doc_id", how="inner")
    print(f"Merged dataset: {len(df)} articles with both data and scores")

    return df_articles, df_scores, df


def print_dataset_summary(
    df_articles: pd.DataFrame, df_scores: pd.DataFrame, df: pd.DataFrame
) -> dict:
    """Print and return dataset summary statistics."""
    total_articles = len(df_articles)
    articles_with_scores = len(df)
    articles_with_highlights = int((df["num_highlights"] > 0).sum()) if len(df) > 0 else 0
    articles_never_opened = (
        int((df["reading_progress"] == 0).sum()) if "reading_progress" in df.columns else 0
    )

    summary = {
        "total_articles": total_articles,
        "total_scores": len(df_scores),
        "merged_count": len(df),
        "articles_with_highlights": articles_with_highlights,
        "articles_never_opened": articles_never_opened,
    }

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total articles:                {total_articles}")
    print(f"  Total scores:                  {len(df_scores)}")
    print(f"  Merged (articles with scores): {len(df)}")
    print(f"  Articles with highlights > 0:  {articles_with_highlights}")
    print(f"  Articles never opened (rp=0):  {articles_never_opened}")

    if len(df) > 0:
        print(
            f"\n  info_score  - mean: {df['info_score'].mean():.1f}, "
            f"median: {df['info_score'].median():.1f}, "
            f"std: {df['info_score'].std():.1f}"
        )
        print(
            f"  num_highlights - mean: {df['num_highlights'].mean():.1f}, "
            f"median: {df['num_highlights'].median():.1f}, "
            f"max: {df['num_highlights'].max()}"
        )

    return summary


def prepare_features(df: pd.DataFrame, exclude_unread: bool = False) -> pd.DataFrame:
    """Add derived features and optionally filter unread articles."""
    df = df.copy()

    # Derived feature: highlights per 1000 words
    df["highlights_per_1000_words"] = np.where(
        (df["word_count"].notna()) & (df["word_count"] > 0),
        (df["total_highlighted_words"] / df["word_count"]) * 1000,
        0.0,
    )

    # Binary: has any highlights
    df["has_highlights"] = (df["num_highlights"] > 0).astype(int)

    # Score quintile bins
    df["score_quintile"] = pd.cut(
        df["info_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
    )

    if exclude_unread and "reading_progress" in df.columns:
        before = len(df)
        df = df[df["reading_progress"] > 0].copy()
        print(f"  Filtered unread: {before} -> {len(df)} articles")

    return df


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Compute Spearman rank correlations between scores and engagement metrics."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS (Spearman)")
    print("=" * 70)

    results = {}

    if len(df) < 3:
        print("  Skipped: need at least 3 data points")
        return results

    # Overall info_score vs each engagement metric
    print("\n--- info_score vs engagement ---")
    for metric in ENGAGEMENT_METRICS:
        valid = df[["info_score", metric]].dropna()
        if len(valid) < 3:
            print(f"  info_score vs {METRIC_LABELS.get(metric, metric)}: insufficient data")
            continue
        rho, pval = spearmanr(valid["info_score"], valid[metric])
        key = f"info_score_vs_{metric}"
        results[key] = {"rho": round(rho, 4), "p_value": round(pval, 6), "n": len(valid)}
        sig = "*" if pval < 0.05 else ""
        print(
            f"  info_score vs {METRIC_LABELS.get(metric, metric):>22s}: "
            f"rho={rho:+.4f}, p={pval:.4f} {sig}  (n={len(valid)})"
        )

    # Per-dimension correlations
    print("\n--- Per-dimension correlations ---")
    dim_results = {}
    for dim in DIMENSIONS:
        dim_results[dim] = {}
        for metric in ENGAGEMENT_METRICS:
            valid = df[[dim, metric]].dropna()
            if len(valid) < 3:
                continue
            rho, pval = spearmanr(valid[dim], valid[metric])
            dim_results[dim][metric] = {
                "rho": round(rho, 4),
                "p_value": round(pval, 6),
                "n": len(valid),
            }
            sig = "*" if pval < 0.05 else ""
            print(
                f"  {DIM_LABELS.get(dim, dim):>15s} vs {METRIC_LABELS.get(metric, metric):>22s}: "
                f"rho={rho:+.4f}, p={pval:.4f} {sig}"
            )

    results["per_dimension"] = dim_results

    # Full correlation matrix (scores + engagement)
    corr_cols = ["info_score"] + DIMENSIONS + ENGAGEMENT_METRICS
    available_cols = [c for c in corr_cols if c in df.columns]
    if len(available_cols) >= 2:
        corr_matrix = df[available_cols].corr(method="spearman")
        print("\n--- Full Spearman Correlation Matrix ---")
        print(corr_matrix.round(3).to_string())
        results["correlation_matrix"] = corr_matrix.round(4).to_dict()

    return results


def regression_analysis(df: pd.DataFrame) -> dict:
    """Run OLS and negative binomial regressions."""
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS")
    print("=" * 70)

    results = {}

    if len(df) < 10:
        print("  Skipped: need at least 10 data points for meaningful regression")
        return results

    # --- OLS: engagement ~ info_score ---
    print("\n--- OLS: num_highlights ~ info_score ---")
    try:
        import statsmodels.api as sm

        valid = df[["info_score", "num_highlights"]].dropna()
        X = sm.add_constant(valid["info_score"])
        y = valid["num_highlights"]
        model = sm.OLS(y, X).fit()

        results["ols_highlights_from_score"] = {
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_pvalue": round(model.f_pvalue, 6) if model.f_pvalue is not None else None,
            "coefficients": {
                name: {
                    "coef": round(model.params[i], 4),
                    "p_value": round(model.pvalues[i], 6),
                    "ci_lower": round(model.conf_int().iloc[i, 0], 4),
                    "ci_upper": round(model.conf_int().iloc[i, 1], 4),
                }
                for i, name in enumerate(model.params.index)
            },
            "n": int(model.nobs),
        }
        print(model.summary().as_text())
    except Exception as e:
        print(f"  OLS (highlights ~ score) failed: {e}")

    # --- OLS: info_score ~ num_highlights + total_highlighted_words ---
    print("\n--- OLS: info_score ~ num_highlights + total_highlighted_words ---")
    try:
        valid = df[["info_score", "num_highlights", "total_highlighted_words"]].dropna()
        X = sm.add_constant(valid[["num_highlights", "total_highlighted_words"]])
        y = valid["info_score"]
        model = sm.OLS(y, X).fit()

        results["ols_score_from_engagement"] = {
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_pvalue": round(model.f_pvalue, 6) if model.f_pvalue is not None else None,
            "coefficients": {
                name: {
                    "coef": round(model.params[i], 4),
                    "p_value": round(model.pvalues[i], 6),
                    "ci_lower": round(model.conf_int().iloc[i, 0], 4),
                    "ci_upper": round(model.conf_int().iloc[i, 1], 4),
                }
                for i, name in enumerate(model.params.index)
            },
            "n": int(model.nobs),
        }
        print(model.summary().as_text())
    except Exception as e:
        print(f"  OLS (score ~ engagement) failed: {e}")

    # --- OLS: engagement ~ info_score + word_count ---
    print("\n--- OLS: total_highlighted_words ~ info_score + word_count ---")
    try:
        valid = df[["total_highlighted_words", "info_score", "word_count"]].dropna()
        valid = valid[valid["word_count"] > 0]
        X = sm.add_constant(valid[["info_score", "word_count"]])
        y = valid["total_highlighted_words"]
        model = sm.OLS(y, X).fit()

        results["ols_highlightwords_from_score_wordcount"] = {
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "f_pvalue": round(model.f_pvalue, 6) if model.f_pvalue is not None else None,
            "coefficients": {
                name: {
                    "coef": round(model.params[i], 4),
                    "p_value": round(model.pvalues[i], 6),
                    "ci_lower": round(model.conf_int().iloc[i, 0], 4),
                    "ci_upper": round(model.conf_int().iloc[i, 1], 4),
                }
                for i, name in enumerate(model.params.index)
            },
            "n": int(model.nobs),
        }
        print(model.summary().as_text())
    except Exception as e:
        print(f"  OLS (highlighted words ~ score + word_count) failed: {e}")

    # --- Negative Binomial: num_highlights ~ info_score + word_count ---
    print("\n--- Negative Binomial: num_highlights ~ info_score + word_count ---")
    try:
        import statsmodels.api as sm

        valid = df[["num_highlights", "info_score", "word_count"]].dropna()
        valid = valid[valid["word_count"] > 0].copy()

        if len(valid) < 10:
            print("  Skipped: insufficient data")
        else:
            X = sm.add_constant(valid[["info_score", "word_count"]])
            y = valid["num_highlights"].astype(int)

            # Try negative binomial; fall back to Poisson if it fails
            try:
                nb_model = sm.NegativeBinomial(y, X).fit(disp=False, maxiter=200)
                results["negative_binomial"] = {
                    "converged": nb_model.mle_retvals.get("converged", None)
                    if hasattr(nb_model, "mle_retvals") and isinstance(nb_model.mle_retvals, dict)
                    else None,
                    "llf": round(nb_model.llf, 4),
                    "aic": round(nb_model.aic, 4),
                    "bic": round(nb_model.bic, 4),
                    "coefficients": {
                        name: {
                            "coef": round(nb_model.params[i], 6),
                            "p_value": round(nb_model.pvalues[i], 6),
                            "ci_lower": round(nb_model.conf_int().iloc[i, 0], 6),
                            "ci_upper": round(nb_model.conf_int().iloc[i, 1], 6),
                        }
                        for i, name in enumerate(nb_model.params.index)
                    },
                    "n": len(valid),
                    "model_type": "NegativeBinomial",
                }
                print(nb_model.summary().as_text())
            except Exception as e_nb:
                print(f"  Negative Binomial failed ({e_nb}), trying Poisson...")
                try:
                    poisson_model = sm.Poisson(y, X).fit(disp=False, maxiter=200)
                    results["negative_binomial"] = {
                        "llf": round(poisson_model.llf, 4),
                        "aic": round(poisson_model.aic, 4),
                        "bic": round(poisson_model.bic, 4),
                        "coefficients": {
                            name: {
                                "coef": round(poisson_model.params[i], 6),
                                "p_value": round(poisson_model.pvalues[i], 6),
                            }
                            for i, name in enumerate(poisson_model.params.index)
                        },
                        "n": len(valid),
                        "model_type": "Poisson (NB fallback)",
                    }
                    print(poisson_model.summary().as_text())
                except Exception as e_pois:
                    print(f"  Poisson also failed: {e_pois}")
    except Exception as e:
        print(f"  Negative Binomial regression failed: {e}")

    return results


def calibration_analysis(df: pd.DataFrame) -> dict:
    """Analyze engagement by score quintile and run chi-square test."""
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS (Score Quintiles)")
    print("=" * 70)

    results = {}

    if len(df) < 5:
        print("  Skipped: need at least 5 data points")
        return results

    # Quintile breakdown
    quintile_stats = []
    print(
        f"\n{'Quintile':>10s}  {'N':>4s}  {'Mean HL':>8s}  {'Med HL':>8s}  "
        f"{'Mean HLW':>9s}  {'% w/HL':>7s}"
    )
    print("-" * 60)

    for quintile in ["0-20", "20-40", "40-60", "60-80", "80-100"]:
        subset = df[df["score_quintile"] == quintile]
        n = len(subset)
        if n == 0:
            quintile_stats.append(
                {
                    "quintile": quintile,
                    "n": 0,
                    "mean_highlights": None,
                    "median_highlights": None,
                    "mean_highlighted_words": None,
                    "pct_with_highlights": None,
                }
            )
            print(f"{quintile:>10s}  {0:>4d}  {'n/a':>8s}  {'n/a':>8s}  {'n/a':>9s}  {'n/a':>7s}")
            continue

        mean_hl = subset["num_highlights"].mean()
        med_hl = subset["num_highlights"].median()
        mean_hlw = subset["total_highlighted_words"].mean()
        pct_hl = (subset["has_highlights"].sum() / n) * 100

        stat = {
            "quintile": quintile,
            "n": int(n),
            "mean_highlights": round(float(mean_hl), 2),
            "median_highlights": round(float(med_hl), 2),
            "mean_highlighted_words": round(float(mean_hlw), 2),
            "pct_with_highlights": round(float(pct_hl), 1),
        }
        quintile_stats.append(stat)

        print(
            f"{quintile:>10s}  {n:>4d}  {mean_hl:>8.2f}  {med_hl:>8.1f}  "
            f"{mean_hlw:>9.1f}  {pct_hl:>6.1f}%"
        )

    results["quintile_stats"] = quintile_stats

    # Chi-square test: does score tier predict any highlights?
    print("\n--- Chi-Square Test: score quintile vs has_highlights ---")
    try:
        # Build contingency table
        ct = pd.crosstab(df["score_quintile"], df["has_highlights"])

        # Need at least 2 rows and 2 columns for chi-square
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            print(
                "  Skipped: contingency table too small (need 2+ quintiles with both highlight states)"
            )
            results["chi_square"] = {"skipped": True, "reason": "contingency table too small"}
        else:
            chi2, p, dof, expected = chi2_contingency(ct)
            results["chi_square"] = {
                "chi2": round(chi2, 4),
                "p_value": round(p, 6),
                "dof": int(dof),
                "significant": p < 0.05,
            }
            sig = "YES" if p < 0.05 else "NO"
            print(f"  chi2 = {chi2:.4f}, p = {p:.6f}, dof = {dof}")
            print(f"  Significant at alpha=0.05: {sig}")
            print("\n  Contingency table:")
            print(ct.to_string())
    except Exception as e:
        print(f"  Chi-square test failed: {e}")
        results["chi_square"] = {"error": str(e)}

    return results


def per_dimension_analysis(df: pd.DataFrame) -> dict:
    """Determine which scoring dimension is most predictive of engagement."""
    print("\n" + "=" * 70)
    print("PER-DIMENSION ANALYSIS")
    print("=" * 70)

    results = {"individual_correlations": {}, "most_predictive": {}}

    if len(df) < 5:
        print("  Skipped: insufficient data")
        return results

    # Individual correlations: each dimension vs num_highlights
    print("\n--- Individual dimension correlations with num_highlights ---")
    dim_corrs = {}
    for dim in DIMENSIONS:
        valid = df[[dim, "num_highlights"]].dropna()
        if len(valid) < 3:
            continue
        rho, pval = spearmanr(valid[dim], valid["num_highlights"])
        dim_corrs[dim] = {"rho": round(rho, 4), "p_value": round(pval, 6)}
        sig = "*" if pval < 0.05 else ""
        print(f"  {DIM_LABELS.get(dim, dim):>15s}: rho={rho:+.4f}, p={pval:.4f} {sig}")

    results["individual_correlations"] = dim_corrs

    # Find most predictive
    if dim_corrs:
        best_dim = max(dim_corrs.keys(), key=lambda d: abs(dim_corrs[d]["rho"]))
        results["most_predictive"] = {
            "dimension": best_dim,
            "label": DIM_LABELS.get(best_dim, best_dim),
            "rho": dim_corrs[best_dim]["rho"],
            "p_value": dim_corrs[best_dim]["p_value"],
        }
        print(
            f"\n  Most predictive dimension: {DIM_LABELS.get(best_dim, best_dim)} "
            f"(rho={dim_corrs[best_dim]['rho']:+.4f})"
        )

    # Multiple regression: num_highlights ~ all 4 sub-scores
    print("\n--- Multiple OLS: num_highlights ~ specificity + novelty + depth + actionability ---")
    try:
        import statsmodels.api as sm

        valid = df[DIMENSIONS + ["num_highlights"]].dropna()
        if len(valid) < len(DIMENSIONS) + 2:
            print("  Skipped: insufficient data for multiple regression")
        else:
            X = sm.add_constant(valid[DIMENSIONS])
            y = valid["num_highlights"]
            model = sm.OLS(y, X).fit()

            results["multiple_regression"] = {
                "r_squared": round(model.rsquared, 4),
                "adj_r_squared": round(model.rsquared_adj, 4),
                "f_pvalue": round(model.f_pvalue, 6) if model.f_pvalue is not None else None,
                "coefficients": {
                    name: {
                        "coef": round(model.params[i], 4),
                        "p_value": round(model.pvalues[i], 6),
                        "ci_lower": round(model.conf_int().iloc[i, 0], 4),
                        "ci_upper": round(model.conf_int().iloc[i, 1], 4),
                    }
                    for i, name in enumerate(model.params.index)
                },
                "n": int(model.nobs),
            }
            print(model.summary().as_text())
    except Exception as e:
        print(f"  Multiple regression failed: {e}")

    # Also run for total_highlighted_words
    print(
        "\n--- Multiple OLS: total_highlighted_words ~ specificity + novelty + depth + actionability ---"
    )
    try:
        valid = df[DIMENSIONS + ["total_highlighted_words"]].dropna()
        if len(valid) < len(DIMENSIONS) + 2:
            print("  Skipped: insufficient data")
        else:
            X = sm.add_constant(valid[DIMENSIONS])
            y = valid["total_highlighted_words"]
            model = sm.OLS(y, X).fit()

            results["multiple_regression_hlw"] = {
                "r_squared": round(model.rsquared, 4),
                "adj_r_squared": round(model.rsquared_adj, 4),
                "f_pvalue": round(model.f_pvalue, 6) if model.f_pvalue is not None else None,
                "coefficients": {
                    name: {
                        "coef": round(model.params[i], 4),
                        "p_value": round(model.pvalues[i], 6),
                    }
                    for i, name in enumerate(model.params.index)
                },
                "n": int(model.nobs),
            }
            print(model.summary().as_text())
    except Exception as e:
        print(f"  Multiple regression (hlw) failed: {e}")

    return results


def create_visualizations(df: pd.DataFrame) -> None:
    """Generate and save all visualization plots."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    if len(df) < 3:
        print("  Skipped: insufficient data for meaningful plots")
        return

    # 1. Scatter: info_score vs num_highlights
    print("  Creating scatter_score_vs_highlights.png ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        df["info_score"], df["num_highlights"], alpha=0.5, edgecolors="k", linewidth=0.3, s=50
    )
    # Trend line
    valid = df[["info_score", "num_highlights"]].dropna()
    if len(valid) >= 2:
        z = np.polyfit(valid["info_score"], valid["num_highlights"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(valid["info_score"].min(), valid["info_score"].max(), 100)
        ax.plot(x_range, p_line(x_range), "r--", linewidth=2, label=f"Trend (slope={z[0]:.3f})")
        ax.legend()
    ax.set_xlabel("Info Score")
    ax.set_ylabel("Number of Highlights")
    ax.set_title("Article Score vs Number of Highlights")
    fig.tight_layout()
    fig.savefig(BACKTEST_DIR / "scatter_score_vs_highlights.png", dpi=150)
    plt.close(fig)

    # 2. Scatter: info_score vs total_highlighted_words
    print("  Creating scatter_score_vs_words.png ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        df["info_score"],
        df["total_highlighted_words"],
        alpha=0.5,
        edgecolors="k",
        linewidth=0.3,
        s=50,
    )
    valid = df[["info_score", "total_highlighted_words"]].dropna()
    if len(valid) >= 2:
        z = np.polyfit(valid["info_score"], valid["total_highlighted_words"], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(valid["info_score"].min(), valid["info_score"].max(), 100)
        ax.plot(x_range, p_line(x_range), "r--", linewidth=2, label=f"Trend (slope={z[0]:.3f})")
        ax.legend()
    ax.set_xlabel("Info Score")
    ax.set_ylabel("Total Highlighted Words")
    ax.set_title("Article Score vs Total Highlighted Words")
    fig.tight_layout()
    fig.savefig(BACKTEST_DIR / "scatter_score_vs_words.png", dpi=150)
    plt.close(fig)

    # 3. Bar chart: mean highlights by score quintile
    print("  Creating quintile_analysis.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    quintile_order = ["0-20", "20-40", "40-60", "60-80", "80-100"]

    for idx, (metric, label) in enumerate(
        [
            ("num_highlights", "Mean # Highlights"),
            ("total_highlighted_words", "Mean Highlighted Words"),
            ("highlights_per_1000_words", "Mean HL/1k Words"),
        ]
    ):
        means = []
        labels_q = []
        for q in quintile_order:
            subset = df[df["score_quintile"] == q]
            if len(subset) > 0:
                means.append(subset[metric].mean())
            else:
                means.append(0)
            labels_q.append(q)

        bars = axes[idx].bar(
            labels_q, means, color=sns.color_palette("viridis", 5), edgecolor="black"
        )
        axes[idx].set_xlabel("Score Quintile")
        axes[idx].set_ylabel(label)
        axes[idx].set_title(label + " by Quintile")

        # Add count annotations on bars
        for i, q in enumerate(quintile_order):
            n = len(df[df["score_quintile"] == q])
            axes[idx].annotate(f"n={n}", xy=(i, means[i]), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Engagement by Score Quintile", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(BACKTEST_DIR / "quintile_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Bar chart: dimension correlations
    print("  Creating dimension_correlations.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    dim_names = []
    corr_hl = []
    corr_hlw = []
    for dim in DIMENSIONS:
        dim_names.append(DIM_LABELS.get(dim, dim))
        valid_hl = df[[dim, "num_highlights"]].dropna()
        valid_hlw = df[[dim, "total_highlighted_words"]].dropna()
        if len(valid_hl) >= 3:
            rho_hl, _ = spearmanr(valid_hl[dim], valid_hl["num_highlights"])
            corr_hl.append(rho_hl)
        else:
            corr_hl.append(0)
        if len(valid_hlw) >= 3:
            rho_hlw, _ = spearmanr(valid_hlw[dim], valid_hlw["total_highlighted_words"])
            corr_hlw.append(rho_hlw)
        else:
            corr_hlw.append(0)

    x = np.arange(len(dim_names))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2, corr_hl, width, label="vs # Highlights", color="#4C72B0", edgecolor="black"
    )
    bars2 = ax.bar(
        x + width / 2,
        corr_hlw,
        width,
        label="vs Highlighted Words",
        color="#55A868",
        edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.set_ylabel("Spearman Correlation (rho)")
    ax.set_title("Scoring Dimension Correlations with Engagement")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va=va,
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(BACKTEST_DIR / "dimension_correlations.png", dpi=150)
    plt.close(fig)

    # 5. Histogram: score distributions for highlighted vs non-highlighted
    print("  Creating score_distribution.png ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    highlighted = df[df["has_highlights"] == 1]["info_score"]
    not_highlighted = df[df["has_highlights"] == 0]["info_score"]

    bins = np.arange(0, 105, 5)
    if len(highlighted) > 0:
        ax.hist(
            highlighted,
            bins=bins,
            alpha=0.6,
            label=f"With Highlights (n={len(highlighted)})",
            color="#E24A33",
            edgecolor="black",
        )
    if len(not_highlighted) > 0:
        ax.hist(
            not_highlighted,
            bins=bins,
            alpha=0.6,
            label=f"No Highlights (n={len(not_highlighted)})",
            color="#348ABD",
            edgecolor="black",
        )

    ax.set_xlabel("Info Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Highlighted vs Non-Highlighted Articles")
    ax.legend()

    # Add mean lines
    if len(highlighted) > 0:
        ax.axvline(
            highlighted.mean(),
            color="#E24A33",
            linestyle="--",
            linewidth=2,
            label=f"Mean (highlighted): {highlighted.mean():.1f}",
        )
    if len(not_highlighted) > 0:
        ax.axvline(
            not_highlighted.mean(),
            color="#348ABD",
            linestyle="--",
            linewidth=2,
            label=f"Mean (no highlights): {not_highlighted.mean():.1f}",
        )
    ax.legend()

    fig.tight_layout()
    fig.savefig(BACKTEST_DIR / "score_distribution.png", dpi=150)
    plt.close(fig)

    print("  All visualizations saved.")


def run_analysis(exclude_unread: bool = False) -> None:
    """Main analysis pipeline."""
    print("=" * 70)
    print("READER TRIAGE BACKTEST ANALYSIS")
    print("Measuring how well article scoring predicts user engagement")
    print("=" * 70)

    # Load data
    df_articles, df_scores, df = load_data()

    if len(df) == 0:
        print("\nERROR: No articles matched between the two datasets. Cannot proceed.")
        sys.exit(1)

    # Summary
    summary = print_dataset_summary(df_articles, df_scores, df)

    # Prepare features
    label = "(excluding unread)" if exclude_unread else "(all articles)"
    print(f"\nPreparing features {label} ...")
    df = prepare_features(df, exclude_unread=exclude_unread)

    if len(df) == 0:
        print("ERROR: No articles remaining after filtering. Cannot proceed.")
        sys.exit(1)

    print(f"Analysis dataset: {len(df)} articles")
    summary["analysis_n"] = len(df)
    summary["exclude_unread"] = exclude_unread

    # Analyses
    corr_results = correlation_analysis(df)
    reg_results = regression_analysis(df)
    cal_results = calibration_analysis(df)
    dim_results = per_dimension_analysis(df)

    # Visualizations
    create_visualizations(df)

    # Build and save report
    report = {
        "summary": summary,
        "correlation_analysis": _make_serializable(corr_results),
        "regression_analysis": _make_serializable(reg_results),
        "calibration_analysis": _make_serializable(cal_results),
        "per_dimension_analysis": _make_serializable(dim_results),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"JSON report saved to: {REPORT_PATH}")
    print(f"Visualizations saved to: {BACKTEST_DIR}/")
    print("=" * 70)


def _make_serializable(obj):
    """Recursively convert numpy/pandas types to Python-native types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if pd.isna(obj) if isinstance(obj, float) else False:
        return None
    return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze article scoring vs engagement")
    parser.add_argument(
        "--exclude-unread",
        action="store_true",
        help="Exclude articles with reading_progress == 0 (never opened)",
    )
    args = parser.parse_args()

    run_analysis(exclude_unread=args.exclude_unread)
