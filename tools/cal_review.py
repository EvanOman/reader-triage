"""Content review module for scoring calibration toolkit.

Provides CLI tools for a data analyst agent to review scoring misses
and investigate individual articles. Works alongside cal_data.py for
data access.

Functions:
    show_misses      - Find articles where score diverges from engagement
    inspect_article  - Deep dive on a single article
    show_theme_summary - Higher-level pattern analysis of misses
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

import pandas as pd

from tools.cal_data import get_article_details, load_article_content, load_dataset

# ── ANSI color codes ────────────────────────────────────────────────

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ── Helpers ─────────────────────────────────────────────────────────


def _tier_label(score: float | int | None) -> str:
    """Return colored tier label for a score."""
    if score is None:
        return f"{DIM}N/A{RESET}"
    s = int(score)
    if s >= 60:
        return f"{GREEN}High{RESET}"
    elif s >= 30:
        return f"{YELLOW}Medium{RESET}"
    else:
        return f"{RED}Low{RESET}"


def _tier_label_plain(score: float | int | None) -> str:
    """Return plain tier label (no color) for a score."""
    if score is None:
        return "N/A"
    s = int(score)
    if s >= 60:
        return "High"
    elif s >= 30:
        return "Medium"
    else:
        return "Low"


def _format_date(val) -> str:
    """Format a date value for display."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            return dt.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            return val
    if isinstance(val, datetime):
        return val.strftime("%b %d, %Y")
    if isinstance(val, pd.Timestamp):
        return val.strftime("%b %d, %Y")
    return str(val)


def _progress_bar(value: int, max_val: int = 25, width: int = 25) -> str:
    """Build a unicode progress bar: filled and empty blocks."""
    if max_val <= 0:
        return "\u2591" * width
    filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _dim_bar(value: int, max_val: int = 25) -> str:
    """Colored dimension bar with value."""
    bar = _progress_bar(value, max_val, width=25)
    ratio = value / max_val if max_val > 0 else 0
    if ratio >= 0.75:
        color = GREEN
    elif ratio >= 0.45:
        color = YELLOW
    else:
        color = RED
    return f"{color}{bar}{RESET}"


def _safe_int(val, default: int = 0) -> int:
    """Safely convert to int."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val, default: str = "") -> str:
    """Safely convert to str, handling NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val)


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _strip_html(html: str) -> str:
    """Strip HTML tags from content."""
    return re.sub(r"<[^>]+>", "", html)


def _compute_miss_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute engagement and score percentiles, plus the gap.

    Returns a copy of df with added columns:
        engagement_percentile, score_percentile, fp_gap, fn_gap
    """
    out = df.copy()

    # Rank-based percentiles (0-100). Ties get average rank.
    out["engagement_percentile"] = out["num_highlights"].rank(pct=True, method="average") * 100
    out["score_percentile"] = out["info_score"].rank(pct=True, method="average") * 100

    out["fp_gap"] = out["score_percentile"] - out["engagement_percentile"]
    out["fn_gap"] = out["engagement_percentile"] - out["score_percentile"]

    return out


# ── Public API ──────────────────────────────────────────────────────


def show_misses(
    top: int = 15,
    miss_type: str = "both",
    since: str | None = None,
    min_progress: float = 0.1,
) -> None:
    """Find and display articles where score most diverges from actual engagement.

    Args:
        top: Number of misses to show per category.
        miss_type: "fp" for false positives, "fn" for false negatives, "both".
        since: ISO date filter on added_at (e.g. "2025-01-01").
        min_progress: Minimum reading progress to include (default 0.1).
    """
    df = load_dataset(since=since, min_reading_progress=min_progress, exclude_content_failed=True)

    if df.empty:
        print(f"{RED}No articles found matching the filters.{RESET}")
        return

    # Need scores and highlight data
    scored = df.dropna(subset=["info_score"]).copy()
    if scored.empty:
        print(f"{RED}No scored articles found.{RESET}")
        return

    # Fill missing highlights with 0
    scored["num_highlights"] = scored["num_highlights"].fillna(0).astype(int)

    data = _compute_miss_data(scored)

    miss_type = miss_type.lower().strip()
    if miss_type not in ("fp", "fn", "both"):
        print(f"{RED}miss_type must be 'fp', 'fn', or 'both'. Got: {miss_type}{RESET}")
        return

    if miss_type in ("fp", "both"):
        _print_miss_section(data, kind="fp", top=top)

    if miss_type == "both":
        print()  # separator between sections

    if miss_type in ("fn", "both"):
        _print_miss_section(data, kind="fn", top=top)


def _print_miss_section(data: pd.DataFrame, kind: str, top: int) -> None:
    """Print one section of misses (FP or FN)."""
    if kind == "fp":
        label = "False Positive"
        gap_col = "fp_gap"
        ascending = False
        description = "Scored high, user didn't engage"
    else:
        label = "False Negative"
        gap_col = "fn_gap"
        ascending = False
        description = "Scored low, user engaged a lot"

    # Filter to positive gaps only and sort
    candidates = data[data[gap_col] > 0].sort_values(gap_col, ascending=ascending)
    shown = candidates.head(top)

    if shown.empty:
        print(f"{CYAN}{BOLD}  No {label.lower()}s found.{RESET}")
        return

    header = f" {label}s ({description}) "
    print(f"\n{CYAN}{BOLD}{'=' * 50}{RESET}")
    print(f"{CYAN}{BOLD}{header:^50}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 50}{RESET}")

    for rank, (_, row) in enumerate(shown.iterrows(), 1):
        score = _safe_int(row.get("info_score"))
        highlights = _safe_int(row.get("num_highlights"))
        sp = _safe_int(row.get("specificity_score"))
        nv = _safe_int(row.get("novelty_score"))
        dp = _safe_int(row.get("depth_score"))
        ac = _safe_int(row.get("actionability_score"))
        score_pct = _safe_float(row.get("score_percentile"))
        eng_pct = _safe_float(row.get("engagement_percentile"))
        gap = _safe_float(row.get(gap_col))
        assessment = _safe_str(row.get("overall_assessment"))
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        tier = _tier_label(score)

        print(f"\n{CYAN}\u2500\u2500 {label} #{rank} {'─' * 35}{RESET}")
        print(f"  {BOLD}Title:{RESET}      {_safe_str(row.get('title'))}")
        print(f"  {BOLD}URL:{RESET}        {_safe_str(row.get('url'))}")
        print(f"  {BOLD}Author:{RESET}     {_safe_str(row.get('author'), 'Unknown')}")
        print(
            f"  {BOLD}Score:{RESET}      {score} ({tier})"
            f"  {DIM}\u2194{RESET}  {BOLD}Highlights:{RESET} {highlights}"
        )
        print(f"  {BOLD}Dimensions:{RESET} Q:{sp} S:{nv} A:{dp} I:{ac}")
        if assessment:
            print(f'  {BOLD}Assessment:{RESET} "{_truncate(assessment, 70)}"')
        if tags:
            print(f"  {BOLD}Tags:{RESET}       {', '.join(tags)}")
        print(f"  {BOLD}Added:{RESET}      {_format_date(row.get('added_at'))}")
        print(
            f"  {BOLD}Gap:{RESET}        Score P{score_pct:.0f} vs Engagement P{eng_pct:.0f}"
            f" ({RED}\u0394{gap:.0f}{RESET})"
        )


def inspect_article(article_id: str, show_content: bool = False) -> None:
    """Deep dive on a single article.

    Shows all metadata, scoring breakdown with visual bars, engagement
    data, and optionally a content preview.

    Args:
        article_id: The Readwise document ID.
        show_content: If True, fetch and display the first 3000 chars
                      of stripped article content.
    """
    details = get_article_details(article_id)
    if details is None:
        print(f"{RED}Article not found: {article_id}{RESET}")
        return

    # Extract fields with safe defaults
    title = _safe_str(details.get("title"), "Untitled")
    author = _safe_str(details.get("author"), "Unknown")
    url = _safe_str(details.get("url"), "N/A")
    word_count = _safe_int(details.get("word_count"))
    category = _safe_str(details.get("category"), "Unknown")
    location = _safe_str(details.get("location"), "Unknown")
    added_at = details.get("added_at")
    published_date = details.get("published_date")
    reading_progress = _safe_float(details.get("reading_progress"))

    info_score = details.get("info_score")
    specificity = _safe_int(details.get("specificity_score"))
    novelty = _safe_int(details.get("novelty_score"))
    depth = _safe_int(details.get("depth_score"))
    actionability = _safe_int(details.get("actionability_score"))
    assessment = _safe_str(details.get("overall_assessment"))
    scoring_version = _safe_str(details.get("scoring_version"), "Unknown")
    scored_at = details.get("scored_at")

    num_highlights = _safe_int(details.get("num_highlights"))
    tags = details.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    # ── Header ──────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'═' * 50}{RESET}")
    print(f"{CYAN}{BOLD}{'  ARTICLE INSPECTION':^50}{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 50}{RESET}")

    # ── Metadata ────────────────────────────────────────────
    print(f"\n{CYAN}\u2500\u2500 Metadata {'─' * 38}{RESET}")
    print(f"  {BOLD}Title:{RESET}      {title}")
    print(f"  {BOLD}Author:{RESET}     {author}")
    print(f"  {BOLD}URL:{RESET}        {url}")
    print(
        f"  {BOLD}Words:{RESET}      {word_count:,}"
        if word_count
        else f"  {BOLD}Words:{RESET}      Unknown"
    )
    print(f"  {BOLD}Category:{RESET}   {category}")
    print(f"  {BOLD}Location:{RESET}   {location}")
    print(f"  {BOLD}Added:{RESET}      {_format_date(added_at)}")
    print(f"  {BOLD}Published:{RESET}  {_format_date(published_date)}")
    print(f"  {BOLD}Progress:{RESET}   {reading_progress:.0%}")

    # ── Scoring ─────────────────────────────────────────────
    print(f"\n{CYAN}\u2500\u2500 Scoring {'─' * 39}{RESET}")

    if info_score is not None:
        total = _safe_int(info_score)
        tier = _tier_label(total)
        print(f"  {BOLD}Total Score:{RESET}    {total} / 100  ({tier})")
        print(f"  {BOLD}Version:{RESET}        {scoring_version}")
        if scored_at:
            print(f"  {BOLD}Scored:{RESET}         {_format_date(scored_at)}")
        print()
        print(f"  {BOLD}Quotability:{RESET}    {specificity:>2} / 25  {_dim_bar(specificity)}")
        print(f"  {BOLD}Surprise:{RESET}       {novelty:>2} / 25  {_dim_bar(novelty)}")
        print(f"  {BOLD}Argument:{RESET}       {depth:>2} / 25  {_dim_bar(depth)}")
        print(f"  {BOLD}Insight:{RESET}        {actionability:>2} / 25  {_dim_bar(actionability)}")

        if assessment:
            print(f"\n  {BOLD}Assessment:{RESET}")
            # Word-wrap assessment at ~68 chars for nice display
            _print_wrapped(assessment, indent=4, width=68)
    else:
        print(f"  {DIM}Not scored yet.{RESET}")

    # ── Engagement ──────────────────────────────────────────
    print(f"\n{CYAN}\u2500\u2500 Engagement {'─' * 36}{RESET}")
    print(f"  {BOLD}Highlights:{RESET}     {num_highlights}")
    print(f"  {BOLD}Reading Prog:{RESET}   {reading_progress:.0%}")

    # ── Tags ────────────────────────────────────────────────
    print(f"\n{CYAN}\u2500\u2500 Tags {'─' * 42}{RESET}")
    if tags:
        print(f"  {', '.join(tags)}")
    else:
        print(f"  {DIM}No tags.{RESET}")

    # ── Content Preview ─────────────────────────────────────
    if show_content:
        print(f"\n{CYAN}\u2500\u2500 Content Preview {'─' * 31}{RESET}")
        html = load_article_content(article_id)
        if html:
            plain = _strip_html(html)
            # Collapse whitespace
            plain = re.sub(r"\s+", " ", plain).strip()
            preview = plain[:3000]
            if len(plain) > 3000:
                preview += "..."
            _print_wrapped(preview, indent=2, width=76)
        else:
            print(f"  {DIM}Content not available.{RESET}")

    print()


def _print_wrapped(text: str, indent: int = 4, width: int = 68) -> None:
    """Print text word-wrapped with given indent and width."""
    prefix = " " * indent
    words = text.split()
    line = prefix
    for word in words:
        if len(line) + len(word) + 1 > width + indent:
            print(line)
            line = prefix + word
        else:
            if line == prefix:
                line += word
            else:
                line += " " + word
    if line.strip():
        print(line)


def show_theme_summary(
    miss_type: str = "both",
    top: int = 20,
    since: str | None = None,
    min_progress: float = 0.1,
) -> None:
    """Higher-level analysis: group misses by common patterns.

    For false positives, shows which tags/categories/dimensions are
    inflated. For false negatives, shows which are deflated.

    Args:
        miss_type: "fp", "fn", or "both".
        top: Number of top misses to analyze per category.
        since: ISO date filter on added_at.
        min_progress: Minimum reading progress to include.
    """
    df = load_dataset(since=since, min_reading_progress=min_progress, exclude_content_failed=True)

    if df.empty:
        print(f"{RED}No articles found matching the filters.{RESET}")
        return

    scored = df.dropna(subset=["info_score"]).copy()
    if scored.empty:
        print(f"{RED}No scored articles found.{RESET}")
        return

    scored["num_highlights"] = scored["num_highlights"].fillna(0).astype(int)
    data = _compute_miss_data(scored)

    # Overall means for comparison
    overall_means = {
        "Quotability": _safe_float(data["specificity_score"].mean()),
        "Surprise": _safe_float(data["novelty_score"].mean()),
        "Argument": _safe_float(data["depth_score"].mean()),
        "Insight": _safe_float(data["actionability_score"].mean()),
    }
    overall_avg_words = _safe_float(data["word_count"].dropna().mean())

    miss_type = miss_type.lower().strip()
    if miss_type not in ("fp", "fn", "both"):
        print(f"{RED}miss_type must be 'fp', 'fn', or 'both'. Got: {miss_type}{RESET}")
        return

    if miss_type in ("fp", "both"):
        _print_theme_section(
            data,
            kind="fp",
            top=top,
            overall_means=overall_means,
            overall_avg_words=overall_avg_words,
        )

    if miss_type == "both":
        print()

    if miss_type in ("fn", "both"):
        _print_theme_section(
            data,
            kind="fn",
            top=top,
            overall_means=overall_means,
            overall_avg_words=overall_avg_words,
        )


def _print_theme_section(
    data: pd.DataFrame,
    kind: str,
    top: int,
    overall_means: dict[str, float],
    overall_avg_words: float,
) -> None:
    """Print theme analysis for one miss type."""
    if kind == "fp":
        label = "False Positive"
        gap_col = "fp_gap"
        dim_label = "Inflated"
    else:
        label = "False Negative"
        gap_col = "fn_gap"
        dim_label = "Deflated"

    candidates = data[data[gap_col] > 0].sort_values(gap_col, ascending=False)
    misses = candidates.head(top)

    if misses.empty:
        print(f"{CYAN}{BOLD}  No {label.lower()}s found.{RESET}")
        return

    header = f" {label} Themes (top {len(misses)}) "
    print(f"\n{CYAN}{BOLD}{'─' * 50}{RESET}")
    print(f"{CYAN}{BOLD}{header}{RESET}")
    print(f"{CYAN}{BOLD}{'─' * 50}{RESET}")

    # ── Common Tags ─────────────────────────────────────────
    tag_counter: Counter = Counter()
    for _, row in misses.iterrows():
        tags = row.get("tags", [])
        if isinstance(tags, list):
            tag_counter.update(tags)

    top_tags = tag_counter.most_common(8)
    if top_tags:
        tag_str = ", ".join(f"{BOLD}{tag}{RESET} ({count})" for tag, count in top_tags)
        print(f"  {BOLD}Common Tags:{RESET}     {tag_str}")
    else:
        print(f"  {BOLD}Common Tags:{RESET}     {DIM}None{RESET}")

    # ── Common Categories ───────────────────────────────────
    cat_counter: Counter = Counter()
    for _, row in misses.iterrows():
        cat = _safe_str(row.get("category"))
        if cat:
            cat_counter[cat] += 1

    top_cats = cat_counter.most_common(5)
    if top_cats:
        cat_str = ", ".join(f"{tag} ({count})" for tag, count in top_cats)
        print(f"  {BOLD}Categories:{RESET}      {cat_str}")

    # ── Average Word Count ──────────────────────────────────
    miss_words = misses["word_count"].dropna()
    if not miss_words.empty:
        miss_avg = miss_words.mean()
        direction = ""
        if kind == "fp" and miss_avg < overall_avg_words * 0.75:
            direction = f" {YELLOW}\u2190 short articles score too high?{RESET}"
        elif kind == "fn" and miss_avg > overall_avg_words * 1.25:
            direction = f" {YELLOW}\u2190 long articles score too low?{RESET}"
        print(
            f"  {BOLD}Avg Word Count:{RESET}  {miss_avg:,.0f}"
            f" (overall avg: {overall_avg_words:,.0f}){direction}"
        )

    # ── Dimension Analysis ──────────────────────────────────
    dim_map = {
        "Quotability": "specificity_score",
        "Surprise": "novelty_score",
        "Argument": "depth_score",
        "Insight": "actionability_score",
    }

    dim_deltas: list[tuple[str, float, float, float]] = []
    for dim_name, col in dim_map.items():
        miss_mean = _safe_float(misses[col].mean())
        overall_mean = overall_means[dim_name]
        delta = miss_mean - overall_mean
        dim_deltas.append((dim_name, miss_mean, overall_mean, delta))

    # Sort: for FP, most positive delta first (inflated); for FN, most negative first (deflated)
    if kind == "fp":
        dim_deltas.sort(key=lambda x: x[3], reverse=True)
    else:
        dim_deltas.sort(key=lambda x: x[3])

    print(f"\n  {BOLD}{dim_label} Dimensions ({label} mean vs overall mean):{RESET}")
    for dim_name, miss_mean, overall_mean, delta in dim_deltas:
        sign = "+" if delta >= 0 else ""
        # Flag suspect dimensions: > 2.0 delta for FP inflated, < -2.0 for FN deflated
        suspect = ""
        if kind == "fp" and delta > 2.0:
            suspect = f" {RED}\u2190 SUSPECT{RESET}"
        elif kind == "fn" and delta < -2.0:
            suspect = f" {RED}\u2190 SUSPECT{RESET}"

        print(
            f"    {dim_name + ':':<14} {miss_mean:>5.1f} vs {overall_mean:>5.1f}"
            f"  ({sign}{delta:.1f}){suspect}"
        )

    # ── Score and Engagement Summary ────────────────────────
    avg_score = _safe_float(misses["info_score"].mean())
    avg_highlights = _safe_float(misses["num_highlights"].mean())
    avg_gap = _safe_float(misses[gap_col].mean())
    print(f"\n  {BOLD}Avg Score:{RESET}       {avg_score:.1f}  ({_tier_label_plain(avg_score)})")
    print(f"  {BOLD}Avg Highlights:{RESET}  {avg_highlights:.1f}")
    print(f"  {BOLD}Avg Gap:{RESET}         {avg_gap:.1f} percentile points")
