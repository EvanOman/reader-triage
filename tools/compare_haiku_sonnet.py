"""Compare Haiku vs Sonnet 4.5 on V4 scoring for 5 articles.

Picks 5 articles spanning the existing V4 score range, re-scores each with
both models using the identical V4 prompt, and prints a detailed comparison.

Usage:
    python -m tools.compare_haiku_sonnet
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from anthropic import Anthropic  # noqa: E402
from anthropic.types import TextBlock  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.scoring_strategy import (  # noqa: E402
    _V4_ARTICLE_PROMPT,
    _V4_SYSTEM_PROMPT,
    V4_DIMENSION_QUESTIONS,
    V4_WEIGHTS,
    compute_v4_dimension,
    compute_v4_total,
)

SONNET = "claude-sonnet-4-5-20250929"
HAIKU = "claude-haiku-4-5-20251001"

DIMENSION_NAMES = {
    "quotability": "Quotability",
    "surprise": "Surprise",
    "argument": "Argument",
    "insight": "Insight",
}


def pick_articles(db_path: Path, n: int = 5) -> list[dict]:
    """Pick n articles spanning the V4 score range."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get score range
    row = conn.execute(
        "SELECT MIN(info_score) as lo, MAX(info_score) as hi FROM article_scores_v4"
    ).fetchone()
    lo, hi = row["lo"], row["hi"]

    # Pick articles at roughly evenly-spaced score points
    targets = [lo + i * (hi - lo) // (n - 1) for i in range(n)]
    articles = []
    used_ids = set()

    for target in targets:
        row = conn.execute(
            """
            SELECT a.id, a.title, a.author, a.word_count, a.content, a.category,
                   v4.info_score as existing_v4
            FROM articles a
            JOIN article_scores_v4 v4 ON v4.article_id = a.id
            WHERE a.id NOT IN ({})
              AND a.content IS NOT NULL
              AND length(a.content) > 200
            ORDER BY ABS(v4.info_score - ?) ASC
            LIMIT 1
            """.format(",".join(f"'{x}'" for x in used_ids) if used_ids else "''"),
            (target,),
        ).fetchone()

        if row:
            articles.append(dict(row))
            used_ids.add(row["id"])

    conn.close()
    return articles


def score_with_model(client: Anthropic, model: str, article: dict) -> dict:
    """Score an article with the given model using the V4 prompt."""
    content = article["content"]
    max_len = 15000
    if len(content) > max_len:
        content = content[:max_len] + "... [content trimmed for evaluation]"

    prompt = _V4_ARTICLE_PROMPT.format(
        title=article["title"],
        author=article["author"] or "Unknown",
        word_count=article["word_count"] or "Unknown",
        content=content,
        content_warning="",
    )

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        system=_V4_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    first_block = response.content[0]
    assert isinstance(first_block, TextBlock)
    text = first_block.text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Regex fallback
        data = {}
        for match in re.finditer(r'"(q\d{1,2})"\s*:\s*(true|false)', text):
            data[match.group(1)] = match.group(2) == "true"
        for match in re.finditer(r'"(q\d{1,2}_evidence)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
            data[match.group(1)] = match.group(2)
        oa_match = re.search(r'"overall_assessment"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if oa_match:
            data["overall_assessment"] = oa_match.group(1)

    responses = {}
    for i in range(1, 25):
        key = f"q{i}"
        responses[key] = bool(data.get(key, False))

    total = compute_v4_total(responses)
    dims = {d: compute_v4_dimension(d, responses) for d in V4_DIMENSION_QUESTIONS}

    return {
        "total": total,
        "dims": dims,
        "responses": responses,
        "data": data,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def print_comparison(articles: list[dict], results: list[tuple[dict, dict]]) -> None:
    """Print a detailed comparison table."""
    print("\n" + "=" * 90)
    print("HAIKU vs SONNET 4.5 — V4 SCORING COMPARISON (5 articles)")
    print("=" * 90)

    total_agree = 0
    total_questions = 0
    total_sonnet_yes = 0
    total_haiku_yes = 0
    sonnet_tokens_in = 0
    sonnet_tokens_out = 0
    haiku_tokens_in = 0
    haiku_tokens_out = 0

    for idx, (article, (sonnet_r, haiku_r)) in enumerate(zip(articles, results, strict=True)):
        print(f"\n{'─' * 90}")
        print(f"Article {idx + 1}: {article['title'][:70]}")
        print(
            f"  Existing V4: {article['existing_v4']}  |  Sonnet: {sonnet_r['total']}  |  Haiku: {haiku_r['total']}  |  Δ = {haiku_r['total'] - sonnet_r['total']:+d}"
        )

        # Dimension comparison
        print(f"  {'Dimension':<14} {'Sonnet':>7} {'Haiku':>7} {'Δ':>5}")
        for dim, label in DIMENSION_NAMES.items():
            s = sonnet_r["dims"][dim]
            h = haiku_r["dims"][dim]
            print(f"  {label:<14} {s:>7} {h:>7} {h - s:>+5}")

        # Question-level disagreements
        disagrees = []
        for i in range(1, 25):
            q = f"q{i}"
            s_val = sonnet_r["responses"][q]
            h_val = haiku_r["responses"][q]
            if s_val != h_val:
                weight = V4_WEIGHTS[q]
                disagrees.append((q, s_val, h_val, weight))

        agree = 24 - len(disagrees)
        total_agree += agree
        total_questions += 24

        s_yes = sum(1 for q in sonnet_r["responses"].values() if q)
        h_yes = sum(1 for q in haiku_r["responses"].values() if q)
        total_sonnet_yes += s_yes
        total_haiku_yes += h_yes

        sonnet_tokens_in += sonnet_r["input_tokens"]
        sonnet_tokens_out += sonnet_r["output_tokens"]
        haiku_tokens_in += haiku_r["input_tokens"]
        haiku_tokens_out += haiku_r["output_tokens"]

        print(f"  Agreement: {agree}/24 questions  |  Sonnet yes: {s_yes}  |  Haiku yes: {h_yes}")

        if disagrees:
            print("  Disagreements:")
            for q, s_val, h_val, weight in disagrees:
                s_str = "YES" if s_val else "no"
                h_str = "YES" if h_val else "no"
                print(f"    {q:>3} (w={weight:>+3}): Sonnet={s_str:<3}  Haiku={h_str:<3}")

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    score_diffs = [r[1]["total"] - r[0]["total"] for r in results]
    abs_diffs = [abs(d) for d in score_diffs]

    print(
        f"  Overall question agreement: {total_agree}/{total_questions} ({100 * total_agree / total_questions:.1f}%)"
    )
    print(f"  Mean absolute score difference: {sum(abs_diffs) / len(abs_diffs):.1f} points")
    print(f"  Max absolute score difference: {max(abs_diffs)} points")
    print(
        f"  Sonnet avg yes-rate: {total_sonnet_yes / (5 * 24) * 100:.1f}%  |  Haiku avg yes-rate: {total_haiku_yes / (5 * 24) * 100:.1f}%"
    )

    # Cost comparison
    # Sonnet: $3/M input, $15/M output; Haiku: $0.80/M input, $4/M output
    sonnet_cost = sonnet_tokens_in * 3 / 1_000_000 + sonnet_tokens_out * 15 / 1_000_000
    haiku_cost = haiku_tokens_in * 0.80 / 1_000_000 + haiku_tokens_out * 4 / 1_000_000

    print("\n  Token usage (5 articles):")
    print(f"    Sonnet: {sonnet_tokens_in:,} in + {sonnet_tokens_out:,} out = ${sonnet_cost:.4f}")
    print(f"    Haiku:  {haiku_tokens_in:,} in + {haiku_tokens_out:,} out = ${haiku_cost:.4f}")
    print(f"    Cost ratio: Haiku is {sonnet_cost / haiku_cost:.1f}x cheaper")

    # Per-article cost extrapolation
    per_article_sonnet = sonnet_cost / 5
    per_article_haiku = haiku_cost / 5
    print(
        f"\n  Per-article cost: Sonnet ${per_article_sonnet:.4f}  |  Haiku ${per_article_haiku:.4f}"
    )
    print(
        f"  For 1,305 articles: Sonnet ${per_article_sonnet * 1305:.2f}  |  Haiku ${per_article_haiku * 1305:.2f}"
    )


def main() -> None:
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    db_path = _PROJECT_ROOT / "reader_triage.db"

    print("Selecting 5 articles spanning the V4 score range...")
    articles = pick_articles(db_path, 5)

    print(f"Selected {len(articles)} articles:")
    for i, a in enumerate(articles):
        print(f"  {i + 1}. [{a['existing_v4']:>2}] {a['title'][:70]}")

    results: list[tuple[dict, dict]] = []
    for i, article in enumerate(articles):
        print(f"\nScoring article {i + 1}/{len(articles)}: {article['title'][:50]}...")

        print("  Sonnet 4.5...", end=" ", flush=True)
        sonnet_r = score_with_model(client, SONNET, article)
        print(f"score={sonnet_r['total']}")

        print("  Haiku 4.5...", end=" ", flush=True)
        haiku_r = score_with_model(client, HAIKU, article)
        print(f"score={haiku_r['total']}")

        results.append((sonnet_r, haiku_r))

    print_comparison(articles, results)


if __name__ == "__main__":
    main()
