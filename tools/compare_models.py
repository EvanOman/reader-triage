"""Compare scoring across multiple models using V4 strategy.

Uses TieredBinaryScoringStrategy with different model_ids to compare
Sonnet 4.5 vs Haiku 4.5 vs GPT-5 Mini on article scoring.

Usage:
    python -m tools.compare_models [--limit N]
"""

import asyncio
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from app.services.scoring_strategy import (  # noqa: E402
    V4_DIMENSION_QUESTIONS,
    TieredBinaryScoringStrategy,
    compute_v4_dimension,
)

MODELS = {
    "Sonnet 4.5": "anthropic/claude-sonnet-4-5-20250929",
    "Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
    "GPT-5 Mini": "openai/gpt-5-mini",
}

DIMENSION_NAMES = {
    "quotability": "Quotability",
    "surprise": "Surprise",
    "argument": "Argument",
    "insight": "Insight",
}

# Per-million-token pricing: (input, output)
MODEL_COSTS = {
    "anthropic/claude-sonnet-4-5-20250929": (3.0, 15.0),
    "anthropic/claude-haiku-4-5-20251001": (0.25, 1.25),
    "openai/gpt-5-mini": (0.15, 0.60),
}


def pick_articles(db_path: Path, n: int = 5) -> list[dict]:
    """Pick n articles spanning the V4 score range."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT MIN(info_score) as lo, MAX(info_score) as hi FROM article_scores_v4"
    ).fetchone()
    lo, hi = row["lo"], row["hi"]
    targets = [lo + i * (hi - lo) // (n - 1) for i in range(n)]
    articles = []
    used_ids: set[str] = set()
    for target in targets:
        placeholder = ",".join(f"'{x}'" for x in used_ids) if used_ids else "''"
        row = conn.execute(
            f"""
            SELECT a.id, a.title, a.author, a.word_count, a.content, a.category,
                   v4.info_score as existing_v4
            FROM articles a
            JOIN article_scores_v4 v4 ON v4.article_id = a.id
            WHERE a.id NOT IN ({placeholder})
              AND a.content IS NOT NULL AND length(a.content) > 200
            ORDER BY ABS(v4.info_score - ?) ASC
            LIMIT 1
            """,
            (target,),
        ).fetchone()
        if row:
            articles.append(dict(row))
            used_ids.add(row["id"])
    conn.close()
    return articles


async def score_with_model(model_id: str, article: dict) -> dict:
    """Score an article using the V4 strategy with a specific model."""
    strategy = TieredBinaryScoringStrategy(model_id=model_id)
    result = await strategy.score(
        title=article["title"],
        author=article["author"],
        content=article["content"],
        word_count=article["word_count"],
        content_type_hint="article",
        entity_id=article["id"],
    )
    if result is None:
        return {
            "total": 0,
            "dims": {d: 0 for d in V4_DIMENSION_QUESTIONS},
            "responses": {},
            "error": True,
        }

    responses = {}
    if result.raw_responses:
        for i in range(1, 25):
            key = f"q{i}"
            responses[key] = bool(result.raw_responses.get(key, False))

    dims = {d: compute_v4_dimension(d, responses) for d in V4_DIMENSION_QUESTIONS}

    # Get token usage from strategy's LM history
    input_tokens = 0
    output_tokens = 0
    if strategy._lm.history:
        usage = strategy._lm.history[-1].get("response", {}).get("usage", {})
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)

    return {
        "total": result.total,
        "dims": dims,
        "responses": responses,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def print_comparison(articles: list[dict], all_results: dict[str, list[dict]]) -> None:
    """Print a detailed comparison table."""
    model_names = list(all_results.keys())

    print("\n" + "=" * 100)
    print(f"MODEL COMPARISON — V4 SCORING ({len(articles)} articles)")
    print("=" * 100)

    totals = {
        name: {"agree": 0, "questions": 0, "yes": 0, "in": 0, "out": 0} for name in model_names
    }

    for idx, article in enumerate(articles):
        print(f"\n{'---' * 34}")
        print(f"Article {idx + 1}: {article['title'][:70]}")

        scores_str = f"  Existing V4: {article['existing_v4']}"
        for name in model_names:
            r = all_results[name][idx]
            scores_str += f"  |  {name}: {r['total']}"
        print(scores_str)

        # Dimension comparison
        header = f"  {'Dimension':<14}"
        for name in model_names:
            header += f" {name:>10}"
        print(header)

        for dim, label in DIMENSION_NAMES.items():
            row = f"  {label:<14}"
            for name in model_names:
                row += f" {all_results[name][idx]['dims'].get(dim, 0):>10}"
            print(row)

        # Token tracking
        for name in model_names:
            r = all_results[name][idx]
            totals[name]["in"] += r.get("input_tokens", 0)
            totals[name]["out"] += r.get("output_tokens", 0)
            if r.get("responses"):
                totals[name]["yes"] += sum(1 for v in r["responses"].values() if v)

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    # Score differences between models
    ref_name = model_names[0]
    for name in model_names[1:]:
        diffs = [
            abs(all_results[name][i]["total"] - all_results[ref_name][i]["total"])
            for i in range(len(articles))
        ]
        print(
            f"  {ref_name} vs {name}: mean abs diff = {sum(diffs) / len(diffs):.1f}, max = {max(diffs)}"
        )

    # Cost comparison
    print(f"\n  Token usage ({len(articles)} articles):")
    for name in model_names:
        model_id = MODELS[name]
        in_rate, out_rate = MODEL_COSTS.get(model_id, (3.0, 15.0))
        t = totals[name]
        cost = t["in"] * in_rate / 1_000_000 + t["out"] * out_rate / 1_000_000
        print(f"    {name}: {t['in']:,} in + {t['out']:,} out = ${cost:.4f}")
        per_article = cost / len(articles) if articles else 0
        print(
            f"      Per-article: ${per_article:.4f}  |  For 1,305 articles: ${per_article * 1305:.2f}"
        )


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare models on V4 scoring")
    parser.add_argument("--limit", type=int, default=5, help="Number of articles")
    args = parser.parse_args()

    db_path = _PROJECT_ROOT / "reader_triage.db"

    print(f"Selecting {args.limit} articles spanning the V4 score range...")
    articles = pick_articles(db_path, args.limit)

    print(f"Selected {len(articles)} articles:")
    for i, a in enumerate(articles):
        print(f"  {i + 1}. [{a['existing_v4']:>2}] {a['title'][:70]}")

    all_results: dict[str, list[dict]] = {}

    for model_name, model_id in MODELS.items():
        print(f"\n--- Scoring with {model_name} ({model_id}) ---")
        results = []
        for i, article in enumerate(articles):
            print(f"  [{i + 1}/{len(articles)}] {article['title'][:50]}...", end=" ", flush=True)
            try:
                r = await score_with_model(model_id, article)
                print(f"score={r['total']}")
            except Exception as e:
                print(f"ERROR: {e}")
                r = {
                    "total": 0,
                    "dims": {d: 0 for d in V4_DIMENSION_QUESTIONS},
                    "responses": {},
                    "error": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            results.append(r)
        all_results[model_name] = results

    print_comparison(articles, all_results)


if __name__ == "__main__":
    asyncio.run(main())
