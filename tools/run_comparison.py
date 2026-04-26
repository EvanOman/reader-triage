"""One-off comparison: Sonnet 4.5 vs GPT-5 Mini on 5 calibrated articles."""

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
    V4_WEIGHTS,
    TieredBinaryScoringStrategy,
    compute_v4_dimension,
)

IDS = [
    "01kefdj6pzqjrnkk3yjcjmy15q",  # 5 highlights, 599 words highlighted
    "01khpmx71y7w7d242mtd22sbg3",  # 4 highlights, 299 words highlighted
    "01kgvf09afyhgprappzf2rc5sb",  # 0 highlights, V4=100 (high score, no engagement)
    "01khwwt669mscsm724edh7z8d8",  # 0 highlights, V4=3 (low score, no engagement)
    "01khh3sj8t681pn2gxcvqap2p4",  # 1 highlight, V4=35 (mismatch)
]

MODELS = {
    "Sonnet 4.5": "anthropic/claude-sonnet-4-5-20250929",
    "Qwen3-32B": "groq/qwen/qwen3-32b",
    "GPT-OSS 20B": "groq/openai/gpt-oss-20b",
}

DIM_NAMES = {
    "quotability": "Quotability",
    "surprise": "Surprise",
    "argument": "Argument",
    "insight": "Insight",
}


async def score_article(model_id: str, article: dict) -> dict:
    strategy = TieredBinaryScoringStrategy(model_id=model_id)
    result = await strategy.score(
        title=article["title"],
        author=article["author"],
        content=article["content"],
        word_count=article["word_count"],
        content_type_hint="article",
        entity_id=article["id"],
    )
    responses = {}
    if result and result.raw_responses:
        for q in range(1, 25):
            responses[f"q{q}"] = bool(result.raw_responses.get(f"q{q}", False))
    dims = {d: compute_v4_dimension(d, responses) for d in V4_DIMENSION_QUESTIONS}

    in_tok = out_tok = 0
    if strategy._lm.history:
        usage = strategy._lm.history[-1].get("response", {}).get("usage", {})
        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        out_tok = usage.get("output_tokens") or usage.get("completion_tokens", 0)

    return {
        "total": result.total if result else 0,
        "dims": dims,
        "responses": responses,
        "in": in_tok,
        "out": out_tok,
    }


async def main() -> None:
    conn = sqlite3.connect(str(_PROJECT_ROOT / "reader_triage.db"))
    conn.row_factory = sqlite3.Row

    articles = []
    for aid in IDS:
        row = conn.execute(
            """
            SELECT a.id, a.title, a.author, a.word_count, a.content, a.category,
                   a.num_highlights, a.highlighted_words,
                   v4.info_score as existing_v4
            FROM articles a
            JOIN article_scores_v4 v4 ON v4.article_id = a.id
            WHERE a.id = ?
            """,
            (aid,),
        ).fetchone()
        if row:
            articles.append(dict(row))
    conn.close()

    print(f"Selected {len(articles)} articles:")
    for i, a in enumerate(articles):
        print(
            f"  {i + 1}. [V4={a['existing_v4']:>3}] hl={a['num_highlights']:>2}"
            f"  hw={a['highlighted_words'] or 0:>4}  {a['title'][:60]}"
        )

    all_results: dict[str, list[dict]] = {}
    for model_name, model_id in MODELS.items():
        print(f"\n--- Scoring with {model_name} ---")
        results = []
        for i, article in enumerate(articles):
            print(f"  [{i + 1}/{len(articles)}] {article['title'][:50]}...", end=" ", flush=True)
            try:
                r = await score_article(model_id, article)
                print(f"score={r['total']}")
            except Exception as e:
                print(f"ERROR: {e}")
                r = {
                    "total": 0,
                    "dims": {d: 0 for d in V4_DIMENSION_QUESTIONS},
                    "responses": {},
                    "in": 0,
                    "out": 0,
                }
            results.append(r)
        all_results[model_name] = results

    # === COMPARISON TABLE ===
    model_names = list(all_results.keys())
    ref = model_names[0]  # Sonnet is the reference
    n = len(articles)

    print()
    print("=" * 130)
    header = f"{'Article':<40} {'HL':>3} {'HW':>5} {'Stored':>6}"
    for name in model_names:
        short = name[:8]
        header += f" {short:>8}"
    print(header)
    print("=" * 130)

    # Per-model agreement totals (vs reference)
    agree_totals = {name: 0 for name in model_names}

    for i, a in enumerate(articles):
        row = (
            f"{a['title'][:39]:<40} {a['num_highlights']:>3}"
            f" {a['highlighted_words'] or 0:>5} {a['existing_v4']:>6}"
        )
        for name in model_names:
            row += f" {all_results[name][i]['total']:>8}"
        print(row)

        # Track agreement vs reference
        for name in model_names:
            agree = 0
            for q in range(1, 25):
                key = f"q{q}"
                if all_results[ref][i]["responses"].get(key) == all_results[name][i][
                    "responses"
                ].get(key):
                    agree += 1
            agree_totals[name] += agree

    # === DIMENSION BREAKDOWN ===
    print()
    for i, a in enumerate(articles):
        title_short = a["title"][:55]
        print(f"--- {title_short} (hl={a['num_highlights']}, hw={a['highlighted_words'] or 0}) ---")
        dim_header = f"  {'Dimension':<14}"
        for name in model_names:
            dim_header += f" {name[:8]:>8}"
        print(dim_header)
        for dim, label in DIM_NAMES.items():
            dim_row = f"  {label:<14}"
            for name in model_names:
                dim_row += f" {all_results[name][i]['dims'].get(dim, 0):>8}"
            print(dim_row)

        # Question disagreements vs reference
        for name in model_names[1:]:
            disagrees = []
            for q in range(1, 25):
                key = f"q{q}"
                rv = all_results[ref][i]["responses"].get(key)
                cv = all_results[name][i]["responses"].get(key)
                if rv != cv:
                    w = V4_WEIGHTS.get(key, 0)
                    disagrees.append((key, rv, cv, w))
            if disagrees:
                print(f"  {name} disagrees with {ref}:")
                for key, rv, cv, w in disagrees:
                    r_str = "YES" if rv else "no"
                    c_str = "YES" if cv else "no"
                    print(f"    {key:>3} (w={w:>+3}): {ref[:6]}={r_str:<3}  {name[:6]}={c_str:<3}")
        print()

    # === SUMMARY ===
    print("=" * 130)
    print("SUMMARY")
    print("=" * 130)

    for name in model_names[1:]:
        diffs = [
            abs(all_results[name][i]["total"] - all_results[ref][i]["total"]) for i in range(n)
        ]
        agree = agree_totals[name]
        print(
            f"  {ref} vs {name}: agreement={agree}/{n * 24} ({100 * agree / (n * 24):.1f}%)"
            f"  mean_diff={sum(diffs) / len(diffs):.1f}  max_diff={max(diffs)}"
        )

    print()
    for name in model_names:
        yes_count = sum(
            sum(1 for v in all_results[name][i]["responses"].values() if v) for i in range(n)
        )
        print(f"  {name:>12} yes-rate: {yes_count}/{n * 24} ({100 * yes_count / (n * 24):.1f}%)")

    # Cost
    print()
    import litellm

    for name, model_id in MODELS.items():
        bare = model_id.split("/", 1)[-1]
        info = litellm.model_cost.get(bare, {})
        in_r = info.get("input_cost_per_token", 3e-6) * 1e6
        out_r = info.get("output_cost_per_token", 15e-6) * 1e6
        total_in = sum(r["in"] for r in all_results[name])
        total_out = sum(r["out"] for r in all_results[name])
        cost = total_in * in_r / 1_000_000 + total_out * out_r / 1_000_000
        per = cost / n
        print(
            f"  {name:>12}: {total_in:>7,} in + {total_out:>6,} out"
            f" = ${cost:.4f} (${per:.4f}/article, ${per * 1305:.2f} for 1,305)"
        )


if __name__ == "__main__":
    asyncio.run(main())
