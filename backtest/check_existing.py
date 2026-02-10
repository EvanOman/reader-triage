"""Check existing reader_triage database for scored articles and report statistics."""

import json
import sqlite3
import statistics
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "reader_triage.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "db_stats.json"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # --- Total articles ---
    c.execute("SELECT COUNT(*) FROM articles")
    total_articles = c.fetchone()[0]
    print(f"Total articles: {total_articles}")

    # --- Location distribution ---
    c.execute("""
        SELECT COALESCE(location, 'NULL') as loc, COUNT(*) as cnt
        FROM articles
        GROUP BY location
        ORDER BY cnt DESC
    """)
    location_dist = {row["loc"]: row["cnt"] for row in c.fetchall()}
    print("\nLocation distribution:")
    for loc, cnt in location_dist.items():
        print(f"  {loc:>10}: {cnt}")

    # --- Articles with scores ---
    c.execute("SELECT COUNT(*) FROM article_scores")
    total_scored = c.fetchone()[0]
    print(f"\nArticles with scores: {total_scored} / {total_articles}")

    # --- Scored articles by location ---
    c.execute("""
        SELECT COALESCE(a.location, 'NULL') as loc, COUNT(*) as cnt
        FROM articles a
        JOIN article_scores s ON a.id = s.article_id
        GROUP BY a.location
        ORDER BY cnt DESC
    """)
    scored_by_location = {row["loc"]: row["cnt"] for row in c.fetchall()}
    print("\nScored articles by location:")
    for loc, cnt in scored_by_location.items():
        print(f"  {loc:>10}: {cnt}")

    # --- Archived articles with scores ---
    c.execute("""
        SELECT COUNT(*) FROM articles a
        JOIN article_scores s ON a.id = s.article_id
        WHERE a.location = 'archive'
    """)
    archived_scored = c.fetchone()[0]
    print(f"\nArchived articles with scores: {archived_scored}")

    # --- Score distributions ---
    score_columns = [
        "info_score",
        "specificity_score",
        "novelty_score",
        "depth_score",
        "actionability_score",
    ]

    distributions = {}
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)

    for col in score_columns:
        c.execute(f"SELECT {col} FROM article_scores WHERE {col} IS NOT NULL")
        values = [row[0] for row in c.fetchall()]

        if not values:
            print(f"\n{col}: no data")
            distributions[col] = None
            continue

        dist = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        }
        distributions[col] = dist

        print(f"\n{col}:")
        print(f"  count:  {dist['count']}")
        print(f"  min:    {dist['min']}")
        print(f"  max:    {dist['max']}")
        print(f"  mean:   {dist['mean']}")
        print(f"  median: {dist['median']}")
        print(f"  stdev:  {dist['stdev']}")

    # --- Histogram of info_score buckets ---
    c.execute("SELECT info_score FROM article_scores WHERE info_score IS NOT NULL")
    info_scores = [row[0] for row in c.fetchall()]
    buckets = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for s in info_scores:
        if s < 20:
            buckets["0-19"] += 1
        elif s < 40:
            buckets["20-39"] += 1
        elif s < 60:
            buckets["40-59"] += 1
        elif s < 80:
            buckets["60-79"] += 1
        else:
            buckets["80-100"] += 1

    print("\ninfo_score histogram:")
    for bucket, cnt in buckets.items():
        bar = "#" * cnt
        print(f"  {bucket:>6}: {cnt:3d} {bar}")

    # --- Value tier breakdown ---
    c.execute("""
        SELECT
            SUM(CASE WHEN info_score >= 60 THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN info_score >= 30 AND info_score < 60 THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN info_score < 30 THEN 1 ELSE 0 END) as low
        FROM article_scores
    """)
    row = c.fetchone()
    tiers = {"high_ge60": row["high"], "medium_30_59": row["medium"], "low_lt30": row["low"]}
    print("\nValue tiers:")
    print(f"  High (>=60):  {tiers['high_ge60']}")
    print(f"  Medium (30-59): {tiers['medium_30_59']}")
    print(f"  Low (<30):    {tiers['low_lt30']}")

    # --- Top 5 highest-scored articles ---
    c.execute("""
        SELECT a.title, s.info_score, s.specificity_score, s.novelty_score,
               s.depth_score, s.actionability_score, a.location, a.url
        FROM article_scores s
        JOIN articles a ON a.id = s.article_id
        ORDER BY s.info_score DESC
        LIMIT 5
    """)
    top5 = [dict(row) for row in c.fetchall()]

    print(f"\n{'=' * 70}")
    print("TOP 5 HIGHEST-SCORED ARTICLES")
    print("=" * 70)
    for i, art in enumerate(top5, 1):
        print(f"\n{i}. [{art['info_score']:.0f}] {art['title']}")
        print(
            f"   Spec={art['specificity_score']} Nov={art['novelty_score']} "
            f"Dep={art['depth_score']} Act={art['actionability_score']}"
        )
        print(f"   Location: {art['location']}  URL: {art['url'][:80]}")

    # --- Bottom 5 lowest-scored articles ---
    c.execute("""
        SELECT a.title, s.info_score, s.specificity_score, s.novelty_score,
               s.depth_score, s.actionability_score, a.location, a.url
        FROM article_scores s
        JOIN articles a ON a.id = s.article_id
        ORDER BY s.info_score ASC
        LIMIT 5
    """)
    bottom5 = [dict(row) for row in c.fetchall()]

    print(f"\n{'=' * 70}")
    print("BOTTOM 5 LOWEST-SCORED ARTICLES")
    print("=" * 70)
    for i, art in enumerate(bottom5, 1):
        print(f"\n{i}. [{art['info_score']:.0f}] {art['title']}")
        print(
            f"   Spec={art['specificity_score']} Nov={art['novelty_score']} "
            f"Dep={art['depth_score']} Act={art['actionability_score']}"
        )
        print(f"   Location: {art['location']}  URL: {art['url'][:80]}")

    # --- Build JSON summary ---
    summary = {
        "total_articles": total_articles,
        "location_distribution": location_dist,
        "total_scored": total_scored,
        "scored_by_location": scored_by_location,
        "archived_scored": archived_scored,
        "score_distributions": distributions,
        "info_score_histogram": buckets,
        "value_tiers": tiers,
        "top_5_highest": [
            {
                "title": a["title"],
                "info_score": a["info_score"],
                "specificity": a["specificity_score"],
                "novelty": a["novelty_score"],
                "depth": a["depth_score"],
                "actionability": a["actionability_score"],
                "location": a["location"],
                "url": a["url"],
            }
            for a in top5
        ],
        "bottom_5_lowest": [
            {
                "title": a["title"],
                "info_score": a["info_score"],
                "specificity": a["specificity_score"],
                "novelty": a["novelty_score"],
                "depth": a["depth_score"],
                "actionability": a["actionability_score"],
                "location": a["location"],
                "url": a["url"],
            }
            for a in bottom5
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"JSON summary saved to: {OUTPUT_PATH}")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
