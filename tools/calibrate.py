"""Scoring calibration toolkit.

Compare predicted scores against actual user engagement (highlights)
to calibrate the scoring model over time.

Usage:
    python -m tools.calibrate sync          # Refresh highlight data from Readwise
    python -m tools.calibrate report        # Calibration report (correlation stats)
    python -m tools.calibrate misses        # Show biggest prediction misses
    python -m tools.calibrate inspect ID    # Deep dive on one article
    python -m tools.calibrate dimensions    # Analyze which dimensions predict best
    python -m tools.calibrate trends        # Temporal calibration tracking

Or via justfile:
    just cal-sync
    just cal-report [--since 2025-01-01] [--min-progress 0.1]
    just cal-misses [--top 10] [--type fp]
    just cal-inspect <article_id> [--content]
    just cal-dimensions [--since 2025-01-01]
    just cal-trends [--window 30]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Scoring calibration toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # sync -- refresh highlight cache
    subparsers.add_parser("sync", help="Refresh highlight data from Readwise")

    # report -- calibration stats
    report_p = subparsers.add_parser("report", help="Calibration report")
    report_p.add_argument("--since", help="Start date (ISO format)")
    report_p.add_argument("--until", help="End date (ISO format)")
    report_p.add_argument(
        "--min-progress", type=float, default=0.0, help="Min reading progress (0-1)"
    )
    report_p.add_argument("--tag", help="Filter by tag slug")
    report_p.add_argument("--category", help="Filter by category")

    # misses -- prediction misses
    misses_p = subparsers.add_parser("misses", help="Show biggest prediction misses")
    misses_p.add_argument("--top", type=int, default=10, help="Number of misses to show")
    misses_p.add_argument(
        "--type",
        choices=["fp", "fn", "both"],
        default="both",
        help="false positives, false negatives, or both",
    )
    misses_p.add_argument("--since", help="Start date")
    misses_p.add_argument("--min-progress", type=float, default=0.1, help="Min reading progress")

    # inspect -- single article deep dive
    inspect_p = subparsers.add_parser("inspect", help="Inspect a single article")
    inspect_p.add_argument("article_id", help="Article ID to inspect")
    inspect_p.add_argument(
        "--content", action="store_true", help="Also fetch and show article content"
    )

    # dimensions -- dimension analysis
    dim_p = subparsers.add_parser("dimensions", help="Analyze scoring dimensions")
    dim_p.add_argument("--since", help="Start date")
    dim_p.add_argument("--min-progress", type=float, default=0.1)

    # trends -- temporal tracking
    trends_p = subparsers.add_parser("trends", help="Temporal calibration trends")
    trends_p.add_argument("--window", type=int, default=30, help="Rolling window in days")
    trends_p.add_argument("--min-progress", type=float, default=0.1)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate module
    if args.command == "sync":
        from tools.cal_data import fetch_highlights

        highlights = fetch_highlights(force=True)
        print(f"\n\033[32mSynced highlights for {len(highlights)} articles\033[0m")

    elif args.command == "report":
        from tools.cal_report import run_report

        run_report(
            since=args.since,
            until=args.until,
            min_progress=args.min_progress,
            tag=args.tag,
            category=args.category,
        )

    elif args.command == "misses":
        from tools.cal_review import show_misses

        show_misses(
            top=args.top,
            miss_type=args.type,
            since=args.since,
            min_progress=args.min_progress,
        )

    elif args.command == "inspect":
        from tools.cal_review import inspect_article

        inspect_article(args.article_id, show_content=args.content)

    elif args.command == "dimensions":
        from tools.cal_report import run_dimensions

        run_dimensions(since=args.since, min_progress=args.min_progress)

    elif args.command == "trends":
        from tools.cal_report import run_trends

        run_trends(window=args.window, min_progress=args.min_progress)


if __name__ == "__main__":
    main()
