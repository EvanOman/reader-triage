"""Collect highlight data from Readwise for archived articles.

Fetches archived articles from v3 API and highlight data from v2 export API,
then matches them by source_url to compute highlight statistics per article.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urldefrag, urlparse, urlunparse

# Load environment
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Read the token (our .env uses READWISE_TOKEN, not READWISE_API_KEY)
import os

token = os.environ.get("READWISE_TOKEN")
if not token:
    print("ERROR: READWISE_TOKEN not found in .env")
    sys.exit(1)

from readwise_sdk import ReadwiseClient
from readwise_sdk.client import READWISE_API_V2_BASE
from readwise_sdk.v2.models import ExportBook


def export_highlights_safe(client: ReadwiseClient):
    """Export highlights with manual pagination to handle integer cursors.

    The SDK's parse_pagination_cursor assumes string cursors, but the v2 export
    endpoint returns integer cursors for nextPageCursor. This works around that.
    """
    url = f"{READWISE_API_V2_BASE}/export/"
    params: dict = {}

    while True:
        response = client.get(url, params=params)
        data = response.json()

        results = data.get("results", [])
        for item in results:
            yield ExportBook.model_validate(item)

        next_cursor = data.get("nextPageCursor")
        if not next_cursor:
            break

        params["pageCursor"] = str(next_cursor)


def normalize_url(url: str | None) -> str | None:
    """Normalize a URL for matching: strip fragment, trailing slash, lowercase host."""
    if not url:
        return None
    # Remove fragment
    url, _ = urldefrag(url)
    # Parse and normalize
    parsed = urlparse(url)
    # Lowercase the scheme and host
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") if parsed.path != "/" else parsed.path,
            parsed.params,
            parsed.query,
            "",  # no fragment
        )
    )
    return normalized


def main() -> None:
    print("Initializing Readwise client...")
    client = ReadwiseClient(api_key=token)

    # Step 1: Fetch all archived articles from v3
    print("\n=== Fetching archived articles from v3 API ===")
    archived_docs = []
    for i, doc in enumerate(client.v3.get_archive(), 1):
        archived_docs.append(doc)
        if i % 50 == 0:
            print(f"  Fetched {i} archived documents...")

    print(f"  Total archived documents: {len(archived_docs)}")

    # Step 2: Fetch all books with highlights from v2 export
    print("\n=== Fetching highlight exports from v2 API ===")
    export_books = []
    for i, book in enumerate(export_highlights_safe(client), 1):
        export_books.append(book)
        if i % 50 == 0:
            print(f"  Fetched {i} export books...")

    print(f"  Total export books: {len(export_books)}")

    # Step 3: Build lookup index from v2 books by source_url and unique_url
    print("\n=== Building URL lookup index ===")
    # Map normalized URL -> ExportBook
    url_to_book: dict[str, list] = {}

    for book in export_books:
        urls_to_index = []
        if book.source_url:
            urls_to_index.append(book.source_url)
        if book.unique_url:
            urls_to_index.append(book.unique_url)

        for url in urls_to_index:
            norm = normalize_url(url)
            if norm:
                if norm not in url_to_book:
                    url_to_book[norm] = []
                url_to_book[norm].append(book)

    print(f"  Indexed {len(url_to_book)} unique URLs from v2 export")

    # Step 4: Match v3 docs to v2 books and compute highlight stats
    print("\n=== Matching articles to highlights ===")
    results = []
    matched = 0
    unmatched = 0

    for doc in archived_docs:
        # Try to find matching v2 book
        matching_book = None

        # Try source_url first, then url
        for candidate_url in [doc.source_url, doc.url]:
            norm = normalize_url(candidate_url)
            if norm and norm in url_to_book:
                # Pick the book with the most highlights if multiple matches
                books = url_to_book[norm]
                matching_book = max(books, key=lambda b: len(b.highlights))
                break

        # Compute highlight statistics
        num_highlights = 0
        total_highlighted_words = 0

        if matching_book:
            matched += 1
            num_highlights = len(matching_book.highlights)
            for hl in matching_book.highlights:
                total_highlighted_words += len(hl.text.split())
        else:
            unmatched += 1

        results.append(
            {
                "doc_id": doc.id,
                "title": doc.title,
                "author": doc.author,
                "url": doc.url,
                "source_url": doc.source_url,
                "word_count": doc.word_count,
                "category": doc.category.value if doc.category else None,
                "site_name": doc.site_name,
                "reading_progress": doc.reading_progress,
                "num_highlights": num_highlights,
                "total_highlighted_words": total_highlighted_words,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
            }
        )

    print(f"  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")

    # Step 5: Save results
    output_path = Path(__file__).resolve().parent / "archived_articles.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Done ===")
    print(f"Saved {len(results)} articles to {output_path}")

    # Print some summary stats
    with_highlights = [r for r in results if r["num_highlights"] > 0]
    print(f"\nArticles with highlights: {len(with_highlights)}")
    if with_highlights:
        avg_highlights = sum(r["num_highlights"] for r in with_highlights) / len(with_highlights)
        avg_words = sum(r["total_highlighted_words"] for r in with_highlights) / len(
            with_highlights
        )
        print(f"Average highlights per article (with highlights): {avg_highlights:.1f}")
        print(f"Average highlighted words per article (with highlights): {avg_words:.1f}")


if __name__ == "__main__":
    main()
