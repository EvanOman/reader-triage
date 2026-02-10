"""Calibration data loading -- merges DB scores with Readwise highlight data.

This module is the data foundation for the calibration toolkit. It provides
functions to load scored articles from the DB, fetch highlight counts from
Readwise, and merge them into a unified dataset for analysis.

Other modules (cal_report, cal_review) import from here.
"""

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urldefrag, urlparse, urlunparse

import pandas as pd
from dotenv import load_dotenv

# Project root and default paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "reader_triage.db"
_DEFAULT_CACHE = Path(__file__).resolve().parent / ".highlight_cache.json"
_ENV_PATH = _PROJECT_ROOT / ".env"


def _get_readwise_token() -> str:
    """Load and return the Readwise API token from .env."""
    load_dotenv(_ENV_PATH)
    token = os.environ.get("READWISE_TOKEN")
    if not token:
        raise RuntimeError("READWISE_TOKEN not found. Ensure it is set in .env or the environment.")
    return token


def _normalize_url(url: str | None) -> str | None:
    """Normalize a URL for matching: strip fragment, trailing slash, lowercase host."""
    if not url:
        return None
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") if parsed.path != "/" else parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return normalized


def load_scores_from_db(db_path: str = "reader_triage.db") -> pd.DataFrame:
    """Load all scored articles from the DB.

    Uses sqlite3 directly (not async SQLAlchemy) for simplicity.
    JOINs articles with article_scores on article_id.

    Returns DataFrame with columns:
        article_id, title, url, author, word_count, category, location,
        reading_progress, added_at, published_date, info_score,
        specificity_score, novelty_score, depth_score, actionability_score,
        overall_assessment, content_fetch_failed, scoring_version, scored_at
    """
    resolved = Path(db_path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / db_path

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            a.id AS article_id,
            a.title,
            a.url,
            a.author,
            a.word_count,
            a.category,
            a.location,
            a.reading_progress,
            a.readwise_created_at AS added_at,
            a.published_date,
            s.info_score,
            s.specificity_score,
            s.novelty_score,
            s.depth_score,
            s.actionability_score,
            s.overall_assessment,
            s.content_fetch_failed,
            s.scoring_version,
            s.scored_at
        FROM articles a
        INNER JOIN article_scores s ON s.article_id = a.id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Parse date columns
    for col in ["added_at", "published_date", "scored_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure boolean type for content_fetch_failed
    if "content_fetch_failed" in df.columns:
        df["content_fetch_failed"] = df["content_fetch_failed"].fillna(0).astype(bool)

    return df


def load_tags_from_db(db_path: str = "reader_triage.db") -> dict[str, list[str]]:
    """Return {article_id: [tag_slug, ...]} mapping from the DB."""
    resolved = Path(db_path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / db_path

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("SELECT article_id, tag_slug FROM article_tags")
    tags: dict[str, list[str]] = {}
    for row in cursor:
        aid = row["article_id"]
        slug = row["tag_slug"]
        tags.setdefault(aid, []).append(slug)
    conn.close()
    return tags


def fetch_highlights(
    cache_path: str | Path = _DEFAULT_CACHE,
    force: bool = False,
) -> dict[str, int]:
    """Fetch highlight counts from Readwise for all articles in the DB.

    Uses Readwise v2 export API to get highlight counts per URL, then
    matches to DB articles by normalized URL.

    Caches results in cache_path. If cache exists and is < 1 hour old,
    returns cached data unless force=True.

    Returns {article_id: num_highlights}
    """
    cache_path = Path(cache_path)

    # Check cache freshness
    if not force and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            if datetime.now(UTC) - fetched_at < timedelta(hours=1):
                return cached["data"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Cache is corrupt, refetch

    token = _get_readwise_token()

    from readwise_sdk import ReadwiseClient
    from readwise_sdk.client import READWISE_API_V2_BASE
    from readwise_sdk.v2.models import ExportBook

    client = ReadwiseClient(api_key=token)

    # Step 1: Load all article URLs from the DB
    df = load_scores_from_db()
    db_url_to_id: dict[str, str] = {}
    for _, row in df.iterrows():
        norm = _normalize_url(row["url"])
        if norm:
            db_url_to_id[norm] = row["article_id"]

    # Step 2: Fetch all books with highlights from v2 export (manual pagination)
    print("Fetching highlight exports from Readwise v2 API...")
    url_to_highlight_count: dict[str, int] = {}
    api_url = f"{READWISE_API_V2_BASE}/export/"
    params: dict = {}
    book_count = 0

    while True:
        response = client.get(api_url, params=params)
        data = response.json()
        results = data.get("results", [])

        for item in results:
            book = ExportBook.model_validate(item)
            book_count += 1
            num_hl = len(book.highlights)

            for candidate_url in [book.source_url, book.unique_url]:
                norm = _normalize_url(candidate_url)
                if norm:
                    # Keep the max highlight count if URL appears multiple times
                    url_to_highlight_count[norm] = max(url_to_highlight_count.get(norm, 0), num_hl)

        if book_count % 100 == 0:
            print(f"  Processed {book_count} books...")

        next_cursor = data.get("nextPageCursor")
        if not next_cursor:
            break
        params["pageCursor"] = str(next_cursor)

    print(f"  Total books processed: {book_count}")

    # Step 3: Match DB articles to highlight counts
    result: dict[str, int] = {}
    matched = 0
    for norm_url, article_id in db_url_to_id.items():
        hl_count = url_to_highlight_count.get(norm_url, 0)
        result[article_id] = hl_count
        if hl_count > 0:
            matched += 1

    print(f"  Matched {matched} articles with highlights out of {len(result)} total")

    # Step 4: Save to cache
    cache_data = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "data": result,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    return result


def load_dataset(
    db_path: str = "reader_triage.db",
    since: str | None = None,
    until: str | None = None,
    min_reading_progress: float = 0.0,
    exclude_content_failed: bool = True,
    location: str | None = None,
) -> pd.DataFrame:
    """Load unified dataset: scores + highlights + tags.

    This is the main entry point for analysis. Returns a DataFrame with all
    score columns plus:
        num_highlights (int) -- highlight count from Readwise
        tags (list[str]) -- tag slugs from the DB

    Filters:
        since/until: filter on added_at (ISO date strings)
        min_reading_progress: exclude articles below this threshold (0 = include all)
        exclude_content_failed: skip articles where scoring had bad content
        location: filter by Readwise location (new, later, archive)
    """
    df = load_scores_from_db(db_path)

    if df.empty:
        df["num_highlights"] = pd.Series(dtype=int)
        df["tags"] = pd.Series(dtype=object)
        return df

    # Apply filters
    if since:
        since_dt = pd.to_datetime(since)
        df = df[df["added_at"] >= since_dt]

    if until:
        until_dt = pd.to_datetime(until)
        df = df[df["added_at"] <= until_dt]

    if min_reading_progress > 0:
        df = df[df["reading_progress"].fillna(0) >= min_reading_progress]

    if exclude_content_failed:
        df = df[~df["content_fetch_failed"]]

    if location:
        df = df[df["location"] == location]

    # Merge highlight counts
    highlights = fetch_highlights()
    df["num_highlights"] = df["article_id"].map(highlights).fillna(0).astype(int)

    # Merge tags
    tags_map = load_tags_from_db(db_path)
    df["tags"] = df["article_id"].map(lambda aid: tags_map.get(aid, []))

    df = df.reset_index(drop=True)
    return df


def load_article_content(article_id: str) -> str | None:
    """Fetch full article content from Readwise v3 API.

    Uses the v3 get endpoint with withHtmlContent=true to retrieve the
    full HTML content for a single article.

    Returns the HTML content string, or None if not available.
    """
    token = _get_readwise_token()

    import httpx

    response = httpx.get(
        "https://readwise.io/api/v3/get/",
        params={"id": article_id, "withHtmlContent": "true"},
        headers={"Authorization": f"Token {token}"},
        timeout=30,
    )

    if response.status_code != 200:
        return None

    data = response.json()
    # v3 get returns the document directly or in results
    if isinstance(data, dict):
        # Try direct content field
        content = data.get("html_content") or data.get("content")
        if content:
            return content
        # Try results list
        results = data.get("results", [])
        if results:
            doc = results[0]
            return doc.get("html_content") or doc.get("content")

    return None


def get_article_details(article_id: str, db_path: str = "reader_triage.db") -> dict | None:
    """Get complete details for one article: metadata, scores, highlights, tags.

    Returns a dict with all fields, or None if the article is not found.
    """
    resolved = Path(db_path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / db_path

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT
            a.id AS article_id,
            a.title,
            a.url,
            a.author,
            a.word_count,
            a.category,
            a.location,
            a.reading_progress,
            a.readwise_created_at AS added_at,
            a.published_date,
            a.site_name,
            s.info_score,
            s.specificity_score,
            s.novelty_score,
            s.depth_score,
            s.actionability_score,
            s.overall_assessment,
            s.score_reasons,
            s.content_fetch_failed,
            s.scoring_version,
            s.scored_at,
            s.priority_score,
            s.author_boost,
            s.skip_recommended,
            s.skip_reason,
            s.model_used
        FROM articles a
        LEFT JOIN article_scores s ON s.article_id = a.id
        WHERE a.id = ?
        """,
        (article_id,),
    ).fetchone()

    if not row:
        conn.close()
        return None

    details = dict(row)

    # Get tags
    tag_rows = conn.execute(
        "SELECT tag_slug FROM article_tags WHERE article_id = ?", (article_id,)
    ).fetchall()
    details["tags"] = [r["tag_slug"] for r in tag_rows]

    conn.close()

    # Get highlight count
    highlights = fetch_highlights()
    details["num_highlights"] = highlights.get(article_id, 0)

    return details
