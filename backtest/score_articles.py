"""Score archived articles using the same Claude scoring method as the main app.

Fetches article content from Readwise v3 API, scores with Claude, and saves results.
Supports resuming from a previous run (skips already-scored articles).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

token = os.environ.get("READWISE_TOKEN")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

if not token:
    print("ERROR: READWISE_TOKEN not found in .env")
    sys.exit(1)
if not anthropic_key:
    print("ERROR: ANTHROPIC_API_KEY not found in .env")
    sys.exit(1)

from anthropic import Anthropic
from readwise_sdk import ReadwiseClient

BACKTEST_DIR = Path(__file__).resolve().parent
ARTICLES_PATH = BACKTEST_DIR / "archived_articles.json"
SCORES_PATH = BACKTEST_DIR / "scores.json"

# Point mappings for categorical responses (same as app/services/scorer.py)
STANDALONE_SCORES = {"none": 0, "a_few": 9, "several": 17, "many": 25}
NOVEL_FRAMING_POINTS = 15
CONTENT_TYPE_SCORES = {
    "original_analysis": 10,
    "opinion_with_evidence": 8,
    "informational_summary": 3,
    "product_review": 2,
    "news_or_roundup": 0,
}
AUTHOR_CONVICTION_POINTS = 12
PRACTITIONER_VOICE_POINTS = 8
COMPLETENESS_SCORES = {"complete": 5, "appears_truncated": 2, "summary_or_excerpt": 0}
NAMED_FRAMEWORK_POINTS = 12
APPLICABLE_SCORES = {"broadly": 13, "narrowly": 7, "not_really": 0}

# Same prompt as app/services/scorer.py
SCORING_PROMPT = """Evaluate this article for capture value — how likely a reader is to want to save and highlight passages.

Answer each question by selecting from the provided options.

1. STANDALONE PASSAGES: How many passages could stand alone as a saved note — a memorable phrasing, crisp claim, or striking example worth revisiting?
   Options: none / a_few / several / many

2. NOVEL FRAMING: Does it reframe a familiar topic or present a surprising, unexpected lens for understanding something?
   Options: true / false

3. CONTENT TYPE: What best describes this content?
   Options: original_analysis / opinion_with_evidence / informational_summary / product_review / news_or_roundup

4. AUTHOR CONVICTION: Does the author argue for a clear position with conviction, rather than just reporting or summarizing?
   Options: true / false

5. PRACTITIONER VOICE: Is this written from first-person practitioner experience sharing hard-won opinions?
   Options: true / false

6. CONTENT COMPLETENESS: Does the available text appear to be a complete piece?
   Options: complete / appears_truncated / summary_or_excerpt

7. NAMED FRAMEWORK: Does it introduce or organize around a named concept, framework, or mental model?
   Options: true / false

8. APPLICABLE IDEAS: Could a reader apply ideas from this in their own work or thinking?
   Options: broadly / narrowly / not_really

{content_warning}Article Title: {title}
Author: {author}
Word Count: {word_count}

Content:
{content}

Respond with ONLY a JSON object (no markdown, no extra text):
{{"standalone_passages": "<none|a_few|several|many>", "quotability_reason": "<brief reason>", "novel_framing": <true or false>, "content_type": "<original_analysis|opinion_with_evidence|informational_summary|product_review|news_or_roundup>", "surprise_reason": "<brief reason>", "author_conviction": <true or false>, "practitioner_voice": <true or false>, "content_completeness": "<complete|appears_truncated|summary_or_excerpt>", "argument_reason": "<brief reason>", "named_framework": <true or false>, "applicable_ideas": "<broadly|narrowly|not_really>", "insight_reason": "<brief reason>", "overall_assessment": "<1-2 sentence summary>"}}"""

MAX_CONTENT_LENGTH = 15000


def score_article(
    client: Anthropic, title: str, author: str, word_count: int, content: str
) -> dict | None:
    """Score a single article using Claude. Returns parsed score dict or None."""
    if not content:
        return None

    # Detect likely truncated/incomplete content from API
    content_warning = ""
    if word_count and word_count > 1000 and len(content) < 500:
        content_warning = (
            "NOTE: The content below appears incomplete (much shorter than the "
            f"reported {word_count} words). Score based only on what is "
            "available, but note this limitation in your assessment.\n\n"
        )

    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH] + "... [truncated]"

    prompt = SCORING_PROMPT.format(
        title=title,
        author=author or "Unknown",
        word_count=word_count or "Unknown",
        content=content,
        content_warning=content_warning,
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()

        data = json.loads(text)

        # Map categorical responses to numeric scores
        quotability = STANDALONE_SCORES.get(data.get("standalone_passages", "none"), 0)
        surprise = (NOVEL_FRAMING_POINTS if data.get("novel_framing") else 0) + (
            CONTENT_TYPE_SCORES.get(data.get("content_type", ""), 0)
        )
        argument = (
            (AUTHOR_CONVICTION_POINTS if data.get("author_conviction") else 0)
            + (PRACTITIONER_VOICE_POINTS if data.get("practitioner_voice") else 0)
            + COMPLETENESS_SCORES.get(data.get("content_completeness", ""), 0)
        )
        insight = (
            NAMED_FRAMEWORK_POINTS if data.get("named_framework") else 0
        ) + APPLICABLE_SCORES.get(data.get("applicable_ideas", ""), 0)

        return {
            "specificity_score": min(25, max(0, quotability)),
            "novelty_score": min(25, max(0, surprise)),
            "depth_score": min(25, max(0, argument)),
            "actionability_score": min(25, max(0, insight)),
            "overall_assessment": data.get("overall_assessment", ""),
            "raw_responses": {
                "standalone_passages": data.get("standalone_passages"),
                "novel_framing": data.get("novel_framing"),
                "content_type": data.get("content_type"),
                "author_conviction": data.get("author_conviction"),
                "practitioner_voice": data.get("practitioner_voice"),
                "content_completeness": data.get("content_completeness"),
                "named_framework": data.get("named_framework"),
                "applicable_ideas": data.get("applicable_ideas"),
            },
        }
    except Exception as e:
        print(f"    Error scoring: {e}")
        return None


def main():
    # Load articles
    with open(ARTICLES_PATH) as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} archived articles")

    # Load existing scores for resume support
    existing_scores = {}
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            for s in json.load(f):
                existing_scores[s["doc_id"]] = s
        print(f"Found {len(existing_scores)} existing scores (will skip)")

    # Initialize clients
    readwise = ReadwiseClient(api_key=token)
    anthropic = Anthropic(api_key=anthropic_key)

    scores = list(existing_scores.values())
    scored_ids = set(existing_scores.keys())
    to_score = [a for a in articles if a["doc_id"] not in scored_ids]

    print(f"Articles to score: {len(to_score)}")
    print()

    failed = 0
    no_content = 0

    for i, article in enumerate(to_score, 1):
        doc_id = article["doc_id"]
        title = article["title"] or "Untitled"
        print(f"[{i}/{len(to_score)}] {title[:60]}...", end=" ", flush=True)

        # Fetch content from Readwise v3 API
        try:
            doc = readwise.v3.get_document(doc_id, with_content=True)
            content = doc.content or doc.summary or ""
        except Exception as e:
            print(f"FETCH ERROR: {e}")
            failed += 1
            continue

        if not content or len(content.strip()) < 50:
            print("NO CONTENT")
            # Save with zero scores so we don't re-fetch
            scores.append(
                {
                    "doc_id": doc_id,
                    "info_score": 0,
                    "specificity_score": 0,
                    "novelty_score": 0,
                    "depth_score": 0,
                    "actionability_score": 0,
                    "overall_assessment": "No content available for scoring",
                }
            )
            no_content += 1
            continue

        # Score with Claude
        result = score_article(
            anthropic,
            title=title,
            author=article.get("author", "Unknown"),
            word_count=article.get("word_count", 0),
            content=content,
        )

        if result is None:
            print("SCORE ERROR")
            failed += 1
            continue

        info_score = (
            result["specificity_score"]
            + result["novelty_score"]
            + result["depth_score"]
            + result["actionability_score"]
        )

        scores.append(
            {
                "doc_id": doc_id,
                "info_score": info_score,
                **result,
            }
        )

        print(f"SCORE={info_score}")

        # Save after every 10 articles for crash recovery
        if i % 10 == 0:
            with open(SCORES_PATH, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"  [checkpoint: {len(scores)} scores saved]")

        # Small delay to be nice to APIs
        time.sleep(0.5)

    # Final save
    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Scored {len(scores)} articles total")
    print(f"  New scores: {len(to_score) - failed - no_content}")
    print(f"  No content: {no_content}")
    print(f"  Failed: {failed}")
    print(f"  Resumed: {len(existing_scores)}")
    print(f"Saved to {SCORES_PATH}")


if __name__ == "__main__":
    main()
