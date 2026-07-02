# Quality Audit — July 2026

One-pass audit of scorer output quality, per the distribution/calibration/quality
goal. Companion to `docs/calibration-log.md`.

## "Why this scored" lines — PASS, no prompt change

Spot-checked the 10 most recently scored articles. Reasons cite article
specifics (the "'great renovation' framing and stats about leader-worker
perception gaps", "the claim about startups competing to make Anthropic lose
the most money", "'lowest possible human surface area' and the 'magic number
is 149' line") rather than restating the rubric. The digest reuses the first
score_reason as its one-line why.

## Slop in the ≥60 tier — FIXED (the big one)

**326 Readwise `highlight` documents and 17 `note` documents were being
LLM-scored as articles**; 64 of them sat in the ≥60 tier as "Untitled".
Scoring the user's own saved passages is circular (they are by definition
quotable) and they were one `first_synced_at` away from wasting daily digest
slots. Also burned summarizer calls (all 4 summaries in the last 30 days were
for highlight/note docs).

Fixes:
- Scanner skips scoring `highlight`/`note` categories (metadata row only) —
  `EXCLUDED_SCORING_CATEGORIES` in `app/services/scorer.py`. Tagger and
  summarizer join through `ArticleScore`, so the gate covers them too.
- Digest selection excludes those categories defensively.
- Existing 340 highlight/note scores demoted with `skip_recommended=1`,
  reason "Readwise highlight/note, not an article". Zero Untitled entries
  remain in the eligible ≥60 tier (399 articles).

## Dedup — FIXED

No duplicate URLs in the ≥60 tier, but 8 title-level duplicates (same article
saved twice under different Readwise IDs). Digest selection now dedupes by
lowercased title, per channel — within a batch and against prior sends.
"Untitled" is exempt from title dedup.

## Tags — PASS, no change

Closed vocabulary of 18 slugs, mean 2.4 tags/article (max 6). Distribution is
sane (ai-agents 370 → parenting 2); no vague or redundant slugs. No tagger
changes.

## Metadata bug — FIXED

Every score row recorded `model_used="claude-sonnet-4-5-20250929"` (hardcoded)
even though scoring runs on GPT-5.4 via LiteLLM. Strategies now expose
`model_id` and the scorer records the actual model.

## 👍/👎 capture — SHIPPED (part of distribution work)

Each digest item carries feedback links to
`/api/feedback/{exposure_id}/{up|down}`, recorded on the exposure event.
Covered by integration tests; endpoint verified reachable through the
Tailscale URL.
