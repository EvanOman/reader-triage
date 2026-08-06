# Persona corpus

Six hand-authored articles and the scoring responses recorded for them. Driven by
`tests/test_persona_judgement.py` (the `persona` marker).

Every article here was written by hand. Nothing in this directory came from the
production database, and nothing should: `reader_triage.db` is a live 53MB
personal reading history, not a fixture source.

## Why this tier exists

The unit, integration and golden tiers pin arithmetic and byte-for-byte output.
None of them notices if the rubric inverts and starts ranking listicles above
practitioner essays — a golden test would happily pin the inverted output. This
tier pins the *shape of the outcome*: what gets skipped, what tier each item
lands in, and what order the surfaced ranking comes out in.

## The corpus

| File | What it is | Expected outcome |
| --- | --- | --- |
| `01_technical_deep_dive.json` | Practitioner post-mortem, named framework, measured evidence | high tier (100), rank #1 |
| `02_listicle.json` | Generic SEO listicle | low tier (18), `skip_recommended` |
| `03_paywalled_stub.json` | Paywalled — Readwise returned only the subscription teaser | 0, `skip_recommended`, `content_fetch_failed` |
| `04_newsletter.json` | Curated weekly newsletter, one useful observation | medium tier (48) |
| `05_non_english.json` | Substantive Spanish-language essay | high tier (84), rank #2 |
| `06_already_read.json` | Good essay, archived and 95% read | high tier (60), **not** surfaced |

Expected values live in `EXPECTED` in the test module, not here. These files
describe only the article and what the scoring LLM said about it; the test
describes what the system is expected to do with that.

## File shape

```jsonc
{
  "id": "...",                          // also the Readwise document id
  "kind": "...",                        // corpus role, for the judge prompt
  "expectation": "...",                 // prose statement of sane behaviour
  "judge_summary": "...",               // neutral description handed to the judge
  "document": { ... },                  // becomes a ReaderDocument via make_document
  "recorded_scoring_response": { ... }  // V2CategoricalOutput fields, or null
}
```

`recorded_scoring_response` is the raw categorical answer the scoring LLM gave —
not a score. The test replays it through the real
`ReweightedCategoricalScoringStrategy`, so the production mapping tables, the v5
reweighting, the author boost and the `< 30` skip threshold are all genuinely
exercised. Only the DSPy predictor is substituted.

`03_paywalled_stub.json` records `null` deliberately: the pipeline detects the
stub and returns a zeroed score *before* reaching the strategy, so there is no
response to record. The test asserts the strategy was never invoked for it.

## Two constraints worth knowing before you edit these

**Word counts.** Stub detection fires when a document reports more than 500 words
and returns less than 15% of them. Every authored body declares its real length
and stays under 500, so only the paywalled stub can trip that path. If you extend
an article past 500 words, update the declared `word_count` to match the real one
or you will silently change which code path it takes.

**Hermeticity.** These fixtures are the only thing standing between this tier and
a live scoring call. The autouse socket guard in `tests/conftest.py` stays armed
throughout — including during the judge test — so a missing or mismatched
recorded response fails loudly rather than falling through to the network.
