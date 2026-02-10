# Reader Triage

Article triage tool for Readwise Reader that surfaces high-value articles using AI-powered scoring.

## Quick Commands

```bash
just dev       # Run dev server with hot reload (port 19000)
just redeploy  # Restart systemd service after code changes
just status    # Check service status
just logs      # Tail service logs
just fc        # Format and lint
```

## Local Deployment

The service can run as a systemd user service:
- **Service:** `reader-triage.service`
- **Port:** 19000

After making code changes:
```bash
just redeploy
```

## Architecture

- **Backend:** FastAPI + SQLAlchemy async + aiosqlite
- **Frontend:** Jinja2 templates + Tailwind CSS (CDN)
- **AI:** Claude (via Anthropic SDK) for scoring and summarization
- **Data:** readwise-plus package for Readwise Reader API

## Scoring Dimensions

Articles are scored 0-100 on four dimensions (25 points each), predicting how likely a reader is to save/highlight passages:
1. **Quotability** - Specific passages worth saving, striking data, memorable phrasings
2. **Surprise Factor** - Challenges assumptions, unexpected findings, reframes topics
3. **Argument Quality** - Well-supported claims, coherent reasoning, strong opinion pieces
4. **Applicable Insight** - Frameworks, mental models, techniques, perspective shifts

Note: DB columns retain legacy names (`specificity_score`, `novelty_score`, `depth_score`, `actionability_score`).

Thresholds:
- Score >= 60: High value
- Score 30-59: Medium value
- Score < 30: Low value (auto-summarized)

## Scoring Calibration

When analyzing or recalibrating scoring, read `docs/calibration.md` for the full toolkit reference.

## Type Safety

Strict typing is enforced via `ty`. Rules:
- **No `Any`** — use `object`, concrete types, or union types instead. `Any` is only acceptable when required by a third-party API signature.
- **No blanket `--ignore` flags** in the ty command. Fix the root cause.
- **No `cast(Any, ...)`** — use `isinstance` narrowing to satisfy the type checker.
- **Anthropic SDK responses** — narrow `response.content[0]` with `assert isinstance(block, TextBlock)` before accessing `.text`. Narrow stream deltas with `isinstance(delta, TextDelta)`.
- **Per-line suppression** (`# type: ignore[specific-rule]`) is acceptable only when the type checker genuinely cannot understand the code (e.g., dynamic mock patching in tests). Always include the specific rule code.

## Environment Variables

Configured in `.env`:
- `READWISE_TOKEN` - Readwise API key
- `ANTHROPIC_API_KEY` - Claude API key
- `DATABASE_URL` - SQLite path (default: ./reader_triage.db)
- `ROOT_PATH` - URL prefix for reverse proxy (default: empty)
- `OTLP_ENDPOINT` - OpenTelemetry collector endpoint (default: http://localhost:4317)
