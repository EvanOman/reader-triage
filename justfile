# Reader Triage justfile
set shell := ["bash", "-cu"]

default:
    @just --list

# Install dependencies
install:
    uv sync --dev

# Run development server on port 19000
dev:
    uv run uvicorn app.main:app --reload --port 19000

# Format code
fmt:
    uv run ruff format .

# Check formatting without changes
format-check:
    uv run ruff format --check .

# Run linter
lint:
    uv run ruff check .

# Lint with auto-fix
lint-fix:
    uv run ruff check . --fix

# Type check
type:
    uv run ty check . --exclude "backtest/*.py" --exclude "tools/cal_*.py" --exclude "tools/embed_articles.py" --exclude "tests/*.py"

# Run tests
test:
    uv run pytest

# Run tests with coverage report
test-cov:
    uv run pytest --cov=app --cov-report=term-missing

# FIX + CHECK: Run before every commit
fc: fmt lint-fix lint type test

# CI mode (no auto-fix)
ci: lint format-check type test

# Redeploy - restart the systemd service after code changes
redeploy:
    systemctl --user restart reader-triage
    @sleep 1
    systemctl --user status reader-triage --no-pager

# Check service status
status:
    systemctl --user status reader-triage --no-pager

# View service logs
logs:
    journalctl --user -u reader-triage -f

# Trigger inbox scan via API
scan:
    curl -X POST http://localhost:19000/api/scan

# Tag all untagged articles
tag:
    curl -X POST http://localhost:19000/inbox-monitor/api/tag | python -m json.tool

# Re-tag all articles (force)
retag:
    curl -X POST "http://localhost:19000/inbox-monitor/api/tag?force=true" | python -m json.tool

# Get top 5 articles
top5:
    curl http://localhost:19000/api/top5 | python -m json.tool

# Calibration tools
cal-sync:
    uv run python -m tools.calibrate sync

cal-report *ARGS:
    uv run python -m tools.calibrate report {{ARGS}}

cal-misses *ARGS:
    uv run python -m tools.calibrate misses {{ARGS}}

cal-inspect ID *ARGS:
    uv run python -m tools.calibrate inspect {{ID}} {{ARGS}}

# Backfill article content from Readwise API
backfill-content:
    uv run python -m tools.backfill_content

# Embed all articles into vector store
embed:
    uv run python -m tools.embed_articles

# Re-embed all articles (delete and recreate)
reembed:
    uv run python -m tools.embed_articles --reindex

cal-dimensions *ARGS:
    uv run python -m tools.calibrate dimensions {{ARGS}}

cal-trends *ARGS:
    uv run python -m tools.calibrate trends {{ARGS}}
