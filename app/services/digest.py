"""Daily digest and weekly roundup builders for Telegram distribution.

Selects high-value articles (info_score >= 60), formats Telegram MarkdownV2
messages with per-item 👍/👎 feedback links, and records every send as an
ExposureEvent — the ground truth for digest precision.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.article import Article, ArticleScore, ExposureEvent

logger = logging.getLogger(__name__)

HIGH_VALUE_THRESHOLD = 60.0
DAILY_CAP = 5
WEEKLY_CAP = 3
# Window for "new" articles in the daily digest. Wider than 24h to absorb
# timestamp skew and late syncs; the per-channel exposure dedup prevents
# any article from being sent twice.
DAILY_WINDOW_HOURS = 36

# Readwise document categories that are the user's own saved passages/notes,
# not articles — never digest material.
EXCLUDED_CATEGORIES = ("highlight", "note")


def _utcnow() -> datetime:
    """Naive UTC now, matching SQLite server_default timestamps."""
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Telegram formatting — nanobot notify -p uses LEGACY parse_mode=Markdown
# (see ~/dev/nanobot-dashboard/cmd/nanobot/main.go). Legacy Markdown has no
# backslash escaping (backslashes render literally) and no nested entities
# (a link inside *bold* breaks parsing). So: minimal styling, sanitize text
# instead of escaping, bare URLs for article links, and inline links only as
# standalone 👍/👎 entities.
# ---------------------------------------------------------------------------

_MD_LEGACY_REPLACEMENTS = {"*": "", "_": " ", "[": "(", "]": ")", "`": "'"}


def sanitize_md(text: str) -> str:
    """Strip characters that break legacy-Markdown entity parsing."""
    for char, replacement in _MD_LEGACY_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


@dataclass
class DigestItem:
    """One article entry in a digest message."""

    exposure_id: int
    article_id: str
    title: str
    url: str
    score: float
    why: str


def one_line_why(score: ArticleScore) -> str:
    """Extract a one-line 'why this scored' from a score record.

    Prefers the first score_reason (usually the most specific), falling back
    to the first sentence of the overall assessment.
    """
    try:
        reasons = json.loads(score.score_reasons)
    except (ValueError, TypeError):
        reasons = []
    line = ""
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], str):
        line = reasons[0].strip()
    if not line and score.overall_assessment:
        line = score.overall_assessment.split(". ")[0].strip()
    if len(line) > 200:
        line = line[:197].rstrip() + "..."
    return line


# ---------------------------------------------------------------------------
# Article selection
# ---------------------------------------------------------------------------


def _already_sent_subquery(channel: str):
    """Subquery of article IDs already sent on a channel (no repeats)."""
    return select(ExposureEvent.article_id).where(ExposureEvent.channel == channel)


async def _sent_titles(session: AsyncSession, channel: str) -> set[str]:
    """Lowercased titles of every article already sent on a channel.

    The same article sometimes gets saved twice under different Readwise IDs;
    ID-based dedup misses those, so digests also dedupe by title. Scoped per
    channel: a daily-digested article that stays unread may still resurface
    in the weekly roundup, matching the ID-level dedup semantics.
    """
    result = await session.execute(
        select(func.lower(Article.title))
        .join(ExposureEvent, ExposureEvent.article_id == Article.id)
        .where(ExposureEvent.channel == channel)
    )
    return {row[0] for row in result if row[0] and row[0] != "untitled"}


def _dedupe_by_title(
    pairs: list[tuple[Article, ArticleScore]], sent_titles: set[str], cap: int
) -> list[tuple[Article, ArticleScore]]:
    """Drop same-title duplicates (within the batch and vs. prior sends)."""
    seen = set(sent_titles)
    result: list[tuple[Article, ArticleScore]] = []
    for article, score in pairs:
        key = (article.title or "").lower()
        if key and key != "untitled":
            if key in seen:
                continue
            seen.add(key)
        result.append((article, score))
        if len(result) >= cap:
            break
    return result


async def select_daily_articles(
    session: AsyncSession, now: datetime | None = None
) -> list[tuple[Article, ArticleScore]]:
    """High-value articles synced in the last day, never sent on the daily channel."""
    now = now or _utcnow()
    cutoff = now - timedelta(hours=DAILY_WINDOW_HOURS)
    result = await session.execute(
        select(Article, ArticleScore)
        .join(ArticleScore, ArticleScore.article_id == Article.id)
        .where(
            ArticleScore.info_score >= HIGH_VALUE_THRESHOLD,
            ArticleScore.skip_recommended.is_(False),
            Article.location != "archive",
            func.coalesce(Article.category, "").not_in(EXCLUDED_CATEGORIES),
            Article.first_synced_at >= cutoff,
            Article.id.not_in(_already_sent_subquery("daily")),
        )
        .order_by(ArticleScore.info_score.desc())
        .limit(DAILY_CAP * 3)
    )
    pairs = [(row[0], row[1]) for row in result.all()]
    return _dedupe_by_title(pairs, await _sent_titles(session, "daily"), DAILY_CAP)


async def select_weekly_articles(
    session: AsyncSession, now: datetime | None = None
) -> list[tuple[Article, ArticleScore]]:
    """Top unread high-scorers older than a week, never sent on the weekly channel."""
    now = now or _utcnow()
    cutoff = now - timedelta(days=7)
    result = await session.execute(
        select(Article, ArticleScore)
        .join(ArticleScore, ArticleScore.article_id == Article.id)
        .where(
            ArticleScore.info_score >= HIGH_VALUE_THRESHOLD,
            ArticleScore.skip_recommended.is_(False),
            Article.location != "archive",
            func.coalesce(Article.category, "").not_in(EXCLUDED_CATEGORIES),
            func.coalesce(Article.reading_progress, 0.0) < 0.05,
            Article.first_synced_at < cutoff,
            Article.id.not_in(_already_sent_subquery("weekly")),
        )
        .order_by(ArticleScore.info_score.desc())
        .limit(WEEKLY_CAP * 3)
    )
    pairs = [(row[0], row[1]) for row in result.all()]
    return _dedupe_by_title(pairs, await _sent_titles(session, "weekly"), WEEKLY_CAP)


async def record_exposures(
    session: AsyncSession,
    pairs: list[tuple[Article, ArticleScore]],
    channel: str,
) -> list[DigestItem]:
    """Create ExposureEvent rows (flushed for IDs) and return digest items.

    Caller is responsible for committing only after the message is actually
    sent, so failed sends leave no exposure records.
    """
    items: list[DigestItem] = []
    for article, score in pairs:
        event = ExposureEvent(
            article_id=article.id,
            score=score.info_score,
            channel=channel,
        )
        session.add(event)
        await session.flush()
        items.append(
            DigestItem(
                exposure_id=event.id,
                article_id=article.id,
                title=article.title or "Untitled",
                url=article.url,
                score=score.info_score,
                why=one_line_why(score),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Pipeline health (for the weekly roundup)
# ---------------------------------------------------------------------------


@dataclass
class PipelineHealth:
    """One-line health stats for the weekly roundup."""

    articles_scored_7d: int
    exposures_28d: int
    engaged_28d: int

    @property
    def precision_line(self) -> str:
        if self.exposures_28d == 0:
            return "no digest sends yet"
        pct = 100.0 * self.engaged_28d / self.exposures_28d
        return f"digest precision {pct:.0f}% ({self.engaged_28d}/{self.exposures_28d})"


async def compute_pipeline_health(
    session: AsyncSession, now: datetime | None = None
) -> PipelineHealth:
    """Compute scoring throughput and digest precision.

    Precision counts a daily-digest exposure as engaged if the article was
    thumbed up, opened (reading_progress >= 0.1), or highlighted. Never-opened
    articles are not negatives per se, but for the send-precision metric an
    unopened send is a wasted slot, so it counts against precision.
    """
    now = now or _utcnow()
    scored_result = await session.execute(
        select(func.count(ArticleScore.id)).where(ArticleScore.scored_at >= now - timedelta(days=7))
    )
    scored_7d = scored_result.scalar() or 0

    exposures_result = await session.execute(
        select(ExposureEvent, Article)
        .join(Article, Article.id == ExposureEvent.article_id)
        .where(
            ExposureEvent.channel == "daily",
            ExposureEvent.sent_at >= now - timedelta(days=28),
        )
    )
    rows = exposures_result.all()
    engaged = 0
    for event, article in rows:
        opened = (article.reading_progress or 0.0) >= 0.1
        highlighted = (article.num_highlights or 0) > 0
        if event.feedback == 1 or opened or highlighted:
            engaged += 1
    return PipelineHealth(
        articles_scored_7d=scored_7d,
        exposures_28d=len(rows),
        engaged_28d=engaged,
    )


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def _format_item(index: int, item: DigestItem, base_url: str) -> str:
    up_url = f"{base_url}/api/feedback/{item.exposure_id}/up"
    down_url = f"{base_url}/api/feedback/{item.exposure_id}/down"
    lines = [f"{index}. {sanitize_md(item.title)} — {item.score:.0f}"]
    if item.why:
        lines.append(sanitize_md(item.why))
    lines.append(item.url)
    lines.append(f"[👍]({up_url}) · [👎]({down_url})")
    return "\n".join(lines)


def format_daily_message(items: list[DigestItem]) -> str:
    """Format the daily digest as Telegram legacy Markdown."""
    base_url = get_settings().public_base_url
    noun = "article" if len(items) == 1 else "articles"
    header = f"*📚 Reader Triage daily — {len(items)} high-value {noun}*"
    parts = [header]
    for i, item in enumerate(items, 1):
        parts.append(_format_item(i, item, base_url))
    return "\n\n".join(parts)


def format_weekly_message(items: list[DigestItem], health: PipelineHealth) -> str:
    """Format the weekly roundup as Telegram legacy Markdown."""
    base_url = get_settings().public_base_url
    header = "*🗓 Reader Triage weekly — top unread from the backlog*"
    parts = [header]
    for i, item in enumerate(items, 1):
        parts.append(_format_item(i, item, base_url))
    health_line = (
        f"Pipeline: {health.articles_scored_7d} articles scored this week, {health.precision_line}"
    )
    parts.append(sanitize_md(health_line))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def check_nanobot_health() -> str | None:
    """Check that the nanobot binary exists and is executable.

    Returns None on success, or an error description string.
    """
    import os
    import stat

    nanobot = get_settings().nanobot_bin
    if not os.path.exists(nanobot):
        return f"nanobot binary not found at {nanobot}"
    st = os.stat(nanobot)
    if not st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        return f"nanobot binary at {nanobot} is not executable"
    return None


async def send_telegram(message: str) -> bool:
    """Send a message via `nanobot notify -p`. Returns True on success."""
    nanobot = get_settings().nanobot_bin
    try:
        proc = await asyncio.create_subprocess_exec(
            nanobot,
            "notify",
            "-p",
            message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return True
        logger.error(
            "nanobot notify failed (exit %d): %s",
            proc.returncode,
            stderr.decode(errors="replace")[:500],
        )
        return False
    except OSError as e:
        logger.error("Failed to run nanobot: %s", e)
        return False


async def send_ntfy(message: str, *, title: str = "Reader Triage digest") -> bool:
    """Send a message via ntfy.sh as a fallback delivery channel.

    Returns True on success, False if ntfy is not configured or the send fails.
    """
    settings = get_settings()
    if not settings.ntfy_topic:
        logger.warning("NTFY_TOPIC not configured — cannot send ntfy fallback")
        return False
    url = f"{settings.ntfy_server.rstrip('/')}/{settings.ntfy_topic}"
    try:
        proc = await asyncio.create_subprocess_exec(
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-H",
            f"Title: {title}",
            "-H",
            "Priority: high",
            "-H",
            "Tags: warning",
            "-d",
            message,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        status = stdout.decode().strip()
        if proc.returncode == 0 and status.startswith("2"):
            logger.info("ntfy fallback sent successfully")
            return True
        logger.error(
            "ntfy send failed (curl exit %d, HTTP %s): %s",
            proc.returncode,
            status,
            stderr.decode(errors="replace")[:300],
        )
        return False
    except OSError as e:
        logger.error("Failed to run curl for ntfy: %s", e)
        return False


async def deliver_digest(message: str) -> bool:
    """Deliver a digest message, trying nanobot first then ntfy as fallback.

    Returns True if the message was delivered via any channel.
    """
    # Pre-flight: warn if nanobot is unhealthy (but still try — it may recover)
    health_err = check_nanobot_health()
    if health_err:
        logger.warning("Nanobot health check failed: %s", health_err)

    if await send_telegram(message):
        return True

    logger.warning("Primary delivery (nanobot) failed — trying ntfy fallback")
    # Strip Markdown formatting for ntfy (plain text channel)
    plain = message.replace("*", "").replace("[", "").replace("]", "")
    if await send_ntfy(plain, title="Reader Triage digest (nanobot failed)"):
        return True

    logger.error("All delivery channels failed — digest not sent")
    return False
