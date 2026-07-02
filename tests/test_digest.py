"""Tests for the digest service and feedback endpoint."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.article import Article, ArticleScore, Base, ExposureEvent
from app.services.digest import (
    DigestItem,
    PipelineHealth,
    compute_pipeline_health,
    escape_md,
    format_daily_message,
    format_weekly_message,
    one_line_why,
    record_exposures,
    select_daily_articles,
    select_weekly_articles,
)

NOW = datetime(2026, 7, 1, 12, 0, 0)


def make_article(
    article_id: str,
    *,
    score: float = 75.0,
    synced_hours_ago: float = 2.0,
    location: str = "new",
    category: str | None = "article",
    reading_progress: float | None = None,
    num_highlights: int | None = None,
    skip_recommended: bool = False,
    title: str | None = None,
) -> tuple[Article, ArticleScore]:
    article = Article(
        id=article_id,
        title=title if title is not None else f"Article {article_id}",
        url=f"https://example.com/{article_id}",
        location=location,
        category=category,
        reading_progress=reading_progress,
        num_highlights=num_highlights,
        first_synced_at=NOW - timedelta(hours=synced_hours_ago),
    )
    art_score = ArticleScore(
        article_id=article_id,
        info_score=score,
        score_reasons=json.dumps([f"Sharp claim about {article_id}."]),
        overall_assessment=f"Assessment for {article_id}. More detail here.",
        skip_recommended=skip_recommended,
        scored_at=NOW - timedelta(hours=synced_hours_ago),
    )
    return article, art_score


@pytest.fixture
async def digest_engine():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def digest_session(digest_engine):
    factory = async_sessionmaker(digest_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


async def seed(session, *pairs):
    for article, score in pairs:
        session.add(article)
        session.add(score)
    await session.commit()


class TestDailySelection:
    async def test_selects_recent_high_scorers(self, digest_session):
        await seed(
            digest_session,
            make_article("recent-high", score=80, synced_hours_ago=5),
            make_article("recent-low", score=45, synced_hours_ago=5),
            make_article("old-high", score=90, synced_hours_ago=100),
        )
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["recent-high"]

    async def test_caps_at_five_ordered_by_score(self, digest_session):
        await seed(
            digest_session,
            *[make_article(f"a{i}", score=60 + i, synced_hours_ago=1) for i in range(7)],
        )
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert len(pairs) == 5
        scores = [s.info_score for _, s in pairs]
        assert scores == sorted(scores, reverse=True)

    async def test_excludes_archived_skip_recommended_and_already_sent(self, digest_session):
        await seed(
            digest_session,
            make_article("ok", score=70),
            make_article("archived", score=80, location="archive"),
            make_article("skip", score=80, skip_recommended=True),
            make_article("sent", score=80),
        )
        digest_session.add(ExposureEvent(article_id="sent", score=80, channel="daily"))
        await digest_session.commit()
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["ok"]

    async def test_excludes_highlight_and_note_categories(self, digest_session):
        await seed(
            digest_session,
            make_article("real-article", score=70),
            make_article("a-highlight", score=90, category="highlight"),
            make_article("a-note", score=90, category="note"),
            make_article("no-category", score=65, category=None),
        )
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["real-article", "no-category"]

    async def test_dedupes_same_title_within_batch_and_vs_prior_sends(self, digest_session):
        await seed(
            digest_session,
            make_article("dup-a", score=80, title="Same Piece Twice"),
            make_article("dup-b", score=75, title="same piece twice"),
            make_article("resent", score=70, title="Sent Last Week"),
            make_article("sent-orig", score=90, title="Sent Last Week", synced_hours_ago=500),
            make_article("fresh", score=65, title="Brand New"),
        )
        digest_session.add(ExposureEvent(article_id="sent-orig", score=90, channel="daily"))
        await digest_session.commit()
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["dup-a", "fresh"]

    async def test_daily_dedup_is_per_channel(self, digest_session):
        await seed(digest_session, make_article("weekly-sent", score=70))
        digest_session.add(ExposureEvent(article_id="weekly-sent", score=70, channel="weekly"))
        await digest_session.commit()
        pairs = await select_daily_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["weekly-sent"]


class TestWeeklySelection:
    async def test_selects_old_unread_high_scorers(self, digest_session):
        await seed(
            digest_session,
            make_article("old-unread", score=85, synced_hours_ago=24 * 10),
            make_article("old-read", score=90, synced_hours_ago=24 * 10, reading_progress=0.5),
            make_article("recent", score=90, synced_hours_ago=24 * 2),
        )
        pairs = await select_weekly_articles(digest_session, now=NOW)
        assert [a.id for a, _ in pairs] == ["old-unread"]

    async def test_caps_at_three(self, digest_session):
        await seed(
            digest_session,
            *[make_article(f"w{i}", score=60 + i, synced_hours_ago=24 * 10) for i in range(5)],
        )
        pairs = await select_weekly_articles(digest_session, now=NOW)
        assert len(pairs) == 3


class TestExposuresAndFormatting:
    async def test_record_exposures_creates_rows_with_ids(self, digest_session):
        await seed(digest_session, make_article("x1", score=70))
        pairs = await select_daily_articles(digest_session, now=NOW)
        items = await record_exposures(digest_session, pairs, "daily")
        await digest_session.commit()

        assert len(items) == 1
        assert items[0].exposure_id > 0
        result = await digest_session.execute(select(ExposureEvent))
        events = result.scalars().all()
        assert len(events) == 1
        assert events[0].article_id == "x1"
        assert events[0].score == 70
        assert events[0].channel == "daily"
        assert events[0].feedback is None

    def test_one_line_why_prefers_score_reasons(self):
        score = ArticleScore(
            score_reasons=json.dumps(["Specific quote about X."]),
            overall_assessment="Generic assessment.",
        )
        assert one_line_why(score) == "Specific quote about X."

    def test_one_line_why_falls_back_to_assessment(self):
        score = ArticleScore(
            score_reasons="[]",
            overall_assessment="First sentence here. Second sentence.",
        )
        assert one_line_why(score) == "First sentence here"

    def test_escape_md(self):
        assert escape_md("a_b*c[d]") == "a\\_b\\*c\\[d\\]"

    def test_daily_message_contains_links_and_feedback(self):
        items = [
            DigestItem(
                exposure_id=42,
                article_id="a1",
                title="Great Article (2026)",
                url="https://example.com/a1",
                score=78.0,
                why="A sharp claim.",
            )
        ]
        msg = format_daily_message(items)
        assert "78" in msg
        assert "https://example.com/a1" in msg
        assert "/api/feedback/42/up" in msg
        assert "/api/feedback/42/down" in msg
        assert "Great Article \\(2026\\)" in msg

    def test_weekly_message_includes_health_line(self):
        health = PipelineHealth(articles_scored_7d=12, exposures_28d=10, engaged_28d=6)
        msg = format_weekly_message([], health)
        assert "12 articles scored" in msg
        assert "60%" in msg

    def test_health_line_no_sends(self):
        health = PipelineHealth(articles_scored_7d=3, exposures_28d=0, engaged_28d=0)
        assert health.precision_line == "no digest sends yet"


class TestPipelineHealth:
    async def test_engagement_counting(self, digest_session):
        await seed(
            digest_session,
            make_article("thumbed", score=70),
            make_article("opened", score=70, reading_progress=0.4),
            make_article("highlighted", score=70, num_highlights=2),
            make_article("ignored", score=70),
        )
        for aid, feedback in [
            ("thumbed", 1),
            ("opened", None),
            ("highlighted", None),
            ("ignored", None),
        ]:
            digest_session.add(
                ExposureEvent(
                    article_id=aid,
                    score=70,
                    channel="daily",
                    sent_at=NOW - timedelta(days=3),
                    feedback=feedback,
                )
            )
        await digest_session.commit()

        health = await compute_pipeline_health(digest_session, now=NOW)
        assert health.exposures_28d == 4
        assert health.engaged_28d == 3


# ---------------------------------------------------------------------------
# Feedback endpoint (integration, reusing the patched-app pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
async def api_engine():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts "
                "USING fts5(article_id UNINDEXED, title, author, content)"
            )
        )
    yield engine
    await engine.dispose()


@pytest.fixture
async def api_client(api_engine):
    import app.models.article as article_mod

    factory = async_sessionmaker(api_engine, expire_on_commit=False)
    mock_sync = MagicMock()
    mock_sync.start = AsyncMock()
    mock_sync.stop = AsyncMock()

    with (
        patch.object(article_mod, "_engine", api_engine),
        patch.object(article_mod, "_session_factory", factory),
        patch("app.main.init_db", new_callable=AsyncMock),
        patch("app.main.rebuild_fts_index", new_callable=AsyncMock),
        patch("app.main.get_background_sync", return_value=mock_sync),
    ):
        from app.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, factory


class TestFeedbackEndpoint:
    async def test_records_thumbs_up(self, api_client):
        client, factory = api_client
        async with factory() as session:
            article, score = make_article("fb1", score=70)
            session.add(article)
            session.add(score)
            event = ExposureEvent(article_id="fb1", score=70, channel="daily")
            session.add(event)
            await session.commit()
            exposure_id = event.id

        resp = await client.get(f"/api/feedback/{exposure_id}/up")
        assert resp.status_code == 200
        assert "👍" in resp.text

        async with factory() as session:
            refreshed = await session.get(ExposureEvent, exposure_id)
            assert refreshed is not None
            assert refreshed.feedback == 1
            assert refreshed.feedback_at is not None

    async def test_records_thumbs_down(self, api_client):
        client, factory = api_client
        async with factory() as session:
            article, score = make_article("fb2", score=70)
            session.add(article)
            session.add(score)
            event = ExposureEvent(article_id="fb2", score=70, channel="daily")
            session.add(event)
            await session.commit()
            exposure_id = event.id

        resp = await client.get(f"/api/feedback/{exposure_id}/down")
        assert resp.status_code == 200

        async with factory() as session:
            refreshed = await session.get(ExposureEvent, exposure_id)
            assert refreshed is not None
            assert refreshed.feedback == -1

    async def test_unknown_exposure_404(self, api_client):
        client, _ = api_client
        resp = await client.get("/api/feedback/99999/up")
        assert resp.status_code == 404

    async def test_bad_verdict_400(self, api_client):
        client, factory = api_client
        async with factory() as session:
            article, score = make_article("fb3", score=70)
            session.add(article)
            session.add(score)
            event = ExposureEvent(article_id="fb3", score=70, channel="daily")
            session.add(event)
            await session.commit()
            exposure_id = event.id

        resp = await client.get(f"/api/feedback/{exposure_id}/sideways")
        assert resp.status_code == 400
