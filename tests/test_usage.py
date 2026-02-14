"""Tests for the API usage tracking service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.models.article import ApiUsageLog
from app.services.usage import DEFAULT_PRICING, MODEL_PRICING, compute_cost, log_usage


# ---------------------------------------------------------------------------
# 1. compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    """Tests for compute_cost pricing calculation."""

    def test_known_model_sonnet(self):
        """Sonnet pricing: $3/Mtok input, $15/Mtok output."""
        model = "claude-sonnet-4-20250514"
        cost = compute_cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(3.0 + 15.0)

    def test_known_model_haiku(self):
        """Haiku pricing: $0.25/Mtok input, $1.25/Mtok output."""
        model = "claude-haiku-4-5-20251001"
        cost = compute_cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.25 + 1.25)

    def test_unknown_model_uses_default(self):
        """Unknown model falls back to DEFAULT_PRICING."""
        cost = compute_cost("unknown-model-xyz", input_tokens=1_000_000, output_tokens=1_000_000)
        input_rate, output_rate = DEFAULT_PRICING
        assert cost == pytest.approx(input_rate + output_rate)

    def test_zero_tokens(self):
        """Zero tokens yields zero cost."""
        cost = compute_cost("claude-sonnet-4-20250514", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_small_token_count(self):
        """Verify cost for small realistic token counts."""
        # 1000 input + 500 output on Sonnet
        cost = compute_cost("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_input_only(self):
        """Cost with only input tokens, zero output."""
        cost = compute_cost("claude-sonnet-4-20250514", input_tokens=500_000, output_tokens=0)
        assert cost == pytest.approx(500_000 * 3.0 / 1_000_000)

    def test_output_only(self):
        """Cost with only output tokens, zero input."""
        cost = compute_cost("claude-sonnet-4-20250514", input_tokens=0, output_tokens=500_000)
        assert cost == pytest.approx(500_000 * 15.0 / 1_000_000)

    def test_model_pricing_dict_has_expected_models(self):
        """Verify the MODEL_PRICING dict contains the expected models."""
        assert "claude-sonnet-4-20250514" in MODEL_PRICING
        assert "claude-haiku-4-5-20251001" in MODEL_PRICING

    def test_all_pricing_tuples_have_two_elements(self):
        """Each pricing entry should be a (input, output) tuple."""
        for model, pricing in MODEL_PRICING.items():
            assert len(pricing) == 2, f"Model {model} has {len(pricing)} pricing elements"
            assert pricing[0] > 0, f"Model {model} has non-positive input rate"
            assert pricing[1] > 0, f"Model {model} has non-positive output rate"


# ---------------------------------------------------------------------------
# 2. log_usage — success path
# ---------------------------------------------------------------------------


class TestLogUsage:
    """Tests for the log_usage async function."""

    async def test_log_usage_creates_entry(self, session_factory, session):
        """log_usage creates an ApiUsageLog record in the database."""
        with patch(
            "app.services.usage.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            await log_usage(
                service="scorer",
                model="claude-sonnet-4-20250514",
                input_tokens=1000,
                output_tokens=500,
                article_id="test-article-001",
            )

        # Verify the record was committed
        result = await session.execute(select(ApiUsageLog))
        entries = result.scalars().all()
        assert len(entries) == 1

        entry = entries[0]
        assert entry.service == "scorer"
        assert entry.model == "claude-sonnet-4-20250514"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.article_id == "test-article-001"
        assert entry.cost_usd == pytest.approx(compute_cost("claude-sonnet-4-20250514", 1000, 500))
        assert isinstance(entry.timestamp, datetime)

    async def test_log_usage_without_article_id(self, session_factory, session):
        """log_usage works when article_id is None."""
        with patch(
            "app.services.usage.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            await log_usage(
                service="chat",
                model="claude-sonnet-4-20250514",
                input_tokens=2000,
                output_tokens=1000,
            )

        result = await session.execute(select(ApiUsageLog))
        entries = result.scalars().all()
        assert len(entries) == 1
        assert entries[0].article_id is None
        assert entries[0].service == "chat"

    async def test_log_usage_computes_correct_cost(self, session_factory, session):
        """The cost_usd field matches compute_cost output."""
        with patch(
            "app.services.usage.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            await log_usage(
                service="tagger",
                model="claude-haiku-4-5-20251001",
                input_tokens=10_000,
                output_tokens=5_000,
            )

        result = await session.execute(select(ApiUsageLog))
        entry = result.scalar_one()
        expected = compute_cost("claude-haiku-4-5-20251001", 10_000, 5_000)
        assert entry.cost_usd == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. log_usage — retry logic
# ---------------------------------------------------------------------------


def _make_mock_factory(mock_session: MagicMock) -> object:
    """Build a callable that returns the mock session as an async context manager.

    Mirrors the real pattern: ``factory = await get_session_factory(); async with factory() as session:``
    """

    def factory_callable() -> object:
        return mock_session

    return factory_callable


class TestLogUsageRetry:
    """Tests for log_usage retry behavior on database errors."""

    async def test_retries_on_transient_error(self):
        """log_usage retries up to 3 times on DB errors."""
        mock_session = MagicMock()
        mock_session.add = MagicMock()

        # First two commits fail, third succeeds
        call_count = 0

        async def commit_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("DB locked")

        mock_session.commit = commit_side_effect
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "app.services.usage.get_session_factory",
                new_callable=AsyncMock,
                return_value=_make_mock_factory(mock_session),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await log_usage(
                service="scorer",
                model="claude-sonnet-4-20250514",
                input_tokens=100,
                output_tokens=50,
            )

        # Should have retried twice (sleeping before attempts 2 and 3)
        assert mock_sleep.await_count == 2
        assert call_count == 3

    async def test_gives_up_after_3_failures(self):
        """After 3 failures, log_usage logs a warning but does not raise."""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=RuntimeError("DB locked"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "app.services.usage.get_session_factory",
                new_callable=AsyncMock,
                return_value=_make_mock_factory(mock_session),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            # Should not raise even after 3 failures
            await log_usage(
                service="scorer",
                model="claude-sonnet-4-20250514",
                input_tokens=100,
                output_tokens=50,
            )

        # All 3 attempts should have called commit
        assert mock_session.commit.await_count == 3

    async def test_sleep_backoff_timing(self):
        """Verify the retry sleep uses 0.5 * (attempt + 1) backoff."""
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock(side_effect=RuntimeError("DB locked"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "app.services.usage.get_session_factory",
                new_callable=AsyncMock,
                return_value=_make_mock_factory(mock_session),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await log_usage(
                service="scorer",
                model="claude-sonnet-4-20250514",
                input_tokens=100,
                output_tokens=50,
            )

        # attempt 0 fails -> sleep(0.5 * 1) = 0.5
        # attempt 1 fails -> sleep(0.5 * 2) = 1.0
        # attempt 2 fails -> no sleep (last attempt)
        assert mock_sleep.await_count == 2
        mock_sleep.assert_any_await(0.5)
        mock_sleep.assert_any_await(1.0)

    async def test_succeeds_on_first_try_no_retry(self, session_factory, session):
        """When the first attempt succeeds, no retries happen."""
        with patch(
            "app.services.usage.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            await log_usage(
                service="summarizer",
                model="claude-sonnet-4-20250514",
                input_tokens=500,
                output_tokens=200,
            )

        result = await session.execute(select(ApiUsageLog))
        entries = result.scalars().all()
        assert len(entries) == 1
        assert entries[0].service == "summarizer"


# ---------------------------------------------------------------------------
# 4. log_usage — multiple entries
# ---------------------------------------------------------------------------


class TestLogUsageMultiple:
    """Test logging multiple usage entries."""

    async def test_multiple_log_entries(self, session_factory, session):
        """Multiple calls to log_usage create separate entries."""
        with patch(
            "app.services.usage.get_session_factory",
            new_callable=AsyncMock,
            return_value=session_factory,
        ):
            await log_usage(
                service="scorer",
                model="claude-sonnet-4-20250514",
                input_tokens=1000,
                output_tokens=500,
                article_id="article-1",
            )
            await log_usage(
                service="tagger",
                model="claude-haiku-4-5-20251001",
                input_tokens=800,
                output_tokens=200,
                article_id="article-2",
            )
            await log_usage(
                service="summarizer",
                model="claude-sonnet-4-20250514",
                input_tokens=3000,
                output_tokens=1500,
                article_id="article-1",
            )

        result = await session.execute(select(ApiUsageLog))
        entries = result.scalars().all()
        assert len(entries) == 3

        services = {e.service for e in entries}
        assert services == {"scorer", "tagger", "summarizer"}
