"""API usage tracking service."""

import logging
from datetime import datetime

from app.models.article import ApiUsageLog, get_session_factory

logger = logging.getLogger(__name__)

# Anthropic pricing per million tokens (as of 2025)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
}

# Default pricing for unknown models
DEFAULT_PRICING = (3.0, 15.0)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for an API call."""
    input_rate, output_rate = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


async def log_usage(
    service: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    article_id: str | None = None,
) -> None:
    """Log an API usage entry to the database."""
    import asyncio

    cost = compute_cost(model, input_tokens, output_tokens)

    for attempt in range(3):
        try:
            factory = await get_session_factory()
            async with factory() as session:
                entry = ApiUsageLog(
                    timestamp=datetime.now(),
                    service=service,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    article_id=article_id,
                )
                session.add(entry)
                await session.commit()
            return
        except Exception as exc:
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                logger.warning("Failed to log API usage after 3 attempts: %s", exc)
