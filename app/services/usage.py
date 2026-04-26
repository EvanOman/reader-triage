"""API usage tracking service."""

import logging
from datetime import datetime

from app.models.article import ApiUsageLog, get_session_factory

logger = logging.getLogger(__name__)

# Pricing per million tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # LiteLLM-format keys (provider/model)
    "anthropic/claude-sonnet-4-5-20250929": (3.0, 15.0),
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
    # OpenAI
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.2": (1.25, 5.0),
    "gpt-5": (0.625, 5.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "openai/gpt-5-mini": (0.25, 2.0),
    # Groq-hosted open-source models
    "groq/qwen/qwen3-32b": (0.29, 0.59),
    "groq/moonshotai/kimi-k2-instruct-0905": (1.00, 3.00),
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = (2.5, 15.0)


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
