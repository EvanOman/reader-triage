"""Application configuration."""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class Settings:
    """Application settings from environment variables."""

    readwise_token: str
    anthropic_api_key: str
    openai_api_key: str
    database_url: str
    groq_api_key: str
    scoring_model: str
    tagger_model: str
    chat_model: str
    llm_gateway_base_url: str
    llm_gateway_api_key: str
    public_base_url: str
    nanobot_bin: str

    def __init__(self):
        self.readwise_token = os.environ.get("READWISE_TOKEN", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.database_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./reader_triage.db")
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.scoring_model = os.environ.get("SCORING_MODEL", "openai/gpt-5.4")
        self.tagger_model = os.environ.get("TAGGER_MODEL", "openai/gpt-4.1-mini")
        # Chat model: bare tier alias (e.g. "tier-smart") for the Anthropic SDK path,
        # or a full Anthropic model id when talking directly to Anthropic.
        self.chat_model = os.environ.get("CHAT_MODEL", "tier-smart")
        # Optional LLM gateway: set both to route litellm calls through it
        # for spend attribution; unset → direct-to-provider (no behavior change).
        self.llm_gateway_base_url = os.environ.get("LLM_GATEWAY_BASE_URL", "")
        self.llm_gateway_api_key = os.environ.get("LLM_GATEWAY_API_KEY", "")
        # Externally reachable base URL (Tailscale serve) used for digest feedback links
        self.public_base_url = os.environ.get(
            "PUBLIC_BASE_URL", "https://omachine.werewolf-universe.ts.net/inbox-monitor"
        )
        self.nanobot_bin = os.environ.get("NANOBOT_BIN", os.path.expanduser("~/bin/nanobot"))
        # ntfy fallback for digest delivery failures
        self.ntfy_server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
        self.ntfy_topic = os.environ.get("NTFY_TOPIC", "")
        # Alert webhook (nanobot) for sync failure notifications
        self.alert_webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "http://localhost:18950/alert")

    def llm_gateway_kwargs(self) -> dict[str, str]:
        """Return api_base/api_key kwargs for litellm when the gateway is configured.

        Returns an empty dict when both LLM_GATEWAY_BASE_URL and
        LLM_GATEWAY_API_KEY are not set, preserving litellm's default
        direct-to-provider behavior.
        """
        if self.llm_gateway_base_url and self.llm_gateway_api_key:
            return {
                "api_base": self.llm_gateway_base_url,
                "api_key": self.llm_gateway_api_key,
            }
        return {}

    def anthropic_gateway_kwargs(self) -> dict[str, str]:
        """Return base_url/api_key kwargs for the Anthropic SDK when the gateway is configured.

        The Anthropic SDK appends its own "/v1" path segment, so any trailing
        "/v1" on LLM_GATEWAY_BASE_URL (which is litellm-style) is stripped.
        Returns an empty dict when the gateway is not configured, preserving
        the SDK's default direct-to-Anthropic behavior (reads ANTHROPIC_API_KEY).
        """
        if self.llm_gateway_base_url and self.llm_gateway_api_key:
            base_url = self.llm_gateway_base_url
            if base_url.endswith("/v1"):
                base_url = base_url[: -len("/v1")]
            return {
                "base_url": base_url,
                "api_key": self.llm_gateway_api_key,
            }
        return {}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
