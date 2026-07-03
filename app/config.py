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
        # Externally reachable base URL (Tailscale serve) used for digest feedback links
        self.public_base_url = os.environ.get(
            "PUBLIC_BASE_URL", "https://omachine.werewolf-universe.ts.net/inbox-monitor"
        )
        self.nanobot_bin = os.environ.get("NANOBOT_BIN", os.path.expanduser("~/bin/nanobot"))
        # ntfy fallback for digest delivery failures
        self.ntfy_server = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
        self.ntfy_topic = os.environ.get("NTFY_TOPIC", "")
        # Alert webhook (nanobot) for sync failure notifications
        self.alert_webhook_url = os.environ.get(
            "ALERT_WEBHOOK_URL", "http://localhost:18950/alert"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
