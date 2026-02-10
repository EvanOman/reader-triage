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
    database_url: str

    def __init__(self):
        self.readwise_token = os.environ.get("READWISE_TOKEN", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.database_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./reader_triage.db")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
