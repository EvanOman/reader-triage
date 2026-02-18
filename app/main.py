"""FastAPI application for Reader Triage."""

import logging
import os
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from fastapi import FastAPI

import app.models.podcast  # noqa: F401  # Register podcast models before init_db
from app.models.article import init_db, rebuild_fts_index
from app.routers import api, chat, pages, podcasts_api, podcasts_pages
from app.services.sync import get_background_sync
from app.tracing import setup_tracing

logger = logging.getLogger(__name__)

_tracer_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and start background sync on startup."""
    await init_db()
    await rebuild_fts_index()

    # Start periodic background sync
    sync = get_background_sync()
    sync.start_periodic()
    logger.info("Background sync started")

    yield

    # Stop periodic background sync
    sync.stop_periodic()
    logger.info("Background sync stopped")

    # Flush remaining traces
    if _tracer_provider:
        _tracer_provider.shutdown()


app = FastAPI(
    title="Reader Triage",
    description="Surface high-value articles from Readwise Reader",
    version="0.1.0",
    lifespan=lifespan,
    root_path=os.environ.get("ROOT_PATH", ""),
)

# Include routers
app.include_router(api.router)
app.include_router(chat.router)
app.include_router(pages.router)
app.include_router(podcasts_api.router)
app.include_router(podcasts_pages.router)

# Set up OpenTelemetry tracing
_tracer_provider = setup_tracing(app)
