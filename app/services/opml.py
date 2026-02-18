"""OPML import service for podcast subscriptions."""

import logging
import xml.etree.ElementTree as ET

from sqlalchemy import select

from app.models.article import get_session_factory
from app.models.podcast import Podcast

logger = logging.getLogger(__name__)


class OPMLImporter:
    """Parses OPML files and upserts Podcast records."""

    def parse_opml(self, opml_content: str) -> list[dict[str, str]]:
        """Parse OPML XML and extract podcast feeds.

        Returns list of dicts with 'title' and 'feed_url' keys.
        """
        root = ET.fromstring(opml_content)
        feeds: list[dict[str, str]] = []

        for outline in root.iter("outline"):
            outline_type = outline.get("type", "").lower()
            xml_url = outline.get("xmlUrl", "")

            if outline_type == "rss" and xml_url:
                title = outline.get("title") or outline.get("text") or "Untitled"
                feeds.append({"title": title, "feed_url": xml_url})

        return feeds

    async def import_opml(self, opml_content: str) -> dict[str, int]:
        """Import podcasts from OPML content.

        Returns dict with 'imported' and 'skipped' counts.
        """
        feeds = self.parse_opml(opml_content)
        imported = 0
        skipped = 0

        factory = await get_session_factory()
        async with factory() as session:
            for feed in feeds:
                # Check if already exists
                result = await session.execute(
                    select(Podcast).where(Podcast.feed_url == feed["feed_url"])
                )
                existing = result.scalar_one_or_none()

                if existing:
                    skipped += 1
                    continue

                podcast = Podcast(
                    title=feed["title"],
                    feed_url=feed["feed_url"],
                )
                session.add(podcast)
                imported += 1

            await session.commit()

        logger.info("OPML import: %d imported, %d skipped", imported, skipped)
        return {"imported": imported, "skipped": skipped}


# Singleton
_importer: OPMLImporter | None = None


def get_opml_importer() -> OPMLImporter:
    """Get or create the OPML importer singleton."""
    global _importer
    if _importer is None:
        _importer = OPMLImporter()
    return _importer
