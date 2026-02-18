"""Tests for OPML import service."""

from unittest.mock import patch

import pytest

from app.services.opml import OPMLImporter

SAMPLE_OPML = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head><title>Podcast Subscriptions</title></head>
  <body>
    <outline text="Tech Podcasts" title="Tech Podcasts">
      <outline type="rss" text="Dwarkesh Podcast" title="Dwarkesh Podcast"
               xmlUrl="https://api.substack.com/feed/podcast/69345.rss" />
      <outline type="rss" text="Lex Fridman Podcast"
               xmlUrl="https://lexfridman.com/feed/podcast/" />
    </outline>
    <outline type="rss" text="Top Level Podcast"
             xmlUrl="https://example.com/feed.rss" />
  </body>
</opml>"""


# ---------------------------------------------------------------------------
# 1. parse_opml (synchronous XML parsing)
# ---------------------------------------------------------------------------


class TestParseOPML:
    def test_parses_nested_outlines(self):
        importer = OPMLImporter()
        feeds = importer.parse_opml(SAMPLE_OPML)
        assert len(feeds) == 3

    def test_extracts_title_and_url(self):
        importer = OPMLImporter()
        feeds = importer.parse_opml(SAMPLE_OPML)
        dwarkesh = next(f for f in feeds if "Dwarkesh" in f["title"])
        assert dwarkesh["feed_url"] == "https://api.substack.com/feed/podcast/69345.rss"

    def test_handles_empty_opml(self):
        importer = OPMLImporter()
        feeds = importer.parse_opml('<?xml version="1.0"?><opml version="2.0"><body></body></opml>')
        assert feeds == []

    def test_handles_invalid_xml(self):
        """Invalid XML raises ParseError; OPMLImporter does not catch it."""
        import xml.etree.ElementTree as ET

        importer = OPMLImporter()
        with pytest.raises(ET.ParseError):
            importer.parse_opml("not valid xml at all")

    def test_skips_non_rss_outlines(self):
        opml = """<?xml version="1.0"?>
        <opml><body>
          <outline type="link" text="Not a podcast" xmlUrl="https://example.com" />
          <outline type="rss" text="Real Podcast" xmlUrl="https://example.com/feed.rss" />
        </body></opml>"""
        importer = OPMLImporter()
        feeds = importer.parse_opml(opml)
        assert len(feeds) == 1
        assert feeds[0]["title"] == "Real Podcast"

    def test_uses_text_when_title_missing(self):
        """When title attr is absent, falls back to text attr."""
        opml = """<?xml version="1.0"?>
        <opml><body>
          <outline type="rss" text="Fallback Name" xmlUrl="https://example.com/feed.rss" />
        </body></opml>"""
        importer = OPMLImporter()
        feeds = importer.parse_opml(opml)
        assert feeds[0]["title"] == "Fallback Name"

    def test_skips_outline_without_url(self):
        """Outlines of type rss but no xmlUrl are skipped."""
        opml = """<?xml version="1.0"?>
        <opml><body>
          <outline type="rss" text="No URL" />
        </body></opml>"""
        importer = OPMLImporter()
        feeds = importer.parse_opml(opml)
        assert feeds == []


# ---------------------------------------------------------------------------
# 2. import_opml (async, DB interaction)
# ---------------------------------------------------------------------------


class TestImportOPML:
    @patch("app.services.opml.get_session_factory")
    async def test_imports_podcasts(self, mock_factory, session_factory):
        mock_factory.return_value = session_factory
        importer = OPMLImporter()
        result = await importer.import_opml(SAMPLE_OPML)
        assert result["imported"] == 3
        assert result["skipped"] == 0

    @patch("app.services.opml.get_session_factory")
    async def test_deduplicates_by_feed_url(self, mock_factory, session_factory):
        mock_factory.return_value = session_factory
        importer = OPMLImporter()
        await importer.import_opml(SAMPLE_OPML)
        result = await importer.import_opml(SAMPLE_OPML)
        # Second import should skip all existing feeds
        assert result["imported"] == 0
        assert result["skipped"] == 3
