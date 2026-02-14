"""Shared test factories and utilities for reader_triage tests."""

import json
from dataclasses import replace
from unittest.mock import MagicMock

from anthropic.types import TextBlock

from app.services.readwise import ReaderDocument


class FakeReadwiseService:
    """In-memory Readwise service for integration tests.

    Implements the same interface as ReadwiseService:
    - get_all_documents()
    - get_inbox_documents()
    - get_document()

    Supports scheduling failures for specific document IDs and tracks call counts.
    """

    def __init__(self) -> None:
        self._documents: dict[str, ReaderDocument] = {}
        self._failures: dict[str, Exception] = {}
        self.get_document_calls: list[str] = []
        self.get_all_documents_calls: int = 0
        self.get_inbox_documents_calls: int = 0

    def add_document(self, doc: ReaderDocument) -> None:
        """Add a document to the in-memory store."""
        self._documents[doc.id] = doc

    def schedule_failure(self, doc_id: str, exc: Exception) -> None:
        """Schedule a one-shot failure for the next get_document() call for this ID.

        The failure is consumed on first use; subsequent calls return normally.
        """
        self._failures[doc_id] = exc

    async def get_all_documents(
        self,
        limit: int = 200,
        with_content: bool = False,
        exclude_archived: bool = True,
    ) -> list[ReaderDocument]:
        self.get_all_documents_calls += 1
        docs = list(self._documents.values())
        if exclude_archived:
            docs = [d for d in docs if d.location != "archive"]
        if not with_content:
            docs = [replace(d, content=None) for d in docs]
        return docs[:limit]

    async def get_inbox_documents(
        self, limit: int = 50, with_content: bool = True
    ) -> list[ReaderDocument]:
        self.get_inbox_documents_calls += 1
        docs = [d for d in self._documents.values() if d.location == "new"]
        if not with_content:
            docs = [replace(d, content=None) for d in docs]
        return docs[:limit]

    async def get_document(
        self, document_id: str, with_content: bool = True
    ) -> ReaderDocument | None:
        self.get_document_calls.append(document_id)
        if document_id in self._failures:
            exc = self._failures.pop(document_id)
            raise exc
        doc = self._documents.get(document_id)
        if doc is None:
            return None
        if not with_content:
            return replace(doc, content=None)
        return doc


def make_document(
    *,
    id: str = "doc-1",
    title: str = "Test Article",
    url: str = "https://example.com/test",
    author: str | None = "Author Name",
    word_count: int | None = 2000,
    content: str | None = "Some real content here. " * 100,
    summary: str | None = None,
    location: str | None = "new",
    category: str | None = "article",
    site_name: str | None = "example.com",
    reading_progress: float | None = 0.0,
    created_at: object = None,
    updated_at: object = None,
    published_date: object = None,
) -> ReaderDocument:
    """Build a ReaderDocument with sensible defaults for testing."""
    return ReaderDocument(
        id=id,
        title=title,
        url=url,
        author=author,
        word_count=word_count,
        content=content,
        summary=summary,
        location=location,
        category=category,
        site_name=site_name,
        reading_progress=reading_progress,
        created_at=created_at,
        updated_at=updated_at,
        published_date=published_date,
    )


def make_claude_response(overrides: dict | None = None) -> dict:
    """Build a Claude JSON response dict with sensible defaults (high-quality article)."""
    base: dict = {
        "standalone_passages": "several",
        "quotability_reason": "Contains memorable phrasings",
        "novel_framing": True,
        "content_type": "original_analysis",
        "surprise_reason": "Reframes familiar topic",
        "author_conviction": True,
        "practitioner_voice": True,
        "content_completeness": "complete",
        "argument_reason": "Strong conviction with evidence",
        "named_framework": True,
        "applicable_ideas": "broadly",
        "insight_reason": "Broadly applicable framework",
        "overall_assessment": "High value article with novel insights.",
    }
    if overrides:
        base.update(overrides)
    return base


def mock_anthropic_response(
    data: dict,
    *,
    input_tokens: int = 500,
    output_tokens: int = 100,
) -> MagicMock:
    """Create a mock Anthropic messages.create() return value."""
    content_block = TextBlock(type="text", text=json.dumps(data))
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response
