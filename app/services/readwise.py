"""Readwise service wrapper for fetching documents."""

from dataclasses import dataclass
from datetime import datetime

from readwise_sdk import AsyncReadwiseClient
from readwise_sdk.v3.models import Document

from app.config import get_settings


@dataclass
class ReaderDocument:
    """Document from Readwise Reader."""

    id: str
    title: str
    url: str
    author: str | None
    word_count: int | None
    content: str | None
    summary: str | None
    location: str | None  # new, later, archive
    category: str | None  # article, email, rss, etc.
    site_name: str | None
    reading_progress: float | None
    created_at: datetime | None
    updated_at: datetime | None
    published_date: datetime | None


class ReadwiseService:
    """Async service for interacting with Readwise Reader API."""

    def __init__(self):
        settings = get_settings()
        self._client = AsyncReadwiseClient(api_key=settings.readwise_token)

    async def get_all_documents(
        self,
        limit: int = 200,
        with_content: bool = False,
        exclude_archived: bool = True,
    ) -> list[ReaderDocument]:
        """Fetch all documents (inbox + later, optionally archive)."""
        documents: list[ReaderDocument] = []
        count = 0

        async for doc in self._client.v3.list_documents(with_content=with_content):
            if exclude_archived and doc.location and doc.location.value == "archive":
                continue

            documents.append(self._to_reader_document(doc))
            count += 1
            if count >= limit:
                break

        return documents

    async def get_inbox_documents(
        self, limit: int = 50, with_content: bool = True
    ) -> list[ReaderDocument]:
        """Fetch documents from the inbox only."""
        documents: list[ReaderDocument] = []
        count = 0

        async for doc in self._client.v3.list_documents(with_content=with_content):
            if doc.location and doc.location.value == "new":
                documents.append(self._to_reader_document(doc))
                count += 1
                if count >= limit:
                    break

        return documents

    async def get_document(
        self, document_id: str, with_content: bool = True
    ) -> ReaderDocument | None:
        """Get a single document by ID."""
        doc = await self._client.v3.get_document(document_id, with_content=with_content)
        if doc:
            return self._to_reader_document(doc)
        return None

    def _to_reader_document(self, doc: Document) -> ReaderDocument:
        """Convert a Readwise Document to ReaderDocument."""
        return ReaderDocument(
            id=doc.id,
            title=doc.title or "Untitled",
            url=doc.url,
            author=doc.author,
            word_count=doc.word_count,
            content=doc.content,
            summary=doc.summary,
            location=doc.location.value if doc.location else None,
            category=doc.category.value if doc.category else None,
            site_name=doc.site_name,
            reading_progress=doc.reading_progress,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            published_date=doc.published_date,
        )


# Singleton instance
_service: ReadwiseService | None = None


def get_readwise_service() -> ReadwiseService:
    """Get or create the Readwise service singleton."""
    global _service
    if _service is None:
        _service = ReadwiseService()
    return _service
