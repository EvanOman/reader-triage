"""Author sync service - syncs author highlight data from Readwise v2 API."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from readwise_sdk import AsyncReadwiseClient
from sqlalchemy import select

from app.config import get_settings
from app.models.article import Author, AuthorBook, get_session_factory


@dataclass
class AuthorStats:
    """Aggregated stats for an author."""

    name: str
    total_highlights: int
    total_books: int
    first_highlighted_at: datetime | None
    last_highlighted_at: datetime | None
    books: list[dict]


@dataclass
class SyncResult:
    """Result of syncing authors from Readwise."""

    total_authors: int
    new_authors: int
    updated_authors: int
    total_books: int


def normalize_author_name(name: str) -> str:
    """Normalize author name for matching."""
    return name.lower().strip()


class AuthorService:
    """Service for managing author data and highlight statistics."""

    def __init__(self):
        settings = get_settings()
        self._client = AsyncReadwiseClient(api_key=settings.readwise_token)

    async def sync_authors_from_readwise(self) -> SyncResult:
        """Sync all authors and their books/highlights from Readwise v2 API.

        Returns:
            SyncResult with sync statistics.
        """
        # Aggregate books by author
        author_data: dict[str, AuthorStats] = defaultdict(
            lambda: AuthorStats(
                name="",
                total_highlights=0,
                total_books=0,
                first_highlighted_at=None,
                last_highlighted_at=None,
                books=[],
            )
        )

        # Fetch all books with highlights
        async for book in self._client.v2.list_books():
            if not book.author or book.num_highlights == 0:
                continue

            author_name = book.author.strip()
            if not author_name:
                continue

            stats = author_data[author_name]
            stats.name = author_name
            stats.total_highlights += book.num_highlights
            stats.total_books += 1

            # Track first/last highlight times
            if book.last_highlight_at:
                if (
                    stats.last_highlighted_at is None
                    or book.last_highlight_at > stats.last_highlighted_at
                ):
                    stats.last_highlighted_at = book.last_highlight_at
                if (
                    stats.first_highlighted_at is None
                    or book.last_highlight_at < stats.first_highlighted_at
                ):
                    stats.first_highlighted_at = book.last_highlight_at

            # Store book info
            stats.books.append(
                {
                    "id": book.id,
                    "title": book.title,
                    "category": book.category.value if book.category else None,
                    "source_url": book.source_url,
                    "cover_image_url": book.cover_image_url,
                    "num_highlights": book.num_highlights,
                    "last_highlight_at": book.last_highlight_at,
                }
            )

        # Save to database
        factory = await get_session_factory()
        new_authors = 0
        updated_authors = 0
        total_books = 0

        async with factory() as session:
            for author_name, stats in author_data.items():
                normalized = normalize_author_name(author_name)

                # Check if author exists
                result = await session.execute(
                    select(Author).where(Author.normalized_name == normalized)
                )
                author = result.scalar_one_or_none()

                if author is None:
                    # Create new author
                    author = Author(
                        name=author_name,
                        normalized_name=normalized,
                        total_highlights=stats.total_highlights,
                        total_books=stats.total_books,
                        first_highlighted_at=stats.first_highlighted_at,
                        last_highlighted_at=stats.last_highlighted_at,
                        last_synced_at=datetime.now(),
                    )
                    session.add(author)
                    await session.flush()  # Get the ID
                    new_authors += 1
                else:
                    # Update existing author
                    author.total_highlights = stats.total_highlights
                    author.total_books = stats.total_books
                    author.first_highlighted_at = stats.first_highlighted_at
                    author.last_highlighted_at = stats.last_highlighted_at
                    author.last_synced_at = datetime.now()
                    updated_authors += 1

                # Sync books for this author
                for book_data in stats.books:
                    book_result = await session.execute(
                        select(AuthorBook).where(AuthorBook.readwise_book_id == book_data["id"])
                    )
                    author_book = book_result.scalar_one_or_none()

                    if author_book is None:
                        author_book = AuthorBook(
                            author_id=author.id,
                            readwise_book_id=book_data["id"],
                            title=book_data["title"],
                            category=book_data["category"],
                            source_url=book_data["source_url"],
                            cover_image_url=book_data["cover_image_url"],
                            num_highlights=book_data["num_highlights"],
                            last_highlight_at=book_data["last_highlight_at"],
                        )
                        session.add(author_book)
                        total_books += 1
                    else:
                        author_book.num_highlights = book_data["num_highlights"]
                        author_book.last_highlight_at = book_data["last_highlight_at"]
                        total_books += 1

            await session.commit()

        return SyncResult(
            total_authors=len(author_data),
            new_authors=new_authors,
            updated_authors=updated_authors,
            total_books=total_books,
        )

    async def get_liked_authors(self, min_highlights: int = 2) -> list[Author]:
        """Get authors with at least min_highlights.

        Args:
            min_highlights: Minimum number of highlights to be considered "liked".

        Returns:
            List of Authors sorted by total_highlights descending.
        """
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Author)
                .where(Author.total_highlights >= min_highlights)
                .order_by(Author.total_highlights.desc())
            )
            return list(result.scalars().all())

    async def get_author_by_name(self, name: str) -> Author | None:
        """Get an author by name (case-insensitive).

        Args:
            name: Author name to look up.

        Returns:
            Author or None.
        """
        factory = await get_session_factory()
        async with factory() as session:
            normalized = normalize_author_name(name)
            result = await session.execute(
                select(Author).where(Author.normalized_name == normalized)
            )
            return result.scalar_one_or_none()

    async def get_author_highlight_count(self, name: str) -> int:
        """Get total highlights for an author.

        Args:
            name: Author name to look up.

        Returns:
            Total highlight count, 0 if author not found.
        """
        author = await self.get_author_by_name(name)
        return author.total_highlights if author else 0

    async def get_top_authors(self, limit: int = 20) -> list[Author]:
        """Get top authors by highlight count.

        Args:
            limit: Maximum number of authors to return.

        Returns:
            List of Authors.
        """
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Author).order_by(Author.total_highlights.desc()).limit(limit)
            )
            return list(result.scalars().all())

    async def mark_favorite(self, author_id: int, is_favorite: bool = True) -> None:
        """Mark an author as favorite/not favorite.

        Args:
            author_id: The author ID.
            is_favorite: Whether to mark as favorite.
        """
        factory = await get_session_factory()
        async with factory() as session:
            author = await session.get(Author, author_id)
            if author:
                author.is_favorite = is_favorite
                await session.commit()


# Singleton instance
_service: AuthorService | None = None


def get_author_service() -> AuthorService:
    """Get or create the author service singleton."""
    global _service
    if _service is None:
        _service = AuthorService()
    return _service
