"""Tests for the author service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.article import Author, AuthorBook
from app.services.authors import (
    AuthorService,
    AuthorStats,
    SyncResult,
    get_author_service,
    normalize_author_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_readwise_book(
    *,
    id: int = 1,
    author: str | None = "Test Author",
    title: str = "Test Book",
    num_highlights: int = 5,
    last_highlight_at: datetime | None = None,
    category: str | None = "articles",
    source_url: str | None = "https://example.com",
    cover_image_url: str | None = None,
) -> MagicMock:
    """Create a mock Readwise book object."""
    book = MagicMock()
    book.id = id
    book.author = author
    book.title = title
    book.num_highlights = num_highlights
    book.last_highlight_at = last_highlight_at
    book.source_url = source_url
    book.cover_image_url = cover_image_url

    # category is an enum with .value
    if category is not None:
        cat_mock = MagicMock()
        cat_mock.value = category
        book.category = cat_mock
    else:
        book.category = None

    return book


async def _insert_author(
    session,
    *,
    name: str = "Jane Smith",
    total_highlights: int = 10,
    total_books: int = 3,
    is_favorite: bool = False,
) -> Author:
    """Insert an author into the test database."""
    author = Author(
        name=name,
        normalized_name=normalize_author_name(name),
        total_highlights=total_highlights,
        total_books=total_books,
        is_favorite=is_favorite,
        last_synced_at=datetime.now(),
    )
    session.add(author)
    await session.commit()
    await session.refresh(author)
    return author


# ---------------------------------------------------------------------------
# 1. normalize_author_name
# ---------------------------------------------------------------------------


class TestNormalizeAuthorName:
    """Test the normalize_author_name utility function."""

    async def test_lowercases(self):
        assert normalize_author_name("Jane Smith") == "jane smith"

    async def test_strips_whitespace(self):
        assert normalize_author_name("  Jane Smith  ") == "jane smith"

    async def test_empty_string(self):
        assert normalize_author_name("") == ""

    async def test_already_normalized(self):
        assert normalize_author_name("jane smith") == "jane smith"

    async def test_mixed_case_and_whitespace(self):
        assert normalize_author_name("  JOHN DOE  ") == "john doe"

    async def test_single_name(self):
        assert normalize_author_name("Madonna") == "madonna"


# ---------------------------------------------------------------------------
# 2. AuthorStats and SyncResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Test dataclass construction."""

    async def test_author_stats_defaults(self):
        stats = AuthorStats(
            name="Test",
            total_highlights=0,
            total_books=0,
            first_highlighted_at=None,
            last_highlighted_at=None,
            books=[],
        )
        assert stats.name == "Test"
        assert stats.total_highlights == 0
        assert stats.books == []

    async def test_sync_result_fields(self):
        result = SyncResult(
            total_authors=5,
            new_authors=2,
            updated_authors=3,
            total_books=15,
        )
        assert result.total_authors == 5
        assert result.new_authors == 2
        assert result.updated_authors == 3
        assert result.total_books == 15


# ---------------------------------------------------------------------------
# 3. get_liked_authors
# ---------------------------------------------------------------------------


class TestGetLikedAuthors:
    """Test get_liked_authors method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_authors_above_threshold(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should return authors with highlights >= min_highlights."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Prolific Author", total_highlights=50)
        await _insert_author(session, name="Occasional Author", total_highlights=5)
        await _insert_author(session, name="Rare Author", total_highlights=1)

        service = AuthorService()
        liked = await service.get_liked_authors(min_highlights=2)

        names = [a.name for a in liked]
        assert "Prolific Author" in names
        assert "Occasional Author" in names
        assert "Rare Author" not in names

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_sorted_by_highlights_descending(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Results should be ordered by total_highlights descending."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Author A", total_highlights=10)
        await _insert_author(session, name="Author B", total_highlights=50)
        await _insert_author(session, name="Author C", total_highlights=25)

        service = AuthorService()
        liked = await service.get_liked_authors(min_highlights=2)

        assert liked[0].name == "Author B"
        assert liked[1].name == "Author C"
        assert liked[2].name == "Author A"

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_empty_when_no_authors(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should return empty list when no authors meet threshold."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        service = AuthorService()
        liked = await service.get_liked_authors(min_highlights=2)
        assert liked == []

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_custom_min_highlights(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Custom min_highlights threshold should be respected."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Author X", total_highlights=8)
        await _insert_author(session, name="Author Y", total_highlights=12)

        service = AuthorService()
        liked = await service.get_liked_authors(min_highlights=10)

        assert len(liked) == 1
        assert liked[0].name == "Author Y"


# ---------------------------------------------------------------------------
# 4. get_author_by_name
# ---------------------------------------------------------------------------


class TestGetAuthorByName:
    """Test get_author_by_name method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_finds_existing_author(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should find an author by name (case-insensitive)."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Jane Smith", total_highlights=15)

        service = AuthorService()
        author = await service.get_author_by_name("Jane Smith")
        assert author is not None
        assert author.name == "Jane Smith"

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_case_insensitive_lookup(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Lookup should be case-insensitive."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Jane Smith", total_highlights=10)

        service = AuthorService()
        author = await service.get_author_by_name("JANE SMITH")
        assert author is not None
        assert author.name == "Jane Smith"

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_none_for_unknown(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should return None for unknown author names."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        service = AuthorService()
        author = await service.get_author_by_name("Unknown Person")
        assert author is None

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_strips_whitespace_in_lookup(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Lookup should strip whitespace."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Jane Smith", total_highlights=10)

        service = AuthorService()
        author = await service.get_author_by_name("  Jane Smith  ")
        assert author is not None


# ---------------------------------------------------------------------------
# 5. get_author_highlight_count
# ---------------------------------------------------------------------------


class TestGetAuthorHighlightCount:
    """Test get_author_highlight_count method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_highlight_count(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should return the total highlights for a known author."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Jane Smith", total_highlights=42)

        service = AuthorService()
        count = await service.get_author_highlight_count("Jane Smith")
        assert count == 42

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_zero_for_unknown_author(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should return 0 for an author not in the database."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        service = AuthorService()
        count = await service.get_author_highlight_count("Unknown Person")
        assert count == 0


# ---------------------------------------------------------------------------
# 6. get_top_authors
# ---------------------------------------------------------------------------


class TestGetTopAuthors:
    """Test get_top_authors method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_top_authors_by_highlights(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should return authors sorted by total_highlights descending."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        await _insert_author(session, name="Author A", total_highlights=10)
        await _insert_author(session, name="Author B", total_highlights=50)
        await _insert_author(session, name="Author C", total_highlights=25)

        service = AuthorService()
        top = await service.get_top_authors(limit=2)

        assert len(top) == 2
        assert top[0].name == "Author B"
        assert top[1].name == "Author C"

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_respects_limit(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Limit parameter should cap the number of results."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        for i in range(5):
            await _insert_author(session, name=f"Author {i}", total_highlights=i * 10)

        service = AuthorService()
        top = await service.get_top_authors(limit=3)
        assert len(top) == 3

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_returns_empty_when_no_authors(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should return empty list when there are no authors."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        service = AuthorService()
        top = await service.get_top_authors()
        assert top == []


# ---------------------------------------------------------------------------
# 7. mark_favorite
# ---------------------------------------------------------------------------


class TestMarkFavorite:
    """Test mark_favorite method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_mark_as_favorite(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should mark an author as favorite."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        author = await _insert_author(session, name="Fav Author", total_highlights=20)
        author_id = author.id

        service = AuthorService()
        await service.mark_favorite(author_id, is_favorite=True)

        async with session_factory() as s:
            refreshed = await s.get(Author, author_id)
            assert refreshed is not None
            assert refreshed.is_favorite is True

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_unmark_favorite(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should unmark an author as favorite."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        author = await _insert_author(session, name="Ex-Fav", total_highlights=10, is_favorite=True)
        author_id = author.id

        service = AuthorService()
        await service.mark_favorite(author_id, is_favorite=False)

        async with session_factory() as s:
            refreshed = await s.get(Author, author_id)
            assert refreshed is not None
            assert refreshed.is_favorite is False

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_mark_nonexistent_author(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Marking a nonexistent author should not raise."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        service = AuthorService()
        # Should not raise
        await service.mark_favorite(99999, is_favorite=True)


# ---------------------------------------------------------------------------
# 8. sync_authors_from_readwise
# ---------------------------------------------------------------------------


class TestSyncAuthorsFromReadwise:
    """Test the sync_authors_from_readwise method."""

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_syncs_new_authors(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should create new authors from Readwise book data."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        dt1 = datetime(2024, 1, 15)
        dt2 = datetime(2024, 6, 20)

        books = [
            _make_readwise_book(
                id=1, author="Author A", title="Book 1", num_highlights=5,
                last_highlight_at=dt1,
            ),
            _make_readwise_book(
                id=2, author="Author A", title="Book 2", num_highlights=3,
                last_highlight_at=dt2,
            ),
            _make_readwise_book(
                id=3, author="Author B", title="Book 3", num_highlights=10,
                last_highlight_at=dt1,
            ),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert isinstance(result, SyncResult)
        assert result.total_authors == 2
        assert result.new_authors == 2
        assert result.updated_authors == 0
        assert result.total_books == 3

        # Verify authors in DB
        async with session_factory() as session:
            from sqlalchemy import select

            all_authors = (await session.execute(select(Author))).scalars().all()
            names = {a.name for a in all_authors}
            assert "Author A" in names
            assert "Author B" in names

            # Check Author A aggregate stats
            author_a_result = await session.execute(
                select(Author).where(Author.normalized_name == "author a")
            )
            author_a = author_a_result.scalar_one()
            assert author_a.total_highlights == 8  # 5 + 3
            assert author_a.total_books == 2
            assert author_a.first_highlighted_at == dt1
            assert author_a.last_highlighted_at == dt2

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_updates_existing_authors(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Should update existing authors when syncing again."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        # Pre-insert an author
        await _insert_author(session, name="Author A", total_highlights=5, total_books=1)

        books = [
            _make_readwise_book(
                id=1, author="Author A", title="Book 1", num_highlights=10,
                last_highlight_at=datetime(2024, 6, 1),
            ),
            _make_readwise_book(
                id=2, author="Author A", title="Book 2", num_highlights=5,
                last_highlight_at=datetime(2024, 8, 1),
            ),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.new_authors == 0
        assert result.updated_authors == 1
        assert result.total_authors == 1

        # Verify updated stats
        async with session_factory() as s:
            from sqlalchemy import select

            author = (
                await s.execute(select(Author).where(Author.normalized_name == "author a"))
            ).scalar_one()
            assert author.total_highlights == 15  # 10 + 5
            assert author.total_books == 2

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_skips_books_with_no_author(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Books with no author should be skipped."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(id=1, author=None, num_highlights=5),
            _make_readwise_book(id=2, author="Real Author", num_highlights=3),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.total_authors == 1
        assert result.new_authors == 1

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_skips_books_with_zero_highlights(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Books with 0 highlights should be skipped."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(id=1, author="Author A", num_highlights=0),
            _make_readwise_book(id=2, author="Author B", num_highlights=5),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.total_authors == 1
        assert result.new_authors == 1

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_skips_books_with_blank_author(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Books with empty/whitespace author should be skipped."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(id=1, author="   ", num_highlights=5),
            _make_readwise_book(id=2, author="Real Author", num_highlights=3),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.total_authors == 1

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_syncs_author_books(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should create AuthorBook records for each book."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(
                id=101, author="Author A", title="First Book", num_highlights=5,
                source_url="https://example.com/1", category="articles",
            ),
            _make_readwise_book(
                id=102, author="Author A", title="Second Book", num_highlights=3,
                source_url="https://example.com/2", category="books",
            ),
        ]

        async def mock_list_books():
            for book in books:
                yield book

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        await service.sync_authors_from_readwise()

        async with session_factory() as s:
            from sqlalchemy import select

            author_books = (await s.execute(select(AuthorBook))).scalars().all()
            assert len(author_books) == 2
            titles = {b.title for b in author_books}
            assert "First Book" in titles
            assert "Second Book" in titles

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_updates_existing_author_books(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory, session
    ):
        """Re-syncing should update existing AuthorBook records."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        # Pre-insert author and book
        author = await _insert_author(session, name="Author A", total_highlights=5)
        book = AuthorBook(
            author_id=author.id,
            readwise_book_id=101,
            title="First Book",
            num_highlights=5,
        )
        session.add(book)
        await session.commit()

        # Now sync with updated highlight count
        new_dt = datetime(2025, 1, 1)
        books = [
            _make_readwise_book(
                id=101, author="Author A", title="First Book",
                num_highlights=15, last_highlight_at=new_dt,
            ),
        ]

        async def mock_list_books():
            for b in books:
                yield b

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.total_books == 1

        async with session_factory() as s:
            from sqlalchemy import select

            ab = (
                await s.execute(
                    select(AuthorBook).where(AuthorBook.readwise_book_id == 101)
                )
            ).scalar_one()
            assert ab.num_highlights == 15
            assert ab.last_highlight_at == new_dt

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_tracks_first_and_last_highlight_times(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Should correctly track earliest and latest highlight times."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        early = datetime(2023, 1, 1)
        middle = datetime(2024, 6, 15)
        late = datetime(2025, 1, 1)

        books = [
            _make_readwise_book(id=1, author="Author A", num_highlights=3, last_highlight_at=middle),
            _make_readwise_book(id=2, author="Author A", num_highlights=2, last_highlight_at=early),
            _make_readwise_book(id=3, author="Author A", num_highlights=5, last_highlight_at=late),
        ]

        async def mock_list_books():
            for b in books:
                yield b

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        await service.sync_authors_from_readwise()

        async with session_factory() as s:
            from sqlalchemy import select

            author = (
                await s.execute(select(Author).where(Author.normalized_name == "author a"))
            ).scalar_one()
            assert author.first_highlighted_at == early
            assert author.last_highlighted_at == late

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_no_highlight_at_fields_when_missing(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """When last_highlight_at is None on books, timestamps stay None."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(id=1, author="Author A", num_highlights=3, last_highlight_at=None),
        ]

        async def mock_list_books():
            for b in books:
                yield b

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        await service.sync_authors_from_readwise()

        async with session_factory() as s:
            from sqlalchemy import select

            author = (
                await s.execute(select(Author).where(Author.normalized_name == "author a"))
            ).scalar_one()
            assert author.first_highlighted_at is None
            assert author.last_highlighted_at is None

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_handles_null_category(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """Books with no category should have None stored."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        books = [
            _make_readwise_book(id=1, author="Author A", num_highlights=5, category=None),
        ]

        async def mock_list_books():
            for b in books:
                yield b

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        await service.sync_authors_from_readwise()

        async with session_factory() as s:
            from sqlalchemy import select

            ab = (await s.execute(select(AuthorBook))).scalar_one()
            assert ab.category is None

    @patch("app.services.authors.get_session_factory")
    @patch("app.services.authors.get_settings")
    async def test_empty_readwise_returns_zero_counts(
        self, mock_settings: MagicMock, mock_get_factory: MagicMock, session_factory
    ):
        """When Readwise has no books, all counts should be zero."""
        mock_settings.return_value = MagicMock(readwise_token="test-token")
        mock_get_factory.return_value = session_factory

        async def mock_list_books():
            return
            yield  # make it an async generator  # type: ignore[misc]

        service = AuthorService()
        service._client = MagicMock()
        service._client.v2.list_books = mock_list_books

        result = await service.sync_authors_from_readwise()

        assert result.total_authors == 0
        assert result.new_authors == 0
        assert result.updated_authors == 0
        assert result.total_books == 0


# ---------------------------------------------------------------------------
# 9. Singleton accessor
# ---------------------------------------------------------------------------


class TestGetAuthorService:
    """Test the singleton accessor."""

    async def test_returns_instance(self):
        import app.services.authors as authors_mod

        original = authors_mod._service
        try:
            authors_mod._service = None
            with patch("app.services.authors.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(readwise_token="test-token")
                instance = get_author_service()
                assert isinstance(instance, AuthorService)
        finally:
            authors_mod._service = original

    async def test_returns_same_instance(self):
        import app.services.authors as authors_mod

        original = authors_mod._service
        try:
            authors_mod._service = None
            with patch("app.services.authors.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock(readwise_token="test-token")
                instance1 = get_author_service()
                instance2 = get_author_service()
                assert instance1 is instance2
        finally:
            authors_mod._service = original
