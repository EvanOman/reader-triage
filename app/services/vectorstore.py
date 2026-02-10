"""Vector store for semantic article search using Qdrant and sentence-transformers."""

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.models.article import Article, ArticleScore, ArticleTag, get_session_factory

logger = logging.getLogger(__name__)

COLLECTION_NAME = "articles"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Chunking strategy interface
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text to embed, linked back to its source article."""

    text: str
    chunk_index: int = 0
    extra_payload: dict = field(default_factory=dict)


class ChunkingStrategy(Protocol):
    """Protocol for article chunking strategies.

    Implementations take article fields and return one or more Chunks.
    Each chunk becomes a separate vector in the store.
    """

    def chunk_article(
        self,
        title: str,
        author: str | None,
        overall_assessment: str | None,
        content_preview: str | None,
        score_reasons: str | None,
    ) -> list[Chunk]: ...


class WholeDocumentChunker:
    """Embeds each article as a single chunk.

    Combines title, author, AI assessment, score reasons, and content preview
    into one string. This is the simplest strategy — one vector per article.
    """

    def chunk_article(
        self,
        title: str,
        author: str | None,
        overall_assessment: str | None,
        content_preview: str | None,
        score_reasons: str | None,
    ) -> list[Chunk]:
        parts = [title]
        if author:
            parts[0] = f"{title} by {author}"
        if overall_assessment:
            parts.append(overall_assessment)
        if score_reasons:
            try:
                reasons = json.loads(score_reasons)
                if isinstance(reasons, list):
                    parts.extend(r for r in reasons if isinstance(r, str) and r)
            except (json.JSONDecodeError, TypeError):
                pass
        if content_preview:
            parts.append(content_preview)
        return [Chunk(text=". ".join(parts))]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def _point_id(article_id: str, chunk_index: int = 0) -> int:
    """Deterministic numeric point ID from article_id and chunk index."""
    return abs(hash(f"{article_id}:{chunk_index}")) % (2**63)


class VectorStore:
    """Qdrant-backed vector store for semantic article search.

    Uses local file-based storage (no server required) and sentence-transformers
    for embedding generation.
    """

    def __init__(
        self,
        path: str = "./qdrant_data",
        chunker: ChunkingStrategy | None = None,
    ):
        self._client = QdrantClient(path=path)
        self._model: SentenceTransformer | None = None
        self._chunker = chunker or WholeDocumentChunker()
        self._ensure_collection()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        return self._model

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self._client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    def _embed_texts(self, texts: list[str], prefix: str = "search_document") -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Nomic models require a task prefix. Use 'search_document' for indexing
        and 'search_query' for queries.
        """
        model = self._get_model()
        prefixed = [f"{prefix}: {t}" for t in texts]
        embeddings = model.encode(prefixed, show_progress_bar=False)
        return embeddings.tolist()

    def embed_article(self, article_id: str, text: str, metadata: dict):
        """Generate embedding for an article and upsert into Qdrant.

        Args:
            article_id: Unique article identifier (used as point ID via hash).
            text: The text to embed.
            metadata: Payload metadata (title, author, info_score, tags, etc.).
        """
        embeddings = self._embed_texts([text])
        point = PointStruct(
            id=_point_id(article_id),
            vector=embeddings[0],
            payload={"article_id": article_id, **metadata},
        )
        self._client.upsert(collection_name=COLLECTION_NAME, points=[point])

    async def embed_all_articles(self, batch_size: int = 32) -> int:
        """Embed all scored articles from the DB that aren't already embedded.

        Uses the configured chunking strategy to produce chunks, then embeds
        and upserts them into Qdrant.

        Args:
            batch_size: Number of articles to embed per batch.

        Returns:
            Number of articles embedded.
        """
        factory = await get_session_factory()

        # Load all scored articles with their tags
        async with factory() as session:
            result = await session.execute(
                select(Article, ArticleScore)
                .join(ArticleScore)
                .where(ArticleScore.overall_assessment.isnot(None))
            )
            rows = result.all()

            articles_data = []
            for article, score in rows:
                tag_result = await session.execute(
                    select(ArticleTag.tag_slug).where(ArticleTag.article_id == article.id)
                )
                tags = [row[0] for row in tag_result.all()]

                articles_data.append(
                    {
                        "article_id": article.id,
                        "title": article.title,
                        "author": article.author,
                        "content_preview": article.content_preview,
                        "overall_assessment": score.overall_assessment,
                        "score_reasons": score.score_reasons,
                        "info_score": score.info_score,
                        "tags": tags,
                    }
                )

        # Get already-embedded article IDs from Qdrant
        existing_ids = set()
        offset = None
        while True:
            results, next_offset = self._client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["article_id"],
                with_vectors=False,
            )
            for point in results:
                if point.payload and "article_id" in point.payload:
                    existing_ids.add(point.payload["article_id"])
            if next_offset is None:
                break
            offset = next_offset

        # Filter to only un-embedded articles
        to_embed = [a for a in articles_data if a["article_id"] not in existing_ids]

        if not to_embed:
            logger.info("All %d articles already embedded", len(articles_data))
            return 0

        logger.info(
            "Embedding %d new articles (%d already in store)",
            len(to_embed),
            len(existing_ids),
        )

        # Chunk all articles, then embed in batches
        chunked_articles: list[tuple[dict, list[Chunk]]] = []
        for a in to_embed:
            chunks = self._chunker.chunk_article(
                title=a["title"],
                author=a["author"],
                overall_assessment=a["overall_assessment"],
                content_preview=a["content_preview"],
                score_reasons=a["score_reasons"],
            )
            chunked_articles.append((a, chunks))

        # Flatten to (article_data, chunk) pairs for batched embedding
        all_pairs = [(a, c) for a, chunks in chunked_articles for c in chunks]

        count = 0
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i : i + batch_size]
            texts = [chunk.text for _, chunk in batch]
            embeddings = self._embed_texts(texts)

            points = []
            for (a, chunk), emb in zip(batch, embeddings, strict=True):
                points.append(
                    PointStruct(
                        id=_point_id(a["article_id"], chunk.chunk_index),
                        vector=emb,
                        payload={
                            "article_id": a["article_id"],
                            "title": a["title"],
                            "author": a["author"],
                            "info_score": a["info_score"],
                            "tags": a["tags"],
                            "chunk_index": chunk.chunk_index,
                            **chunk.extra_payload,
                        },
                    )
                )

            self._client.upsert(collection_name=COLLECTION_NAME, points=points)
            count += len(batch)
            logger.info("Embedded %d/%d chunks", count, len(all_pairs))

        articles_count = len(chunked_articles)
        if len(all_pairs) != articles_count:
            logger.info(
                "Chunking produced %d chunks from %d articles (avg %.1f chunks/article)",
                len(all_pairs),
                articles_count,
                len(all_pairs) / articles_count,
            )

        return articles_count

    def search(self, query: str, limit: int = 10, min_score: float = 0.0) -> list[dict]:
        """Semantic search for articles similar to a query.

        When multiple chunks from the same article match, only the
        highest-scoring chunk is returned.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            min_score: Minimum cosine similarity score (0.0-1.0).

        Returns:
            List of dicts with article_id, title, author, info_score, tags,
            and similarity score.
        """
        query_embedding = self._embed_texts([query], prefix="search_query")[0]

        # Fetch extra results to account for deduplication across chunks
        fetch_limit = limit * 3

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=fetch_limit,
            score_threshold=min_score if min_score > 0 else None,
            with_payload=True,
        )

        # Deduplicate: keep highest-scoring chunk per article
        seen: dict[str, dict] = {}
        for hit in results.points:
            if not hit.payload:
                continue
            aid = hit.payload.get("article_id", "")
            if aid not in seen or hit.score > seen[aid]["similarity"]:
                seen[aid] = {
                    "article_id": aid,
                    "title": hit.payload.get("title", ""),
                    "author": hit.payload.get("author"),
                    "info_score": hit.payload.get("info_score", 0),
                    "tags": hit.payload.get("tags", []),
                    "similarity": hit.score,
                }

        ranked = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
        return ranked[:limit]

    def collection_count(self) -> int:
        """Return the number of points in the collection."""
        info = self._client.get_collection(COLLECTION_NAME)
        return info.points_count or 0


# Singleton instance
_vectorstore: VectorStore | None = None


def get_vectorstore(path: str | None = None) -> VectorStore:
    """Get or create the VectorStore singleton.

    Args:
        path: Override the default Qdrant storage path. Only used on first call.

    Returns:
        The VectorStore singleton instance.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore(path=path or "./qdrant_data")
    return _vectorstore
