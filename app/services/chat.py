"""Chat service for article discovery using Anthropic API with tool use."""

import json
import logging
import re
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic
from anthropic.types import TextDelta, ToolParam, ToolUseBlock
from sqlalchemy import func, select

from app.models.article import (
    Article,
    ArticleScore,
    ArticleTag,
    Summary,
    get_session_factory,
    search_articles_fts,
)
from app.services.tagger import get_all_tags

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a reading assistant for a curated article inbox. You help the user find \
articles to read based on their interests, mood, or questions. You have access to a database of \
scored and tagged articles.

When the user describes what they want to read, use your tools to search and browse the article \
collection. Present your findings conversationally, explaining why each article might be a good fit.

For each recommended article, include:
- The title and author
- The score and a brief note on why it scored that way
- The tags
- A brief description of what makes it relevant to the user's request

You can search by keywords, browse by tags, look at article scores, and read article content to \
give informed recommendations.

Response Style:
- Be concise and conversational. This is a chat, not an essay.
- Keep initial responses brief. Use bullet points and bold for scannability.
- After your initial response, invite the user to go deeper if relevant.
- Only give lengthy, detailed responses when the user explicitly asks for depth."""

CHAT_TOOLS: list[ToolParam] = [
    {
        "name": "search",
        "description": (
            "Search articles by keywords, concepts, or natural language. "
            "Combines keyword search (title, author, full article text) with "
            "semantic search (meaning-based over AI summaries). Use this as "
            "your primary search tool for any query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or natural language search query",
                },
                "modes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["keyword", "semantic"]},
                    "description": "Search modes to use (default: both)",
                    "default": ["keyword", "semantic"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_article",
        "description": (
            "Get full details for a specific article by its ID, including scores, "
            "tags, assessment, and summary if available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "The article ID to look up",
                }
            },
            "required": ["article_id"],
        },
    },
    {
        "name": "read_article_content",
        "description": (
            "Fetch the full article text from Readwise for deep reading. "
            "Use when you need to read the actual content to answer a question "
            "about what the article says."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "string",
                    "description": "The article ID to read content for",
                }
            },
            "required": ["article_id"],
        },
    },
    {
        "name": "browse_by_tag",
        "description": (
            "List articles with a specific tag, sorted by score. "
            "Use when the user wants to explore a topic area."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "The tag slug to browse (e.g. 'ai-agents', 'startups')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles to return (default 10)",
                    "default": 10,
                },
            },
            "required": ["tag"],
        },
    },
    {
        "name": "list_tags",
        "description": (
            "Get all available tags with article counts. "
            "Use to understand what topics are available in the collection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "browse_top_articles",
        "description": (
            "Get the highest-scored articles, optionally filtered by minimum score. "
            "Use when the user wants to see the best articles overall."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of articles to return (default 10)",
                    "default": 10,
                },
                "min_score": {
                    "type": "integer",
                    "description": "Minimum info_score to include (optional)",
                },
            },
        },
    },
]


def _serialize_content_blocks(content_blocks: list) -> list[dict]:
    """Serialize Anthropic content blocks (SDK objects) to plain dicts."""
    result = []
    for block in content_blocks:
        if hasattr(block, "type"):
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            else:
                if hasattr(block, "model_dump"):
                    result.append(block.model_dump())
                else:
                    result.append({"type": str(block.type)})
        elif isinstance(block, dict):
            result.append(block)
    return result


class ChatService:
    """Service for chatting about articles using Claude with tool use."""

    def __init__(self, chat_model: str = "claude-opus-4-6"):
        self._client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        self._chat_model = chat_model
        self.tool_messages: list[dict] = []

    @staticmethod
    def _tool_result_summary(tool_name: str, result: dict) -> str:
        """Build a short human-readable summary of a tool result."""
        if "error" in result:
            return result["error"]
        if tool_name == "search":
            n = len(result.get("results", []))
            modes = ", ".join(result.get("modes_used", []))
            return f"Found {n} article{'s' if n != 1 else ''} via {modes}"
        if tool_name == "get_article":
            title = result.get("title", "Unknown")
            return f"Loaded: {title[:50]}"
        if tool_name == "read_article_content":
            length = len(result.get("content", ""))
            return f"Read {length} chars of content"
        if tool_name == "browse_by_tag":
            n = len(result.get("articles", []))
            tag = result.get("tag", "")
            return f"Found {n} article{'s' if n != 1 else ''} tagged '{tag}'"
        if tool_name == "list_tags":
            n = len(result.get("tags", []))
            return f"Found {n} tag{'s' if n != 1 else ''}"
        if tool_name == "browse_top_articles":
            n = len(result.get("articles", []))
            return f"Found {n} top article{'s' if n != 1 else ''}"
        return "Done"

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call and return results."""
        logger.info("Tool call: %s(%s)", tool_name, tool_input)

        try:
            if tool_name == "search":
                return await self._tool_search(tool_input)
            if tool_name == "get_article":
                return await self._tool_get_article(tool_input)
            if tool_name == "read_article_content":
                return await self._tool_read_article_content(tool_input)
            if tool_name == "browse_by_tag":
                return await self._tool_browse_by_tag(tool_input)
            if tool_name == "list_tags":
                return await self._tool_list_tags()
            if tool_name == "browse_top_articles":
                return await self._tool_browse_top_articles(tool_input)
            return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.exception("Tool execution error for %s", tool_name)
            return {"error": f"Tool error: {e!s}"}

    async def _tool_search(self, tool_input: dict) -> dict:
        """Unified search combining keyword (FTS) and semantic (vector) search with RRF."""
        query = tool_input["query"]
        modes = tool_input.get("modes", ["keyword", "semantic"])
        limit = tool_input.get("limit", 10)
        fetch_limit = 50  # fetch more from each source for better RRF merging

        # Collect ranked lists: {article_id: rank} per mode
        rankings: list[dict[str, int]] = []
        modes_used = []

        if "keyword" in modes:
            try:
                keyword_ids = await search_articles_fts(query, limit=fetch_limit)
                if keyword_ids:
                    rankings.append({aid: rank for rank, aid in enumerate(keyword_ids)})
                    modes_used.append("keyword")
            except Exception as e:
                logger.warning("Keyword search failed: %s", e)

        if "semantic" in modes:
            try:
                from app.services.vectorstore import get_vectorstore

                vs = get_vectorstore()
                sem_results = vs.search(query, limit=fetch_limit)
                if sem_results:
                    rankings.append({r["article_id"]: rank for rank, r in enumerate(sem_results)})
                    modes_used.append("semantic")
            except Exception as e:
                logger.warning("Semantic search unavailable: %s", e)

        if not rankings:
            return {"results": [], "query": query, "modes_used": modes_used}

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for ranking in rankings:
            for aid, rank in ranking.items():
                rrf_scores[aid] = rrf_scores.get(aid, 0.0) + 1.0 / (60 + rank)

        # Sort by RRF score descending, take top `limit`
        sorted_ids = sorted(rrf_scores, key=lambda aid: rrf_scores[aid], reverse=True)[:limit]

        # Fetch article details
        factory = await get_session_factory()
        async with factory() as session:
            results = []
            for aid in sorted_ids:
                article = await session.get(Article, aid)
                if not article:
                    continue
                score_result = await session.execute(
                    select(ArticleScore).where(ArticleScore.article_id == aid)
                )
                score = score_result.scalar_one_or_none()
                tag_result = await session.execute(
                    select(ArticleTag.tag_slug).where(ArticleTag.article_id == aid)
                )
                tags = [row[0] for row in tag_result.all()]

                results.append(
                    {
                        "id": article.id,
                        "title": article.title,
                        "author": article.author,
                        "url": article.url,
                        "word_count": article.word_count,
                        "location": article.location,
                        "info_score": score.info_score if score else 0,
                        "overall_assessment": score.overall_assessment if score else None,
                        "tags": tags,
                    }
                )

        logger.info("search(%s, modes=%s) returned %d results", query, modes_used, len(results))
        return {"results": results, "query": query, "modes_used": modes_used}

    async def _tool_get_article(self, tool_input: dict) -> dict:
        """Get full details for a single article."""
        article_id = tool_input["article_id"]
        factory = await get_session_factory()
        async with factory() as session:
            article = await session.get(Article, article_id)
            if not article:
                return {"error": f"Article not found: {article_id}"}

            score_result = await session.execute(
                select(ArticleScore).where(ArticleScore.article_id == article_id)
            )
            score = score_result.scalar_one_or_none()

            summary_result = await session.execute(
                select(Summary).where(Summary.article_id == article_id)
            )
            summary = summary_result.scalar_one_or_none()

            tag_result = await session.execute(
                select(ArticleTag.tag_slug).where(ArticleTag.article_id == article_id)
            )
            tags = [row[0] for row in tag_result.all()]

            result: dict[str, object] = {
                "id": article.id,
                "title": article.title,
                "author": article.author,
                "url": article.url,
                "word_count": article.word_count,
                "location": article.location,
                "category": article.category,
                "site_name": article.site_name,
                "tags": tags,
            }

            if score:
                result.update(
                    {
                        "info_score": score.info_score,
                        "priority_score": score.priority_score,
                        "specificity_score": score.specificity_score,
                        "novelty_score": score.novelty_score,
                        "depth_score": score.depth_score,
                        "actionability_score": score.actionability_score,
                        "overall_assessment": score.overall_assessment,
                        "score_reasons": json.loads(score.score_reasons)
                        if score.score_reasons
                        else [],
                        "skip_recommended": score.skip_recommended,
                        "skip_reason": score.skip_reason,
                    }
                )

            if summary:
                result.update(
                    {
                        "summary_text": summary.summary_text,
                        "key_points": json.loads(summary.key_points) if summary.key_points else [],
                    }
                )

        return result

    async def _tool_read_article_content(self, tool_input: dict) -> dict:
        """Read article content, preferring local DB copy over Readwise API."""
        article_id = tool_input["article_id"]

        factory = await get_session_factory()
        async with factory() as session:
            article = await session.get(Article, article_id)
            if not article:
                return {"error": f"Article not found: {article_id}"}

            # Use locally stored content if available
            if article.content:
                content = article.content
                if len(content) > 10000:
                    content = content[:10000] + "\n\n[Content truncated at 10,000 characters]"
                return {
                    "article_id": article_id,
                    "title": article.title,
                    "content": content,
                }

        # Fall back to Readwise API
        try:
            from app.services.readwise import get_readwise_service

            service = get_readwise_service()
            doc = await service.get_document(article_id, with_content=True)
            if not doc or not doc.content:
                return {
                    "error": "Could not fetch content from Readwise",
                    "article_id": article_id,
                }

            # Strip HTML tags
            content = re.sub(r"<[^>]+>", "", doc.content)
            if len(content) > 10000:
                content = content[:10000] + "\n\n[Content truncated at 10,000 characters]"

            return {
                "article_id": article_id,
                "title": doc.title,
                "content": content,
            }
        except Exception as e:
            logger.exception("Failed to fetch article content for %s", article_id)
            return {"error": f"Failed to fetch content: {e!s}", "article_id": article_id}

    async def _tool_browse_by_tag(self, tool_input: dict) -> dict:
        """List articles with a specific tag, sorted by score."""
        tag = tool_input["tag"]
        limit = tool_input.get("limit", 10)

        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Article, ArticleScore)
                .join(ArticleScore)
                .join(ArticleTag)
                .where(ArticleTag.tag_slug == tag)
                .where(Article.location != "archive")
                .order_by(ArticleScore.info_score.desc())
                .limit(limit)
            )
            rows = result.all()

            articles = []
            for article, score in rows:
                tag_result = await session.execute(
                    select(ArticleTag.tag_slug).where(ArticleTag.article_id == article.id)
                )
                all_tags = [row[0] for row in tag_result.all()]

                articles.append(
                    {
                        "id": article.id,
                        "title": article.title,
                        "author": article.author,
                        "info_score": score.info_score,
                        "overall_assessment": score.overall_assessment,
                        "tags": all_tags,
                    }
                )

        logger.info("browse_by_tag(%s) returned %d results", tag, len(articles))
        return {"tag": tag, "articles": articles}

    async def _tool_list_tags(self) -> dict:
        """Get all available tags with article counts."""
        factory = await get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(ArticleTag.tag_slug, func.count(ArticleTag.id))
                .join(Article)
                .where(Article.location != "archive")
                .group_by(ArticleTag.tag_slug)
            )
            db_counts = dict(result.all())

        all_tags = get_all_tags()
        tags = [
            {
                "slug": t.slug,
                "name": t.name,
                "description": t.description,
                "article_count": db_counts.get(t.slug, 0),
            }
            for t in all_tags
            if db_counts.get(t.slug, 0) > 0
        ]

        return {"tags": tags}

    async def _tool_browse_top_articles(self, tool_input: dict) -> dict:
        """Get highest-scored articles."""
        limit = tool_input.get("limit", 10)
        min_score = tool_input.get("min_score")

        factory = await get_session_factory()
        async with factory() as session:
            query = (
                select(Article, ArticleScore)
                .join(ArticleScore)
                .where(Article.location != "archive")
                .order_by(ArticleScore.info_score.desc())
                .limit(limit)
            )
            if min_score is not None:
                query = query.where(ArticleScore.info_score >= min_score)

            result = await session.execute(query)
            rows = result.all()

            articles = []
            for article, score in rows:
                tag_result = await session.execute(
                    select(ArticleTag.tag_slug).where(ArticleTag.article_id == article.id)
                )
                tags = [row[0] for row in tag_result.all()]

                articles.append(
                    {
                        "id": article.id,
                        "title": article.title,
                        "author": article.author,
                        "info_score": score.info_score,
                        "overall_assessment": score.overall_assessment,
                        "tags": tags,
                    }
                )

        logger.info("browse_top_articles returned %d results", len(articles))
        return {"articles": articles}

    async def send_message(self, history: list) -> AsyncGenerator[str, None]:
        """Stream a response, handling tool use loops.

        Yields text chunks for streaming, plus __tool_use__ and __tool_done__ markers.
        """
        self.tool_messages = []
        messages: list = list(history)
        max_tool_rounds = 5

        try:
            for _round in range(max_tool_rounds):
                if _round > 0:
                    yield "\n\n"

                async with self._client.messages.stream(
                    model=self._chat_model,
                    max_tokens=16384,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=CHAT_TOOLS,
                ) as stream:
                    async for event in stream:
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, TextDelta):
                            yield delta.text

                    final_message = await stream.get_final_message()

                if final_message.stop_reason == "tool_use":
                    # Extract tool use blocks
                    tool_use_blocks = [
                        b for b in final_message.content if isinstance(b, ToolUseBlock)
                    ]

                    # Serialize content blocks for persistence
                    serialized_content = _serialize_content_blocks(final_message.content)

                    # Append the full assistant message (with both text + tool_use blocks)
                    messages.append({"role": "assistant", "content": final_message.content})

                    # Execute each tool, caching results, yielding status markers
                    tool_result_content = []
                    for block in tool_use_blocks:
                        yield f"__tool_use__:{json.dumps({'tool': block.name, 'input': block.input})}"
                        tool_result = await self._execute_tool(block.name, block.input)
                        summary = self._tool_result_summary(block.name, tool_result)
                        yield f"__tool_done__:{json.dumps({'tool': block.name, 'summary': summary})}"
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(tool_result),
                            }
                        )

                    # Store serialized tool messages for persistence by the API layer
                    self.tool_messages.append(
                        {"role": "assistant", "content_blocks": serialized_content}
                    )
                    self.tool_messages.append(
                        {"role": "user", "content_blocks": tool_result_content}
                    )

                    messages.append({"role": "user", "content": tool_result_content})
                    continue  # loop back to stream again with tool results
                else:
                    break  # got a final text response

            # Warn if response was truncated
            if final_message and final_message.stop_reason == "max_tokens":
                yield "\n\n---\n*[Response truncated due to length limit]*"

            # Warn if all rounds used tools without a final text response
            if final_message and final_message.stop_reason == "tool_use":
                yield "\n\n---\n*[Reached maximum tool use rounds. Try a more specific question.]*"

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield f"I apologize, but I encountered an error: {e!s}"
