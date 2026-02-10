"""API endpoints for chat functionality."""

import json
import logging
from typing import cast

from anthropic.types import MessageParam
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.models.article import (
    ChatMessage,
    ChatThread,
    get_session_factory,
)
from app.services.chat import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _build_history(messages: list[ChatMessage]) -> list[MessageParam]:
    """Reconstruct Anthropic API message format from DB messages.

    Handles plain text messages and structured content_blocks
    (tool_use / tool_result) when present.
    """
    history: list[MessageParam] = []
    for m in messages:
        if m.content_blocks:
            blocks = json.loads(m.content_blocks)
            history.append(cast(MessageParam, {"role": m.role, "content": blocks}))
        else:
            history.append(cast(MessageParam, {"role": m.role, "content": m.content}))
    return history


@router.post("/message")
async def send_message(request: Request):
    """Send a message and stream the response via SSE."""
    body = await request.json()
    message = body["message"]
    thread_id = body.get("thread_id")

    factory = await get_session_factory()

    # Create or get thread, save user message, load history
    async with factory() as session:
        if not thread_id:
            thread = ChatThread(title=message[:50])
            session.add(thread)
            await session.flush()
            thread_id = thread.id
        else:
            # Update timestamp on existing thread
            thread_result = await session.execute(
                select(ChatThread).where(ChatThread.id == thread_id)
            )
            thread = thread_result.scalar_one_or_none()
            if not thread:
                return {"error": "Thread not found"}, 404

        # Save user message
        user_msg = ChatMessage(thread_id=thread_id, role="user", content=message)
        session.add(user_msg)
        await session.commit()

        # Load full conversation history
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at)
        )
        db_messages = result.scalars().all()

    # Build Anthropic message history from DB messages
    history = _build_history(db_messages)

    chat_service = ChatService()

    # Capture thread_id for the closure
    _thread_id = thread_id

    async def generate():
        full_response = ""
        async for chunk in chat_service.send_message(history):
            # Detect tool-use markers emitted by ChatService
            if chunk.startswith("__tool_use__:"):
                payload = chunk[len("__tool_use__:") :]
                yield f"event: tool_use\ndata: {payload}\n\n"
                continue
            if chunk.startswith("__tool_done__:"):
                payload = chunk[len("__tool_done__:") :]
                yield f"event: tool_done\ndata: {payload}\n\n"
                continue

            full_response += chunk
            for line in chunk.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

        # Save tool messages and assistant response using a fresh session
        try:
            async with factory() as session:
                # Save intermediate tool messages (assistant tool_use + user tool_result)
                for tool_msg in chat_service.tool_messages:
                    blocks_json = json.dumps(tool_msg["content_blocks"])
                    display = "[tool call]" if tool_msg["role"] == "assistant" else "[tool result]"
                    msg = ChatMessage(
                        thread_id=_thread_id,
                        role=tool_msg["role"],
                        content=display,
                        content_blocks=blocks_json,
                    )
                    session.add(msg)

                # Save final assistant text response
                assistant_msg = ChatMessage(
                    thread_id=_thread_id,
                    role="assistant",
                    content=full_response,
                )
                session.add(assistant_msg)
                await session.commit()
        except Exception:
            logger.exception("Failed to save assistant message for thread %s", _thread_id)
            yield "event: error\ndata: Failed to save response\n\n"

        # Signal completion with thread_id
        yield f"event: done\ndata: {_thread_id}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/threads")
async def list_threads():
    """List all chat threads, most recently updated first."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(select(ChatThread).order_by(ChatThread.updated_at.desc()))
        threads = result.scalars().all()

        return [
            {
                "id": t.id,
                "title": t.title,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }
            for t in threads
        ]


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: int):
    """Get all messages in a thread (excluding tool use/result messages from display)."""
    factory = await get_session_factory()
    async with factory() as session:
        # Verify thread exists
        thread_result = await session.execute(select(ChatThread).where(ChatThread.id == thread_id))
        thread = thread_result.scalar_one_or_none()
        if not thread:
            return {"error": "Thread not found"}, 404

        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at)
        )
        messages = result.scalars().all()

        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
            if not m.content_blocks  # Hide tool_use/tool_result messages from display
        ]


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: int):
    """Delete a chat thread and all its messages."""
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(select(ChatThread).where(ChatThread.id == thread_id))
        thread = result.scalar_one_or_none()
        if not thread:
            return {"error": "Thread not found"}, 404

        await session.delete(thread)
        await session.commit()

    return {"ok": True}
