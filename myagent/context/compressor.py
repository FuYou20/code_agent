"""Compress oversized conversation history while retaining recent messages."""

from __future__ import annotations

import json

from ..agent.state import SessionState
from ..llm.types import Message


def _serialize(messages: list[Message]) -> str:
    rows: list[str] = []
    for message in messages:
        item: dict = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            item["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ]
        if message.reasoning_content is not None:
            item["reasoning_content"] = message.reasoning_content
        rows.append(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(rows)


def _estimate_tokens(messages: list[Message]) -> int:
    return int(sum(len(str(message)) for message in messages) / 4)


def compress_if_needed(state: SessionState, max_tokens: int = 60000) -> None:
    """Replace old middle history with an LLM summary only after the limit."""
    if _estimate_tokens(state.history) <= max_tokens:
        return

    first_system = state.history[:1] if state.history and state.history[0].role == "system" else []
    middle_start = 1 if first_system else 0
    tail_start = max(middle_start, len(state.history) - 10)
    middle = state.history[middle_start:tail_start]
    recent = state.history[tail_start:]
    if not middle:
        return

    prompt = (
        "用 200 字总结这段对话的关键决策。保留已确认的需求、修改、文件、工具结果、"
        "测试结论和未解决问题；不要编造。\n\n"
        + _serialize(middle)
    )
    client = state.compression_client
    if client is not None:
        response = client.chat([Message(role="user", content=prompt)], tools=[])
        summary = (response.content or "").strip()
    else:
        # Direct callers can still reduce history; normal agent execution always supplies an LLM.
        compact = " ".join(message.content.strip() for message in middle if message.content.strip())
        summary = compact[:200]
    if not summary:
        summary = "较早对话已压缩；模型未返回可用摘要。"

    state.history = [
        *first_system,
        Message(role="assistant", content=f"[历史摘要]\n{summary}"),
        *recent,
    ]

