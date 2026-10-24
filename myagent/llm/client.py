"""Provide the provider-neutral LLM protocol and an OpenAI implementation."""

from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI

from .types import LLMResponse, Message, ToolCall


class LLMClient(Protocol):
    def chat(self, messages: list[Message], tools: list[dict]) -> LLMResponse: ...


class OpenAIClient:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        options = {"api_key": api_key}
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)

    @staticmethod
    def _message_to_openai(message: Message) -> dict:
        item: dict = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls is not None:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        if message.reasoning_content is not None:
            item["reasoning_content"] = message.reasoning_content
        return item

    def chat(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        # Canonicalize once so equivalent schemas always have identical bytes/order.
        canonical_tools = json.dumps(tools, sort_keys=True, separators=(",", ":"))
        request = {
            "model": self.model,
            "messages": [self._message_to_openai(message) for message in messages],
        }
        if tools:
            request["tools"] = json.loads(canonical_tools)
        response = self._client.chat.completions.create(**request)
        choice = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for raw_call in choice.tool_calls or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": raw_call.function.arguments}
            tool_calls.append(
                ToolCall(
                    id=raw_call.id,
                    name=raw_call.function.name,
                    arguments=arguments,
                )
            )

        usage: dict[str, int] = {}
        if response.usage is not None:
            usage["prompt_tokens"] = response.usage.prompt_tokens
            usage["completion_tokens"] = response.usage.completion_tokens
            usage["total_tokens"] = response.usage.total_tokens
            details = getattr(response.usage, "prompt_tokens_details", None)
            detail_cached = (
                details.get("cached_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "cached_tokens", 0)
            )
            usage["cached_tokens"] = detail_cached or (
                getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0
            )
        return LLMResponse(
            content=choice.content,
            tool_calls=tool_calls,
            usage=usage,
            reasoning_content=getattr(choice, "reasoning_content", None),
        )

