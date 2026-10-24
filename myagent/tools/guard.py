"""Enforce scope and approval checks before every tool execution."""

from __future__ import annotations

from typing import Callable

from ..audit.log import log
from ..llm.types import ToolCall
from .registry import execute, lookup
from .scope import Scope, allows


def _audit(call: ToolCall, result: dict) -> None:
    log(
        {
            "tool_call": {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            },
            "result": result,
        }
    )


def check_and_execute(
    call: ToolCall,
    scope: Scope,
    approve_fn: Callable[[ToolCall], bool],
) -> dict:
    """Check permission and approval, execute, audit, and return the result."""
    if not allows(scope, call.name, call.arguments):
        result = {
            "error": "scope_denied",
            "tool": call.name,
            "message": "当前 Scope 不允许此工具调用或参数",
        }
        _audit(call, result)
        return result

    try:
        tool = lookup(call.name)
    except KeyError:
        result = {"error": "unknown_tool", "tool": call.name}
        _audit(call, result)
        return result

    if tool.risk == "high" and not approve_fn(call):
        result = {"error": "user_denied", "tool": call.name}
        _audit(call, result)
        return result

    result = execute(call.name, call.arguments)
    _audit(call, result)
    return result

