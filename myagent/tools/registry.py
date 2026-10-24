"""Maintain the deterministic registry for the six built-in tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class ToolDef:
    name: str
    schema: dict
    fn: Callable
    risk: Literal["low", "medium", "high"]


TOOLS: dict[str, ToolDef] = {}


def register(name, schema, fn, risk="low") -> None:
    """Register or replace one tool definition."""
    if risk not in {"low", "medium", "high"}:
        raise ValueError(f"invalid tool risk: {risk}")
    TOOLS[name] = ToolDef(name=name, schema=schema, fn=fn, risk=risk)


def lookup(name) -> ToolDef:
    """Return a registered tool or raise ``KeyError``."""
    return TOOLS[name]


def execute(name, args: dict) -> dict:
    """Execute a tool and turn exceptions into serializable error results."""
    try:
        result = lookup(name).fn(**args)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as exc:  # Tool failures are data returned to the model.
        return {"error": type(exc).__name__, "message": str(exc)}


def schemas(allowed_names: set[str] | None = None) -> list[dict]:
    """Return sorted OpenAI schemas, optionally limited to an effective scope."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.schema.get("description", ""),
                "parameters": {
                    key: value
                    for key, value in tool.schema.items()
                    if key != "description"
                },
            },
        }
        for tool in sorted(TOOLS.values(), key=lambda item: item.name)
        if allowed_names is None or tool.name in allowed_names
    ]

