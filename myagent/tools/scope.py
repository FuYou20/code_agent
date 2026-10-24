"""Define non-escalating tool scopes and parameter-level permission checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolRule:
    """A tool permission whose predicate must accept call parameters."""

    tool_name: str
    matches: Callable[[dict], bool]
    description: str = ""


@dataclass(frozen=True)
class Scope:
    tools: dict[str, ToolRule]
    max_actions: int = 100
    expiry_ts: float | None = None


class ScopeEscalationDenied(Exception):
    """Raised when a child scope requests a tool absent from its parent."""


def intersect(parent: Scope, child: Scope) -> Scope:
    """Return a child scope constrained by the parent without permitting escalation."""
    extra_tools = set(child.tools) - set(parent.tools)
    if extra_tools:
        names = ", ".join(sorted(extra_tools))
        raise ScopeEscalationDenied(f"child scope requests unavailable tools: {names}")

    combined: dict[str, ToolRule] = {}
    for name, child_rule in child.tools.items():
        parent_rule = parent.tools[name]

        def combined_matches(
            params: dict,
            parent_matches=parent_rule.matches,
            child_matches=child_rule.matches,
        ) -> bool:
            return parent_matches(params) and child_matches(params)

        descriptions = filter(None, [parent_rule.description, child_rule.description])
        combined[name] = ToolRule(
            tool_name=name,
            matches=combined_matches,
            description=" AND ".join(descriptions),
        )

    expiries = [value for value in (parent.expiry_ts, child.expiry_ts) if value is not None]
    return Scope(
        tools=combined,
        max_actions=min(parent.max_actions, child.max_actions),
        expiry_ts=min(expiries) if expiries else None,
    )


def allows(scope: Scope, tool_name: str, params: dict) -> bool:
    """Return whether the scope currently allows the parameterized tool call."""
    if scope.expiry_ts is not None and time.time() > scope.expiry_ts:
        return False
    rule = scope.tools.get(tool_name)
    if rule is None:
        return False
    try:
        return bool(rule.matches(params))
    except Exception:
        return False

