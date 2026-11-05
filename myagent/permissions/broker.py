"""Authorize and execute tool calls through the atomic permission tree."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..audit.log import log
from ..llm.types import ToolCall
from ..tools.registry import execute, lookup
from .model import PermissionAtom
from .telemetry import atom_payload, emit_permission_event
from .tree import PermissionTree


class ToolPermissionAdapter:
    """Translate concrete tool arguments into complete permission atoms.

    This adapter is trusted code.  A tool must not be registered for runtime use
    until its filesystem effects are represented here.  Unknown tools are
    treated as file-less, which is suitable only for explicitly registered
    pure/custom tools whose root policy also uses ``without_files``.
    """

    _WORKSPACE_READ = {"git_read"}
    _WORKSPACE_READ_WRITE = {"git_commit", "terminal_exec", "test_run"}

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def _logical_path(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tool path must be a non-empty string")
        candidate = Path(value)
        target = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve(strict=False)
        )
        try:
            relative = target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("tool path escapes the workspace") from exc
        rendered = relative.as_posix()
        return rendered or "."

    def request_atom(self, call: ToolCall) -> PermissionAtom:
        if call.name == "file_read":
            return PermissionAtom.read(call.name, self._logical_path(call.arguments.get("path")))
        if call.name == "file_write":
            return PermissionAtom.write(call.name, self._logical_path(call.arguments.get("path")))
        if call.name in self._WORKSPACE_READ:
            return PermissionAtom.read(call.name, ".")
        if call.name in self._WORKSPACE_READ_WRITE:
            return PermissionAtom.read_write(call.name, read=(".",), write=(".",))
        return PermissionAtom.without_files(call.name)


def advertised_tools(tree: PermissionTree, agent_id: str) -> set[str]:
    """Return only tool names present in the agent's bound permission node."""
    tools = {atom.tool for atom in tree.policy_for_agent(agent_id)}
    emit_permission_event(
        "tool.visibility.filtered",
        agent_id=agent_id,
        visible_tools=sorted(tools),
    )
    return tools


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
    tree: PermissionTree,
    agent_id: str,
    adapter: ToolPermissionAdapter,
    approve_fn: Callable[[ToolCall], bool],
) -> dict:
    """Authorize one complete atom, request approval, execute, and audit."""
    emit_permission_event(
        "tool.call.received",
        agent_id=agent_id,
        call_id=call.id,
        tool=call.name,
        argument_names=sorted(call.arguments),
    )
    try:
        lookup(call.name)
    except KeyError:
        result = {"error": "unknown_tool", "tool": call.name}
        emit_permission_event(
            "tool.call.denied",
            agent_id=agent_id,
            tool=call.name,
            reason="unknown_tool",
        )
        _audit(call, result)
        return result

    try:
        requested = adapter.request_atom(call)
    except (TypeError, ValueError, OSError) as exc:
        result = {
            "error": "permission_denied",
            "tool": call.name,
            "message": str(exc),
        }
        emit_permission_event(
            "tool.request_atom.rejected",
            agent_id=agent_id,
            tool=call.name,
            reason="invalid_arguments_or_path",
            error_type=type(exc).__name__,
        )
        _audit(call, result)
        return result
    emit_permission_event(
        "tool.request_atom.built",
        agent_id=agent_id,
        call_id=call.id,
        requested_atom=atom_payload(requested),
    )

    if not tree.allows(agent_id, requested):
        result = {
            "error": "permission_denied",
            "tool": call.name,
            "message": "当前 Agent 的原子权限节点不允许该工具与文件访问组合",
        }
        emit_permission_event(
            "tool.call.denied",
            agent_id=agent_id,
            tool=call.name,
            reason="permission_denied",
        )
        _audit(call, result)
        return result

    tool = lookup(call.name)
    approval_required = tool.risk == "high"
    approved = not approval_required or approve_fn(call)
    emit_permission_event(
        "tool.approval.checked",
        agent_id=agent_id,
        tool=call.name,
        risk=tool.risk,
        required=approval_required,
        approved=approved,
    )
    if not approved:
        result = {"error": "user_denied", "tool": call.name}
        emit_permission_event(
            "tool.call.denied",
            agent_id=agent_id,
            tool=call.name,
            reason="user_denied",
        )
        _audit(call, result)
        return result

    emit_permission_event(
        "tool.execution.started",
        agent_id=agent_id,
        call_id=call.id,
        tool=call.name,
    )
    result = execute(call.name, call.arguments)
    emit_permission_event(
        "tool.execution.finished",
        agent_id=agent_id,
        call_id=call.id,
        tool=call.name,
        succeeded="error" not in result,
        error=str(result.get("error", "")),
    )
    _audit(call, result)
    return result
