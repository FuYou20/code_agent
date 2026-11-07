"""Small persistence wrapper around the existing synchronous approval prompt."""

from __future__ import annotations

from collections.abc import Callable

from ..llm.types import ToolCall
from .models import AgentTask
from .store import DurableGatewayStore


def request_persisted_approval(
    store: DurableGatewayStore,
    *,
    task: AgentTask,
    agent_id: str,
    call: ToolCall,
    delegate: Callable[[ToolCall], bool],
    channel: str,
) -> bool:
    """Record PENDING, run the current prompt, then store its decision."""
    approval_id = store.create_approval(
        task_id=task.task_id,
        session_id=task.session_id,
        agent_id=agent_id,
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=call.arguments,
        requested_by=channel,
    )
    try:
        approved = bool(delegate(call))
    except BaseException as exc:
        store.resolve_approval(
            approval_id,
            False,
            decided_by=channel,
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise
    store.resolve_approval(
        approval_id,
        approved,
        decided_by=channel,
    )
    return approved
