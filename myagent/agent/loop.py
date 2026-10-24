"""Run one synchronous user turn through model and guarded tool calls."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable

from ..audit.log import log as audit_log
from ..context.assembler import build_messages
from ..context.compressor import compress_if_needed
from ..llm.types import Message, ToolCall
from ..memory.retrieve import retrieve as retrieve_memory
from ..memory.store import insert_decision
from ..skills.manager import activate, load_skill
from ..skills.retriever import retrieve_relevant
from ..tools import registry
from ..tools.guard import check_and_execute
from ..tools.scope import Scope, ScopeEscalationDenied
from .state import SessionState


def _prepare_skill_turn(
    state: SessionState,
    user_input: str,
    parent_scope: Scope,
) -> tuple[list[dict], str, Scope]:
    """Retrieve L0 candidates and activate at most one high-confidence L1 skill."""
    if not user_input.strip() or not state.skill_index:
        return [], "", parent_scope

    matches = retrieve_relevant(user_input, state.skill_index, k=state.skill_top_k)
    by_name = {str(item.get("name")): item for item in state.skill_index}
    selected = (
        list(state.skill_index)
        if state.skill_strategy == "full_sidecar"
        else [by_name[match.name] for match in matches if match.name in by_name]
    )
    audit_log(
        {
            "event": "skill_retrieved",
            "session_id": state.session_id,
            "query_sha256": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "query_chars": len(user_input),
            "matches": [
                {
                    "name": match.name,
                    "score": match.score,
                    "reasons": list(match.reasons),
                }
                for match in matches
            ],
        }
    )

    if not matches or matches[0].score < state.skill_activation_min_score:
        return selected, "", parent_scope

    primary = matches[0]
    item = by_name.get(primary.name)
    try:
        if item is None or not item.get("path"):
            raise ValueError("skill index entry has no trusted path")
        skill = load_skill(Path(str(item["path"])))
        if skill.frontmatter.name != primary.name:
            raise ValueError("skill name changed after index loading")
        effective_scope = activate(skill, parent_scope)
    except (OSError, ValueError, ScopeEscalationDenied) as exc:
        audit_log(
            {
                "event": "skill_denied",
                "session_id": state.session_id,
                "skill": primary.name,
                "reason": type(exc).__name__,
                "message": str(exc),
            }
        )
        return selected, "", parent_scope

    audit_log(
        {
            "event": "skill_activated",
            "session_id": state.session_id,
            "skill": skill.frontmatter.name,
            "score": primary.score,
            "effective_tools": sorted(effective_scope.tools),
        }
    )
    return selected, skill.body, effective_scope


def _record_successful_change(
    state: SessionState,
    final_content: str,
    changes: list[tuple[ToolCall, dict]],
) -> None:
    if state.memory_conn is None:
        return
    files: list[str] = []
    actions: list[str] = []
    for call, result in changes:
        if call.name == "file_write":
            path = call.arguments.get("path")
            if isinstance(path, str) and path not in files:
                files.append(path)
            actions.append(f"file_write({path})")
        elif call.name == "git_commit":
            commit = result.get("commit", "unknown")
            actions.append(f"git_commit({commit})")
    summary = "成功变更：" + ", ".join(actions)
    if final_content.strip():
        summary += "\n最终结果：" + final_content.strip()
    insert_decision(
        state.memory_conn,
        state.session_id,
        "diff",
        summary,
        files,
        "success",
    )


def run_turn(
    state: SessionState,
    user_input: str,
    client,
    scope: Scope,
    approve_fn: Callable[[ToolCall], bool],
) -> str:
    """Run model/tool steps synchronously until a final answer or step limit."""
    max_steps = state.max_steps
    state.compression_client = client
    original_input = user_input
    recorded_input = False
    retrieved = (
        retrieve_memory(state.memory_conn, original_input)
        if state.memory_conn is not None and original_input.strip()
        else ""
    )
    skill_items, skill_body, effective_scope = _prepare_skill_turn(
        state,
        original_input,
        scope,
    )
    successful_changes: list[tuple[ToolCall, dict]] = []
    turn_had_error = False

    for _ in range(max_steps):
        state.step_count += 1
        compress_if_needed(state)
        messages = build_messages(
            state,
            user_input,
            retrieved,
            skill_items=skill_items,
            skill_body=skill_body,
        )
        response = client.chat(
            messages,
            tools=registry.schemas(set(effective_scope.tools)),
        )

        if not recorded_input:
            state.history.append(Message(role="user", content=original_input))
            recorded_input = True

        assistant = Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls or None,
            reasoning_content=response.reasoning_content,
        )
        state.history.append(assistant)

        if not response.tool_calls:
            if successful_changes and not turn_had_error:
                _record_successful_change(state, response.content or "", successful_changes)
            return response.content or ""

        for call in response.tool_calls:
            result = check_and_execute(call, effective_scope, approve_fn)
            if "error" in result:
                turn_had_error = True
            elif call.name in {"file_write", "git_commit"}:
                successful_changes.append((call, result))
            state.history.append(
                Message(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False, sort_keys=True),
                    tool_call_id=call.id,
                )
            )
        user_input = ""

    return "达到步数上限"

