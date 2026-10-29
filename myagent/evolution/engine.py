"""Turn verified execution evidence into memory and future Skill candidates."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from ..llm.types import ToolCall
from ..memory.store import insert_decision


@dataclass(frozen=True)
class ToolObservation:
    call: ToolCall
    result: dict


@dataclass(frozen=True)
class EvolutionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    changed_files: tuple[str, ...] = ()


def evaluate_turn(
    final_content: str,
    observations: list[ToolObservation],
) -> EvolutionDecision:
    """Apply one shared success gate before any durable evolution output."""
    reasons: list[str] = []
    changed_files: list[str] = []
    durable_change = False

    if not final_content.strip():
        reasons.append("missing_final_answer")

    for observation in observations:
        call, result = observation.call, observation.result
        if "error" in result:
            reasons.append(f"tool_error:{call.name}")
            continue
        if call.name == "file_write":
            durable_change = True
            path = call.arguments.get("path")
            if isinstance(path, str) and path not in changed_files:
                changed_files.append(path)
        elif call.name == "git_commit":
            durable_change = True
        elif call.name == "test_run" and result.get("returncode") != 0:
            reasons.append("verification_failed:test_run")

    if not durable_change:
        reasons.append("no_durable_change")

    return EvolutionDecision(
        eligible=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        changed_files=tuple(changed_files),
    )


def record_experience(
    conn: sqlite3.Connection | None,
    session_id: str,
    final_content: str,
    observations: list[ToolObservation],
    decision: EvolutionDecision,
) -> bool:
    """Persist a compact successful experience after the shared gate passes."""
    if conn is None or not decision.eligible:
        return False

    actions: list[str] = []
    for observation in observations:
        call, result = observation.call, observation.result
        if call.name == "file_write" and "error" not in result:
            actions.append(f"file_write({call.arguments.get('path')})")
        elif call.name == "git_commit" and "error" not in result:
            actions.append(f"git_commit({result.get('commit', 'unknown')})")

    summary = "成功变更：" + ", ".join(actions)
    summary += "\n最终结果：" + final_content.strip()
    insert_decision(
        conn,
        session_id,
        "experience",
        summary,
        list(decision.changed_files),
        "success",
    )
    return True
