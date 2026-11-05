"""Turn verified execution evidence into memory and future Skill candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3

from ..llm.types import ToolCall


@dataclass(frozen=True)
class ToolObservation:
    call: ToolCall
    result: dict


@dataclass(frozen=True)
class EvolutionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    changed_files: tuple[str, ...] = ()


_CARD_TASK_LIMIT = 140
_CARD_RESULT_LIMIT = 240


def _compact_text(value: str, limit: int) -> str:
    """Collapse model prose into a deterministic, bounded memory-card field."""
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def build_memory_card(
    user_input: str,
    final_content: str,
    observations: list[ToolObservation],
) -> str:
    """Build a compact reusable record without spending another model call."""
    verification = "test_run=passed" if any(
        observation.call.name == "test_run"
        and "error" not in observation.result
        and observation.result.get("returncode") == 0
        for observation in observations
    ) else "durable_change_recorded"
    return "\n".join(
        (
            "[经验卡]",
            f"task: {_compact_text(user_input, _CARD_TASK_LIMIT)}",
            f"lesson: {_compact_text(final_content, _CARD_RESULT_LIMIT)}",
            f"evidence: {verification}",
        )
    )


def evaluate_turn(
    final_content: str,
    observations: list[ToolObservation],
) -> EvolutionDecision:
    """Apply one shared success gate before any durable evolution output."""
    reasons: list[str] = []
    changed_files: list[str] = []
    durable_change = False
    verification_results: list[int] = []

    if not final_content.strip():
        reasons.append("missing_final_answer")

    for observation in observations:
        call, result = observation.call, observation.result
        if "error" in result:
            # A failed mutation means the durable result is untrustworthy.
            # Recoverable exploration failures (for example a denied shell
            # listing followed by bounded file reads) do not invalidate a
            # later hidden-test-passing repair and are excluded at distillation.
            if call.name in {"file_write", "git_commit"}:
                reasons.append(f"tool_error:{call.name}")
            continue
        if call.name == "file_write":
            durable_change = True
            path = call.arguments.get("path")
            if isinstance(path, str) and path not in changed_files:
                changed_files.append(path)
        elif call.name == "git_commit":
            durable_change = True
        elif call.name == "test_run":
            verification_results.append(int(result.get("returncode", 1)))

    # A red-green-refactor trajectory is still a successful trajectory when
    # its final verification passes. Earlier failing runs are useful evidence,
    # not a reason to discard the repaired result.
    if verification_results and verification_results[-1] != 0:
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
    user_input: str,
    final_content: str,
    observations: list[ToolObservation],
    decision: EvolutionDecision,
    scope_key: str = "",
) -> int | None:
    """Persist a compact successful experience after the shared gate passes."""
    if conn is None or not decision.eligible:
        return None

    # Local import keeps the storage layer free to initialize the shared
    # artifact schema without creating an evolution<->memory import cycle.
    from ..memory.store import insert_decision

    summary = build_memory_card(
        user_input,
        final_content,
        observations,
    )
    return insert_decision(
        conn,
        session_id,
        "experience",
        summary,
        list(decision.changed_files),
        "success",
        scope_key,
    )
