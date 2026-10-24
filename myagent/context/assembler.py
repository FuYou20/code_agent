"""Assemble stable, context, and per-turn sidecar prompt layers."""

from __future__ import annotations

import subprocess
from datetime import datetime

from ..agent.state import SessionState
from ..llm.types import Message
from . import stable


def _diff_stat(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def build_messages(
    state: SessionState,
    user_input: str,
    retrieved: str = "",
    skill_items: list[dict] | None = None,
    skill_body: str = "",
) -> list[Message]:
    """Return stable system, chronological history, then an ephemeral sidecar."""
    context_parts = ["## 仓库快照", state.repo_snapshot or "(空)"]
    if state.agents_content:
        context_parts.extend(["## AGENTS.md", state.agents_content])
    system_content = stable.STABLE_PREFIX + "\n\n" + "\n".join(context_parts)

    selected_skills = state.skill_index if skill_items is None else skill_items
    if not user_input and not retrieved and not selected_skills and not skill_body:
        return [Message(role="system", content=system_content), *state.history]

    sidecar_parts = ["## 当前轮 Sidecar"]
    if retrieved:
        sidecar_parts.extend(["### 检索结果", retrieved])
    diff = _diff_stat(state.cwd)
    sidecar_parts.extend(["### Git diff --stat", diff or "(无)"])
    sidecar_parts.extend(["### 日期", datetime.now().strftime("%Y-%m-%d")])
    if selected_skills:
        sidecar_parts.append("## 可用技能（L0）")
        sidecar_parts.extend(
            f"- {item['name']}: {item['description']}" for item in selected_skills
        )
    if skill_body:
        sidecar_parts.extend(["## 当前主技能（L1）", skill_body])
    if user_input:
        sidecar_parts.extend(["## 用户输入", user_input])
    current = Message(role="user", content="\n".join(sidecar_parts))
    return [Message(role="system", content=system_content), *state.history, current]

