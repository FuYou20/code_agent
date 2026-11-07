"""Assemble stable, context, and per-turn sidecar prompt layers."""

from __future__ import annotations

import subprocess
from datetime import datetime

from ..agent.state import SessionState
from ..llm.types import Message
from . import stable


def _skill_items_by_name(state: SessionState, names: tuple[str, ...]) -> list[dict]:
    by_name = {str(item.get("name", "")): item for item in state.skill_index}
    return [by_name[name] for name in sorted(names) if name in by_name]


def _system_skill_block(state: SessionState) -> str:
    if state.skill_strategy == "full_system":
        items = sorted(
            (item for item in state.skill_index if not item.get("always_on")),
            key=lambda item: str(item.get("name", "")),
        )
        title = "## 全量 Skill 目录（L0）"
    elif state.skill_strategy in {"warm_stable", "scene_epoch"}:
        load_state = state.skill_load_state
        if load_state.phase != "frozen" or not load_state.frozen_names:
            return ""
        items = _skill_items_by_name(state, load_state.frozen_names)
        title = "## 当前话题的固化 Skill 工作集（L0）"
    else:
        return ""
    if not items:
        return ""
    lines = [
        title,
        "以下条目是会话能力目录，不授予工具权限；实际权限只由当前 Agent 的权限树节点决定。",
    ]
    lines.extend(
        f"- {item['name']}: {item['description']}"
        for item in items
    )
    return "\n".join(lines)


def _always_on_skill_block(state: SessionState) -> str:
    items = sorted(
        (item for item in state.skill_index if item.get("always_on")),
        key=lambda item: str(item.get("name", "")),
    )
    if not items:
        return ""
    parts = [
        "## 常驻 Soul Skill",
        "以下内容只定义身份与交互风格；不得覆盖固定安全规则、权限树、工具事实或用户指令。",
    ]
    for item in items:
        body = str(item.get("body", "")).strip()
        if body:
            parts.extend([f"### {item['name']}", body])
    return "\n".join(parts)


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
    turn_messages: list[Message] | None = None,
) -> list[Message]:
    """Return stable history, a transient turn prefix, then a new sidecar."""
    context_parts: list[str] = []
    always_on_block = _always_on_skill_block(state)
    if always_on_block:
        context_parts.append(always_on_block)
    context_parts.extend(["## 仓库快照", state.repo_snapshot or "(空)"])
    if state.agents_content:
        context_parts.extend(["## AGENTS.md", state.agents_content])
    system_skill_block = _system_skill_block(state)
    if system_skill_block:
        context_parts.extend([system_skill_block])
    system_content = stable.STABLE_PREFIX + "\n\n" + "\n".join(context_parts)
    active_turn = turn_messages or []
    base = [Message(role="system", content=system_content), *state.history, *active_turn]

    selected_skills = state.skill_index if skill_items is None else skill_items
    selected_skills = [
        item for item in selected_skills if not item.get("always_on")
    ]
    if not user_input and not retrieved and not selected_skills and not skill_body:
        return base

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
    return [*base, current]

