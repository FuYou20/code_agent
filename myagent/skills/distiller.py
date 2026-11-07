"""Distill verified tool trajectories into versioned procedural Skills."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import yaml

from ..home import get_home
from ..llm.types import Message
from .schema import Skill, SkillFrontmatter


DISTILL_PROMPT = """你是一个技能提炼器。下面是完成一个 coding 任务的执行轨迹。
请生成一个 SKILL.md，包含：
1. frontmatter（name, description, scene, triggers, permission_requirements）
2. 分步骤的操作指南

约束：
- permission_requirements 只能描述轨迹中实际调用过的工具；它不是授权
- scene 使用稳定的任务场景标签，例如 code-repair、code-development、documentation
- 步骤要具体到工具名和参数模式
- 不要编造轨迹中没出现的工具

任务语义：
{task_context}

轨迹：
{trajectory}
"""


def _filtered_trajectory(history: list[Message]) -> tuple[str, set[str]]:
    rows: list[str] = []
    used_tools: set[str] = set()
    successful_call_ids: set[str] = set()
    for message in history:
        if message.role != "tool" or not message.tool_call_id:
            continue
        try:
            result = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(result, dict) and "error" not in result:
            successful_call_ids.add(message.tool_call_id)

    for message in history:
        if message.role == "assistant" and message.tool_calls:
            calls = []
            for call in message.tool_calls:
                if call.id not in successful_call_ids:
                    continue
                used_tools.add(call.name)
                calls.append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
            if calls:
                rows.append(
                    json.dumps(
                        {"role": "assistant", "tool_calls": calls},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        elif message.role == "tool" and message.tool_call_id in successful_call_ids:
            rows.append(
                json.dumps(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    return "\n".join(rows), used_tools


def _parse_generated(text: str) -> tuple[SkillFrontmatter, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:markdown|md|yaml)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    delimiters = list(re.finditer(r"(?m)^---\s*$", cleaned))
    if len(delimiters) < 2:
        raise ValueError("distilled skill is missing YAML frontmatter delimiters")
    raw_yaml = cleaned[delimiters[0].end() : delimiters[1].start()]
    raw_frontmatter = yaml.safe_load(raw_yaml)
    if not isinstance(raw_frontmatter, dict):
        raise ValueError("distilled skill frontmatter must be a mapping")
    frontmatter = SkillFrontmatter.model_validate(raw_frontmatter)
    body = cleaned[delimiters[1].end() :].lstrip("\r\n")
    return frontmatter, body


def _safe_name(name: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in name.strip()
    )
    normalized = re.sub(r"-+", "-", normalized).strip("-_")
    if not normalized:
        raise ValueError("distilled skill name has no safe path characters")
    return normalized


def _render(frontmatter: SkillFrontmatter, body: str) -> str:
    dumped = yaml.safe_dump(
        frontmatter.model_dump(exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    return f"---\n{dumped}\n---\n\n{body.rstrip()}\n"


def _archive_existing(skill_file: Path) -> tuple[int, Path | None]:
    if not skill_file.exists():
        return 1, None
    history_dir = skill_file.parent / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)
    versions: list[int] = []
    for candidate in history_dir.glob("v*.md"):
        match = re.fullmatch(r"v(\d+)\.md", candidate.name)
        if match:
            versions.append(int(match.group(1)))
    archived_version = max(versions, default=0) + 1
    archived_path = history_dir / f"v{archived_version}.md"
    skill_file.replace(archived_path)
    return archived_version + 1, archived_path


def distill(
    history: list[Message],
    client,
    *,
    task_context: str = "",
    artifact_manager=None,
    scope_key: str = "",
) -> Skill:
    """Create a versioned skill whose tools are limited to the real trajectory."""
    trajectory, used_tools = _filtered_trajectory(history) # 从完整对话历史中提取经过真实工具结果确认的操作轨迹，并统计实际使用过的工具
    response = client.chat(
        [
            Message(
                role="user",
                content=DISTILL_PROMPT.format(
                    task_context=task_context.strip()[:4000] or "(未提供)",
                    trajectory=trajectory,
                ),
            )
        ],
        tools=[],
    )
    if not response.content:
        raise ValueError("distillation model returned empty content")
    frontmatter, body = _parse_generated(response.content)

    accepted = []
    removed: set[str] = set()
    for item in frontmatter.permission_requirements:
        if item.tool in used_tools:
            accepted.append(item)
        else:
            removed.add(item.tool)
    if removed:
        warnings.warn(
            "已剔除轨迹中未调用的工具权限：" + ", ".join(sorted(removed)),
            UserWarning,
            stacklevel=2,
        )

    safe_name = _safe_name(frontmatter.name)
    frontmatter = frontmatter.model_copy(
        update={"name": safe_name, "permission_requirements": accepted}
    )
    skill_dir = (get_home() / "skills_store" / safe_name).resolve()
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    version, archived_path = _archive_existing(skill_file)
    if artifact_manager is not None and archived_path is not None:
        try:
            previous = artifact_manager.get_by_key(
                "skill", frontmatter.name, version - 1
            )
        except KeyError:
            previous = None
        if previous is not None:
            artifact_manager.update_content_ref(
                previous.artifact_id,
                str(archived_path),
            )
    rendered = _render(frontmatter, body)
    skill_file.write_text(rendered, encoding="utf-8")
    skill = Skill(
        frontmatter=frontmatter,
        body=body,
        path=str(skill_dir),
        version=version,
    )
    if artifact_manager is not None:
        from ..evolution.artifacts import TokenUsage

        artifact_manager.register_skill(
            skill.frontmatter.name,
            version=skill.version,
            scope_key=scope_key,
            content=rendered,
            content_ref=str(skill_file),
            creation_usage=TokenUsage.from_mapping(response.usage),
        )
    return skill

