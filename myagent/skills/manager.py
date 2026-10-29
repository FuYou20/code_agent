"""Load and index Skill knowledge without participating in authorization."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import Skill, SkillFrontmatter


def _skill_file(skill_path: Path) -> Path:
    return skill_path / "SKILL.md" if skill_path.is_dir() else skill_path


def _parse_skill(skill_file: Path) -> Skill:
    text = skill_file.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"missing YAML frontmatter: {skill_file}")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"unterminated YAML frontmatter: {skill_file}")
    raw_frontmatter = yaml.safe_load(parts[1])
    if not isinstance(raw_frontmatter, dict):
        raise ValueError(f"frontmatter must be a mapping: {skill_file}")
    frontmatter = SkillFrontmatter.model_validate(raw_frontmatter)
    return Skill(
        frontmatter=frontmatter,
        body=parts[2].lstrip("\r\n"),
        path=str(skill_file.parent.resolve()),
    )


def load_index(store: Path) -> list[dict]:
    """Scan ``store/*/SKILL.md`` and return a stable L0 metadata index."""
    if not store.is_dir():
        return []
    index: list[dict] = []
    for skill_file in sorted(store.glob("*/SKILL.md"), key=lambda path: path.as_posix()):
        try:
            skill = _parse_skill(skill_file)
        except (OSError, ValueError, yaml.YAMLError, ValidationError):
            continue
        index.append(
            {
                "name": skill.frontmatter.name,
                "description": skill.frontmatter.description,
                "scene": skill.frontmatter.scene,
                "triggers": list(skill.frontmatter.triggers),
                "path": skill.path,
            }
        )
    return index


def load_body(skill_path: Path) -> str:
    """Read a skill's complete ``SKILL.md`` for L1 disclosure."""
    return _skill_file(skill_path).read_text(encoding="utf-8")


def load_skill(skill_path: Path) -> Skill:
    """Load and validate a complete skill for activation and L1 disclosure."""
    return _parse_skill(_skill_file(skill_path))


def find_relevant(user_input: str, index: list[dict]) -> list[str]:
    """Return skill names whose non-empty trigger occurs in the user input."""
    query = user_input.casefold()
    relevant: list[str] = []
    for item in index:
        triggers = item.get("triggers", [])
        if any(str(trigger).casefold() in query for trigger in triggers if str(trigger).strip()):
            name = str(item["name"])
            if name not in relevant:
                relevant.append(name)
    return relevant

