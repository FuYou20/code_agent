"""Convert skill permission manifests into non-escalating effective scopes."""

from __future__ import annotations

import fnmatch
import posixpath
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..tools.scope import Scope, ToolRule, intersect
from .schema import ScopeManifestItem, Skill, SkillFrontmatter


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


def _normalize_path(value: str) -> str:
    return posixpath.normpath(value.replace("\\", "/"))


def _path_allowed(path: object, allowed_paths: list[str]) -> bool:
    if not isinstance(path, str):
        return False
    target = _normalize_path(path)
    if target == ".." or target.startswith("../"):
        return False
    target_is_absolute = target.startswith("/") or bool(re.match(r"^[A-Za-z]:/", target))

    for configured in allowed_paths:
        allowed = _normalize_path(configured)
        if allowed in {"", "."}:
            return not target_is_absolute
        if any(character in allowed for character in "*?["):
            if fnmatch.fnmatchcase(target, allowed):
                return True
            continue
        prefix = allowed.rstrip("/") + "/"
        if target == allowed or target.startswith(prefix):
            return True
    return False


def _item_matcher(item: ScopeManifestItem):
    command_regex = re.compile(item.command_pattern) if item.command_pattern is not None else None
    allowed_paths = list(item.paths) if item.paths is not None else None

    def matches(params: dict) -> bool:
        if allowed_paths is not None and not _path_allowed(params.get("path"), allowed_paths):
            return False
        if command_regex is not None:
            command = params.get("command")
            if not isinstance(command, str) or command_regex.match(command) is None:
                return False
        return True

    return matches


def activate(skill: Skill, parent_scope: Scope) -> Scope:
    """Intersect a skill manifest with its parent scope without adding tools."""
    grouped: dict[str, list[ScopeManifestItem]] = {}
    for item in skill.frontmatter.scope_manifest:
        grouped.setdefault(item.tool, []).append(item)

    rules: dict[str, ToolRule] = {}
    for tool_name, items in grouped.items():
        matchers = [_item_matcher(item) for item in items]

        def matches(params: dict, item_matchers=matchers) -> bool:
            return any(matcher(params) for matcher in item_matchers)

        rules[tool_name] = ToolRule(
            tool_name=tool_name,
            matches=matches,
            description=f"skill:{skill.frontmatter.name}",
        )

    child = Scope(tools=rules)
    return intersect(parent_scope, child)

