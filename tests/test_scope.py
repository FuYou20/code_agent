"""Verify scope intersection, skill activation, and pre-execution denial."""

import json
import time

import pytest

from myagent.audit.log import verify
from myagent.llm.types import ToolCall
from myagent.skills.manager import activate
from myagent.skills.schema import Skill, SkillFrontmatter, ScopeManifestItem
from myagent.tools.guard import check_and_execute
from myagent.tools.registry import TOOLS, register
from myagent.tools.scope import (
    Scope,
    ScopeEscalationDenied,
    ToolRule,
    allows,
    intersect,
)


def test_unknown_tool_is_denied() -> None:
    assert not allows(Scope(tools={}), "file_read", {"path": "README.md"})


def test_child_cannot_add_tool() -> None:
    parent = Scope({"file_read": ToolRule("file_read", lambda _params: True)})
    child = Scope({"terminal_exec": ToolRule("terminal_exec", lambda _params: True)})
    with pytest.raises(ScopeEscalationDenied):
        intersect(parent, child)


def test_intersection_ands_parameter_rules() -> None:
    parent = Scope({"file_read": ToolRule("file_read", lambda p: p["path"].endswith(".py"))})
    child = Scope({"file_read": ToolRule("file_read", lambda p: p["path"].startswith("src/"))})
    effective = intersect(parent, child)
    assert allows(effective, "file_read", {"path": "src/app.py"})
    assert not allows(effective, "file_read", {"path": "tests/app.py"})
    assert not allows(effective, "file_read", {"path": "src/readme.md"})


def test_child_stricter_path_rule_wins() -> None:
    parent = Scope({"file_read": ToolRule("file_read", lambda p: p["path"].endswith(".py"))})
    child = Scope({"file_read": ToolRule("file_read", lambda p: p["path"].startswith("src/"))})
    effective = intersect(parent, child)
    assert allows(effective, "file_read", {"path": "src/main.py"})
    assert not allows(effective, "file_read", {"path": "tests/main.py"})


def test_child_wider_rule_cannot_bypass_parent() -> None:
    parent = Scope({"file_read": ToolRule("file_read", lambda p: p["path"] == "safe.txt")})
    child = Scope({"file_read": ToolRule("file_read", lambda _p: True)})
    effective = intersect(parent, child)
    assert allows(effective, "file_read", {"path": "safe.txt"})
    assert not allows(effective, "file_read", {"path": "secret.txt"})


def test_max_actions_uses_minimum() -> None:
    rule = ToolRule("file_read", lambda _p: True)
    effective = intersect(
        Scope({"file_read": rule}, max_actions=12),
        Scope({"file_read": rule}, max_actions=4),
    )
    assert effective.max_actions == 4


def test_expiry_uses_earlier_timestamp() -> None:
    rule = ToolRule("file_read", lambda _p: True)
    later = time.time() + 1000
    earlier = time.time() + 500
    effective = intersect(
        Scope({"file_read": rule}, expiry_ts=later),
        Scope({"file_read": rule}, expiry_ts=earlier),
    )
    assert effective.expiry_ts == earlier


def test_skill_activation_applies_path_manifest() -> None:
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="read-source",
            description="Read source files",
            scope_manifest=[ScopeManifestItem(tool="file_read", paths=["src"])],
        ),
        body="Read only source files.",
        path="C:/skills/read-source",
    )
    parent = Scope({"file_read": ToolRule("file_read", lambda _p: True)})
    effective = activate(skill, parent)
    assert allows(effective, "file_read", {"path": "src/app.py"})
    assert not allows(effective, "file_read", {"path": "tests/test_app.py"})
    assert not allows(effective, "file_read", {"path": "../src/app.py"})


def test_skill_activation_rejects_unavailable_tool() -> None:
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="run-command",
            description="Run one command",
            scope_manifest=[
                ScopeManifestItem(tool="terminal_exec", command_pattern="^pytest")
            ],
        ),
        body="Run tests.",
        path="C:/skills/run-command",
    )
    parent = Scope({"file_read": ToolRule("file_read", lambda _p: True)})
    with pytest.raises(ScopeEscalationDenied):
        activate(skill, parent)


def test_skill_activation_applies_command_pattern() -> None:
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="test-only",
            description="Only run pytest",
            scope_manifest=[
                ScopeManifestItem(tool="terminal_exec", command_pattern=r"^pytest(?:\s|$)")
            ],
        ),
        body="Run pytest.",
        path="C:/skills/test-only",
    )
    parent = Scope({"terminal_exec": ToolRule("terminal_exec", lambda _p: True)})
    effective = activate(skill, parent)
    assert allows(effective, "terminal_exec", {"command": "pytest -q"})
    assert not allows(effective, "terminal_exec", {"command": "pytestevil"})
    assert not allows(effective, "terminal_exec", {"command": "git status"})


def test_scope_denial_happens_before_execution_and_is_audited(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path))
    executed = False
    approved = False

    def dangerous(command: str) -> dict:
        nonlocal executed
        executed = True
        return {"command": command}

    def approve(_call: ToolCall) -> bool:
        nonlocal approved
        approved = True
        return True

    TOOLS.clear()
    register(
        "terminal_exec",
        {"type": "object", "properties": {"command": {"type": "string"}}},
        dangerous,
        risk="high",
    )
    parent = Scope({"file_read": ToolRule("file_read", lambda _p: True)})
    result = check_and_execute(
        ToolCall(id="denied-1", name="terminal_exec", arguments={"command": "pytest"}),
        parent,
        approve,
    )

    assert result["error"] == "scope_denied"
    assert not executed
    assert not approved
    assert verify()
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["entry"]["result"]["error"] == "scope_denied"
    TOOLS.clear()

