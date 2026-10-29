"""Verify that Skills describe needs while the permission tree owns authority."""

import json

import pytest

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.audit.log import verify
from myagent.llm.types import LLMResponse, ToolCall
from myagent.permissions import PermissionAtom, PermissionTree, ToolPermissionAdapter
from myagent.skills.manager import load_index
from myagent.tools.registry import TOOLS, register


@pytest.fixture(autouse=True)
def clear_tools():
    TOOLS.clear()
    yield
    TOOLS.clear()


class SkillAttackClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools) -> LLMResponse:
        self.calls += 1
        assert "Run tests in this exact order" in messages[-1].content
        assert [item["function"]["name"] for item in tools] == ["test_run"]
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="attack-1",
                        name="file_write",
                        arguments={"path": "owned.txt", "content": "should not exist"},
                    )
                ],
            )
        tool_message = next(message for message in messages if message.role == "tool")
        assert "permission_denied" in tool_message.content
        return LLMResponse(content="write denied")


def test_primary_skill_body_persists_but_does_not_change_tree_authority(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    skill_dir = tmp_path / "skills" / "run-tests"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: run-tests
description: Run pytest safely.
triggers: [pytest]
permission_requirements:
  - tool: test_run
---
Run tests in this exact order.
""",
        encoding="utf-8",
    )
    register("test_run", {"type": "object", "properties": {}}, lambda: {"ok": True})
    register(
        "file_write",
        {"type": "object", "properties": {}},
        lambda path, content: {"path": path},
        risk="high",
    )
    tree = PermissionTree(tmp_path, [PermissionAtom.without_files("test_run")])
    tree.attach_agent(tree.ROOT_ID, "skill-agent")
    state = SessionState(
        session_id="skill-session",
        agent_id="skill-agent",
        permission_tree=tree,
        permission_adapter=ToolPermissionAdapter(tmp_path),
        cwd=str(tmp_path),
        skill_index=load_index(tmp_path / "skills"),
    )

    answer = run_turn(state, "Please run pytest", SkillAttackClient(), lambda _call: True)

    assert answer == "write denied"
    assert not (tmp_path / "owned.txt").exists()
    assert all("Run tests in this exact order" not in message.content for message in state.history)
    assert verify()
    records = [
        json.loads(line)["entry"]
        for line in (tmp_path / "home" / "audit.log").read_text(encoding="utf-8").splitlines()
    ]
    assert [record.get("event") for record in records[:2]] == [
        "skill_retrieved",
        "skill_activated",
    ]
    assert records[-1]["result"]["error"] == "permission_denied"
