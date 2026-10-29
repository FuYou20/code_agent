"""Verify trajectory filtering, scope validation, persistence, and versioning."""

from pathlib import Path

import pytest
import yaml

from myagent.llm.types import LLMResponse, Message, ToolCall
from myagent.skills.distiller import distill
from myagent.skills.manager import load_index


class FakeDistillClient:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.requests = []

    def chat(self, messages, tools) -> LLMResponse:
        self.requests.append((messages, tools))
        return LLMResponse(content=next(self.outputs))


def _history() -> list[Message]:
    return [
        Message(role="user", content="PRIVATE CHATTER"),
        Message(role="assistant", content="unrelated small talk"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="file_read",
                    arguments={"path": "src/app.py"},
                )
            ],
        ),
        Message(
            role="tool",
            content='{"content":"source"}',
            tool_call_id="call-1",
        ),
    ]


def _generated(body: str, include_invalid_tool: bool = False) -> str:
    invalid = "\n  - tool: terminal_exec\n    command_pattern: '^rm '" if include_invalid_tool else ""
    return f"""```markdown
---
name: learned-reader
description: Read one source file safely
triggers:
  - inspect source
permission_requirements:
  - tool: file_read
    paths:
      - src
{invalid}
---

{body}
```"""


def test_distill_filters_chatter_and_removes_invented_tools(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    client = FakeDistillClient([_generated("Use file_read.", include_invalid_tool=True)])

    with pytest.warns(UserWarning, match="terminal_exec"):
        skill = distill(_history(), client)

    assert [item.tool for item in skill.frontmatter.permission_requirements] == ["file_read"]
    assert skill.version == 1
    assert Path(skill.path).is_absolute()
    request_message, request_tools = client.requests[0]
    assert request_tools == []
    prompt = request_message[0].content
    assert "file_read" in prompt
    assert "PRIVATE CHATTER" not in prompt
    assert "unrelated small talk" not in prompt

    stored = Path(skill.path, "SKILL.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(stored.split("---", 2)[1])
    assert [item["tool"] for item in frontmatter["permission_requirements"]] == ["file_read"]
    assert "terminal_exec" not in stored.split("---", 2)[1]
    restarted_index = load_index(tmp_path / "home" / "skills_store")
    assert [item["name"] for item in restarted_index] == ["learned-reader"]


def test_distill_archives_previous_versions(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    client = FakeDistillClient(
        [_generated("First version."), _generated("Second version.")]
    )

    first = distill(_history(), client)
    second = distill(_history(), client)

    assert first.version == 1
    assert second.version == 2
    skill_dir = Path(second.path)
    assert "First version." in (skill_dir / ".history" / "v1.md").read_text(
        encoding="utf-8"
    )
    assert "Second version." in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_distill_removes_all_permissions_when_trajectory_has_no_tools(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    client = FakeDistillClient([_generated("No tool steps.")])
    history = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    with pytest.warns(UserWarning, match="file_read"):
        skill = distill(history, client)
    assert skill.frontmatter.permission_requirements == []


def test_distill_sanitizes_generated_skill_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    output = _generated("Safe path.").replace("name: learned-reader", "name: ../../reader")
    skill = distill(_history(), FakeDistillClient([output]))
    assert skill.frontmatter.name == "reader"
    assert Path(skill.path).parent == (tmp_path / "home" / "skills_store").resolve()

