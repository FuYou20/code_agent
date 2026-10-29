"""Verify runtime authorization uses only atomic permission-tree bindings."""

import json

import pytest

from myagent.llm.types import ToolCall
from myagent.permissions import (
    PermissionAtom,
    PermissionTree,
    ToolPermissionAdapter,
    advertised_tools,
    check_and_execute,
)
from myagent.tools.registry import TOOLS, register


@pytest.fixture(autouse=True)
def clear_tools():
    TOOLS.clear()
    yield
    TOOLS.clear()


def test_bound_node_alone_controls_advertised_and_executable_tools(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    register("file_read", {"type": "object", "properties": {}}, lambda path: {"path": path})
    register("file_write", {"type": "object", "properties": {}}, lambda path, content: {"path": path})
    tree = PermissionTree(tmp_path, [PermissionAtom.read("file_read", "src")])
    tree.attach_agent(tree.ROOT_ID, "reader")
    adapter = ToolPermissionAdapter(tmp_path)

    assert advertised_tools(tree, "reader") == {"file_read"}
    allowed = check_and_execute(
        ToolCall(id="1", name="file_read", arguments={"path": "src/app.py"}),
        tree,
        "reader",
        adapter,
        lambda _call: True,
    )
    denied = check_and_execute(
        ToolCall(
            id="2",
            name="file_write",
            arguments={"path": "src/app.py", "content": "x"},
        ),
        tree,
        "reader",
        adapter,
        lambda _call: True,
    )

    assert allowed == {"path": "src/app.py"}
    assert denied["error"] == "permission_denied"


def test_path_escape_is_denied_before_tool_execution(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    calls = []
    register(
        "file_read",
        {"type": "object", "properties": {}},
        lambda path: calls.append(path) or {"path": path},
    )
    tree = PermissionTree(tmp_path, [PermissionAtom.read("file_read", ".")])
    tree.attach_agent(tree.ROOT_ID, "reader")

    result = check_and_execute(
        ToolCall(id="1", name="file_read", arguments={"path": "../secret.txt"}),
        tree,
        "reader",
        ToolPermissionAdapter(tmp_path),
        lambda _call: True,
    )

    assert result["error"] == "permission_denied"
    assert calls == []


def test_high_risk_approval_is_separate_from_authority(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    register(
        "file_write",
        {"type": "object", "properties": {}},
        lambda path, content: {"path": path},
        risk="high",
    )
    tree = PermissionTree(tmp_path, [PermissionAtom.write("file_write", ".")])
    tree.attach_agent(tree.ROOT_ID, "writer")
    result = check_and_execute(
        ToolCall(
            id="1",
            name="file_write",
            arguments={"path": "out.txt", "content": "x"},
        ),
        tree,
        "writer",
        ToolPermissionAdapter(tmp_path),
        lambda _call: False,
    )
    assert result["error"] == "user_denied"
    records = [
        json.loads(line)["entry"]
        for line in (tmp_path / "home" / "audit.log").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["result"]["error"] == "user_denied"
