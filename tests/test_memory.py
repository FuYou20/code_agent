"""Verify FTS5 decision storage, successful-only retrieval, and loop integration."""

import sqlite3

import pytest

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.context import stable
from myagent.llm.types import LLMResponse, ToolCall
from myagent.memory.retrieve import retrieve
from myagent.memory.store import init_db, insert_decision
from myagent.permissions import PermissionAtom, PermissionTree, ToolPermissionAdapter
from myagent.tools.registry import TOOLS, register


@pytest.fixture(autouse=True)
def clear_tools():
    TOOLS.clear()
    yield
    TOOLS.clear()


def test_retrieve_returns_only_successful_decisions(tmp_path) -> None:
    conn = init_db(tmp_path / "memory.db")
    insert_decision(
        conn,
        "s1",
        "diff",
        "Updated the parser implementation",
        ["src/parser.py"],
        "success",
    )
    insert_decision(
        conn,
        "s1",
        "diff",
        "Failed parser experiment",
        ["src/parser.py"],
        "failed",
    )

    result = retrieve(conn, "parser.py")

    assert "Updated the parser implementation" in result
    assert "Failed parser experiment" not in result
    assert "src/parser.py" in result
    conn.close()


def test_retrieve_handles_empty_or_missing_queries(tmp_path) -> None:
    conn = init_db(tmp_path / "memory.db")
    assert retrieve(conn, "") == ""
    assert retrieve(conn, "!!!") == ""
    assert retrieve(conn, "missing-file") == ""
    conn.close()


class WriteThenAnswerClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="file_write",
                        arguments={"path": "src/example.py", "content": "value = 1\n"},
                    )
                ],
            )
        return LLMResponse(content="Updated src/example.py successfully.")


class CaptureAnswerClient:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, tools) -> LLMResponse:
        self.messages = messages
        return LLMResponse(content="The previous decision was recalled.")


class StaticAnswerClient:
    def chat(self, messages, tools) -> LLMResponse:
        return LLMResponse(content="ack")


def _bind_write_permission(state: SessionState, workspace) -> None:
    tree = PermissionTree(workspace, [PermissionAtom.write("file_write", ".")])
    tree.attach_agent(tree.ROOT_ID, state.agent_id)
    state.permission_tree = tree
    state.permission_adapter = ToolPermissionAdapter(workspace)


def test_successful_write_is_recalled_in_sidecar_but_not_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "stable"
    conn = init_db(tmp_path / "memory.db")
    register(
        "file_write",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        },
        lambda path, content: {"path": path, "bytes_written": len(content)},
        risk="high",
    )
    state = SessionState(
        session_id="session-1",
        agent_id="root-1",
        cwd=str(tmp_path),
        memory_conn=conn,
    )
    _bind_write_permission(state, tmp_path)

    answer = run_turn(
        state,
        "Update the example module",
        WriteThenAnswerClient(),
        lambda _call: True,
    )
    assert answer == "Updated src/example.py successfully."
    recalled = retrieve(conn, "example.py")
    assert "src/example.py" in recalled
    assert "Updated src/example.py successfully." in recalled

    capture = CaptureAnswerClient()
    run_turn(state, "What changed in example.py?", capture, lambda _call: True)
    current_sidecar = capture.messages[-1].content
    assert "[决策 kind=experience files=src/example.py]" in current_sidecar
    assert all("[决策 kind=experience" not in message.content for message in state.history)
    conn.close()


def test_failed_write_is_not_recorded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    conn = init_db(tmp_path / "memory.db")

    def fail_write(path: str, content: str) -> dict:
        raise OSError(f"cannot write {path}: {len(content)}")

    register(
        "file_write",
        {"type": "object", "properties": {}},
        fail_write,
        risk="high",
    )
    state = SessionState(
        session_id="session-2",
        agent_id="root-2",
        cwd=str(tmp_path),
        memory_conn=conn,
    )
    _bind_write_permission(state, tmp_path)
    run_turn(
        state,
        "Update broken.py",
        WriteThenAnswerClient(),
        lambda _call: True,
    )
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert count == 0
    conn.close()


def test_init_db_is_idempotent(tmp_path) -> None:
    path = tmp_path / "nested" / "memory.db"
    first = init_db(path)
    first.close()
    second = init_db(path)
    assert isinstance(second, sqlite3.Connection)
    second.close()


def test_twenty_turn_session_still_recalls_earlier_file_change(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "stable"
    conn = init_db(tmp_path / "memory.db")
    register(
        "file_write",
        {"type": "object", "properties": {}},
        lambda path, content: {"path": path, "bytes_written": len(content)},
        risk="high",
    )
    state = SessionState(
        session_id="long-session",
        agent_id="root-long",
        cwd=str(tmp_path),
        memory_conn=conn,
    )
    _bind_write_permission(state, tmp_path)
    run_turn(
        state,
        "Update the example module",
        WriteThenAnswerClient(),
        lambda _call: True,
    )
    for turn in range(2, 20):
        run_turn(
            state,
            f"unrelated conversation turn {turn}",
            StaticAnswerClient(),
            lambda _call: True,
        )

    capture = CaptureAnswerClient()
    run_turn(
        state,
        "On turn 20, what changed in example.py?",
        capture,
        lambda _call: True,
    )
    assert "[决策 kind=experience files=src/example.py]" in capture.messages[-1].content
    assert all("[决策 kind=experience" not in message.content for message in state.history)
    conn.close()
