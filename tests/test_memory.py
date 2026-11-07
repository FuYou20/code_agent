"""Verify FTS5 decision storage, successful-only retrieval, and loop integration."""

import sqlite3

import pytest

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.context import stable
from myagent.llm.types import LLMResponse, ToolCall
from myagent.evolution.artifacts import EvolutionArtifactManager, TokenUsage
from myagent.memory.retrieve import (
    mark_recalled,
    memory_stats,
    record_feedback,
    retrieve,
    retrieve_hits,
)
from myagent.memory.store import init_db, insert_decision
from myagent.permissions import PermissionAtom, PermissionTree, ToolPermissionAdapter
from myagent.skills.manager import load_index
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


def test_production_retrieval_requires_token_verified_active_status(tmp_path) -> None:
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "learn",
        "experience",
        "Repair parser cache invalidation",
        ["src/parser.py"],
        "success",
    )
    assert retrieve(
        conn,
        "parser cache",
        artifact_statuses=("active",),
    ) == ""

    manager = EvolutionArtifactManager(conn, candidate_trials=3)
    artifact = manager.get_by_key("memory", str(decision_id), 1)
    for index in range(3):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"parser-transfer-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=TokenUsage(prompt_tokens=1000),
            treatment_usage=TokenUsage(prompt_tokens=700),
        )

    assert "Repair parser cache invalidation" in retrieve(
        conn,
        "parser cache",
        artifact_statuses=("active",),
    )
    conn.close()


def test_success_feedback_does_not_promote_candidate_without_ab(
    tmp_path,
) -> None:
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "learn",
        "experience",
        "Repair retry max_attempts accounting",
        ["request_retry.py"],
        "success",
        "retry-budget",
    )

    mark_recalled(conn, (decision_id,))
    after_recall = memory_stats(conn, decision_id)
    assert after_recall["status"] == "candidate"
    assert after_recall["success_count"] == 0
    assert after_recall["recall_count"] == 1

    for _ in range(3):
        record_feedback(conn, (decision_id,), success=True)
        stats = memory_stats(conn, decision_id)
        assert stats["status"] == "candidate"
        assert stats["lifecycle_status"] == "candidate"
    conn.close()


def test_retry_failure_discards_memory_and_suppresses_next_recall(tmp_path) -> None:
    """Regression: an incomplete max_attempts fix must not poison every variant."""
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "retry-learn",
        "experience",
        (
            "Changed range(max_attempts - 1) to range(max_attempts). "
            "For max_attempts=0 this still raises RuntimeError."
        ),
        ["request_retry.py"],
        "success",
        "retry-budget",
    )
    first = retrieve_hits(
        conn,
        "repair retry max_attempts in request_retry.py",
        k=1,
        scope_key="retry-budget",
    )
    assert [hit.decision_id for hit in first] == [decision_id]
    mark_recalled(conn, (decision_id,))

    # An evaluator-only boundary test proves the recalled guidance incomplete.
    record_feedback(conn, (decision_id,), success=False)
    discarded = memory_stats(conn, decision_id)
    assert discarded["status"] == "discarded"
    assert discarded["lifecycle_status"] == "discarded"
    assert discarded["failure_count"] == 1

    second = retrieve_hits(
        conn,
        "repair retry max_attempts in upload_retry.py",
        k=1,
        scope_key="retry-budget",
    )
    assert second == []

    conn.close()


def test_decisive_source_failure_discards_new_memory_immediately(tmp_path) -> None:
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "retry-learn",
        "experience",
        "Incomplete retry max_attempts repair",
        ["request_retry.py"],
        "success",
        "retry-budget",
    )

    record_feedback(conn, (decision_id,), success=False, decisive=True)

    stats = memory_stats(conn, decision_id)
    assert stats["status"] == "discarded"
    assert stats["failure_count"] == 1
    assert retrieve_hits(
        conn,
        "retry max_attempts repair",
        k=1,
        scope_key="retry-budget",
    ) == []
    conn.close()


def test_concrete_experience_does_not_cross_memory_scope(tmp_path) -> None:
    conn = init_db(tmp_path / "memory.db")
    insert_decision(
        conn,
        "range-learn",
        "experience",
        "Repair max_attempts range accounting",
        ["range_validation.py"],
        "success",
        "range-validation",
    )

    assert retrieve_hits(
        conn,
        "repair max_attempts range accounting",
        k=1,
        scope_key="retry-budget",
    ) == []
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


class MultiStepMemoryClient:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def chat(self, messages, tools) -> LLMResponse:
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="write-once",
                        name="file_write",
                        arguments={"path": "src/example.py", "content": "value = 2\n"},
                    )
                ],
            )
        return LLMResponse(content="done")


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
        evolution_artifact_statuses=("candidate", "active"),
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


def test_memory_sidecar_is_reused_once_inside_multi_step_turn(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "stable"
    conn = init_db(tmp_path / "memory.db")
    insert_decision(
        conn,
        "old-session",
        "experience",
        "Repair example.py using the previous verified change",
        ["src/example.py"],
        "success",
    )
    register(
        "file_write",
        {"type": "object", "properties": {}},
        lambda path, content: {"path": path, "bytes_written": len(content)},
        risk="high",
    )
    state = SessionState(
        session_id="reuse-once",
        agent_id="root-reuse-once",
        cwd=str(tmp_path),
        memory_conn=conn,
        evolution_artifact_statuses=("candidate", "active"),
    )
    _bind_write_permission(state, tmp_path)
    client = MultiStepMemoryClient()

    run_turn(state, "Repair example.py", client, lambda _call: True)

    assert len(client.calls) == 2
    for messages in client.calls:
        assert sum(
            "[决策 kind=experience" in message.content for message in messages
        ) == 1
    assert all("[决策 kind=experience" not in message.content for message in state.history)
    conn.close()


def test_activated_skill_suppresses_duplicate_memory(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "stable"
    conn = init_db(tmp_path / "memory.db")
    insert_decision(
        conn,
        "old-session",
        "experience",
        "Historical pytest advice that should not be duplicated",
        ["test_example.py"],
        "success",
    )
    skill_dir = tmp_path / "skills" / "pytest-repair"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: pytest-repair
description: Repair pytest failures.
triggers: [pytest]
permission_requirements: []
---
Use the activated pytest procedure.
""",
        encoding="utf-8",
    )
    tree = PermissionTree(tmp_path, [])
    tree.attach_agent(tree.ROOT_ID, "root-skill-memory")
    state = SessionState(
        session_id="skill-memory",
        agent_id="root-skill-memory",
        permission_tree=tree,
        permission_adapter=ToolPermissionAdapter(tmp_path),
        cwd=str(tmp_path),
        memory_conn=conn,
        evolution_artifact_statuses=("candidate", "active"),
        skill_index=load_index(tmp_path / "skills"),
        skill_activation_min_score=80,
    )
    capture = CaptureAnswerClient()

    run_turn(state, "Please repair this pytest failure", capture, lambda _call: True)

    sidecar = capture.messages[-1].content
    assert "Use the activated pytest procedure." in sidecar
    assert "Historical pytest advice" not in sidecar
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
        evolution_artifact_statuses=("candidate", "active"),
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
