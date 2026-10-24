"""Verify three-layer prompt placement and stable-prefix determinism."""

from myagent.agent.state import SessionState
from myagent.context import stable
from myagent.context.assembler import build_messages
from myagent.llm.types import Message


def test_stable_prefix_is_deterministic() -> None:
    left = stable.build_stable_prefix([{"b": 2, "a": 1}])
    right = stable.build_stable_prefix([{"a": 1, "b": 2}])
    assert left == right


def test_process_prefix_initializes_only_once(monkeypatch) -> None:
    monkeypatch.setattr(stable, "STABLE_PREFIX", "")
    first = stable.initialize_stable_prefix([{"name": "first"}])
    second = stable.initialize_stable_prefix([{"name": "changed"}])
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_sidecar_is_not_written_to_history(tmp_path) -> None:
    stable.STABLE_PREFIX = "stable"
    prior = Message(role="assistant", content="prior")
    state = SessionState(
        session_id="s1",
        history=[prior],
        cwd=str(tmp_path),
        repo_snapshot="README.md",
    )
    messages = build_messages(state, "hello", retrieved="remembered")
    assert messages[0].role == "system"
    assert messages[1] is prior
    assert messages[2].role == "user"
    assert "remembered" in messages[2].content
    assert state.history == [prior]
    assert "remembered" not in state.history[0].content


def test_skill_l0_and_l1_are_ephemeral_and_follow_history(tmp_path) -> None:
    stable.STABLE_PREFIX = "stable"
    prior = Message(role="assistant", content="prior")
    state = SessionState(session_id="s1", history=[prior], cwd=str(tmp_path))
    messages = build_messages(
        state,
        "fix it",
        skill_items=[{"name": "demo", "description": "Demo workflow"}],
        skill_body="Secret L1 procedure",
    )
    assert messages == [messages[0], prior, messages[-1]]
    assert "demo: Demo workflow" in messages[-1].content
    assert "Secret L1 procedure" in messages[-1].content
    assert all("Secret L1 procedure" not in message.content for message in state.history)


def test_empty_followup_uses_history_without_new_sidecar(tmp_path) -> None:
    stable.STABLE_PREFIX = "stable"
    prior = Message(role="user", content="raw input")
    state = SessionState(session_id="s1", history=[prior], cwd=str(tmp_path))
    messages = build_messages(state, "")
    assert messages == [messages[0], prior]

