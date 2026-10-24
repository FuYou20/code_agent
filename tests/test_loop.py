"""Exercise a complete model-tool-model P0 turn without a network API."""

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.audit.log import verify
from myagent.config import DEFAULT_CONFIG
from myagent.context.stable import initialize_stable_prefix
from myagent.llm.types import LLMResponse, ToolCall
from myagent.tools.builtin import default_scope, register_all
from myagent.tools.registry import TOOLS, schemas


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"})],
            )
        tool_message = next(message for message in messages if message.role == "tool")
        assert "hello" in tool_message.content
        return LLMResponse(content="README 内容是 hello")


def test_read_file_turn_and_audit(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    TOOLS.clear()
    register_all(DEFAULT_CONFIG, tmp_path)
    initialize_stable_prefix(schemas())
    scope = default_scope(DEFAULT_CONFIG, tmp_path)
    state = SessionState(session_id="session", cwd=str(tmp_path), scope=scope)

    answer = run_turn(state, "读一下 README.md", FakeClient(), scope, lambda _call: False)

    assert answer == "README 内容是 hello"
    assert verify()
    assert (tmp_path / "home" / "audit.log").read_text(encoding="utf-8").count("\n") == 1
    TOOLS.clear()

