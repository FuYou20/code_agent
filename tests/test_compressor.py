"""Verify thresholded LLM history compression and retained message ordering."""

from myagent.agent.state import SessionState
from myagent.context.compressor import compress_if_needed
from myagent.llm.types import LLMResponse, Message


class SummaryClient:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, tools) -> LLMResponse:
        self.calls.append((messages, tools))
        return LLMResponse(content="保留了关键决策与测试结论。")


def test_history_below_limit_is_unchanged() -> None:
    history = [Message(role="user", content="short")]
    state = SessionState(session_id="s", history=history.copy())
    compress_if_needed(state, max_tokens=100)
    assert state.history == history


def test_compression_keeps_first_system_and_last_ten() -> None:
    first = Message(role="system", content="system")
    middle = [Message(role="user", content=f"old-{index}") for index in range(4)]
    recent = [Message(role="assistant", content=f"recent-{index}") for index in range(10)]
    client = SummaryClient()
    state = SessionState(
        session_id="s",
        history=[first, *middle, *recent],
        compression_client=client,
    )

    compress_if_needed(state, max_tokens=1)

    assert state.history[0] is first
    assert state.history[1].role == "assistant"
    assert state.history[1].content == "[历史摘要]\n保留了关键决策与测试结论。"
    assert state.history[2:] == recent
    assert len(client.calls) == 1
    assert client.calls[0][1] == []
    assert "old-0" in client.calls[0][0][0].content
    assert "recent-0" not in client.calls[0][0][0].content


def test_compression_has_deterministic_fallback_without_client() -> None:
    history = [Message(role="user", content=f"message-{index}") for index in range(12)]
    state = SessionState(session_id="s", history=history)
    compress_if_needed(state, max_tokens=1)
    assert state.history[0].content.startswith("[历史摘要]\nmessage-0")
    assert state.history[1:] == history[-10:]
