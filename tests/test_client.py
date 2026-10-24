"""Verify OpenAI-compatible base URLs and provider cache-usage fields."""

from types import SimpleNamespace

from myagent.llm import client as client_module
from myagent.llm.client import OpenAIClient
from myagent.llm.types import Message


def test_custom_base_url_is_forwarded(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    OpenAIClient("deepseek-flash", "secret", base_url="https://api.deepseek.com")
    assert captured == {"api_key": "secret", "base_url": "https://api.deepseek.com"}


def test_deepseek_cache_hit_tokens_are_normalized(monkeypatch) -> None:
    class Completions:
        def create(self, **request):
            assert "tools" not in request
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="ok",
                            tool_calls=None,
                            reasoning_content="internal reasoning",
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=1,
                    total_tokens=101,
                    prompt_tokens_details=None,
                    prompt_cache_hit_tokens=80,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    client = OpenAIClient("deepseek-flash", "secret")
    response = client.chat([Message(role="user", content="hello")], tools=[])
    assert response.usage["cached_tokens"] == 80
    assert response.reasoning_content == "internal reasoning"


def test_reasoning_content_is_returned_in_assistant_message() -> None:
    message = Message(
        role="assistant",
        content="answer",
        reasoning_content="preserved reasoning",
    )
    converted = OpenAIClient._message_to_openai(message)
    assert converted["reasoning_content"] == "preserved reasoning"
