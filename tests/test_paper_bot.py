from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from myagent.context import stable
from myagent.llm.types import LLMResponse
from myagent.paper_bot import reader
from myagent.paper_bot import tools as paper_tools
from myagent.paper_bot.reader import PaperSource, extract_focus_excerpt, resolve_pdf_url
from myagent.paper_bot.runtime import PaperAgentRuntime
from myagent.paper_bot.service import PaperBotService, format_feishu_digest, parse_command
from myagent.paper_bot.store import PaperBotStore, StoredPaper
from myagent.paper_digest import PaperRecommendation
from myagent.tools.registry import TOOLS, execute


@pytest.fixture(autouse=True)
def isolate_agent_globals():
    tools_before = dict(TOOLS)
    prefix_before = stable.STABLE_PREFIX
    yield
    TOOLS.clear()
    TOOLS.update(tools_before)
    stable.STABLE_PREFIX = prefix_before


def recommendation(title: str = "ConSurv") -> PaperRecommendation:
    return PaperRecommendation(
        topic="pathology",
        title=title,
        venue="AAAI 2026",
        arxiv_month="2025-11",
        problem="持续学习时会遗忘旧医院的病理知识",
        url="https://papernotes.org/example/",
        is_intersection=True,
        arxiv_id="2511.09853",
    )


def test_command_parser() -> None:
    assert parse_command("绑定").name == "bind"
    assert parse_command("今日推荐").name == "digest"
    assert parse_command("今天心情怎么样").name == "chat"
    command = parse_command("请仔细看一下 P1 的消融实验")
    assert command.name == "read"
    assert command.label == "P1"
    assert command.focus == "实验"
    command = parse_command("总结 p2 的局限")
    assert command.name == "read"
    assert command.label == "P2"
    assert command.focus == "局限"


def test_store_binding_digest_and_dedup(tmp_path) -> None:
    store = PaperBotStore(tmp_path / "paper-bot.sqlite3")
    assert store.bind("chat-1", "owner-1") == (True, "绑定成功")
    assert store.bind("chat-2", "owner-2")[0] is False
    assert store.is_bound("chat-1")
    assert store.enabled_chats() == ["chat-1"]
    assert store.reserve_message("message-1")
    assert not store.reserve_message("message-1")

    store.save_digest("chat-1", date(2026, 11, 6), [recommendation()])
    paper = store.latest_paper("chat-1", "p1")
    assert paper is not None
    assert paper.title == "ConSurv"
    assert paper.arxiv_id == "2511.09853"
    assert store.digest_for_day("chat-1", date(2026, 11, 6)) == [paper]

    store.set_active_label("chat-1", "p1")
    assert store.active_label("chat-1") == "P1"
    for index in range(8):
        store.append_chat_message("chat-1", "user", f"question-{index}", keep=4)
    assert store.recent_chat_messages("chat-1") == [
        ("user", "question-4"),
        ("user", "question-5"),
        ("user", "question-6"),
        ("user", "question-7"),
    ]
    store.close()


def test_digest_format_contains_interactive_commands() -> None:
    item = StoredPaper(
        label="P1",
        topic="pathology",
        title="ConSurv",
        venue="AAAI 2026",
        arxiv_month="2025-11",
        arxiv_id="2511.09853",
        problem="解决多医院任务流中的灾难性遗忘",
        url="https://papernotes.org/example/",
        is_intersection=True,
    )
    text = format_feishu_digest([item], date(2026, 11, 6))
    assert "P1｜病理图像 × 交叉方向" in text
    assert "精读 P1 实验" in text


def test_arxiv_resolution_needs_no_page_fetch() -> None:
    paper = StoredPaper(
        label="P1",
        topic="pathology",
        title="ConSurv",
        venue="AAAI 2026",
        arxiv_month="2025-11",
        arxiv_id="2511.09853",
        problem="",
        url="https://papernotes.org/example/",
        is_intersection=True,
    )
    assert resolve_pdf_url(paper) == "https://arxiv.org/pdf/2511.09853"


def test_focus_extraction_selects_experiment_pages(tmp_path, monkeypatch) -> None:
    pages = [
        "Introduction and related work",
        "3 Method framework architecture",
        "4 Experiments\nDatasets and baselines are described here.",
        "4.2 Results and ablation study",
        "5 Conclusion",
    ]
    monkeypatch.setattr(reader, "_page_texts", lambda _: pages)
    source = extract_focus_excerpt(tmp_path / "fake.pdf", "实验", max_pages=3)
    assert 3 in source.selected_pages
    assert "[PDF p.3]" in source.excerpt


def test_paper_tools_recommend_then_read_with_bound_context(tmp_path, monkeypatch) -> None:
    store = PaperBotStore(tmp_path / "paper-bot.sqlite3")
    monkeypatch.setattr(paper_tools, "recommend_daily", lambda _path: [recommendation()])
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF test")
    monkeypatch.setattr(
        paper_tools,
        "download_pdf",
        lambda _paper, _cache: (fake_pdf, "https://arxiv.org/pdf/2511.09853"),
    )
    monkeypatch.setattr(
        paper_tools,
        "extract_focus_excerpt",
        lambda _path, _focus: PaperSource(
            pdf_url="",
            pdf_path=fake_pdf,
            selected_pages=(3, 4),
            excerpt="[PDF p.3]\nDataset A outperforms baseline B.",
        ),
    )
    paper_tools.register_paper_tools()
    context = paper_tools.PaperToolContext(
        store=store,
        chat_id="chat-1",
        history_path=tmp_path / "history.json",
        cache_dir=tmp_path / "cache",
    )

    with paper_tools.bind_paper_tool_context(context):
        digest = execute("paper_recommend_today", {})
        details = execute("paper_read", {"label": "P1", "focus": "实验"})

    assert digest["papers"][0]["title"] == "ConSurv"
    assert details["selected_pages"] == [3, 4]
    assert "[PDF p.3]" in details["excerpt"]
    store.close()


def test_feishu_runtime_activates_skill_and_executes_paper_tool(
    tmp_path, monkeypatch
) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                assert [item["function"]["name"] for item in tools] == [
                    "paper_read",
                    "paper_recommend_today",
                ]
                assert any("每日论文推荐" in message.content for message in messages)
                from myagent.llm.types import ToolCall

                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="paper-1",
                            name="paper_recommend_today",
                            arguments={},
                        )
                    ],
                )
            assert any(
                message.role == "tool" and "ConSurv" in message.content
                for message in messages
            )
            return LLMResponse(content="今天推荐 ConSurv，可以继续精读 P1 实验。")

    monkeypatch.setattr(paper_tools, "recommend_daily", lambda _path: [recommendation()])
    store = PaperBotStore(tmp_path / "paper-bot.sqlite3")
    runtime = PaperAgentRuntime(
        store=store,
        history_path=tmp_path / "history.json",
        cache_dir=tmp_path / "cache",
        workspace=tmp_path,
    )

    answer = runtime.run("chat-1", "给我今日推荐", Client())

    assert "ConSurv" in answer
    assert store.latest_paper("chat-1", "P1") is not None
    store.close()


def test_normal_chat_uses_lulu_soul_and_keeps_short_history(tmp_path) -> None:
    class Channel:
        def __init__(self) -> None:
            self.sent = []

        def on(self, event, handler) -> None:
            self.handler = handler

        async def send(self, chat_id, payload, opts=None):
            self.sent.append((chat_id, payload, opts))
            return SimpleNamespace(success=True, error=None)

    class Client:
        def __init__(self) -> None:
            self.messages = []

        def chat(self, messages, tools):
            self.messages = messages
            return LLMResponse(content="当然，可以正常聊。")

    channel = Channel()
    store = PaperBotStore(tmp_path / "paper-bot.sqlite3")
    service = PaperBotService(
        channel=channel,
        store=store,
        history_path=tmp_path / "history.json",
        cache_dir=tmp_path / "cache",
    )
    client = Client()
    service._make_llm_client = lambda: client

    asyncio.run(service._chat("chat-1", "message-1", "今天想聊点代码"))

    assert "你叫 **Lulu**" in client.messages[0].content
    assert "今天想聊点代码" in client.messages[-1].content
    assert store.recent_chat_messages("chat-1") == [
        ("user", "今天想聊点代码"),
        ("assistant", "当然，可以正常聊。"),
    ]
    assert channel.sent[0][1]["markdown"] == "当然，可以正常聊。"
    store.close()
