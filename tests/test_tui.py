from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from myagent.llm.types import LLMResponse, Message
from myagent.permissions import PermissionAtom
from myagent.tui import MyAgentTui, TuiTelemetryClient, _runtime_permission_atoms


class _Backend:
    cwd = Path.cwd()

    def set_update_callback(self, _callback) -> None:
        pass


class _Client:
    def chat(self, _messages, _tools) -> LLMResponse:
        return LLMResponse(
            content="done",
            usage={
                "prompt_tokens": 100,
                "cached_tokens": 60,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        )


def test_tui_builds_without_a_real_terminal() -> None:
    status = {
        "provider": "deepseek",
        "model": "deepseek-flash",
        "strategy": "scene_epoch",
        "phase": "warming",
        "scene": "code",
        "epoch": 1,
        "frozen": [],
        "last_skill": "-",
        "last_memory": [],
        "steps": 0,
        "router_tokens": 0,
        "system_changes": 0,
    }
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with create_app_session(input=DummyInput(), output=DummyOutput()):
            tui = MyAgentTui(_Backend(), executor, status)  # type: ignore[arg-type]
            assert tui.status_snapshot["phase"] == "warming"
            assert "Enter send" in str(tui._statusbar_text())
            assert tui.app.full_screen is True
    finally:
        executor.shutdown()


def test_tui_telemetry_counts_usage_and_extracts_sidecar() -> None:
    client = TuiTelemetryClient(_Client())  # type: ignore[arg-type]
    client.begin_turn()
    response = client.chat(
        [
            Message(role="system", content="你是 myagent"),
            Message(
                role="user",
                content=(
                    "## 当前轮 Sidecar\n"
                    "### 检索结果\n一条具体经验\n"
                    "## 可用技能（L0）\n- run-tests: run pytest\n"
                    "## 当前主技能（L1）\nfollow the workflow\n"
                    "## 用户输入\n运行测试"
                ),
            ),
        ],
        [],
    )
    snapshot = client.snapshot()
    assert response.content == "done"
    assert snapshot["token_turn"]["uncached_tokens"] == 40
    assert snapshot["token_session"]["completion_tokens"] == 10
    assert snapshot["sidecar_has_memory"] is True
    assert snapshot["sidecar_has_l1"] is True
    assert snapshot["sidecar_l0"] == ["run-tests"]


def test_tui_telemetry_stops_after_public_token_limit(monkeypatch) -> None:
    monkeypatch.setenv("MYAGENT_DEMO_TOKEN_LIMIT", "100")
    client = TuiTelemetryClient(_Client())  # type: ignore[arg-type]
    client.chat([Message(role="user", content="first")], [])
    with pytest.raises(RuntimeError, match="Token"):
        client.chat([Message(role="user", content="second")], [])


def test_public_tui_only_exposes_read_only_permission_atoms(monkeypatch) -> None:
    monkeypatch.setenv("MYAGENT_PUBLIC_DEMO", "readonly")
    narrowed = _runtime_permission_atoms(
        (
            PermissionAtom.read("file_read", "."),
            PermissionAtom.write("file_write", "."),
            PermissionAtom.read("git_read", "."),
            PermissionAtom.read_write("test_run", read=(".",), write=(".",)),
        )
    )
    assert {atom.tool for atom in narrowed} == {"file_read", "git_read"}
