"""Full-screen prompt-toolkit interface for MyAgent.

The model loop remains synchronous.  The TUI owns one worker thread so model
calls do not freeze terminal rendering while SQLite state stays on the same
thread that created it.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from .agent.loop import run_turn
from .agent.state import SessionState
from .cli import _load_skill_index, _read_agents_once, _repo_snapshot
from .config import load_config
from .context.stable import initialize_stable_prefix
from .evolution.artifacts import EvolutionArtifactManager, TokenWeights
from .home import get_home
from .llm.client import OpenAIClient
from .llm.types import ToolCall
from .memory.store import init_db
from .permissions import PermissionTree, ToolPermissionAdapter
from .skills.distiller import distill
from .tools.builtin import default_permission_atoms, register_all
from .tools.registry import schemas


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


_SIDECAR_HEADING_RE = re.compile(r"^(?:##|###) (.+)$", re.MULTILINE)


def _sidecar_sections(sidecar: str) -> dict[str, str]:
    """Split the rendered Sidecar into display-friendly named sections."""
    if not sidecar:
        return {}
    matches = list(_SIDECAR_HEADING_RE.finditer(sidecar))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sidecar)
        sections[match.group(1).strip()] = sidecar[start:end].strip()
    return sections


def _runtime_permission_atoms(atoms):
    """Narrow root capabilities for the anonymous public demo."""
    if os.environ.get("MYAGENT_PUBLIC_DEMO", "").lower() != "readonly":
        return atoms
    public_tools = {"file_read", "git_read"}
    return tuple(atom for atom in atoms if atom.tool in public_tools)


class TuiTelemetryClient:
    """Collect provider-reported usage and the actual main-Agent Sidecar."""

    def __init__(self, inner: OpenAIClient) -> None:
        self.inner = inner
        self.on_update: Callable[[dict[str, object]], None] | None = None
        self.session = self._empty_usage()
        self.turn = self._empty_usage()
        self.last = self._empty_usage()
        self.last_call_kind = "-"
        self.last_sidecar = ""

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "calls": 0,
            "main_calls": 0,
            "aux_calls": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "uncached_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _is_main_agent_call(messages) -> bool:
        return bool(
            messages
            and messages[0].role == "system"
            and "你是 myagent" in messages[0].content
        )

    @staticmethod
    def _find_sidecar(messages) -> str:
        marker = "## 当前轮 Sidecar"
        for message in reversed(messages):
            if message.role == "user" and marker in message.content:
                return message.content[message.content.index(marker) :]
        return ""

    def begin_turn(self) -> None:
        self.turn = self._empty_usage()
        self.last = self._empty_usage()
        self.last_call_kind = "-"
        self.last_sidecar = ""
        self._notify()

    def _notify(self) -> None:
        if self.on_update is not None:
            self.on_update(self.snapshot())

    def chat(self, messages, tools):
        token_limit = int(os.environ.get("MYAGENT_DEMO_TOKEN_LIMIT", "0") or 0)
        if token_limit > 0 and self.session["total_tokens"] >= token_limit:
            raise RuntimeError(
                f"在线演示已达到{token_limit} Token的会话上限，请刷新页面开启新会话。"
            )
        is_main = self._is_main_agent_call(messages)
        if is_main:
            rendered = self._find_sidecar(messages)
            if rendered:
                self.last_sidecar = rendered
                self._notify()
        response = self.inner.chat(messages, tools)
        prompt = int(response.usage.get("prompt_tokens", 0))
        cached = int(response.usage.get("cached_tokens", 0))
        completion = int(response.usage.get("completion_tokens", 0))
        values = {
            "calls": 1,
            "main_calls": int(is_main),
            "aux_calls": int(not is_main),
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "uncached_tokens": max(0, prompt - cached),
            "completion_tokens": completion,
            "total_tokens": int(
                response.usage.get("total_tokens", prompt + completion)
            ),
        }
        for bucket in (self.session, self.turn):
            for key, value in values.items():
                bucket[key] += value
        self.last = dict(values)
        self.last_call_kind = "main" if is_main else "router/aux"
        self._notify()
        return response

    def snapshot(self) -> dict[str, object]:
        sections = _sidecar_sections(self.last_sidecar)
        l0_lines = [
            line[2:].split(":", 1)[0].strip()
            for line in sections.get("可用技能（L0）", "").splitlines()
            if line.startswith("- ")
        ]
        memory = sections.get("检索结果", "")
        user = sections.get("用户输入", "")
        return {
            "token_session": dict(self.session),
            "token_turn": dict(self.turn),
            "token_last": dict(self.last),
            "last_call_kind": self.last_call_kind,
            "sidecar": self.last_sidecar,
            "sidecar_chars": len(self.last_sidecar),
            "sidecar_sections": list(sections),
            "sidecar_l0": l0_lines,
            "sidecar_has_l1": bool(sections.get("当前主技能（L1）")),
            "sidecar_has_memory": bool(memory),
            "sidecar_memory_preview": re.sub(r"\s+", " ", memory)[:120],
            "sidecar_user_chars": len(user),
        }


class TuiBackend:
    """Own the stateful agent runtime used by the terminal UI."""

    def __init__(
        self,
        *,
        config: dict,
        cwd: Path,
        state: SessionState,
        client: TuiTelemetryClient,
        artifact_manager: EvolutionArtifactManager,
        last_gc_at: float,
    ) -> None:
        self.config = config
        self.cwd = cwd
        self.state = state
        self.client = client
        self.artifact_manager = artifact_manager
        self.last_gc_at = last_gc_at
        self._update_callback: Callable[[dict[str, object]], None] | None = None
        self.client.on_update = self._telemetry_updated

    @classmethod
    def create(cls) -> "TuiBackend":
        config = load_config()
        provider = str(config["llm"].get("provider", ""))
        if provider not in {"openai", "deepseek"}:
            raise ValueError("llm.provider 必须是 openai 或 deepseek")
        key_name = str(config["llm"]["api_key_env"])
        api_key = os.environ.get(key_name)
        if not api_key:
            raise RuntimeError(f"缺少环境变量：{key_name}")

        skills = config["skills"]
        strategy = str(skills["strategy"])
        allowed_strategies = {
            "full_sidecar",
            "full_system",
            "top_k",
            "warm_stable",
            "scene_epoch",
        }
        if strategy not in allowed_strategies:
            raise ValueError("未知的 skills.strategy：" + strategy)
        positive_values = {
            "top_k": int(skills["top_k"]),
            "candidate_k": int(skills["candidate_k"]),
            "max_frozen": int(skills["max_frozen"]),
            "warm_turns": int(skills["warm_turns"]),
            "idle_seconds": float(skills["idle_seconds"]),
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError("skills配置必须大于0：" + ", ".join(invalid))
        activation_min_score = float(skills["activation_min_score"])
        if activation_min_score < 0:
            raise ValueError("skills.activation_min_score不能小于0")

        cwd = Path.cwd().resolve()
        register_all(config, cwd)
        initialize_stable_prefix(schemas())
        agent_id = f"root-{uuid.uuid4().hex}"
        root_atoms = _runtime_permission_atoms(default_permission_atoms())
        permission_tree = PermissionTree(cwd, root_atoms)
        permission_tree.attach_agent(permission_tree.ROOT_ID, agent_id)
        memory_conn = init_db(get_home() / "memory.db")
        evolution = config["evolution"]
        artifact_manager = EvolutionArtifactManager(
            memory_conn,
            token_weights=TokenWeights(
                uncached_input=float(evolution["uncached_input_weight"]),
                cached_input=float(evolution["cached_input_weight"]),
                output=float(evolution["output_weight"]),
            ),
            min_canary_trials=int(evolution["min_canary_trials"]),
            negative_streak=int(evolution["negative_streak"]),
        )
        if bool(evolution["gc_on_startup"]):
            artifact_manager.garbage_collect(
                candidate_days=int(evolution["candidate_retention_days"]),
                inactive_days=int(evolution["inactive_days"]),
                quarantine_days=int(evolution["quarantine_retention_days"]),
                retired_days=int(evolution["retired_retention_days"]),
                purge_content=True,
                skill_store=get_home() / "skills_store",
            )
        gc_interval_seconds = float(evolution["gc_interval_seconds"])
        if gc_interval_seconds <= 0:
            raise ValueError("evolution.gc_interval_seconds必须大于0")

        state = SessionState(
            session_id=uuid.uuid4().hex,
            agent_id=agent_id,
            permission_tree=permission_tree,
            permission_adapter=ToolPermissionAdapter(cwd),
            cwd=str(cwd),
            repo_snapshot=_repo_snapshot(cwd),
            agents_content=_read_agents_once(cwd),
            skill_index=_load_skill_index(artifact_manager),
            skill_strategy=strategy,
            skill_top_k=int(skills["top_k"]),
            skill_candidate_k=int(skills["candidate_k"]),
            skill_activation_min_score=activation_min_score,
            skill_max_frozen=int(skills["max_frozen"]),
            skill_warm_turns=int(skills["warm_turns"]),
            skill_idle_seconds=float(skills["idle_seconds"]),
            max_steps=int(config["agent"]["max_steps"]),
            memory_conn=memory_conn,
            artifact_manager=artifact_manager,
            evolution_artifact_statuses=("active",),
        )
        base_url = config["llm"].get("base_url")
        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com"
        client = TuiTelemetryClient(
            OpenAIClient(
                model=str(config["llm"]["model"]),
                api_key=api_key,
                base_url=base_url,
            )
        )
        return cls(
            config=config,
            cwd=cwd,
            state=state,
            client=client,
            artifact_manager=artifact_manager,
            last_gc_at=time.monotonic(),
        )

    def _garbage_collect_if_due(self) -> None:
        evolution = self.config["evolution"]
        interval = float(evolution["gc_interval_seconds"])
        if time.monotonic() - self.last_gc_at < interval:
            return
        self.artifact_manager.garbage_collect(
            candidate_days=int(evolution["candidate_retention_days"]),
            inactive_days=int(evolution["inactive_days"]),
            quarantine_days=int(evolution["quarantine_retention_days"]),
            retired_days=int(evolution["retired_retention_days"]),
            purge_content=True,
            skill_store=get_home() / "skills_store",
        )
        self.state.skill_index = _load_skill_index(self.artifact_manager)
        self.last_gc_at = time.monotonic()

    def set_update_callback(
        self, callback: Callable[[dict[str, object]], None] | None
    ) -> None:
        self._update_callback = callback

    def _telemetry_updated(self, _telemetry: dict[str, object]) -> None:
        if self._update_callback is not None:
            self._update_callback(self.status())

    def handle(self, text: str, approve: Callable[[ToolCall], bool]) -> str:
        """Handle one chat turn or slash command and return display text."""
        self._garbage_collect_if_due()
        command = text.strip()
        if command == "/help":
            return (
                "/skill-status  查看Skill工作集\n"
                "/skill-reset   重置当前Skill Epoch\n"
                "/scene <name>  切换场景\n"
                "/learn         蒸馏最近成功轨迹\n"
                "/evolution-status  查看进化资产\n"
                "/evolution-gc      立即执行生命周期清理\n"
                "/telemetry     查看Token与缓存明细\n"
                "/sidecar       查看最近一次实际Sidecar\n"
                "/clear         清空TUI显示（不清会话）\n"
                "/exit          退出"
            )
        if command == "/learn":
            state = self.state
            if not state.last_evolution_eligible:
                reasons = ", ".join(state.last_evolution_reasons) or "尚无已验证变更"
                return "本轮轨迹未通过进化门禁：" + reasons
            task_context = next(
                (
                    message.content
                    for message in state.last_evolution_history
                    if message.role == "user" and message.content.strip()
                ),
                "",
            )
            skill = distill(
                state.last_evolution_history,
                self.client,
                task_context=task_context,
                artifact_manager=self.artifact_manager,
                scope_key=state.memory_scope,
            )
            state.last_evolution_eligible = False
            state.last_evolution_reasons = ("trajectory_already_distilled",)
            state.last_evolution_history = []
            state.skill_index = _load_skill_index(self.artifact_manager)
            artifact = self.artifact_manager.get_by_key(
                "skill", skill.frontmatter.name, skill.version
            )
            return (
                f"已蒸馏候选技能：{skill.frontmatter.name}\n"
                f"artifact={artifact.artifact_id}, status=candidate"
            )
        if command == "/evolution-status":
            return _json(
                [
                    {
                        "id": item.artifact_id,
                        "type": item.artifact_type,
                        "key": item.subject_key,
                        "version": item.version,
                        "status": item.status,
                        "economics": self.artifact_manager.economics(item.artifact_id),
                    }
                    for item in self.artifact_manager.list_artifacts()
                ]
            )
        if command == "/evolution-gc":
            evolution = self.config["evolution"]
            result = self.artifact_manager.garbage_collect(
                candidate_days=int(evolution["candidate_retention_days"]),
                inactive_days=int(evolution["inactive_days"]),
                quarantine_days=int(evolution["quarantine_retention_days"]),
                retired_days=int(evolution["retired_retention_days"]),
                purge_content=True,
                skill_store=get_home() / "skills_store",
            )
            self.state.skill_index = _load_skill_index(self.artifact_manager)
            self.last_gc_at = time.monotonic()
            return _json(result)
        if command == "/telemetry":
            return _json(self.client.snapshot())
        if command == "/sidecar":
            return self.client.last_sidecar or "当前会话尚未生成Sidecar。"
        if command == "/skill-status":
            load_state = self.state.skill_load_state
            return _json(
                {
                    "strategy": self.state.skill_strategy,
                    "phase": load_state.phase,
                    "epoch_id": load_state.epoch_id,
                    "scene": load_state.scene,
                    "frozen_skills": list(load_state.frozen_names),
                    "confirmed_fallbacks": list(load_state.confirmed_fallback_names),
                    "router_prompt_tokens": self.state.skill_router_prompt_tokens,
                    "system_prompt_changes": self.state.skill_cache_epoch_changes,
                }
            )
        if command == "/skill-reset":
            was_frozen = self.state.skill_load_state.phase == "frozen"
            event = self.state.skill_load_state.reset(reason="user_reset")
            if was_frozen:
                self.state.skill_cache_epoch_changes += 1
            return "Skill Epoch已重置：" + event
        if command.startswith("/scene "):
            scene = command.split(maxsplit=1)[1].strip()
            allowed = {
                str(item.get("scene", "general")) for item in self.state.skill_index
            }
            if scene not in allowed:
                return "未知场景。可选：" + ", ".join(sorted(allowed))
            was_frozen = self.state.skill_load_state.phase == "frozen"
            self.state.skill_load_state.reset(reason="user_scene", scene=scene)
            if was_frozen:
                self.state.skill_cache_epoch_changes += 1
            return f"已切换场景并开始新Epoch：{scene}"
        self.client.begin_turn()
        return run_turn(self.state, text, self.client, approve)

    def status(self) -> dict[str, object]:
        load_state = self.state.skill_load_state
        warming_scores = sorted(
            load_state.candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[: self.state.skill_max_frozen]
        return {
            "provider": self.config["llm"].get("provider", ""),
            "model": self.config["llm"].get("model", ""),
            "cwd": str(self.cwd),
            "strategy": self.state.skill_strategy,
            "phase": load_state.phase,
            "scene": load_state.scene or "-",
            "epoch": load_state.epoch_id,
            "epoch_turn": load_state.epoch_turn,
            "warm_turns": self.state.skill_warm_turns,
            "warming_scores": warming_scores,
            "freeze_turn": load_state.freeze_turn,
            "frozen": list(load_state.frozen_names),
            "fallbacks": list(load_state.confirmed_fallback_names),
            "last_skill": self.state.last_activated_skill_name or "-",
            "last_memory": list(self.state.last_recalled_memory_ids),
            "router_tokens": self.state.skill_router_prompt_tokens,
            "system_changes": self.state.skill_cache_epoch_changes,
            "steps": self.state.step_count,
            **self.client.snapshot(),
        }

    def close(self) -> None:
        if self.state.memory_conn is not None:
            self.state.memory_conn.close()


@dataclass
class ApprovalRequest:
    call: ToolCall
    event: threading.Event
    decision: bool = False


class MyAgentTui:
    """Render the full-screen chat application."""

    def __init__(
        self,
        backend: TuiBackend,
        executor: ThreadPoolExecutor,
        initial_status: dict[str, object],
    ) -> None:
        self.backend = backend
        self.executor = executor
        self.status_snapshot = initial_status
        self.busy = False
        self.pending_approval: ApprovalRequest | None = None
        self.footer_message = ""
        self.backend.set_update_callback(self._on_status_update)
        self._chat_text = (
            "MyAgent TUI\n"
            "输入任务开始对话，输入 /help 查看命令。\n\n"
        )
        self.chat = TextArea(
            text=self._chat_text,
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            focusable=True,
        )
        self.input = TextArea(
            height=Dimension(min=3, preferred=4, max=8),
            multiline=True,
            wrap_lines=True,
            prompt="❯ ",
        )
        self.sidebar = FormattedTextControl(self._sidebar_text)
        self.statusbar = FormattedTextControl(self._statusbar_text)
        self.header = FormattedTextControl(self._header_text)
        self.bindings = self._build_bindings()
        body = VSplit(
            [
                Frame(self.chat, title="Conversation"),
                Frame(
                    Window(
                        content=self.sidebar,
                        width=Dimension(min=26, preferred=34),
                        wrap_lines=True,
                        right_margins=[ScrollbarMargin(display_arrows=True)],
                    ),
                    title="Runtime",
                ),
            ],
            padding=1,
        )
        root = HSplit(
            [
                Window(self.header, height=1, style="class:header"),
                body,
                Frame(self.input, title="Message"),
                Window(self.statusbar, height=1, style="class:status"),
            ]
        )
        self.app: Application = Application(
            layout=Layout(root, focused_element=self.input),
            key_bindings=self.bindings,
            full_screen=True,
            mouse_support=True,
            style=Style.from_dict(
                {
                    "header": "bg:#1f2937 #f9fafb bold",
                    "status": "bg:#111827 #d1d5db",
                    "user": "#60a5fa bold",
                    "assistant": "#34d399 bold",
                    "error": "#f87171 bold",
                    "approval": "bg:#78350f #fef3c7 bold",
                    "frame.label": "#93c5fd bold",
                }
            ),
        )

    def _header_text(self) -> FormattedText:
        model = self.status_snapshot.get("model", "")
        provider = self.status_snapshot.get("provider", "")
        return FormattedText(
            [("class:header", f" MyAgent  {provider}/{model}  |  {self.backend.cwd} ")]
        )

    def _sidebar_text(self) -> FormattedText:
        s = self.status_snapshot
        frozen = list(s.get("frozen", []))
        turn_usage = dict(s.get("token_turn", {}))
        session_usage = dict(s.get("token_session", {}))
        lines: list[tuple[str, str]] = [
            ("class:assistant", "Skill\n"),
            ("", f" strategy  {s.get('strategy', '-')}\n"),
            ("", f" phase     {s.get('phase', '-')}\n"),
            ("", f" scene     {s.get('scene', '-')}\n"),
            (
                "",
                f" epoch     {s.get('epoch', 0)} "
                f"({s.get('epoch_turn', 0)}/{s.get('warm_turns', 0)})\n",
            ),
            ("", f" frozen    {len(frozen)}\n"),
        ]
        for name in frozen[:8]:
            lines.append(("", f"   · {name}\n"))
        warming_scores = list(s.get("warming_scores", []))
        if not frozen and warming_scores:
            lines.append(("", " candidates\n"))
            for name, score in warming_scores[:8]:
                lines.append(("", f"   · {name} ({float(score):.0f})\n"))
        fallbacks = list(s.get("fallbacks", []))
        if fallbacks:
            lines.append(("", f" fallback   {', '.join(map(str, fallbacks[:3]))}\n"))
        lines.extend(
            [
                ("class:assistant", "\nCurrent turn\n"),
                ("", f" skill     {s.get('last_skill', '-')}\n"),
                ("", f" memory    {s.get('last_memory', [])}\n"),
                ("", f" steps     {s.get('steps', 0)}\n"),
                ("class:assistant", "\nToken · current turn\n"),
                ("", f" calls     {turn_usage.get('calls', 0)}\n"),
                ("", f" input     {turn_usage.get('prompt_tokens', 0)}\n"),
                ("", f" cached    {turn_usage.get('cached_tokens', 0)}\n"),
                ("", f" uncached  {turn_usage.get('uncached_tokens', 0)}\n"),
                ("", f" output    {turn_usage.get('completion_tokens', 0)}\n"),
                ("class:assistant", "\nToken · session\n"),
                ("", f" input     {session_usage.get('prompt_tokens', 0)}\n"),
                ("", f" cached    {session_usage.get('cached_tokens', 0)}\n"),
                ("", f" uncached  {session_usage.get('uncached_tokens', 0)}\n"),
                ("", f" output    {session_usage.get('completion_tokens', 0)}\n"),
                ("class:assistant", "\nSidecar\n"),
                ("", f" chars     {s.get('sidecar_chars', 0)}\n"),
                ("", f" memory    {'yes' if s.get('sidecar_has_memory') else 'no'}\n"),
                ("", f" L1 Skill  {'yes' if s.get('sidecar_has_l1') else 'no'}\n"),
                ("", f" L0 Skill  {len(list(s.get('sidecar_l0', [])))}\n"),
            ]
        )
        for name in list(s.get("sidecar_l0", []))[:4]:
            lines.append(("", f"   · {name}\n"))
        memory_preview = str(s.get("sidecar_memory_preview", ""))
        if memory_preview:
            lines.append(("", f" memo: {memory_preview}\n"))
        lines.extend(
            [
                ("class:assistant", "\nCache signals\n"),
                ("", f" router    {s.get('router_tokens', 0)} tok\n"),
                ("", f" sys Δ     {s.get('system_changes', 0)}\n"),
            ]
        )
        if self.pending_approval is not None:
            call = self.pending_approval.call
            lines.extend(
                [
                    ("class:approval", "\n APPROVAL REQUIRED \n"),
                    ("class:approval", f" {call.name}\n"),
                    ("", _json(call.arguments) + "\n"),
                    ("class:approval", " Press Y / N "),
                ]
            )
        return FormattedText(lines)

    def _on_status_update(self, status: dict[str, object]) -> None:
        """Receive provider-call updates from the single runtime worker."""
        self.status_snapshot = status
        if hasattr(self, "app"):
            self.app.invalidate()

    def _statusbar_text(self) -> FormattedText:
        if self.pending_approval is not None:
            return FormattedText(
                [("class:approval", " Permission approval: Y allow · N deny ")]
            )
        if self.footer_message:
            return FormattedText([("class:error", f" {self.footer_message} ")])
        state = "RUNNING" if self.busy else "READY"
        return FormattedText(
            [
                (
                    "class:status",
                    f" {state}  Enter send · Esc+Enter newline · Tab focus · "
                    "Ctrl+L clear · Ctrl+Q quit ",
                )
            ]
        )

    def _build_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        approval_pending = Condition(lambda: self.pending_approval is not None)

        @bindings.add("y", filter=approval_pending, is_global=True)
        def _approve(_event) -> None:
            self._finish_approval(True)

        @bindings.add("n", filter=approval_pending, is_global=True)
        def _deny(_event) -> None:
            self._finish_approval(False)

        @bindings.add("enter")
        def _send(event) -> None:
            if not event.app.layout.has_focus(self.input):
                return
            if self.pending_approval is not None:
                return
            if self.busy:
                self._set_footer_message("当前任务仍在执行，请等待完成。")
                return
            text = self.input.text.strip()
            if not text:
                return
            self.input.buffer.set_document(Document("", 0), bypass_readonly=True)
            if text in {"/exit", "/quit"}:
                event.app.exit()
                return
            if text == "/clear":
                self._chat_text = ""
                self._render_chat()
                return
            self._append_chat("You", text)
            event.app.create_background_task(self._process(text))

        @bindings.add("escape", "enter")
        def _newline(event) -> None:
            if event.app.layout.has_focus(self.input):
                event.current_buffer.insert_text("\n")

        @bindings.add("tab")
        def _focus_next(event) -> None:
            event.app.layout.focus_next()

        @bindings.add("s-tab")
        def _focus_previous(event) -> None:
            event.app.layout.focus_previous()

        @bindings.add("c-l")
        def _clear(_event) -> None:
            self._chat_text = ""
            self._render_chat()

        @bindings.add("c-q")
        def _quit(event) -> None:
            if self.busy:
                self._set_footer_message("任务执行中，完成后再退出。")
                return
            event.app.exit()

        return bindings

    def _set_footer_message(self, message: str) -> None:
        self.footer_message = message
        self.app.invalidate()

    def _append_chat(self, speaker: str, content: str) -> None:
        self._chat_text += f"{speaker}\n{content.rstrip()}\n\n"
        self._render_chat()

    def _render_chat(self) -> None:
        self.chat.buffer.set_document(
            Document(self._chat_text, cursor_position=len(self._chat_text)),
            bypass_readonly=True,
        )
        self.app.invalidate()

    async def _process(self, text: str) -> None:
        self.footer_message = ""
        self.busy = True
        self.app.invalidate()
        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(
                self.executor,
                self.backend.handle,
                text,
                self._request_approval,
            )
            self.status_snapshot = await loop.run_in_executor(
                self.executor, self.backend.status
            )
            self._append_chat("MyAgent", answer)
        except Exception as exc:  # keep the TUI alive after one failed turn
            self._append_chat("Error", f"{type(exc).__name__}: {exc}")
        finally:
            self.busy = False
            self.footer_message = ""
            self.app.invalidate()

    def _request_approval(self, call: ToolCall) -> bool:
        request = ApprovalRequest(call=call, event=threading.Event())
        self.pending_approval = request
        self.app.invalidate()
        request.event.wait()
        return request.decision

    def _finish_approval(self, decision: bool) -> None:
        request = self.pending_approval
        if request is None:
            return
        request.decision = decision
        self.pending_approval = None
        request.event.set()
        self.app.invalidate()

    def run(self) -> None:
        try:
            self.app.run()
        finally:
            if self.pending_approval is not None:
                self._finish_approval(False)


def main() -> None:
    """Start the full-screen terminal UI."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="myagent-tui")
    backend: TuiBackend | None = None
    try:
        backend = executor.submit(TuiBackend.create).result()
        initial_status = executor.submit(backend.status).result()
        MyAgentTui(backend, executor, initial_status).run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"MyAgent TUI启动失败：{exc}") from exc
    finally:
        if backend is not None:
            executor.submit(backend.close).result()
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
