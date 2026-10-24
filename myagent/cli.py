"""Provide the Rich and prompt-toolkit command-line interface."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel

from .agent.loop import run_turn
from .agent.state import SessionState
from .config import load_config
from .context.stable import initialize_stable_prefix
from .llm.client import OpenAIClient
from .llm.types import ToolCall
from .memory.store import init_db
from .home import get_home
from .skills.manager import load_index
from .skills.distiller import distill
from .tools.builtin import default_scope, register_all
from .tools.registry import schemas


def _repo_snapshot(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return "\n".join(result.stdout.splitlines()[:200])


def _read_agents_once(cwd: Path) -> str:
    path = cwd / "AGENTS.md"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def _load_skill_index() -> list[dict]:
    bundled_store = Path(__file__).resolve().parents[1] / "skills_store"
    runtime_store = get_home() / "skills_store"
    runtime_store.mkdir(parents=True, exist_ok=True)
    merged = {item["name"]: item for item in load_index(bundled_store)}
    merged.update({item["name"]: item for item in load_index(runtime_store)})
    return [merged[name] for name in sorted(merged)]


def _make_approver(console: Console, prompt: PromptSession):
    def approve(call: ToolCall) -> bool:
        console.print(
            Panel.fit(
                f"工具：{call.name}\n参数：{call.arguments}",
                title="高风险工具审批",
                border_style="yellow",
            )
        )
        try:
            answer = prompt.prompt("允许执行？[y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in {"y", "yes"}

    return approve


def main() -> None:
    """Initialize one process and handle either one-shot or interactive input."""
    config = load_config()
    provider = config["llm"].get("provider")
    if provider not in {"openai", "deepseek"}:
        raise ValueError("llm.provider 必须是 openai 或 deepseek")
    key_name = config["llm"]["api_key_env"]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"缺少环境变量：{key_name}")

    cwd = Path.cwd().resolve()
    skill_strategy = str(config["skills"]["strategy"])
    if skill_strategy not in {"full_sidecar", "top_k"}:
        raise ValueError("skills.strategy 必须是 full_sidecar 或 top_k")
    skill_top_k = int(config["skills"]["top_k"])
    if skill_top_k <= 0:
        raise ValueError("skills.top_k 必须大于 0")
    skill_activation_min_score = float(config["skills"]["activation_min_score"])
    if skill_activation_min_score < 0:
        raise ValueError("skills.activation_min_score 不能小于 0")
    register_all(config, cwd)
    initialize_stable_prefix(schemas())
    scope = default_scope(config, cwd)
    state = SessionState(
        session_id=uuid.uuid4().hex,
        scope=scope,
        cwd=str(cwd),
        repo_snapshot=_repo_snapshot(cwd),
        agents_content=_read_agents_once(cwd),
        skill_index=_load_skill_index(),
        skill_strategy=skill_strategy,
        skill_top_k=skill_top_k,
        skill_activation_min_score=skill_activation_min_score,
        max_steps=int(config["agent"]["max_steps"]),
        memory_conn=init_db(get_home() / "memory.db"),
    )
    base_url = config["llm"].get("base_url")
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    client = OpenAIClient(
        model=config["llm"]["model"],
        api_key=api_key,
        base_url=base_url,
    )
    console = Console()
    prompt: PromptSession = PromptSession()
    approve = _make_approver(console, prompt)

    def handle(text: str) -> None:
        if text.strip() == "/learn":
            skill = distill(state.history, client)
            state.skill_index = _load_skill_index()
            console.print(f"[green]已蒸馏技能：{skill.frontmatter.name}[/green]")
            return
        answer = run_turn(state, text, client, scope, approve)
        console.print(answer)

    if len(sys.argv) > 1:
        handle(" ".join(sys.argv[1:]))
        return

    while True:
        try:
            user_input = prompt.prompt("myagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return
        handle(user_input)

