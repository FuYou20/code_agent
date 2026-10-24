"""Measure OpenAI prompt-cache ratios across five turns in one session."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.cli import _load_skill_index
from myagent.config import load_config
from myagent.context.stable import initialize_stable_prefix
from myagent.llm.client import OpenAIClient
from myagent.tools.builtin import default_scope, register_all
from myagent.tools.registry import schemas


class MeasuringClient:
    """Delegate chat calls while accumulating usage for the current user turn."""

    def __init__(self, inner: OpenAIClient):
        self.inner = inner
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.cached_tokens = 0

    def chat(self, messages, tools):
        response = self.inner.chat(messages, tools)
        self.prompt_tokens += response.usage.get("prompt_tokens", 0)
        self.cached_tokens += response.usage.get("cached_tokens", 0)
        return response


def _snapshot(cwd: Path) -> str:
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
    return "\n".join(result.stdout.splitlines()[:200]) if result.returncode == 0 else ""


def main() -> None:
    config = load_config()
    provider = config["llm"].get("provider")
    if provider not in {"openai", "deepseek"}:
        raise SystemExit("llm.provider 必须是 openai 或 deepseek")
    key_name = config["llm"]["api_key_env"]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"缺少环境变量：{key_name}")

    cwd = Path.cwd().resolve()
    register_all(config, cwd)
    initialize_stable_prefix(schemas())
    scope = default_scope(config, cwd)
    state = SessionState(
        session_id=uuid.uuid4().hex,
        scope=scope,
        cwd=str(cwd),
        repo_snapshot=_snapshot(cwd),
        skill_index=_load_skill_index(),
        skill_strategy=str(config["skills"]["strategy"]),
        skill_top_k=int(config["skills"]["top_k"]),
        skill_activation_min_score=float(config["skills"]["activation_min_score"]),
        max_steps=int(config["agent"]["max_steps"]),
    )
    base_url = config["llm"].get("base_url")
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    client = MeasuringClient(
        OpenAIClient(
            model=config["llm"]["model"],
            api_key=api_key,
            base_url=base_url,
        )
    )

    for turn in range(1, 6):
        client.reset()
        prompt = f"这是缓存测量的第 {turn} 轮。不要调用工具，只回复：收到。"
        run_turn(state, prompt, client, scope, lambda _call: False)
        ratio = client.cached_tokens / client.prompt_tokens if client.prompt_tokens else 0.0
        print(
            f"turn={turn} prompt_tokens={client.prompt_tokens} "
            f"cached_tokens={client.cached_tokens} ratio={ratio:.2%}"
        )


if __name__ == "__main__":
    main()
