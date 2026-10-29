"""Run a provider-reported Full-System versus Scene-Epoch A/B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.config import load_config
from myagent.context import stable
from myagent.llm.client import OpenAIClient
from myagent.permissions import PermissionTree, ToolPermissionAdapter
from myagent.skills.manager import load_index
from scripts.evaluate_skill_loading import build_skill_index


@dataclass(frozen=True)
class EvalTurn:
    query: str
    required_skill: str


def build_conversation_scenarios() -> dict[str, tuple[EvalTurn, ...]]:
    """Return three small conversations that isolate the cache-state behavior."""
    test_skill = "code-pytest-failure"
    writing_skill = "writing-release-notes"
    test_turns = (
        EvalTurn("Run pytest and identify the first failing assertion.", test_skill),
        EvalTurn("Explain why that first test is failing.", test_skill),
        EvalTurn("Inspect the related implementation and propose the smallest fix.", test_skill),
        EvalTurn("Check whether the fix covers the failing edge case.", test_skill),
        EvalTurn("Run the relevant test again and interpret the result.", test_skill),
        EvalTurn("Summarize the remaining pytest risk.", test_skill),
    )
    writing_turns = (
        EvalTurn("Create concise release notes for this change.", writing_skill),
        EvalTurn("Rewrite the release notes for end users.", writing_skill),
        EvalTurn("Add a short compatibility warning to the release notes.", writing_skill),
    )
    return {
        "stable_topic": test_turns,
        "one_turn_aside": test_turns[:4] + writing_turns[:1] + test_turns[4:5],
        "sustained_drift": test_turns[:4] + writing_turns,
    }


class MeasuringClient:
    """Count every provider input and inspect the final Agent prompt per turn."""

    def __init__(self, inner: OpenAIClient) -> None:
        self.inner = inner
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.main_prompt_tokens = 0
        self.router_prompt_tokens = 0
        self.main_cached_tokens = 0
        self.router_cached_tokens = 0
        self.calls = 0
        self.main_calls = 0
        self.required_turns = 0
        self.visible_turns = 0
        self._required_skill = ""
        self.visibility_rows: list[dict] = []

    def begin_turn(self, turn: EvalTurn) -> None:
        self._required_skill = turn.required_skill

    @staticmethod
    def _is_main_agent_call(messages) -> bool:
        return bool(
            messages
            and messages[0].role == "system"
            and "你是 myagent" in messages[0].content
        )

    def chat(self, messages, tools):
        response = self.inner.chat(messages, tools)
        used = int(response.usage.get("prompt_tokens", 0))
        cached = int(response.usage.get("cached_tokens", 0))
        self.calls += 1
        self.prompt_tokens += used
        self.cached_tokens += cached

        if self._is_main_agent_call(messages):
            self.main_calls += 1
            self.main_prompt_tokens += used
            self.main_cached_tokens += cached
            prompt_text = "\n".join(message.content for message in messages)
            visible = bool(
                self._required_skill and self._required_skill in prompt_text
            )
            self.required_turns += 1
            self.visible_turns += int(visible)
            self.visibility_rows.append(
                {
                    "required_skill": self._required_skill,
                    "visible": visible,
                }
            )
        else:
            self.router_prompt_tokens += used
            self.router_cached_tokens += cached
        return response

    def metrics(self) -> dict:
        return {
            "api_calls": self.calls,
            "main_calls": self.main_calls,
            "prompt_tokens": self.prompt_tokens,
            "main_prompt_tokens": self.main_prompt_tokens,
            "router_prompt_tokens": self.router_prompt_tokens,
            "main_cached_tokens": self.main_cached_tokens,
            "router_cached_tokens": self.router_cached_tokens,
            "cached_tokens": self.cached_tokens,
            "uncached_prompt_tokens": max(0, self.prompt_tokens - self.cached_tokens),
            "cache_ratio": (
                self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
            ),
            "main_cache_ratio": (
                self.main_cached_tokens / self.main_prompt_tokens
                if self.main_prompt_tokens
                else 0.0
            ),
            "required_skill_visible": self.visible_turns,
            "required_skill_turns": self.required_turns,
            "required_skill_visibility": (
                self.visible_turns / self.required_turns
                if self.required_turns
                else 1.0
            ),
            "visibility_rows": list(self.visibility_rows),
        }


def _write_synthetic_store(root: Path) -> list[dict]:
    for item in build_skill_index():
        name = str(item["name"])
        directory = root / name
        directory.mkdir(parents=True)
        triggers = json.dumps(item["triggers"], ensure_ascii=False)
        (directory / "SKILL.md").write_text(
            (
                "---\n"
                f"name: {name}\n"
                f"description: {item['description']}\n"
                f"scene: {item['scene']}\n"
                f"triggers: {triggers}\n"
                "permission_requirements: []\n"
                "---\n\n"
                f"Skill: {name}. Follow this workflow without calling tools.\n"
            ),
            encoding="utf-8",
        )
    return load_index(root)


def _run_session(
    strategy: str,
    scenario_name: str,
    turns: tuple[EvalTurn, ...],
    client: MeasuringClient,
    index: list[dict],
    cwd: Path,
    run_id: str,
) -> dict:
    agent_id = f"eval-{strategy}-{scenario_name}-{uuid.uuid4().hex}"
    permission_tree = PermissionTree(cwd, ())
    permission_tree.attach_agent(permission_tree.ROOT_ID, agent_id)
    state = SessionState(
        session_id=agent_id,
        agent_id=agent_id,
        permission_tree=permission_tree,
        permission_adapter=ToolPermissionAdapter(cwd),
        cwd=str(cwd),
        repo_snapshot=f"evaluation={run_id}:{strategy}:{scenario_name}",
        skill_index=index,
        skill_strategy=strategy,
        skill_top_k=5,
        skill_candidate_k=20,
        skill_max_frozen=8,
        max_steps=1,
    )
    for turn in turns:
        client.begin_turn(turn)
        prompt = f"{turn.query}\nDo not call tools. Reply with exactly: OK"
        run_turn(state, prompt, client, lambda _call: False)
    return {
        "scenario": scenario_name,
        "turns": len(turns),
        **client.metrics(),
        "system_prompt_changes": state.skill_cache_epoch_changes,
        "final_phase": state.skill_load_state.phase,
        "frozen_skills": list(state.skill_load_state.frozen_names),
    }


def _sum_arm(strategy: str, sessions: list[dict]) -> dict:
    summed = {
        key: sum(int(session[key]) for session in sessions)
        for key in (
            "turns",
            "api_calls",
            "main_calls",
            "prompt_tokens",
            "main_prompt_tokens",
            "router_prompt_tokens",
            "main_cached_tokens",
            "router_cached_tokens",
            "cached_tokens",
            "uncached_prompt_tokens",
            "required_skill_visible",
            "required_skill_turns",
            "system_prompt_changes",
        )
    }
    summed["strategy"] = strategy
    summed["cache_ratio"] = (
        summed["cached_tokens"] / summed["prompt_tokens"]
        if summed["prompt_tokens"]
        else 0.0
    )
    summed["main_cache_ratio"] = (
        summed["main_cached_tokens"] / summed["main_prompt_tokens"]
        if summed["main_prompt_tokens"]
        else 0.0
    )
    summed["required_skill_visibility"] = (
        summed["required_skill_visible"] / summed["required_skill_turns"]
        if summed["required_skill_turns"]
        else 1.0
    )
    summed["sessions"] = sessions
    return summed


def _client(config: dict, api_key: str) -> MeasuringClient:
    base_url = config["llm"].get("base_url")
    if config["llm"].get("provider") == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"
    return MeasuringClient(
        OpenAIClient(str(config["llm"]["model"]), api_key, base_url)
    )


def evaluate_live(config: dict, api_key: str) -> dict:
    """Run both arms over the same 19 grounded turns and aggregate the result."""
    scenarios = build_conversation_scenarios()
    run_id = uuid.uuid4().hex
    stable.STABLE_PREFIX = stable.build_stable_prefix([])
    previous_home = os.environ.get("MYAGENT_HOME")
    try:
        with tempfile.TemporaryDirectory(prefix="myagent-skill-ab-") as temp:
            temp_root = Path(temp)
            os.environ["MYAGENT_HOME"] = str(temp_root / "home")
            index = _write_synthetic_store(temp_root / "skills")
            arms: dict[str, dict] = {}
            for strategy in ("full_system", "scene_epoch"):
                sessions = [
                    _run_session(
                        strategy,
                        scenario_name,
                        turns,
                        _client(config, api_key),
                        index,
                        ROOT,
                        run_id,
                    )
                    for scenario_name, turns in scenarios.items()
                ]
                arms[strategy] = _sum_arm(strategy, sessions)
    finally:
        if previous_home is None:
            os.environ.pop("MYAGENT_HOME", None)
        else:
            os.environ["MYAGENT_HOME"] = previous_home

    full = arms["full_system"]
    warm = arms["scene_epoch"]
    return {
        "method": {
            "skill_count": 100,
            "scenarios": len(scenarios),
            "turns": sum(len(turns) for turns in scenarios.values()),
            "token_scope": (
                "all provider prompt tokens, including query expansion and reranking"
            ),
            "visibility_definition": (
                "required Skill name is present in the main Agent System/Sidecar/L1 prompt"
            ),
        },
        "provider": config["llm"].get("provider"),
        "model": config["llm"].get("model"),
        "full_system": full,
        "scene_epoch": warm,
        "prompt_token_reduction": (
            1 - warm["prompt_tokens"] / full["prompt_tokens"]
            if full["prompt_tokens"]
            else 0.0
        ),
        "uncached_prompt_token_reduction": (
            1 - warm["uncached_prompt_tokens"] / full["uncached_prompt_tokens"]
            if full["uncached_prompt_tokens"]
            else 0.0
        ),
    }


def _print_compact(result: dict) -> None:
    print(
        "strategy\tvisible\tprompt_tokens\tuncached_tokens\tmain_cache_ratio"
        "\trouter_tokens\tsystem_changes"
    )
    for key in ("full_system", "scene_epoch"):
        row = result[key]
        print(
            f"{key}\t{row['required_skill_visible']}/{row['required_skill_turns']}"
            f"\t{row['prompt_tokens']}\t{row['uncached_prompt_tokens']}"
            f"\t{row['main_cache_ratio']:.2%}\t{row['router_prompt_tokens']}"
            f"\t{row['system_prompt_changes']}"
        )
    print(f"prompt_token_reduction={result['prompt_token_reduction']:.2%}")
    print(
        "uncached_prompt_token_reduction="
        f"{result['uncached_prompt_token_reduction']:.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config()
    key_name = str(config["llm"]["api_key_env"])
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(
            f"缺少环境变量：{key_name}。请只在当前终端设置后重新运行，"
            "不要把密钥写入仓库。"
        )
    result = evaluate_live(config, api_key)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.json:
        print(rendered)
    else:
        _print_compact(result)


if __name__ == "__main__":
    main()
