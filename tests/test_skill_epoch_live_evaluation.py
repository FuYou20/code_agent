"""Protect the simple 18-turn provider A/B measurement contract."""

from myagent.llm.types import LLMResponse, Message
from scripts.evaluate_skill_epoch_live import (
    EvalTurn,
    MeasuringClient,
    _sum_arm,
    build_conversation_scenarios,
)


class StaticInner:
    def chat(self, messages, tools):
        return LLMResponse(
            content="OK",
            usage={"prompt_tokens": 100, "cached_tokens": 40},
        )


def test_three_scenarios_have_nineteen_grounded_turns() -> None:
    scenarios = build_conversation_scenarios()
    assert tuple(scenarios) == (
        "stable_topic",
        "one_turn_aside",
        "sustained_drift",
    )
    assert sum(len(turns) for turns in scenarios.values()) == 19
    assert all(turn.required_skill for turns in scenarios.values() for turn in turns)


def test_measuring_client_counts_router_cost_and_main_visibility() -> None:
    client = MeasuringClient(StaticInner())
    turn = EvalTurn("run tests", "code-pytest-failure")
    client.begin_turn(turn)
    client.chat([Message(role="system", content="router")], [])
    client.chat(
        [
            Message(role="system", content="你是 myagent"),
            Message(role="user", content="code-pytest-failure"),
        ],
        [],
    )
    metrics = client.metrics()
    assert metrics["prompt_tokens"] == 200
    assert metrics["router_prompt_tokens"] == 100
    assert metrics["main_prompt_tokens"] == 100
    assert metrics["router_cached_tokens"] == 40
    assert metrics["main_cached_tokens"] == 40
    assert metrics["main_cache_ratio"] == 0.4
    assert metrics["required_skill_visibility"] == 1.0


def test_arm_summary_uses_weighted_not_average_ratios() -> None:
    sessions = [
        {
            "turns": 2,
            "api_calls": 2,
            "main_calls": 2,
            "prompt_tokens": 200,
            "main_prompt_tokens": 200,
            "router_prompt_tokens": 0,
            "main_cached_tokens": 100,
            "router_cached_tokens": 0,
            "cached_tokens": 100,
            "uncached_prompt_tokens": 100,
            "required_skill_visible": 2,
            "required_skill_turns": 2,
            "system_prompt_changes": 0,
        },
        {
            "turns": 1,
            "api_calls": 1,
            "main_calls": 1,
            "prompt_tokens": 100,
            "main_prompt_tokens": 100,
            "router_prompt_tokens": 0,
            "main_cached_tokens": 0,
            "router_cached_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 100,
            "required_skill_visible": 0,
            "required_skill_turns": 1,
            "system_prompt_changes": 1,
        },
    ]
    summary = _sum_arm("scene_epoch", sessions)
    assert summary["cache_ratio"] == 1 / 3
    assert summary["required_skill_visibility"] == 2 / 3
