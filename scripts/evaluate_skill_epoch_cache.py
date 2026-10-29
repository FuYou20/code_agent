"""Measure structural Skill cost for full System versus Scene-Epoch System L0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.skills.retriever import SkillMatch
from myagent.skills.session_state import SkillLoadState, skill_index_generation
from scripts.evaluate_skill_loading import (
    build_skill_index,
    render_l0,
    token_proxy,
)
from scripts.evaluate_skill_epoch_live import build_conversation_scenarios


def _topic(name: str) -> str:
    return name.split("-", 1)[0]


def _potential_workset(required: tuple[str, ...], all_names: list[str]) -> list[str]:
    """Return a deterministic topic workset for the structural cache experiment.

    This deliberately models a successful workset planner. Retrieval quality is
    measured separately by ``evaluate_skill_rag.py``.
    """
    if not required:
        return []
    topics = {_topic(name) for name in required}
    related = [name for name in all_names if _topic(name) in topics]
    ordered = [*required, *(name for name in related if name not in required)]
    return ordered[:8]


def _matches(names: list[str], required: tuple[str, ...]) -> list[SkillMatch]:
    required_set = set(required)
    return [
        SkillMatch(
            name=name,
            score=(1000.0 if name in required_set else 100.0) - index,
            reasons=("structural-workset",),
        )
        for index, name in enumerate(names)
    ]


def evaluate() -> dict:
    index = build_skill_index()
    by_name = {item["name"]: item for item in index}
    all_names = sorted(by_name)
    generation = skill_index_generation(index)
    rows: list[dict] = []

    for scenario_name, turns in build_conversation_scenarios().items():
        full_uncached = token_proxy(render_l0(all_names, by_name))
        frozen_uncached = 0
        frozen_raw = 0
        available = 0
        required_total = 0
        system_changes = 0
        temporary_injections = 0
        state = SkillLoadState()
        previous_system_names: tuple[str, ...] = ()

        for number, turn in enumerate(turns, start=1):
            required = (turn.required_skill,)
            required_total += 1
            potential = _potential_workset(required, all_names)
            events = state.observe(
                _matches(potential, required),
                generation=generation,
                now=float(number * 60),
            )

            system_names = state.frozen_names if state.phase == "frozen" else ()
            if system_names != previous_system_names:
                system_changes += 1
                if system_names:
                    frozen_uncached += token_proxy(render_l0(system_names, by_name))
                previous_system_names = system_names

            if state.phase == "warming":
                sidecar_names = tuple(potential)
            else:
                frozen_set = set(state.frozen_names)
                sidecar_names = tuple(
                    name for name in required if name not in frozen_set
                )
                temporary_injections += len(sidecar_names)
            if sidecar_names:
                frozen_uncached += token_proxy(render_l0(sidecar_names, by_name))

            frozen_raw += token_proxy(render_l0(system_names, by_name))
            frozen_raw += token_proxy(render_l0(sidecar_names, by_name))
            visible = set(system_names) | set(sidecar_names)
            available += int(turn.required_skill in visible)

        reduction = 1 - frozen_uncached / full_uncached if full_uncached else 0.0
        rows.append(
            {
                "scenario": scenario_name,
                "turns": len(turns),
                "full_system_uncached_skill_token_proxy": full_uncached,
                "epoch_system_raw_skill_token_proxy": frozen_raw,
                "epoch_system_uncached_skill_token_proxy": frozen_uncached,
                "uncached_reduction": reduction,
                "required_skill_availability": (
                    available / required_total if required_total else 1.0
                ),
                "system_prompt_changes": system_changes,
                "temporary_injections": temporary_injections,
            }
        )

    full_total = sum(row["full_system_uncached_skill_token_proxy"] for row in rows)
    epoch_total = sum(row["epoch_system_uncached_skill_token_proxy"] for row in rows)
    return {
        "method": "deterministic structural proxy; retrieval quality excluded",
        "skill_count": len(index),
        "scenario_count": len(rows),
        "full_system_uncached_skill_token_proxy": full_total,
        "epoch_system_uncached_skill_token_proxy": epoch_total,
        "uncached_reduction": 1 - epoch_total / full_total if full_total else 0.0,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"skills={result['skill_count']} scenarios={result['scenario_count']} "
        f"full_uncached={result['full_system_uncached_skill_token_proxy']} "
        f"epoch_uncached={result['epoch_system_uncached_skill_token_proxy']} "
        f"reduction={result['uncached_reduction']:.2%}"
    )
    for row in result["results"]:
        print(
            f"{row['scenario']}: reduction={row['uncached_reduction']:.2%} "
            f"availability={row['required_skill_availability']:.2%} "
            f"system_changes={row['system_prompt_changes']} "
            f"temporary={row['temporary_injections']}"
        )


if __name__ == "__main__":
    main()
