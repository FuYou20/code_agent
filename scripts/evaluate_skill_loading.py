"""Run deterministic 100-skill loading experiments without an LLM call."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.skills.retriever import retrieve_relevant
from myagent.skills.session_state import SkillLoadState, skill_index_generation


TOPIC_ACTIONS = {
    "code": [
        "circular import", "pytest failure", "type checking", "dependency upgrade",
        "api refactor", "async deadlock", "memory leak", "database migration",
        "cli parsing", "logging setup", "package build", "code review",
        "performance profile", "exception handling", "unit test design",
    ],
    "writing": [
        "executive summary", "copy edit", "argument outline", "release notes",
        "technical tutorial", "meeting minutes", "press release", "grant abstract",
        "style guide", "email rewrite", "blog introduction", "product brief",
        "proofreading", "tone adjustment", "citation prose",
    ],
    "data": [
        "csv cleanup", "missing values", "schema validation", "sql analysis",
        "dashboard metrics", "outlier detection", "time series", "data merge",
        "sampling plan", "feature engineering", "pivot table", "data dictionary",
        "cohort analysis", "chart selection",
    ],
    "devops": [
        "docker build", "kubernetes rollout", "ci pipeline", "terraform plan",
        "service health", "deployment rollback", "secret rotation", "log shipping",
        "cloud cost", "release automation", "container scan", "dns incident",
        "backup restore", "on call runbook",
    ],
    "security": [
        "threat model", "dependency audit", "access review", "sql injection",
        "xss review", "incident response", "credential leak", "encryption design",
        "permission boundary", "security headers", "malware triage", "risk register",
        "oauth review", "supply chain",
    ],
    "research": [
        "literature search", "paper critique", "citation verify", "experiment design",
        "hypothesis test", "peer review", "systematic review", "method comparison",
        "research question", "evidence table", "replication plan", "abstract screen",
        "bias assessment", "result synthesis",
    ],
    "documents": [
        "pdf extract", "word formatting", "slide outline", "spreadsheet formula",
        "table cleanup", "document redline", "figure caption", "form filling",
        "layout audit", "template design", "ocr correction", "presentation notes",
        "file conversion", "accessible document",
    ],
}


@dataclass(frozen=True)
class Turn:
    query: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected: str | None
    kind: str


def _slug(value: str) -> str:
    return value.replace(" ", "-")


def build_skill_index() -> list[dict]:
    """Build exactly 100 realistic L0 records with intentionally close neighbours."""
    index: list[dict] = []
    for topic, actions in TOPIC_ACTIONS.items():
        for action in actions:
            index.append(
                {
                    "name": f"{topic}-{_slug(action)}",
                    "scene": topic,
                    "description": (
                        f"A {topic} workflow for {action}, including verification, "
                        f"failure analysis, and a concise handoff."
                    ),
                    "triggers": [action, f"{topic} {action}"],
                    "path": f"/synthetic/{topic}/{_slug(action)}",
                }
            )
    assert len(index) == 100
    return sorted(index, key=lambda item: item["name"])


def _turn(topic: str, action: str, wording: str | None = None) -> Turn:
    name = f"{topic}-{_slug(action)}"
    return Turn(wording or f"Help me with {topic} {action}.", (name,))


def build_scenarios() -> dict[str, list[Turn]]:
    code = [_turn("code", action) for action in TOPIC_ACTIONS["code"]]
    writing = [_turn("writing", action) for action in TOPIC_ACTIONS["writing"]]
    data = [_turn("data", action) for action in TOPIC_ACTIONS["data"]]
    return {
        "A_code_long": code,
        "B_writing_long": writing,
        "C_code_to_writing": code[:8] + writing[:7],
        "D_short": code[:3],
        "E_one_turn_aside": code[:4] + [writing[0]] + code[4:9],
        "F_no_skill_chat": [Turn(text, ()) for text in (
            "你好，今天心情怎么样？", "讲一个简短的笑话。", "谢谢，回头见。"
        )],
        "G_data_long": data,
        "H_rare_one_shot": [
            Turn("Please perform supply chain analysis now.", ("security-supply-chain",))
        ],
    }


def build_retrieval_cases() -> list[RetrievalCase]:
    """Return a small challenge set that is not used to generate skill metadata."""
    return [
        RetrievalCase("Fix this circular import", "code-circular-import", "exact"),
        RetrievalCase("Run pytest and explain the first failure", "code-pytest-failure", "exact"),
        RetrievalCase("Create release notes", "writing-release-notes", "exact"),
        RetrievalCase("Validate the data schema", "data-schema-validation", "exact"),
        RetrievalCase("Two Python modules depend on one another while starting", "code-circular-import", "paraphrase"),
        RetrievalCase("The test suite's first assertion is red", "code-pytest-failure", "paraphrase"),
        RetrievalCase("Turn these rough paragraphs into a concise leadership overview", "writing-executive-summary", "paraphrase"),
        RetrievalCase("Check whether the tabular fields conform to the expected contract", "data-schema-validation", "paraphrase"),
        RetrievalCase("Production is bad; restore the previously deployed version", "devops-deployment-rollback", "paraphrase"),
        RetrievalCase("Map attacker goals, assets, and trust boundaries", "security-threat-model", "paraphrase"),
        RetrievalCase("Find prior studies across academic databases", "research-literature-search", "paraphrase"),
        RetrievalCase("Pull text out of a scanned report", "documents-pdf-extract", "paraphrase"),
        RetrievalCase("Write a tutorial about circular imports", "writing-technical-tutorial", "hard_negative"),
        RetrievalCase("Create release notes for a database migration", "writing-release-notes", "hard_negative"),
        RetrievalCase("你好，今天心情怎么样？", None, "no_skill"),
        RetrievalCase("Tell me a short joke", None, "no_skill"),
    ]


def evaluate_retrieval(index: list[dict], top_k: int = 5) -> dict:
    rows = []
    for case in build_retrieval_cases():
        matches = retrieve_relevant(case.query, index, k=top_k)
        names = [match.name for match in matches]
        rows.append(
            {
                "kind": case.kind,
                "query": case.query,
                "expected": case.expected,
                "matches": names,
                "candidate_hit": case.expected in names if case.expected else not names,
                "primary_hit": names[:1] == [case.expected] if case.expected else not names,
            }
        )
    positive = [row for row in rows if row["expected"] is not None]
    negative = [row for row in rows if row["expected"] is None]
    return {
        "candidate_recall_at_5": sum(row["candidate_hit"] for row in positive) / len(positive),
        "primary_accuracy": sum(row["primary_hit"] for row in positive) / len(positive),
        "no_skill_accuracy": sum(row["candidate_hit"] for row in negative) / len(negative),
        "cases": rows,
    }


def render_l0(names: set[str] | tuple[str, ...] | list[str], by_name: dict[str, dict]) -> str:
    lines = []
    for name in sorted(names):
        item = by_name[name]
        triggers = ", ".join(map(str, item["triggers"]))
        lines.append(f"- {name}: {item['description']} [triggers: {triggers}]")
    return "\n".join(lines)


def token_proxy(text: str) -> int:
    """A deterministic UTF-8 proxy used only for relative offline comparisons."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def run_strategy(
    strategy: str,
    turns: list[Turn],
    index: list[dict],
    *,
    top_k: int = 5,
) -> dict:
    by_name = {item["name"]: item for item in index}
    all_names = set(by_name)
    generation = skill_index_generation(index)
    state = SkillLoadState()
    relevant_total = 0
    available_total = 0
    total_skill_tokens = 0
    irrelevant_skill_tokens = 0
    system_hashes: list[str] = []
    transitions: list[dict] = []
    first_switch_available = None

    previous_topic = None
    for number, turn in enumerate(turns, start=1):
        matches = retrieve_relevant(turn.query, index, k=top_k)
        retrieved_names = {match.name for match in matches}
        expected = set(turn.relevant)

        if strategy == "full_sidecar":
            system_names: set[str] = set()
            sidecar_names = all_names
        elif strategy == "full_system":
            system_names = all_names
            sidecar_names = set()
        elif strategy == "top_k":
            system_names = set()
            sidecar_names = retrieved_names
        elif strategy == "oracle":
            system_names = set()
            sidecar_names = expected
        elif strategy in {"warm_stable", "scene_epoch"}:
            events = state.observe(matches, generation=generation, now=float(number * 60))
            transitions.extend({"turn": number, "event": event, "epoch": state.epoch_id} for event in events)
            system_names = set(state.frozen_names)
            sidecar_names = retrieved_names - system_names
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        visible = system_names | sidecar_names
        relevant_total += len(expected)
        available_total += len(expected & visible)
        system_text = render_l0(system_names, by_name)
        sidecar_text = render_l0(sidecar_names, by_name)
        total_skill_tokens += token_proxy(system_text) + token_proxy(sidecar_text)
        irrelevant_skill_tokens += token_proxy(render_l0(visible - expected, by_name))
        system_hashes.append(hashlib.sha256(system_text.encode("utf-8")).hexdigest())

        topic = next(iter(expected)).split("-", 1)[0] if expected else "none"
        if previous_topic not in (None, "none") and topic not in (previous_topic, "none"):
            first_switch_available = expected <= visible
        previous_topic = topic

    stable_windows = sum(
        len(set(system_hashes[start : start + 3])) == 1
        for start in range(max(0, len(system_hashes) - 2))
    )
    return {
        "strategy": strategy,
        "turns": len(turns),
        "total_skill_token_proxy": total_skill_tokens,
        "irrelevant_skill_token_proxy": irrelevant_skill_tokens,
        "availability_recall": 1.0 if relevant_total == 0 else available_total / relevant_total,
        "stable_3_turn_windows": stable_windows,
        "first_switch_available": first_switch_available,
        "transitions": transitions,
        "final_frozen_size": (
            len(state.frozen_names)
            if strategy in {"warm_stable", "scene_epoch"}
            else None
        ),
    }


def evaluate() -> dict:
    index = build_skill_index()
    scenarios = build_scenarios()
    strategies = ("full_sidecar", "full_system", "top_k", "scene_epoch", "oracle")
    rows = [
        {"scenario": scenario, **run_strategy(strategy, turns, index)}
        for scenario, turns in scenarios.items()
        for strategy in strategies
    ]
    return {
        "method": {
            "skill_count": len(index),
            "token_measure": "ceil(UTF-8 bytes / 4), a relative offline proxy, not provider billing tokens",
            "top_k": 5,
            "note": "Availability measures whether relevant L0 is visible; it does not simulate model application success.",
        },
        "retrieval_challenge": evaluate_retrieval(index),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of a compact table")
    args = parser.parse_args()
    report = evaluate()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    challenge = report["retrieval_challenge"]
    print(
        "retrieval_challenge "
        f"candidate_recall@5={challenge['candidate_recall_at_5']:.3f} "
        f"primary_accuracy={challenge['primary_accuracy']:.3f} "
        f"no_skill_accuracy={challenge['no_skill_accuracy']:.3f}"
    )
    print("scenario\tstrategy\ttokens\tirrelevant\tavailability\tfrozen\ttransitions")
    for row in report["results"]:
        changes = ",".join(f"{item['event']}@{item['turn']}" for item in row["transitions"])
        print(
            f"{row['scenario']}\t{row['strategy']}\t{row['total_skill_token_proxy']}\t"
            f"{row['irrelevant_skill_token_proxy']}\t{row['availability_recall']:.3f}\t"
            f"{row['final_frozen_size']}\t{changes}"
        )


if __name__ == "__main__":
    main()
