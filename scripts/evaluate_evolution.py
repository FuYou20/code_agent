"""Run Fixed-Skill versus Evolvable-Agent on the 30-task family benchmark.

The learning phase contains one task per family.  The transfer phase contains
two fresh variants per family and is the primary score.  Both arms see the same
tasks in the same order; only the evolvable arm persists memory and distills a
Skill after a genuinely passing learning task.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import uuid

from benchmarks.evolution import TASKS, EvolutionTask
from benchmarks.evolution.harness import grade, materialize
from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.config import load_config
from myagent.context.stable import initialize_stable_prefix
from myagent.evolution.artifacts import (
    EvolutionArtifactManager,
    TokenUsage,
    TokenWeights,
)
from myagent.llm.client import OpenAIClient
from myagent.memory.store import init_db
from myagent.memory.retrieve import memory_stats, record_feedback
from myagent.observability import LangfuseReporter
from myagent.permissions import PermissionTree, ToolPermissionAdapter
from myagent.skills.distiller import distill
from myagent.skills.manager import load_index
from myagent.tools.builtin import default_permission_atoms, register_all
from myagent.tools.registry import schemas


class MeteredClient:
    """Accumulate provider-reported usage across main-Agent and distillation calls."""

    def __init__(
        self,
        inner: OpenAIClient,
        reporter: LangfuseReporter | None = None,
    ):
        self.inner = inner
        self.reporter = reporter
        self.api_calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
        }

    @staticmethod
    def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {key: after[key] - before[key] for key in before}

    def chat(self, messages, tools):
        generation = (
            self.reporter.start_generation(messages, tools, model=self.inner.model)
            if self.reporter is not None
            else None
        )
        try:
            response = self.inner.chat(messages, tools)
        except Exception as exc:
            if self.reporter is not None:
                self.reporter.finish_generation(generation, None, error=exc)
            raise
        if self.reporter is not None:
            self.reporter.finish_generation(generation, response)
        self.api_calls += 1
        self.prompt_tokens += int(response.usage.get("prompt_tokens", 0))
        self.cached_tokens += int(response.usage.get("cached_tokens", 0))
        self.completion_tokens += int(response.usage.get("completion_tokens", 0))
        self.tool_calls += len(response.tool_calls)
        return response


@contextmanager
def _agent_home(path: Path):
    previous = os.environ.get("MYAGENT_HOME")
    os.environ["MYAGENT_HOME"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MYAGENT_HOME", None)
        else:
            os.environ["MYAGENT_HOME"] = previous


def _skill_index(
    home: Path,
    lifecycle: EvolutionArtifactManager | None = None,
) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    bundled = {item["name"]: item for item in load_index(root / "skills_store")}
    runtime = {
        item["name"]: item
        for item in load_index(
            home / "skills_store",
            lifecycle=lifecycle,
            allowed_statuses=("candidate", "active"),
            trusted_unregistered=lifecycle is None,
        )
    }
    bundled.update(runtime)
    return [bundled[name] for name in sorted(bundled)]


def _new_state(
    workspace: Path,
    home: Path,
    config: dict,
    memory_conn: sqlite3.Connection | None,
    memory_scope: str,
    artifact_manager: EvolutionArtifactManager | None = None,
) -> SessionState:
    register_all(config, workspace)
    agent_id = f"eval-{uuid.uuid4().hex}"
    tree = PermissionTree(workspace, default_permission_atoms())
    tree.attach_agent(tree.ROOT_ID, agent_id)
    return SessionState(
        session_id=uuid.uuid4().hex,
        agent_id=agent_id,
        permission_tree=tree,
        permission_adapter=ToolPermissionAdapter(workspace),
        cwd=str(workspace),
        repo_snapshot="\n".join(sorted(path.name for path in workspace.iterdir())),
        skill_index=_skill_index(home, artifact_manager),
        skill_strategy="top_k",
        skill_top_k=5,
        skill_candidate_k=20,
        # Keep the production confidence gate.  Forcing the threshold to zero
        # makes every weak lexical match inject a full L1 body and invalidates
        # the cost comparison.
        skill_activation_min_score=float(config["skills"]["activation_min_score"]),
        max_steps=int(config["agent"]["max_steps"]),
        memory_conn=memory_conn,
        memory_scope=memory_scope,
        artifact_manager=artifact_manager,
        evolution_artifact_statuses=("candidate", "active"),
    )


def _artifact_manager(
    conn: sqlite3.Connection,
    config: dict,
) -> EvolutionArtifactManager:
    values = config["evolution"]
    return EvolutionArtifactManager(
        conn,
        token_weights=TokenWeights(
            uncached_input=float(values["uncached_input_weight"]),
            cached_input=float(values["cached_input_weight"]),
            output=float(values["output_weight"]),
        ),
        candidate_trials=int(values["candidate_trials"]),
    )


def _ordered_tasks(
    family_limit: int | None,
    family_name: str | None = None,
) -> list[EvolutionTask]:
    families = list(dict.fromkeys(task.family for task in TASKS))
    if family_name is not None:
        families = [family_name]
    elif family_limit is not None:
        families = families[:family_limit]
    selected = [task for task in TASKS if task.family in set(families)]
    return [task for task in selected if task.stage == "learn"] + [
        task for task in selected if task.stage == "transfer"
    ]


def _summarize(rows: list[dict], meter: MeteredClient) -> dict:
    transfer = [row for row in rows if row["stage"] == "transfer"]
    learning = [row for row in rows if row["stage"] == "learn"]

    def phase(items: list[dict]) -> dict:
        total = len(items)
        passed = sum(bool(item["passed"]) for item in items)
        prompt_tokens = sum(item["usage"]["prompt_tokens"] for item in items)
        cached_tokens = sum(item["usage"]["cached_tokens"] for item in items)
        return {
            "tasks": total,
            "passed": passed,
            "success_rate": passed / total if total else 0.0,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "uncached_tokens": prompt_tokens - cached_tokens,
            "cache_hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0.0,
            "completion_tokens": sum(item["usage"]["completion_tokens"] for item in items),
            "tool_calls": sum(item["usage"]["tool_calls"] for item in items),
            "wall_seconds": sum(item["wall_seconds"] for item in items),
        }

    return {
        "learning": phase(learning),
        "transfer": phase(transfer),
        "provider_total": meter.snapshot(),
        "distilled_skills": sum(bool(row.get("distilled_skill")) for row in learning),
    }


def _run_arm(
    arm: str,
    tasks: list[EvolutionTask],
    work_root: Path,
    config: dict,
    client: MeteredClient,
    reporter: LangfuseReporter | None = None,
) -> dict:
    evolved = arm == "evolved"
    arm_root = work_root / arm
    home = arm_root / "home"
    tasks_root = arm_root / "tasks"
    home.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    memory_conn = init_db(home / "memory.db") if evolved else None
    artifact_manager = (
        _artifact_manager(memory_conn, config) if memory_conn is not None else None
    )
    rows: list[dict] = []

    with _agent_home(home):
        for index, task in enumerate(tasks, start=1):
            task_span = (
                reporter.start_task(task, arm=arm, model=client.inner.model)
                if reporter is not None
                else None
            )
            workspace = tasks_root / task.task_id
            if workspace.exists():
                shutil.rmtree(workspace)
            materialize(task, workspace)
            state = _new_state(
                workspace,
                home,
                config,
                memory_conn,
                task.family,
                artifact_manager,
            )
            before = client.snapshot()
            started = time.perf_counter()
            error = ""
            answer = ""
            try:
                answer = run_turn(state, task.prompt, client, lambda _call: True)
            except Exception as exc:  # Preserve the rest of an expensive run.
                error = f"{type(exc).__name__}: {exc}"
            wall_seconds = time.perf_counter() - started
            usage = MeteredClient.delta(before, client.snapshot())
            result = grade(task, workspace)
            feedback_ids: tuple[int, ...] = ()
            feedback_stats: list[dict[str, object]] = []
            if (
                evolved
                and memory_conn is not None
                and state.last_recalled_memory_ids
                and state.last_recalled_memory_scope == task.family
            ):
                feedback_ids = state.last_recalled_memory_ids
                record_feedback(memory_conn, feedback_ids, success=result.passed)
                feedback_stats = [memory_stats(memory_conn, item) for item in feedback_ids]
            recorded_memory_feedback: dict[str, object] = {}
            if evolved and memory_conn is not None and state.last_recorded_memory_id is not None:
                # The evaluator checks the task that produced this memory directly.
                # A hidden-test failure is therefore decisive evidence and the new
                # memory must not be allowed to reproduce itself under a fresh id.
                record_feedback(
                    memory_conn,
                    (state.last_recorded_memory_id,),
                    success=result.passed,
                    decisive=not result.passed,
                )
                recorded_memory_feedback = memory_stats(
                    memory_conn,
                    state.last_recorded_memory_id,
                )
            row = {
                "task_id": task.task_id,
                "family": task.family,
                "stage": task.stage,
                "passed": result.passed,
                "wall_seconds": wall_seconds,
                "usage": usage,
                "evolution_eligible": state.last_evolution_eligible,
                "evolution_reasons": list(state.last_evolution_reasons),
                "answer": answer,
                "error": error,
                "grade_output": result.output,
                "distilled_skill": "",
                "distillation_usage": {},
                "recalled_memory_ids": list(state.last_recalled_memory_ids),
                "feedback_memory_ids": list(feedback_ids),
                "feedback_stats": feedback_stats,
                "recorded_memory_id": state.last_recorded_memory_id,
                "recorded_memory_feedback": recorded_memory_feedback,
                "activated_skill": state.last_activated_skill_name,
                "activated_skill_version": state.last_activated_skill_version,
            }

            if (
                evolved
                and task.stage == "learn"
                and result.passed
                and state.last_evolution_eligible
            ):
                distill_before = client.snapshot()
                try:
                    skill = distill(
                        state.last_evolution_history,
                        client,
                        task_context=task.prompt,
                        artifact_manager=artifact_manager,
                        scope_key=task.family,
                    )
                    row["distilled_skill"] = skill.frontmatter.name
                except Exception as exc:
                    row["error"] = (row["error"] + "; " if row["error"] else "") + (
                        f"distill:{type(exc).__name__}: {exc}"
                    )
                row["distillation_usage"] = MeteredClient.delta(
                    distill_before, client.snapshot()
                )

            rows.append(row)
            if reporter is not None and task_span is not None:
                reporter.finish_task(task_span, row)
            print(
                f"arm={arm} task={index}/{len(tasks)} id={task.task_id} "
                f"passed={result.passed} tools={usage['tool_calls']} "
                f"prompt_tokens={usage['prompt_tokens']}"
            )

    if memory_conn is not None:
        memory_conn.close()
    return {"arm": arm, "summary": _summarize(rows, client), "tasks": rows}


def _comparison(arms: dict[str, dict]) -> dict:
    if set(arms) != {"fixed", "evolved"}:
        return {}
    fixed = arms["fixed"]["summary"]["transfer"]
    evolved = arms["evolved"]["summary"]["transfer"]
    fixed_total = arms["fixed"]["summary"]["provider_total"]
    evolved_total = arms["evolved"]["summary"]["provider_total"]

    def reduction(metric: str) -> float | None:
        base = fixed[metric]
        return 1 - evolved[metric] / base if base else None

    return {
        "transfer_success_rate_change_pp": 100
        * (evolved["success_rate"] - fixed["success_rate"]),
        "transfer_prompt_token_reduction": reduction("prompt_tokens"),
        "transfer_uncached_token_reduction": reduction("uncached_tokens"),
        "transfer_cache_hit_rate_change_pp": 100
        * (evolved["cache_hit_rate"] - fixed["cache_hit_rate"]),
        "transfer_tool_call_reduction": reduction("tool_calls"),
        "transfer_wall_time_reduction": reduction("wall_seconds"),
        "total_prompt_token_change_including_distillation": (
            evolved_total["prompt_tokens"] / fixed_total["prompt_tokens"] - 1
            if fixed_total["prompt_tokens"]
            else None
        ),
        "total_api_call_change_including_distillation": (
            evolved_total["api_calls"] - fixed_total["api_calls"]
        ),
    }


def _record_artifact_trials(
    arms: dict[str, dict],
    work_root: Path,
    config: dict,
) -> list[dict[str, object]]:
    """Pair control/treatment rows and feed token evidence to used artifacts."""
    if set(arms) != {"fixed", "evolved"}:
        return []
    database = work_root / "evolved" / "home" / "memory.db"
    if not database.exists():
        return []
    conn = init_db(database)
    manager = _artifact_manager(conn, config)
    fixed = {row["task_id"]: row for row in arms["fixed"]["tasks"]}
    try:
        for treatment in arms["evolved"]["tasks"]:
            if treatment["stage"] != "transfer":
                continue
            control = fixed.get(treatment["task_id"])
            if control is None:
                continue
            artifacts: set[int] = set()
            for decision_id in treatment.get("recalled_memory_ids", []):
                try:
                    artifacts.add(
                        manager.get_by_key("memory", str(int(decision_id)), 1).artifact_id
                    )
                except KeyError:
                    continue
            skill_name = str(treatment.get("activated_skill", ""))
            if skill_name:
                try:
                    artifacts.add(
                        manager.get_by_key(
                            "skill",
                            skill_name,
                            treatment.get("activated_skill_version"),
                        ).artifact_id
                    )
                except KeyError:
                    pass
            for artifact_id in artifacts:
                if manager.get(artifact_id).status != "candidate":
                    continue
                manager.record_ab_trial(
                    artifact_id,
                    trial_key=f"{treatment['task_id']}:{artifact_id}",
                    task_key=str(treatment["task_id"]),
                    control_correct=bool(control["passed"]),
                    treatment_correct=bool(treatment["passed"]),
                    control_usage=TokenUsage.from_mapping(control["usage"]),
                    treatment_usage=TokenUsage.from_mapping(treatment["usage"]),
                    metadata={"benchmark": "evolution", "family": treatment["family"]},
                )
        return [
            {
                "artifact": item.__dict__,
                "economics": manager.economics(item.artifact_id),
            }
            for item in manager.list_artifacts()
        ]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("fixed", "evolved", "both"), default="both")
    family_group = parser.add_mutually_exclusive_group()
    family_group.add_argument("--families", type=int, default=None)
    family_group.add_argument(
        "--family",
        choices=tuple(dict.fromkeys(task.family for task in TASKS)),
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/evolution-ab.json"),
    )
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="report task traces and deterministic scores to LANGFUSE_BASE_URL",
    )
    args = parser.parse_args()
    if args.families is not None and not 1 <= args.families <= 10:
        parser.error("--families must be between 1 and 10")

    config = load_config()
    provider = str(config["llm"].get("provider"))
    key_name = str(config["llm"]["api_key_env"])
    api_key = os.environ.get(key_name)
    if provider not in {"openai", "deepseek"}:
        raise SystemExit("llm.provider must be openai or deepseek")
    if not api_key:
        raise SystemExit(f"missing environment variable: {key_name}")
    base_url = config["llm"].get("base_url")
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"

    tasks = _ordered_tasks(args.families, args.family)
    # Schemas are invariant across task workspaces, so one process-level prefix is enough.
    with tempfile.TemporaryDirectory(prefix="evolution-schema-") as schema_root:
        register_all(config, schema_root)
        initialize_stable_prefix(schemas())

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reporter = LangfuseReporter.from_env(
        enabled=args.langfuse,
        run_name=output.stem,
    )
    generated_work_root = args.work_root is None
    work_root = (
        Path(tempfile.mkdtemp(prefix="myagent-evolution-"))
        if generated_work_root
        else args.work_root.resolve()
    )
    work_root.mkdir(parents=True, exist_ok=True)

    arm_names = ("fixed", "evolved") if args.arm == "both" else (args.arm,)
    arms: dict[str, dict] = {}
    try:
        for arm in arm_names:
            meter = MeteredClient(
                OpenAIClient(
                    model=str(config["llm"]["model"]),
                    api_key=api_key,
                    base_url=base_url,
                ),
                reporter,
            )
            arms[arm] = _run_arm(
                arm,
                tasks,
                work_root,
                config,
                meter,
                reporter,
            )
        artifact_lifecycle = _record_artifact_trials(arms, work_root, config)
        payload = {
            "dataset": {
                "families": len({task.family for task in tasks}),
                "tasks": len(tasks),
                "learning_tasks": sum(task.stage == "learn" for task in tasks),
                "transfer_tasks": sum(task.stage == "transfer" for task in tasks),
            },
            "provider": provider,
            "model": config["llm"]["model"],
            "arms": arms,
            "comparison": _comparison(arms),
            "artifact_lifecycle": artifact_lifecycle,
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
        print(f"report={output}")
    finally:
        if reporter is not None:
            reporter.flush()
        if generated_work_root:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
