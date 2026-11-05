"""Run a Langfuse-backed A/B test of procedural Skill reuse on long tasks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from langfuse import Evaluation, Langfuse

from benchmarks.long_skill_reuse import TASKS
from benchmarks.long_skill_reuse.harness import grade, materialize
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
from myagent.observability import LangfuseReporter
from myagent.permissions import PermissionTree, ToolPermissionAdapter
from myagent.skills.distiller import distill
from myagent.skills.manager import load_index
from myagent.tools.builtin import default_permission_atoms, register_all
from myagent.tools.registry import schemas


DATASET_NAME = "myagent-long-skill-reuse-v1"


@dataclass(frozen=True)
class CallEvent:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    tool_calls: tuple[tuple[str, str], ...]


class InstrumentedClient:
    """Trace model calls and retain deterministic planning-cost events."""

    def __init__(self, inner: OpenAIClient, reporter: LangfuseReporter):
        self.inner = inner
        self.reporter = reporter
        self.events: list[CallEvent] = []

    def mark(self) -> int:
        return len(self.events)

    def since(self, mark: int) -> list[CallEvent]:
        return self.events[mark:]

    def chat(self, messages, tools):
        generation = self.reporter.start_generation(
            messages, tools, model=self.inner.model
        )
        started = time.perf_counter()
        try:
            response = self.inner.chat(messages, tools)
        except Exception as exc:
            self.reporter.finish_generation(generation, None, error=exc)
            raise
        elapsed = time.perf_counter() - started
        self.reporter.finish_generation(generation, response)
        calls = tuple(
            (
                call.name,
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for call in response.tool_calls
        )
        self.events.append(
            CallEvent(
                prompt_tokens=int(response.usage.get("prompt_tokens", 0)),
                cached_tokens=int(response.usage.get("cached_tokens", 0)),
                completion_tokens=int(response.usage.get("completion_tokens", 0)),
                elapsed_seconds=elapsed,
                tool_calls=calls,
            )
        )
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


def _event_usage(events: list[CallEvent]) -> dict[str, int]:
    prompt = sum(event.prompt_tokens for event in events)
    cached = sum(event.cached_tokens for event in events)
    return {
        "api_calls": len(events),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_tokens": prompt - cached,
        "completion_tokens": sum(event.completion_tokens for event in events),
        "tool_calls": sum(len(event.tool_calls) for event in events),
    }


def _planning_metrics(events: list[CallEvent]) -> dict[str, float | int | bool]:
    first_write_index = next(
        (
            index
            for index, event in enumerate(events)
            if any(name == "file_write" for name, _ in event.tool_calls)
        ),
        None,
    )
    prewrite = events if first_write_index is None else events[:first_write_index]
    exploration_names = {"file_read", "git_read", "terminal_exec", "test_run"}
    exploration = [
        call
        for event in prewrite
        for call in event.tool_calls
        if call[0] in exploration_names
    ]
    through_first_write = (
        events if first_write_index is None else events[: first_write_index + 1]
    )
    return {
        "first_write_observed": first_write_index is not None,
        "planning_api_calls": len(prewrite),
        "planning_prompt_tokens": sum(event.prompt_tokens for event in prewrite),
        "planning_uncached_tokens": sum(
            event.prompt_tokens - event.cached_tokens for event in prewrite
        ),
        "prewrite_exploration_calls": len(exploration),
        "repeated_prewrite_exploration": len(exploration) - len(set(exploration)),
        "seconds_to_first_write": sum(
            event.elapsed_seconds for event in through_first_write
        ),
    }


def _new_state(
    workspace: Path,
    home: Path,
    config: dict,
    artifact_manager: EvolutionArtifactManager | None,
) -> SessionState:
    register_all(config, workspace)
    agent_id = f"long-eval-{uuid.uuid4().hex}"
    tree = PermissionTree(workspace, default_permission_atoms())
    tree.attach_agent(tree.ROOT_ID, agent_id)
    snapshot = "\n".join(
        sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file()
        )
    )
    return SessionState(
        session_id=uuid.uuid4().hex,
        agent_id=agent_id,
        permission_tree=tree,
        permission_adapter=ToolPermissionAdapter(workspace),
        cwd=str(workspace),
        repo_snapshot=snapshot,
        # Runtime Skills only and no experience memory: this isolates Skill reuse.
        skill_index=load_index(
            home / "skills_store",
            lifecycle=artifact_manager,
            allowed_statuses=("candidate", "canary", "active"),
            trusted_unregistered=artifact_manager is None,
        ),
        skill_strategy="top_k",
        skill_top_k=5,
        skill_candidate_k=20,
        skill_activation_min_score=float(config["skills"]["activation_min_score"]),
        max_steps=int(config["agent"]["max_steps"]),
        memory_conn=None,
        memory_scope="",
        artifact_manager=artifact_manager,
        evolution_artifact_statuses=("candidate", "canary", "active"),
    )


def _ensure_dataset(client: Langfuse) -> None:
    try:
        dataset = client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "Multi-file learn/transfer benchmark for testing whether a "
                "distilled procedural Skill reduces repeated planning."
            ),
            metadata={"benchmark_version": 1, "hidden_tests": True},
        )
        dataset = client.get_dataset(DATASET_NAME)

    existing_task_ids = {
        str(item.input.get("task_id"))
        for item in dataset.items
        if isinstance(item.input, dict) and item.input.get("task_id")
    }

    namespace = uuid.UUID("62f39cc9-3273-4bfa-bbc3-a1c7d563c80a")
    for order, task in enumerate(TASKS):
        if task.task_id in existing_task_ids:
            continue
        client.create_dataset_item(
            id=str(uuid.uuid5(namespace, task.task_id)),
            dataset_name=DATASET_NAME,
            input={"task_id": task.task_id, "prompt": task.prompt},
            expected_output={"hidden_tests_pass": True},
            metadata={
                "order": order,
                "family": task.family,
                "stage": task.stage,
                "files": len(task.buggy_files),
            },
        )


def _evaluations(*, output, **_kwargs):
    numeric = (
        "prompt_tokens",
        "uncached_tokens",
        "tool_calls",
        "planning_api_calls",
        "planning_prompt_tokens",
        "planning_uncached_tokens",
        "prewrite_exploration_calls",
        "repeated_prewrite_exploration",
        "seconds_to_first_write",
        "wall_seconds",
    )
    values = [
        Evaluation(
            name="task_passed",
            value=bool(output["passed"]),
            data_type="BOOLEAN",
        ),
        Evaluation(
            name="skill_reused",
            value=bool(output["skill_reused"]),
            data_type="BOOLEAN",
        ),
    ]
    values.extend(Evaluation(name=name, value=float(output[name])) for name in numeric)
    return values


class ArmRunner:
    def __init__(self, *, arm, home, tasks_root, config, client):
        self.arm = arm
        self.home = home
        self.tasks_root = tasks_root
        self.config = config
        self.client = client
        self.by_id = {task.task_id: task for task in TASKS}
        home.mkdir(parents=True, exist_ok=True)
        tasks_root.mkdir(parents=True, exist_ok=True)
        self.lifecycle_conn = init_db(home / "evolution.db") if arm == "evolved" else None
        if self.lifecycle_conn is None:
            self.artifact_manager = None
        else:
            values = config["evolution"]
            self.artifact_manager = EvolutionArtifactManager(
                self.lifecycle_conn,
                token_weights=TokenWeights(
                    uncached_input=float(values["uncached_input_weight"]),
                    cached_input=float(values["cached_input_weight"]),
                    output=float(values["output_weight"]),
                ),
                min_canary_trials=int(values["min_canary_trials"]),
                negative_streak=int(values["negative_streak"]),
            )

    def __call__(self, *, item, **_kwargs):
        task = self.by_id[str(item.input["task_id"])]
        workspace = self.tasks_root / task.task_id
        if workspace.exists():
            shutil.rmtree(workspace)
        materialize(task, workspace)
        state = _new_state(
            workspace,
            self.home,
            self.config,
            self.artifact_manager,
        )
        mark = self.client.mark()
        started = time.perf_counter()
        error = ""
        answer = ""
        with _agent_home(self.home):
            try:
                answer = run_turn(state, task.prompt, self.client, lambda _call: True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        execution_events = self.client.since(mark)
        wall_seconds = time.perf_counter() - started
        result = grade(task, workspace)
        row: dict[str, object] = {
            "task_id": task.task_id,
            "family": task.family,
            "stage": task.stage,
            "arm": self.arm,
            "passed": result.passed,
            "error": error,
            "answer_chars": len(answer),
            "grade_output": result.output,
            "wall_seconds": wall_seconds,
            **_event_usage(execution_events),
            **_planning_metrics(execution_events),
            "skill_reused": bool(state.last_activated_skill_name),
            "activated_skill": state.last_activated_skill_name,
            "activated_skill_version": state.last_activated_skill_version,
            "evolution_eligible": state.last_evolution_eligible,
            "evolution_reasons": list(state.last_evolution_reasons),
            "distilled_skill": "",
            "distillation_usage": {},
        }

        if (
            self.arm == "evolved"
            and task.stage == "learn"
            and result.passed
            and state.last_evolution_eligible
        ):
            distill_mark = self.client.mark()
            try:
                with _agent_home(self.home):
                    skill = distill(
                        state.last_evolution_history,
                        self.client,
                        task_context=task.prompt,
                        artifact_manager=self.artifact_manager,
                        scope_key=task.family,
                    )
                row["distilled_skill"] = skill.frontmatter.name
            except Exception as exc:
                prefix = str(row["error"]) + "; " if row["error"] else ""
                row["error"] = prefix + f"distill:{type(exc).__name__}: {exc}"
            row["distillation_usage"] = _event_usage(
                self.client.since(distill_mark)
            )

        print(
            f"arm={self.arm} task={task.task_id} passed={result.passed} "
            f"skill={row['activated_skill'] or '-'} "
            f"planning_tokens={row['planning_prompt_tokens']}"
        )
        return row

    def close(self) -> None:
        if self.lifecycle_conn is not None:
            self.lifecycle_conn.close()


def _record_long_trials(
    runner: ArmRunner,
    fixed_rows: list[dict],
    evolved_rows: list[dict],
) -> list[dict[str, object]]:
    manager = runner.artifact_manager
    if manager is None:
        return []
    controls = {row["task_id"]: row for row in fixed_rows}
    for treatment in evolved_rows:
        if treatment["stage"] != "transfer" or not treatment["activated_skill"]:
            continue
        control = controls[treatment["task_id"]]
        artifact = manager.get_by_key(
            "skill",
            str(treatment["activated_skill"]),
            treatment.get("activated_skill_version"),
        )
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=str(treatment["task_id"]),
            task_key=str(treatment["task_id"]),
            control_correct=bool(control["passed"]),
            treatment_correct=bool(treatment["passed"]),
            control_usage=TokenUsage(
                prompt_tokens=int(control["prompt_tokens"]),
                cached_tokens=int(control["cached_tokens"]),
                completion_tokens=int(control["completion_tokens"]),
            ),
            treatment_usage=TokenUsage(
                prompt_tokens=int(treatment["prompt_tokens"]),
                cached_tokens=int(treatment["cached_tokens"]),
                completion_tokens=int(treatment["completion_tokens"]),
            ),
            metadata={"benchmark": "long-skill-reuse"},
        )
    return [
        {
            "artifact": item.__dict__,
            "economics": manager.economics(item.artifact_id),
        }
        for item in manager.list_artifacts()
    ]


def _summarize(rows: list[dict]) -> dict[str, float | int]:
    transfer = [row for row in rows if row["stage"] == "transfer"]
    keys = (
        "prompt_tokens",
        "cached_tokens",
        "uncached_tokens",
        "completion_tokens",
        "tool_calls",
        "planning_api_calls",
        "planning_prompt_tokens",
        "planning_uncached_tokens",
        "prewrite_exploration_calls",
        "repeated_prewrite_exploration",
        "seconds_to_first_write",
        "wall_seconds",
    )
    passed = sum(bool(row["passed"]) for row in transfer)
    skill_hits = sum(bool(row["skill_reused"]) for row in transfer)
    summary: dict[str, float | int] = {
        "transfer_tasks": len(transfer),
        "passed": passed,
        "success_rate": passed / len(transfer) if transfer else 0.0,
        "skill_hits": skill_hits,
        "skill_hit_rate": skill_hits / len(transfer) if transfer else 0.0,
    }
    for key in keys:
        summary[key] = sum(float(row[key]) for row in transfer)
    return summary


def _comparison(fixed: dict, evolved: dict) -> dict[str, float | None]:
    keys = (
        "prompt_tokens",
        "uncached_tokens",
        "tool_calls",
        "planning_api_calls",
        "planning_prompt_tokens",
        "planning_uncached_tokens",
        "prewrite_exploration_calls",
        "seconds_to_first_write",
        "wall_seconds",
    )
    result: dict[str, float | None] = {
        "success_rate_change_pp": 100
        * (float(evolved["success_rate"]) - float(fixed["success_rate"])),
        "skill_hit_rate": float(evolved["skill_hit_rate"]),
    }
    for key in keys:
        base = float(fixed[key])
        result[f"{key}_reduction"] = 1 - float(evolved[key]) / base if base else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/long-skill-reuse-langfuse.json"),
    )
    parser.add_argument("--work-root", type=Path, default=None)
    args = parser.parse_args()

    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("missing Langfuse variables: " + ", ".join(missing))
    config = load_config()
    key_name = str(config["llm"]["api_key_env"])
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"missing model variable: {key_name}")

    langfuse = Langfuse()
    if not langfuse.auth_check():
        raise SystemExit("Langfuse authentication failed")
    _ensure_dataset(langfuse)
    dataset = langfuse.get_dataset(DATASET_NAME)
    dataset.items.sort(key=lambda item: int((item.metadata or {}).get("order", 999)))

    with tempfile.TemporaryDirectory(prefix="long-reuse-schema-") as schema_root:
        register_all(config, schema_root)
        initialize_stable_prefix(schemas())

    generated_root = args.work_root is None
    work_root = (
        Path(tempfile.mkdtemp(prefix="myagent-long-reuse-"))
        if generated_root
        else args.work_root.resolve()
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    arms: dict[str, dict] = {}
    artifact_lifecycle: list[dict[str, object]] = []
    try:
        for arm in ("fixed", "evolved"):
            reporter = LangfuseReporter(
                langfuse,
                run_name=f"long-skill-reuse-{arm}",
                inherit_current_context=True,
            )
            instrumented = InstrumentedClient(
                OpenAIClient(
                    model=str(config["llm"]["model"]),
                    api_key=api_key,
                    base_url=config["llm"].get("base_url"),
                ),
                reporter,
            )
            runner = ArmRunner(
                arm=arm,
                home=work_root / arm / "home",
                tasks_root=work_root / arm / "tasks",
                config=config,
                client=instrumented,
            )
            experiment = dataset.run_experiment(
                name=f"MyAgent long Skill reuse ({arm})",
                run_name=f"long-skill-reuse-{run_stamp}-{arm}",
                description=(
                    "Fresh-session transfer test. Evolved arm may distill and "
                    "reuse one procedural Skill; memory retrieval is disabled."
                ),
                task=runner,
                evaluators=[_evaluations],
                max_concurrency=1,
                metadata={
                    "arm": arm,
                    "model": str(config["llm"]["model"]),
                    "treatment": "procedural-skill-only",
                },
            )
            rows = [item.output for item in experiment.item_results]
            arms[arm] = {
                "dataset_run_id": experiment.dataset_run_id,
                "dataset_run_url": experiment.dataset_run_url,
                "summary": _summarize(rows),
                "tasks": rows,
            }
            if arm == "evolved":
                artifact_lifecycle = _record_long_trials(
                    runner,
                    arms["fixed"]["tasks"],
                    rows,
                )
            runner.close()
            langfuse.flush()

        payload = {
            "dataset": DATASET_NAME,
            "model": config["llm"]["model"],
            "method": {
                "fresh_session_per_task": True,
                "memory_retrieval": False,
                "hidden_tests": True,
                "planning_definition": "provider calls before first file_write",
            },
            "arms": arms,
            "comparison": _comparison(
                arms["fixed"]["summary"], arms["evolved"]["summary"]
            ),
            "artifact_lifecycle": artifact_lifecycle,
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
        print(f"fixed_url={arms['fixed']['dataset_run_url']}")
        print(f"evolved_url={arms['evolved']['dataset_run_url']}")
        print(f"report={output}")
    finally:
        langfuse.flush()
        if generated_root:
            shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
