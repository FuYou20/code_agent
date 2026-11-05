"""Sync and execute the 21 permission-governance cases in Langfuse."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.permissions import CASES, run_case


DATASET_NAME = "myagent-permission-governance-v1"


def _load_env(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ensure_dataset(client) -> None:
    try:
        dataset = client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "Deterministic checks for MyAgent atomic permission delegation, "
                "runtime authorization, child isolation, revocation, and distillation governance."
            ),
            metadata={
                "benchmark_version": 1,
                "case_count": len(CASES),
                "uses_llm": False,
                "truth_source": "pytest",
            },
        )
        dataset = client.get_dataset(DATASET_NAME)

    existing = {
        str(item.input.get("case_id"))
        for item in dataset.items
        if isinstance(item.input, dict) and item.input.get("case_id")
    }
    namespace = uuid.UUID("700d4b31-a862-4f4d-92ac-67cb045697f0")
    for order, case in enumerate(CASES):
        if case.case_id in existing:
            continue
        client.create_dataset_item(
            id=str(uuid.uuid5(namespace, case.case_id)),
            dataset_name=DATASET_NAME,
            input={
                "case_id": case.case_id,
                "scenario": case.scenario,
                "pytest_node_id": case.node_id,
            },
            expected_output={
                "decision": case.expected_decision,
                "check_passes": True,
            },
            metadata={
                "order": order,
                "category": case.category,
                "deterministic": True,
                "uses_llm": False,
            },
        )


def _evaluations(*, output, **_kwargs):
    from langfuse import Evaluation

    return [
        Evaluation(
            name="permission_check_passed",
            value=bool(output["passed"]),
            data_type="BOOLEAN",
        ),
        Evaluation(
            name="decision_matches_expected",
            value=output["actual_decision"] == output["expected_decision"],
            data_type="BOOLEAN",
        ),
        Evaluation(
            name="duration_seconds",
            value=float(output["duration_seconds"]),
            data_type="NUMERIC",
        ),
        Evaluation(
            name="permission_event_count",
            value=float(output["permission_event_count"]),
            data_type="NUMERIC",
        ),
    ]


class PermissionRunner:
    def __init__(self, python: str, client) -> None:
        self.python = python
        self.client = client
        self.by_id = {case.case_id: case for case in CASES}

    def __call__(self, *, item, **_kwargs):
        case_id = str(item.input["case_id"])
        result = run_case(self.by_id[case_id], ROOT, python=self.python)
        for event in result["permission_events"]:
            observation = self.client.start_observation(
                name=str(event["name"]),
                as_type="span",
                input=event["data"],
                metadata={
                    "sequence": int(event["sequence"]),
                    "thread_id": int(event["thread_id"]),
                    "permission_case_id": case_id,
                },
            )
            observation.update(output={"recorded": True})
            observation.end()
        print(
            f"case={case_id} category={result['category']} "
            f"passed={result['passed']} duration={result['duration_seconds']}s",
            flush=True,
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/permission-langfuse.json"),
    )
    args = parser.parse_args()
    _load_env(args.env_file)

    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("missing Langfuse variables: " + ", ".join(missing))
    try:
        from langfuse import Langfuse
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Langfuse SDK is unavailable; install with: pip install -e .[observability]"
        ) from exc

    client = Langfuse()
    if not client.auth_check():
        raise SystemExit("Langfuse authentication failed")
    _ensure_dataset(client)
    dataset = client.get_dataset(DATASET_NAME)
    dataset.items.sort(key=lambda item: int((item.metadata or {}).get("order", 999)))

    payload: dict[str, object] = {
        "dataset": DATASET_NAME,
        "items": len(dataset.items),
        "categories": sorted({case.category for case in CASES}),
    }
    if not args.dataset_only:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        experiment = dataset.run_experiment(
            name="MyAgent permission governance",
            run_name=f"permission-governance-{stamp}",
            description=(
                "Runs the original deterministic permission tests and records "
                "one Langfuse trace and business score set per case."
            ),
            task=PermissionRunner(args.python, client),
            evaluators=[_evaluations],
            max_concurrency=1,
            metadata={
                "benchmark": "permission-governance",
                "cases": len(CASES),
                "uses_llm": False,
            },
        )
        rows = [item.output for item in experiment.item_results]
        payload.update(
            {
                "dataset_run_id": experiment.dataset_run_id,
                "dataset_run_url": experiment.dataset_run_url,
                "passed": sum(bool(row["passed"]) for row in rows),
                "failed": sum(not bool(row["passed"]) for row in rows),
                "results": rows,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    client.flush()
    print(json.dumps({key: payload[key] for key in payload if key != "results"}, ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
