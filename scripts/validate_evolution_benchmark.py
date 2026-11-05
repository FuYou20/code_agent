"""Validate that all evolution tasks fail before and pass after the reference fix."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.evolution import TASKS
from benchmarks.evolution.harness import grade, materialize, run_public_tests


def validate() -> dict[str, int]:
    counts = {"tasks": len(TASKS), "buggy_failures": 0, "reference_passes": 0}
    failures: list[str] = []

    for task in TASKS:
        with tempfile.TemporaryDirectory(prefix=f"buggy-{task.task_id}-") as raw_buggy:
            buggy_workspace = Path(raw_buggy)
            materialize(task, buggy_workspace)
            public = run_public_tests(buggy_workspace)
            if not public.passed:
                counts["buggy_failures"] += 1
            else:
                failures.append(
                    f"{task.task_id}: buggy implementation unexpectedly passed public tests"
                )

        with tempfile.TemporaryDirectory(prefix=f"fixed-{task.task_id}-") as raw_fixed:
            fixed_workspace = Path(raw_fixed)
            materialize(task, fixed_workspace, reference=True)
            result = grade(task, fixed_workspace)
            if result.passed:
                counts["reference_passes"] += 1
            else:
                failures.append(f"{task.task_id}: reference failed\n{result.output}")

    if failures:
        raise SystemExit("\n\n".join(failures))
    return counts


def main() -> None:
    result = validate()
    print(
        f"tasks={result['tasks']} "
        f"buggy_failures={result['buggy_failures']} "
        f"reference_passes={result['reference_passes']}"
    )


if __name__ == "__main__":
    main()
