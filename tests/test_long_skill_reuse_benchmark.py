"""Validate the long Skill-reuse benchmark fixtures."""

from pathlib import Path

from benchmarks.long_skill_reuse import TASKS
from benchmarks.long_skill_reuse.harness import grade, materialize


def test_reference_implementations_pass_hidden_contracts(tmp_path: Path) -> None:
    for task in TASKS:
        workspace = tmp_path / task.task_id
        materialize(task, workspace, reference=True)
        result = grade(task, workspace)
        assert result.passed, f"{task.task_id}: {result.output}"


def test_buggy_implementations_fail_contracts(tmp_path: Path) -> None:
    for task in TASKS:
        workspace = tmp_path / task.task_id
        materialize(task, workspace)
        result = grade(task, workspace)
        assert not result.passed, task.task_id
