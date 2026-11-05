"""Materialize and grade isolated long Skill-reuse tasks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from .dataset import LongSkillTask


@dataclass(frozen=True)
class TestResult:
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def materialize(task: LongSkillTask, workspace: Path, *, reference: bool = False) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    files = task.fixed_files if reference else task.buggy_files
    for relative, content in files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (workspace / "test_public.py").write_text(task.public_tests, encoding="utf-8")
    (workspace / "TASK.md").write_text(task.prompt + "\n", encoding="utf-8")


def _pytest(workspace: Path, paths: list[str]) -> TestResult:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--rootdir",
            str(workspace),
            "--confcutdir",
            str(workspace),
            *paths,
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )
    output = "\n".join((completed.stdout + completed.stderr).splitlines()[-100:])
    return TestResult(completed.returncode, output)


def grade(task: LongSkillTask, workspace: Path) -> TestResult:
    hidden = workspace / "test_evaluator_hidden.py"
    hidden.write_text(task.hidden_tests, encoding="utf-8")
    try:
        return _pytest(workspace, ["test_public.py", hidden.name])
    finally:
        hidden.unlink(missing_ok=True)
