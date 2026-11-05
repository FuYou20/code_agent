"""Materialize and grade evolution benchmark tasks without leaking hidden tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from .dataset import EvolutionTask


@dataclass(frozen=True)
class TestResult:
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def materialize(task: EvolutionTask, workspace: Path, *, reference: bool = False) -> None:
    """Create the Agent-visible task workspace.

    Hidden tests and the reference implementation are deliberately excluded.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    code = task.fixed_code if reference else task.buggy_code
    (workspace / task.filename).write_text(code, encoding="utf-8")
    (workspace / "test_public.py").write_text(task.public_tests, encoding="utf-8")
    (workspace / "TASK.md").write_text(task.prompt + "\n", encoding="utf-8")


def _run_pytest(workspace: Path, paths: list[str], timeout: int = 30) -> TestResult:
    environment = dict(os.environ)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(workspace) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
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
        timeout=timeout,
        check=False,
    )
    output = "\n".join((completed.stdout + completed.stderr).splitlines()[-100:])
    return TestResult(completed.returncode, output)


def run_public_tests(workspace: Path) -> TestResult:
    return _run_pytest(workspace, ["test_public.py"])


def grade(task: EvolutionTask, workspace: Path) -> TestResult:
    """Inject hidden tests only after the Agent has returned, then remove them."""
    hidden = workspace / "test_evaluator_hidden.py"
    hidden.write_text(task.hidden_tests, encoding="utf-8")
    try:
        return _run_pytest(workspace, ["test_public.py", hidden.name])
    finally:
        hidden.unlink(missing_ok=True)
