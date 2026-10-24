"""Register the fixed pytest test-suite runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..registry import register as register_tool


def register(cwd: Path) -> None:
    def run() -> dict:
        completed = subprocess.run(
            ["pytest", "-q"],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        lines = (completed.stdout + completed.stderr).splitlines()
        return {"returncode": completed.returncode, "output": "\n".join(lines[-100:])}

    register_tool(
        "test_run",
        {
            "description": "运行 pytest -q 并返回最后 100 行",
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        run,
        risk="medium",
    )

