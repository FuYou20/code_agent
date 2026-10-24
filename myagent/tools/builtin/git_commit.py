"""Register the approved Git add-and-commit operation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..registry import register as register_tool


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def register(cwd: Path) -> None:
    def commit(message: str) -> dict:
        added = _run(["git", "add", "-A"], cwd)
        if added.returncode != 0:
            return {
                "error": "git_add_failed",
                "returncode": added.returncode,
                "stderr": added.stderr,
            }
        committed = _run(["git", "commit", "-m", message], cwd)
        if committed.returncode != 0:
            return {
                "error": "git_commit_failed",
                "returncode": committed.returncode,
                "stdout": committed.stdout,
                "stderr": committed.stderr,
            }
        revision = _run(["git", "rev-parse", "HEAD"], cwd)
        return {
            "commit": revision.stdout.strip(),
            "stdout": committed.stdout,
            "stderr": committed.stderr,
        }

    register_tool(
        "git_commit",
        {
            "description": "暂存全部改动并创建 Git 提交",
            "type": "object",
            "properties": {"message": {"type": "string", "minLength": 1}},
            "required": ["message"],
            "additionalProperties": False,
        },
        commit,
        risk="high",
    )

