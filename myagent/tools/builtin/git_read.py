"""Register read-only Git status, diff, and log commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..registry import register as register_tool


def register(cwd: Path) -> None:
    def read(subcommand: str) -> dict:
        if subcommand not in {"status", "diff", "log"}:
            raise ValueError("subcommand must be status, diff, or log")
        args = ["git", subcommand]
        if subcommand == "status":
            args.append("--short")
        elif subcommand == "log":
            args.extend(["--oneline", "-20"])
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    register_tool(
        "git_read",
        {
            "description": "读取 Git status、diff 或最近日志",
            "type": "object",
            "properties": {
                "subcommand": {"type": "string", "enum": ["status", "diff", "log"]}
            },
            "required": ["subcommand"],
            "additionalProperties": False,
        },
        read,
        risk="low",
    )

