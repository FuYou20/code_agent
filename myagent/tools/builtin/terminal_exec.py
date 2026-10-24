"""Register the whitelist-constrained subprocess execution tool."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from ..registry import register as register_tool


def register(cwd: Path, whitelist: list[str]) -> None:
    patterns = [re.compile(pattern) for pattern in whitelist]

    def run(command: str) -> dict:
        if not any(pattern.match(command) for pattern in patterns):
            raise PermissionError("command does not match terminal_whitelist")
        argv = shlex.split(command, posix=os.name != "nt")
        if not argv:
            raise ValueError("command is empty")
        completed = subprocess.run(
            argv,
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
        "terminal_exec",
        {
            "description": "执行配置白名单内的单个命令",
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        run,
        risk="high",
    )

