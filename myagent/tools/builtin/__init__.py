"""Register all six built-in tools and construct their default scope."""

from __future__ import annotations

import re
from pathlib import Path

from ..registry import TOOLS
from ..scope import Scope, ToolRule
from . import file_read, file_write, git_commit, git_read, terminal_exec, test_run


def _is_within(path_value: object, cwd: Path) -> bool:
    if not isinstance(path_value, str):
        return False
    candidate = Path(path_value)
    resolved = (cwd / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(cwd.resolve())
        return True
    except ValueError:
        return False


def register_all(config: dict, cwd: str | Path = ".") -> None:
    """Configure and register the fixed built-in tool set."""
    workdir = Path(cwd).resolve()
    patterns = list(config["tools"]["terminal_whitelist"])
    file_read.register(workdir)
    file_write.register(workdir)
    terminal_exec.register(workdir, patterns)
    git_read.register(workdir)
    git_commit.register(workdir)
    test_run.register(workdir)


def default_scope(config: dict, cwd: str | Path = ".") -> Scope:
    """Allow built-ins while retaining parameter checks and later high-risk approval."""
    workdir = Path(cwd).resolve()
    patterns = [re.compile(pattern) for pattern in config["tools"]["terminal_whitelist"]]

    def command_allowed(params: dict) -> bool:
        command = params.get("command")
        return isinstance(command, str) and any(pattern.match(command) for pattern in patterns)

    rules: dict[str, ToolRule] = {}
    for name in sorted(TOOLS):
        matcher = lambda _params: True
        description = "built-in tool"
        if name == "terminal_exec":
            matcher = command_allowed
            description = "command must match terminal_whitelist"
        elif name == "file_write":
            matcher = lambda params: _is_within(params.get("path"), workdir)
            description = f"path must stay under {workdir}"
        rules[name] = ToolRule(name, matcher, description)
    return Scope(tools=rules)

