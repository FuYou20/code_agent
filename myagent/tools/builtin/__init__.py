"""Register built-in tools and declare their root permission atoms."""

from __future__ import annotations

from pathlib import Path

from ...permissions.model import PermissionAtom
from ..registry import TOOLS
from . import file_read, file_write, git_commit, git_read, terminal_exec, test_run


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


def default_permission_atoms() -> tuple[PermissionAtom, ...]:
    """Return the root policy for the fixed built-in tool set.

    Command whitelisting and high-risk approval remain execution constraints;
    filesystem authority itself is represented only by these atoms.
    """
    known = {
        "file_read": PermissionAtom.read("file_read", "."),
        "file_write": PermissionAtom.write("file_write", "."),
        "git_read": PermissionAtom.read("git_read", "."),
        "git_commit": PermissionAtom.read_write(
            "git_commit", read=(".",), write=(".",)
        ),
        "terminal_exec": PermissionAtom.read_write(
            "terminal_exec", read=(".",), write=(".",)
        ),
        "test_run": PermissionAtom.read_write(
            "test_run", read=(".",), write=(".",)
        ),
    }
    return tuple(
        known.get(name, PermissionAtom.without_files(name))
        for name in sorted(TOOLS)
    )

