"""Define indivisible tool-and-file capability atoms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import posixpath
import re


AccessMode = Literal["read", "write"]
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:/")


def normalize_logical_path(value: str) -> str:
    """Return a canonical workspace-relative path or reject an escaping path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("permission path must be a non-empty string")
    raw = value.strip().replace("\\", "/")
    if "\x00" in raw or raw.startswith("/") or _WINDOWS_ABSOLUTE.match(raw):
        raise ValueError(f"permission path must be workspace-relative: {value!r}")
    normalized = posixpath.normpath(raw)
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"permission path escapes workspace: {value!r}")
    return "." if normalized in {"", "."} else normalized


@dataclass(frozen=True, order=True)
class FileGrant:
    """One read or write subtree bound to the enclosing tool capability."""

    mode: AccessMode
    root: str

    def __post_init__(self) -> None:
        if self.mode not in {"read", "write"}:
            raise ValueError(f"unsupported file access mode: {self.mode}")
        object.__setattr__(self, "root", normalize_logical_path(self.root))


@dataclass(frozen=True)
class PermissionAtom:
    """An indivisible tool permission and its complete filesystem footprint."""

    tool: str
    files: tuple[FileGrant, ...]

    def __post_init__(self) -> None:
        name = self.tool.strip() if isinstance(self.tool, str) else ""
        if not name:
            raise ValueError("permission atom tool must be a non-empty string")
        canonical_files = tuple(sorted(set(self.files)))
        object.__setattr__(self, "tool", name)
        object.__setattr__(self, "files", canonical_files)

    @classmethod
    def read(cls, tool: str, *roots: str) -> "PermissionAtom":
        return cls(tool=tool, files=tuple(FileGrant("read", root) for root in roots))

    @classmethod
    def write(cls, tool: str, *roots: str) -> "PermissionAtom":
        return cls(tool=tool, files=tuple(FileGrant("write", root) for root in roots))

    @classmethod
    def read_write(
        cls,
        tool: str,
        *,
        read: tuple[str, ...] = (),
        write: tuple[str, ...] = (),
    ) -> "PermissionAtom":
        return cls(
            tool=tool,
            files=tuple(FileGrant("read", root) for root in read)
            + tuple(FileGrant("write", root) for root in write),
        )

    @classmethod
    def without_files(cls, tool: str) -> "PermissionAtom":
        """Represent a tool whose trusted adapter declares no file effects."""
        return cls(tool=tool, files=())
