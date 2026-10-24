"""Register the workspace-confined text-file writer."""

from __future__ import annotations

from pathlib import Path

from ..registry import register as register_tool


def register(cwd: Path) -> None:
    root = cwd.resolve()

    def write(path: str, content: str) -> dict:
        candidate = Path(path)
        target = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"path escapes cwd: {path}") from exc

        existed = target.exists()
        previous = target.read_text(encoding="utf-8") if existed else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "path": str(target),
            "created": not existed,
            "bytes_written": len(content.encode("utf-8")),
            "summary": {
                "old_lines": len(previous.splitlines()),
                "new_lines": len(content.splitlines()),
                "changed": previous != content,
            },
        }

    register_tool(
        "file_write",
        {
            "description": "在当前工作目录内写入 UTF-8 文本文件",
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        write,
        risk="high",
    )

