"""Register the bounded UTF-8 text-file reader."""

from __future__ import annotations

from pathlib import Path

from ..registry import register as register_tool

MAX_BYTES = 50 * 1024


def register(cwd: Path) -> None:
    def read(path: str) -> dict:
        candidate = Path(path)
        target = (cwd / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        data = target.read_bytes()
        truncated = len(data) > MAX_BYTES
        content = data[:MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            content += f"\n\n[内容已截断：原文件 {len(data)} 字节，仅显示前 {MAX_BYTES} 字节]"
        return {"path": str(target), "content": content, "truncated": truncated}

    register_tool(
        "file_read",
        {
            "description": "读取 UTF-8 文本文件，最多返回 50KB",
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        read,
        risk="low",
    )

