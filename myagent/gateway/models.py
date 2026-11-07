"""Stable task and result contracts shared by external Agent channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskSource = Literal["feishu", "cli", "tui", "scheduler", "api"]


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    session_id: str
    source: TaskSource
    user_id: str
    content: str
    reply_target: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "session_id", "source", "content", "reply_target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AgentTask.{name} must be a non-empty string")


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    session_id: str
    content: str = ""
    success: bool = True
    duplicate: bool = False
    error_type: str = ""
    error_message: str = ""
