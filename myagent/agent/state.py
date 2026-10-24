"""Define mutable state retained across turns in one CLI session."""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from ..llm.types import Message
from ..tools.scope import Scope


@dataclass
class SessionState:
    session_id: str
    history: list[Message] = field(default_factory=list)
    scope: Scope | None = None
    step_count: int = 0
    cwd: str = "."
    repo_snapshot: str = ""
    agents_content: str = ""
    skill_index: list[dict] = field(default_factory=list)
    skill_strategy: str = "full_sidecar"
    skill_top_k: int = 5
    skill_activation_min_score: float = 80.0
    max_steps: int = 30
    compression_client: object | None = field(default=None, repr=False)
    memory_conn: sqlite3.Connection | None = field(default=None, repr=False)

