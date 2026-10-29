"""Define mutable state retained across turns in one CLI session."""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3

from ..llm.types import Message
from ..permissions import PermissionTree, ToolPermissionAdapter
from ..skills.session_state import SkillLoadState


@dataclass
class SessionState:
    session_id: str
    history: list[Message] = field(default_factory=list)
    agent_id: str = ""
    permission_tree: PermissionTree | None = field(default=None, repr=False)
    permission_adapter: ToolPermissionAdapter | None = field(default=None, repr=False)
    step_count: int = 0
    cwd: str = "."
    repo_snapshot: str = ""
    agents_content: str = ""
    skill_index: list[dict] = field(default_factory=list)
    skill_strategy: str = "full_system"
    skill_top_k: int = 5
    skill_candidate_k: int = 20
    skill_activation_min_score: float = 80.0
    skill_max_frozen: int = 8
    skill_warm_turns: int = 3
    skill_idle_seconds: float = 6 * 60 * 60
    skill_load_state: SkillLoadState = field(default_factory=SkillLoadState)
    skill_router_prompt_tokens: int = 0
    skill_cache_epoch_changes: int = 0
    last_evolution_eligible: bool = False
    last_evolution_reasons: tuple[str, ...] = ()
    last_evolution_history: list[Message] = field(default_factory=list, repr=False)
    max_steps: int = 30
    compression_client: object | None = field(default=None, repr=False)
    memory_conn: sqlite3.Connection | None = field(default=None, repr=False)

