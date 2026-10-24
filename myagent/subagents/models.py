"""Define immutable child-agent requests, contexts, and result envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..permissions import PermissionAtom


class AgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.TIMED_OUT,
        }


@dataclass(frozen=True)
class ToolDescription:
    name: str
    description: str


@dataclass(frozen=True)
class SkillDescription:
    name: str
    description: str
    required_atoms: tuple[PermissionAtom, ...] = ()


@dataclass(frozen=True)
class CreateAgentRequest:
    parent_agent_id: str
    task: str
    requested_atoms: tuple[PermissionAtom, ...]
    parent_context_summary: str = ""
    requested_skills: tuple[str, ...] = ()
    agent_id: str | None = None


@dataclass(frozen=True)
class ChildContext:
    agent_id: str
    parent_agent_id: str | None
    inherited_prompt: str
    task_prompt: str
    parent_context_summary: str
    permission_atoms: tuple[PermissionAtom, ...]
    tools: tuple[ToolDescription, ...]
    skills: tuple[SkillDescription, ...]


@dataclass(frozen=True)
class AgentOutput:
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChildResult:
    agent_id: str
    parent_agent_id: str | None
    status: AgentStatus
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    error: str | None = None
