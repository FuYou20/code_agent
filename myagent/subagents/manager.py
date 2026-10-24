"""Create, mount, run, cancel, and unload permission-bound child agents."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Callable
import uuid

from ..permissions import PermissionAtom, PermissionTree, UnknownAgent
from .budget import BudgetReservation, GlobalBudgetLedger
from .models import (
    AgentOutput,
    AgentStatus,
    ChildContext,
    ChildResult,
    CreateAgentRequest,
    SkillDescription,
    ToolDescription,
)


class UnknownManagedAgent(KeyError):
    """Raised when the manager does not own the requested agent."""


class InvalidAgentState(Exception):
    """Raised when a lifecycle operation is invalid for the current status."""


class ContextVisibilityDenied(Exception):
    """Raised when a child requests a Skill unavailable to its parent or policy."""


@dataclass
class _AgentRecord:
    agent_id: str
    parent_agent_id: str | None
    node_id: str
    context: ChildContext
    status: AgentStatus = AgentStatus.CREATED
    children: set[str] = field(default_factory=set)
    cancel_event: Event = field(default_factory=Event)
    done_event: Event = field(default_factory=Event)
    result: ChildResult | None = None
    future: Future | None = None
    mounted: bool = True


class AgentExecution:
    """The narrow interface supplied to a child worker instead of the manager."""

    def __init__(self, manager: "AgentManager", agent_id: str) -> None:
        self._manager = manager
        self.agent_id = agent_id

    @property
    def context(self) -> ChildContext:
        return self._manager.context(self.agent_id)

    def is_cancelled(self) -> bool:
        return self._manager.is_cancelled(self.agent_id)

    def wait_cancelled(self, timeout: float | None = None) -> bool:
        return self._manager._cancel_event(self.agent_id).wait(timeout)

    def allows(self, requested_atom: PermissionAtom) -> bool:
        return self._manager.allows(self.agent_id, requested_atom)

    def reserve_budget(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> BudgetReservation:
        return self._manager._reserve_budget(
            self.agent_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )

    def settle_budget(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
    ) -> None:
        self._manager._settle_budget(
            self.agent_id,
            reservation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )

    def release_budget(self, reservation_id: str) -> None:
        self._manager._release_budget(self.agent_id, reservation_id)


Worker = Callable[[AgentExecution], AgentOutput | str]


class AgentManager:
    """Own child contexts and ensure results survive permission-node unloading."""

    def __init__(
        self,
        permission_tree: PermissionTree,
        budget: GlobalBudgetLedger,
        *,
        max_workers: int = 4,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        self.permission_tree = permission_tree
        self.budget = budget
        self._records: dict[str, _AgentRecord] = {}
        self._mailboxes: dict[str, list[ChildResult]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="myagent-child",
        )
        self._lock = RLock()

    def _record(self, agent_id: str) -> _AgentRecord:
        record = self._records.get(agent_id)
        if record is None:
            raise UnknownManagedAgent(agent_id)
        return record

    @staticmethod
    def _tool_names(atoms: tuple[PermissionAtom, ...]) -> set[str]:
        return {atom.tool for atom in atoms}

    def _skill_compatible(
        self,
        skill: SkillDescription,
        policy: tuple[PermissionAtom, ...],
    ) -> bool:
        return all(
            any(
                self.permission_tree.atom_intersects(policy_atom, required_atom)
                for policy_atom in policy
            )
            for required_atom in skill.required_atoms
        )

    def register_root_agent(
        self,
        agent_id: str,
        *,
        prompt: str,
        tools: tuple[ToolDescription, ...],
        skills: tuple[SkillDescription, ...] = (),
    ) -> None:
        """Register a long-lived parent agent on the permanent root node."""
        with self._lock:
            if agent_id in self._records:
                raise ValueError(f"agent already registered: {agent_id}")
            self.permission_tree.attach_agent(self.permission_tree.ROOT_ID, agent_id)
            policy = self.permission_tree.policy_for_agent(agent_id)
            visible_tools = tuple(
                tool for tool in tools if tool.name in self._tool_names(policy)
            )
            visible_skills = tuple(
                skill for skill in skills if self._skill_compatible(skill, policy)
            )
            context = ChildContext(
                agent_id=agent_id,
                parent_agent_id=None,
                inherited_prompt=prompt,
                task_prompt="",
                parent_context_summary="",
                permission_atoms=policy,
                tools=visible_tools,
                skills=visible_skills,
            )
            self._records[agent_id] = _AgentRecord(
                agent_id=agent_id,
                parent_agent_id=None,
                node_id=self.permission_tree.ROOT_ID,
                context=context,
                status=AgentStatus.RUNNING,
            )

    def create_agent(self, request: CreateAgentRequest) -> str:
        """Create and mount a child after permission and context filtering."""
        with self._lock:
            parent = self._record(request.parent_agent_id)
            if parent.status.terminal:
                raise InvalidAgentState("cannot create a child from a terminal parent")
            if not request.task.strip():
                raise ValueError("child task must be non-empty")
            agent_id = request.agent_id or f"agent-{uuid.uuid4().hex}"
            if agent_id in self._records:
                raise ValueError(f"agent already registered: {agent_id}")

            node_id = self.permission_tree.get_or_create_child(
                parent.node_id,
                request.requested_atoms,
            )
            self.permission_tree.attach_agent(node_id, agent_id)
            policy = self.permission_tree.policy_for_agent(agent_id)
            tool_names = self._tool_names(policy)
            tools = tuple(
                tool for tool in parent.context.tools if tool.name in tool_names
            )

            parent_skills = {skill.name: skill for skill in parent.context.skills}
            explicitly_requested = bool(request.requested_skills)
            requested_names = set(request.requested_skills) if explicitly_requested else {
                name
                for name, skill in parent_skills.items()
                if self._skill_compatible(skill, policy)
            }
            unknown = requested_names - set(parent_skills)
            if unknown:
                self.permission_tree.detach_agent(agent_id)
                names = ", ".join(sorted(unknown))
                raise ContextVisibilityDenied(f"parent cannot delegate Skills: {names}")
            incompatible = (
                [
                    parent_skills[name]
                    for name in sorted(requested_names)
                    if not self._skill_compatible(parent_skills[name], policy)
                ]
                if explicitly_requested
                else []
            )
            if incompatible:
                self.permission_tree.detach_agent(agent_id)
                names = ", ".join(skill.name for skill in incompatible)
                raise ContextVisibilityDenied(
                    f"Skills require atoms absent from child policy: {names}"
                )
            skills = tuple(parent_skills[name] for name in sorted(requested_names))
            context = ChildContext(
                agent_id=agent_id,
                parent_agent_id=parent.agent_id,
                inherited_prompt=parent.context.inherited_prompt,
                task_prompt=request.task,
                parent_context_summary=request.parent_context_summary,
                permission_atoms=policy,
                tools=tools,
                skills=skills,
            )
            self._records[agent_id] = _AgentRecord(
                agent_id=agent_id,
                parent_agent_id=parent.agent_id,
                node_id=node_id,
                context=context,
            )
            parent.children.add(agent_id)
            return agent_id

    def context(self, agent_id: str) -> ChildContext:
        with self._lock:
            return self._record(agent_id).context

    def status(self, agent_id: str) -> AgentStatus:
        with self._lock:
            return self._record(agent_id).status

    def _cancel_event(self, agent_id: str) -> Event:
        with self._lock:
            return self._record(agent_id).cancel_event

    def is_cancelled(self, agent_id: str) -> bool:
        with self._lock:
            return self._record(agent_id).cancel_event.is_set()

    def allows(self, agent_id: str, requested_atom: PermissionAtom) -> bool:
        return self.permission_tree.allows(agent_id, requested_atom)

    def _reserve_budget(
        self,
        agent_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
    ) -> BudgetReservation:
        with self._lock:
            record = self._record(agent_id)
            if record.status != AgentStatus.RUNNING or record.cancel_event.is_set():
                raise InvalidAgentState("only a running agent may reserve budget")
            return self.budget.reserve(
                agent_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )

    def _settle_budget(
        self,
        agent_id: str,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
    ) -> None:
        with self._lock:
            record = self._record(agent_id)
            if record.status != AgentStatus.RUNNING or record.cancel_event.is_set():
                raise InvalidAgentState("only a running agent may settle budget")
            self.budget.settle(
                reservation_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )

    def _release_budget(self, agent_id: str, reservation_id: str) -> None:
        with self._lock:
            record = self._record(agent_id)
            if record.status != AgentStatus.RUNNING or record.cancel_event.is_set():
                raise InvalidAgentState("only a running agent may release budget")
            self.budget.release(reservation_id)

    @staticmethod
    def _normalise_output(output: AgentOutput | str) -> AgentOutput:
        return output if isinstance(output, AgentOutput) else AgentOutput(summary=str(output))

    def _start(self, agent_id: str) -> _AgentRecord:
        with self._lock:
            record = self._record(agent_id)
            if record.status != AgentStatus.CREATED:
                raise InvalidAgentState(
                    f"agent {agent_id} cannot start from {record.status.value}"
                )
            record.status = AgentStatus.RUNNING
            return record

    def _deliver(self, record: _AgentRecord, result: ChildResult) -> None:
        record.result = result
        if record.parent_agent_id is not None:
            self._mailboxes.setdefault(record.parent_agent_id, []).append(result)
            parent = self._records.get(record.parent_agent_id)
            if parent is not None:
                parent.children.discard(record.agent_id)
        record.done_event.set()

    def _unmount(self, record: _AgentRecord) -> None:
        if not record.mounted:
            return
        self.budget.release_for_agent(record.agent_id)
        try:
            self.permission_tree.detach_agent(record.agent_id)
        except UnknownAgent:
            pass
        record.mounted = False

    def _cancel_children(self, record: _AgentRecord, reason: str) -> None:
        for child_id in sorted(tuple(record.children)):
            self._cancel_locked(child_id, AgentStatus.CANCELLED, reason)

    def _finish_success(self, agent_id: str, output: AgentOutput) -> ChildResult:
        with self._lock:
            record = self._record(agent_id)
            if record.status.terminal:
                assert record.result is not None
                return record.result
            self._cancel_children(record, "parent completed before child")
            record.status = AgentStatus.SUCCEEDED
            result = ChildResult(
                agent_id=record.agent_id,
                parent_agent_id=record.parent_agent_id,
                status=record.status,
                summary=output.summary,
                data=output.data,
                artifacts=output.artifacts,
                unresolved=output.unresolved,
            )
            self._deliver(record, result)
            self._unmount(record)
            return result

    def _finish_failure(self, agent_id: str, exc: Exception) -> ChildResult:
        with self._lock:
            record = self._record(agent_id)
            if record.status.terminal:
                assert record.result is not None
                return record.result
            self._cancel_children(record, "parent failed")
            record.status = AgentStatus.FAILED
            result = ChildResult(
                agent_id=record.agent_id,
                parent_agent_id=record.parent_agent_id,
                status=record.status,
                summary="child agent failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self._deliver(record, result)
            self._unmount(record)
            return result

    def _execute(self, agent_id: str, worker: Worker) -> ChildResult:
        try:
            output = self._normalise_output(worker(AgentExecution(self, agent_id)))
        except Exception as exc:
            return self._finish_failure(agent_id, exc)
        return self._finish_success(agent_id, output)

    def run_agent(self, agent_id: str, worker: Worker) -> ChildResult:
        """Run one child synchronously and unload it after result persistence."""
        self._start(agent_id)
        return self._execute(agent_id, worker)

    def start_agent(self, agent_id: str, worker: Worker) -> Future:
        """Start one child asynchronously in the shared worker pool."""
        record = self._start(agent_id)
        future = self._executor.submit(self._execute, agent_id, worker)
        with self._lock:
            record.future = future
        return future

    def _cancel_locked(
        self,
        agent_id: str,
        status: AgentStatus,
        reason: str,
        *,
        already_unmounted: bool = False,
    ) -> ChildResult:
        record = self._record(agent_id)
        if record.status.terminal:
            assert record.result is not None
            return record.result
        self._cancel_children(record, reason)
        record.cancel_event.set()
        if record.future is not None:
            record.future.cancel()
        record.status = status
        result = ChildResult(
            agent_id=record.agent_id,
            parent_agent_id=record.parent_agent_id,
            status=status,
            summary=reason,
            error=reason,
        )
        self._deliver(record, result)
        if already_unmounted:
            self.budget.release_for_agent(record.agent_id)
            record.mounted = False
        else:
            self._unmount(record)
        return result

    def cancel_agent(self, agent_id: str, reason: str = "cancelled by manager") -> ChildResult:
        with self._lock:
            return self._cancel_locked(agent_id, AgentStatus.CANCELLED, reason)

    def wait(self, agent_id: str, timeout: float | None = None) -> ChildResult:
        """Wait for completion; timeout cancels and unloads the entire child subtree."""
        record = self._record(agent_id)
        if not record.done_event.wait(timeout):
            with self._lock:
                return self._cancel_locked(
                    agent_id,
                    AgentStatus.TIMED_OUT,
                    "child agent timed out",
                )
        with self._lock:
            result = self._record(agent_id).result
            assert result is not None
            return result

    def revoke_permission_subtree(self, node_id: str) -> tuple[ChildResult, ...]:
        """Revoke a permission subtree, then cancel every affected managed agent."""
        affected = self.permission_tree.revoke_subtree(node_id)
        results: list[ChildResult] = []
        with self._lock:
            for agent_id in affected:
                if agent_id not in self._records:
                    continue
                results.append(
                    self._cancel_locked(
                        agent_id,
                        AgentStatus.CANCELLED,
                        "permission subtree revoked",
                        already_unmounted=True,
                    )
                )
        return tuple(results)

    def drain_results(self, parent_agent_id: str) -> tuple[ChildResult, ...]:
        """Return and clear results delivered to one parent's mailbox."""
        with self._lock:
            self._record(parent_agent_id)
            return tuple(self._mailboxes.pop(parent_agent_id, []))

    def shutdown(self, wait: bool = True) -> None:
        """Cancel all non-root live agents and stop accepting async work."""
        with self._lock:
            for record in tuple(self._records.values()):
                if record.parent_agent_id is not None and not record.status.terminal:
                    self._cancel_locked(
                        record.agent_id,
                        AgentStatus.CANCELLED,
                        "agent manager shutdown",
                    )
        self._executor.shutdown(wait=wait, cancel_futures=True)
