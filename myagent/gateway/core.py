"""Normalize delivery guarantees before work reaches an Agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from threading import RLock

from .models import AgentResult, AgentTask


class AgentGateway:
    """Deduplicate tasks and serialize execution within each Agent session."""

    def __init__(
        self,
        runner: Callable[[AgentTask], str],
        *,
        claim_task: Callable[[str], bool] | None = None,
    ) -> None:
        self._runner = runner
        self._claim_task = claim_task
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._lock_guard = asyncio.Lock()
        self._sync_session_locks: dict[str, RLock] = {}
        self._sync_lock_guard = RLock()

    async def _session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._lock_guard:
            return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def submit(self, task: AgentTask) -> AgentResult:
        if self._claim_task is not None and not self._claim_task(task.task_id):
            return AgentResult(
                task_id=task.task_id,
                session_id=task.session_id,
                success=True,
                duplicate=True,
            )

        lock = await self._session_lock(task.session_id)
        async with lock:
            try:
                content = await asyncio.to_thread(self._runner, task)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Agent runner returned an empty result")
                return AgentResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    content=content.strip(),
                )
            except Exception as exc:
                return AgentResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

    def submit_sync(self, task: AgentTask) -> AgentResult:
        """Submit from synchronous CLI/TUI adapters using the same contract."""
        if self._claim_task is not None and not self._claim_task(task.task_id):
            return AgentResult(
                task_id=task.task_id,
                session_id=task.session_id,
                success=True,
                duplicate=True,
            )
        with self._sync_lock_guard:
            lock = self._sync_session_locks.setdefault(task.session_id, RLock())
        with lock:
            try:
                content = self._runner(task)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Agent runner returned an empty result")
                return AgentResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    content=content.strip(),
                )
            except Exception as exc:
                return AgentResult(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
