"""Single-runtime worker for the durable Agent task queue."""

from __future__ import annotations

from collections.abc import Callable
import uuid

from .models import AgentTask
from .store import DurableGatewayStore


class DurableGatewayWorker:
    def __init__(
        self,
        store: DurableGatewayStore,
        runner: Callable[[AgentTask], str],
        *,
        worker_id: str | None = None,
    ) -> None:
        self.store = store
        self.runner = runner
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex}"

    def run_once(self) -> bool:
        claimed = self.store.claim_next(self.worker_id)
        if claimed is None:
            return False
        try:
            content = self.runner(claimed.task)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Agent runner returned an empty result")
            completed = self.store.complete_task(
                claimed.task.task_id,
                claimed.lease_token,
                content.strip(),
            )
            if completed is None:
                raise RuntimeError("task lease was lost before completion")
        except Exception as exc:
            self.store.fail_task(
                claimed.task.task_id,
                claimed.lease_token,
                exc,
            )
        return True
