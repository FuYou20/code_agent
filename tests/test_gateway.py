from __future__ import annotations

import asyncio
from threading import Lock
import time

from myagent.gateway import AgentGateway, AgentTask


def task(task_id: str, session_id: str = "session-1") -> AgentTask:
    return AgentTask(
        task_id=task_id,
        session_id=session_id,
        source="feishu",
        user_id="user-1",
        content="hello",
        reply_target="chat-1",
    )


def test_gateway_deduplicates_before_agent_execution() -> None:
    claimed: set[str] = set()
    calls: list[str] = []

    def claim(task_id: str) -> bool:
        if task_id in claimed:
            return False
        claimed.add(task_id)
        return True

    def run(item: AgentTask) -> str:
        calls.append(item.task_id)
        return "done"

    gateway = AgentGateway(run, claim_task=claim)

    async def scenario():
        return await gateway.submit(task("same")), await gateway.submit(task("same"))

    first, second = asyncio.run(scenario())
    assert first.success and first.content == "done"
    assert second.duplicate
    assert calls == ["same"]


def test_gateway_serializes_same_session_but_allows_other_sessions() -> None:
    active_by_session: dict[str, int] = {}
    overlap_between_sessions = False
    same_session_overlap = False
    guard = Lock()

    def run(item: AgentTask) -> str:
        nonlocal overlap_between_sessions, same_session_overlap
        with guard:
            if active_by_session.get(item.session_id, 0):
                same_session_overlap = True
            active_by_session[item.session_id] = active_by_session.get(item.session_id, 0) + 1
            if len([value for value in active_by_session.values() if value]) > 1:
                overlap_between_sessions = True
        time.sleep(0.05)
        with guard:
            active_by_session[item.session_id] -= 1
        return item.task_id

    gateway = AgentGateway(run)

    async def scenario():
        return await asyncio.gather(
            gateway.submit(task("a", "same")),
            gateway.submit(task("b", "same")),
            gateway.submit(task("c", "other")),
        )

    results = asyncio.run(scenario())
    assert [result.content for result in results] == ["a", "b", "c"]
    assert not same_session_overlap
    assert overlap_between_sessions


def test_gateway_returns_structured_failure() -> None:
    def fail(_task: AgentTask) -> str:
        raise RuntimeError("provider unavailable")

    result = asyncio.run(AgentGateway(fail).submit(task("failure")))
    assert not result.success
    assert result.error_type == "RuntimeError"
    assert result.error_message == "provider unavailable"


def test_gateway_sync_submission_uses_same_result_contract() -> None:
    seen: list[tuple[str, str]] = []

    def run(item: AgentTask) -> str:
        seen.append((item.source, item.content))
        return "sync done"

    result = AgentGateway(run).submit_sync(
        AgentTask(
            task_id="cli-1",
            session_id="cli-session",
            source="cli",
            user_id="local-user",
            content="inspect repository",
            reply_target="stdout",
        )
    )

    assert result.success
    assert result.content == "sync done"
    assert seen == [("cli", "inspect repository")]
