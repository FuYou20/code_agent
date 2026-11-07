from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from myagent.gateway import (
    AgentTask,
    DurableGatewayStore,
    DurableGatewayWorker,
    request_persisted_approval,
)
from myagent.llm.types import ToolCall


def task(number: int, *, session_id: str = "personal-assistant") -> AgentTask:
    return AgentTask(
        task_id=f"task-{number}",
        session_id=session_id,
        source="api",
        user_id="owner",
        content=f"message-{number}",
        reply_target="owner",
    )


def complete(
    store: DurableGatewayStore,
    item: AgentTask,
    content: str = "done",
    *,
    now: float = 100,
) -> str:
    assert store.enqueue(item, available_at=now)
    claimed = store.claim_next("worker", now=now)
    assert claimed is not None
    result_id = store.complete_task(
        claimed.task.task_id,
        claimed.lease_token,
        content,
        now=now,
    )
    assert result_id is not None
    return result_id


def test_durable_queue_is_fifo_and_deduplicates_task_ids(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    assert store.enqueue(task(2), available_at=100)
    assert store.enqueue(task(1), available_at=100)
    assert not store.enqueue(task(2), available_at=100)

    first = store.claim_next("worker", now=100)
    assert first is not None and first.task.task_id == "task-2"
    assert store.complete_task("task-2", first.lease_token, "two", now=100)
    second = store.claim_next("worker", now=100)
    assert second is not None and second.task.task_id == "task-1"
    store.close()


def test_expired_safe_task_recovers_but_manual_task_needs_review(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    assert store.enqueue(task(1), retry_policy="safe", available_at=100)
    first = store.claim_next("dead-worker", lease_seconds=5, now=100)
    assert first is not None

    recovered = store.claim_next("new-worker", now=106)
    assert recovered is not None and recovered.task.task_id == "task-1"
    assert recovered.attempts == 2

    assert store.enqueue(task(2), retry_policy="manual", available_at=200)
    assert store.complete_task(
        recovered.task.task_id,
        recovered.lease_token,
        "recovered",
        now=106,
    )
    manual = store.claim_next("dead-worker", lease_seconds=5, now=200)
    assert manual is not None and manual.task.task_id == "task-2"
    assert store.claim_next("new-worker", now=206) is None
    assert store.task_status("task-2") == "NEEDS_REVIEW"
    store.close()


def test_live_windows_receive_broadcast_deliveries(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    store.register_window(
        "personal-assistant", "tui", "tui-1", "tui-1", now=100
    )
    store.register_window(
        "personal-assistant", "feishu", "feishu-1", "chat-1", now=100
    )
    result_id = complete(store, task(1), "broadcast", now=100)

    tui = store.claim_deliveries("tui", "tui-1", now=100)
    feishu = store.claim_deliveries("feishu", "feishu-1", now=100)
    assert [item.content for item in tui] == ["broadcast"]
    assert [item.content for item in feishu] == ["broadcast"]
    assert store.mark_delivery_sent(tui[0].delivery_id, tui[0].lease_token, now=101)
    assert store.result_status(result_id) == "ROUTED"
    assert store.mark_delivery_sent(
        feishu[0].delivery_id, feishu[0].lease_token, now=101
    )
    assert store.result_status(result_id) == "DELIVERED"
    store.close()


def test_offline_result_is_consumed_once_by_first_reconnected_window(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    result_id = complete(store, task(1), "offline", now=100)
    assert store.result_status(result_id) == "WAITING_CONSUMER"

    store.register_window(
        "personal-assistant", "tui", "tui-1", "tui-1", now=110
    )
    delivery = store.claim_deliveries("tui", "tui-1", now=110)
    assert len(delivery) == 1
    assert store.mark_delivery_sent(
        delivery[0].delivery_id,
        delivery[0].lease_token,
        now=111,
    )

    store.register_window(
        "personal-assistant", "feishu", "feishu-1", "chat-1", now=120
    )
    assert store.claim_deliveries("feishu", "feishu-1", now=120) == []
    assert store.result_status(result_id) == "DELIVERED"
    store.close()


def test_tui_wins_offline_result_when_tui_and_feishu_are_both_live(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    result_id = complete(store, task(1), "priority", now=100)
    store.register_window(
        "personal-assistant",
        "feishu",
        "feishu-1",
        "chat-1",
        route_waiting=False,
        now=110,
    )
    store.register_window(
        "personal-assistant",
        "tui",
        "tui-1",
        "tui-1",
        route_waiting=False,
        now=110,
    )

    assert store.route_waiting_results("personal-assistant", now=110) == 1
    assert len(store.claim_deliveries("tui", "tui-1", now=110)) == 1
    assert store.claim_deliveries("feishu", "feishu-1", now=110) == []
    assert store.result_status(result_id) == "ROUTED"
    store.close()


def test_failed_delivery_returns_to_outbox_without_duplicate_consumption(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    store.register_window(
        "personal-assistant", "tui", "tui-1", "tui-1", now=100
    )
    complete(store, task(1), "retry me", now=100)
    first = store.claim_deliveries("tui", "tui-1", now=100)
    assert len(first) == 1
    assert store.release_delivery(first[0].delivery_id, first[0].lease_token)
    second = store.claim_deliveries("tui", "tui-1", now=101)
    assert len(second) == 1
    assert second[0].delivery_id == first[0].delivery_id
    assert second[0].attempts == 2
    store.close()


def test_worker_persists_result_instead_of_returning_in_memory_only(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    store.register_window(
        "personal-assistant", "tui", "tui-1", "tui-1"
    )
    assert store.enqueue(task(1))
    worker = DurableGatewayWorker(store, lambda item: f"answer:{item.content}")

    assert worker.run_once()
    assert store.task_status("task-1") == "SUCCEEDED"
    deliveries = store.claim_deliveries("tui", "tui-1")
    assert [item.content for item in deliveries] == ["answer:message-1"]
    store.close()


def test_two_store_connections_cannot_claim_the_same_task(tmp_path) -> None:
    database = tmp_path / "gateway.sqlite3"
    producer = DurableGatewayStore(database)
    worker_a = DurableGatewayStore(database)
    worker_b = DurableGatewayStore(database)
    assert producer.enqueue(task(1), available_at=100)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda pair: pair[0].claim_next(pair[1], now=100),
                ((worker_a, "worker-a"), (worker_b, "worker-b")),
            )
        )

    assert sum(claim is not None for claim in claims) == 1
    producer.close()
    worker_a.close()
    worker_b.close()


def test_approval_is_persisted_before_prompt_and_kept_after_restart(tmp_path) -> None:
    database = tmp_path / "gateway.sqlite3"
    store = DurableGatewayStore(database)
    approval_ids: list[str] = []

    def approve(_call: ToolCall) -> bool:
        pending = store.pending_approvals()
        assert len(pending) == 1
        assert pending[0].status == "PENDING"
        assert pending[0].tool_name == "file_write"
        approval_ids.append(pending[0].approval_id)
        return True

    approved = request_persisted_approval(
        store,
        task=task(1),
        agent_id="root-agent",
        call=ToolCall(
            id="tool-call-1",
            name="file_write",
            arguments={"path": "README.md", "content": "updated"},
        ),
        delegate=approve,
        channel="tui",
    )
    assert approved
    store.close()

    reopened = DurableGatewayStore(database)
    record = reopened.get_approval(approval_ids[0])
    assert record is not None
    assert record.status == "APPROVED"
    assert record.decided_by == "tui"
    assert record.arguments["path"] == "README.md"
    assert reopened.pending_approvals() == []
    reopened.close()


def test_approval_resolution_is_first_decision_wins(tmp_path) -> None:
    store = DurableGatewayStore(tmp_path / "gateway.sqlite3")
    approval_id = store.create_approval(
        task_id="task-1",
        session_id="session-1",
        agent_id="root-agent",
        tool_call_id="call-1",
        tool_name="shell",
        arguments={"command": "git status"},
        requested_by="cli",
    )

    assert store.resolve_approval(approval_id, False, decided_by="cli")
    assert not store.resolve_approval(approval_id, True, decided_by="tui")
    record = store.get_approval(approval_id)
    assert record is not None
    assert record.status == "REJECTED"
    assert record.decided_by == "cli"
    store.close()
