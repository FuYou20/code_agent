"""SQLite-backed FIFO tasks, live-window routing, and durable outbox."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
import uuid

from .models import AgentTask


WINDOW_PRIORITIES = {"tui": 100, "feishu": 50, "cli": 10, "api": 0}


@dataclass(frozen=True)
class ClaimedTask:
    sequence: int
    task: AgentTask
    attempts: int
    lease_token: str


@dataclass(frozen=True)
class OutboundDelivery:
    delivery_id: str
    result_id: str
    channel: str
    client_id: str
    target: str
    content: str
    attempts: int
    lease_token: str


@dataclass(frozen=True)
class StoredApproval:
    approval_id: str
    task_id: str
    session_id: str
    agent_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    status: str
    requested_by: str
    decided_by: str
    error_message: str


class DurableGatewayStore:
    """Own the durable state used by one Personal Agent Runtime."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = RLock()
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,
            timeout=5,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gateway_tasks (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reply_target TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK(status IN (
                            'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED',
                            'RETRY_WAIT', 'CANCELLED', 'NEEDS_REVIEW'
                        )),
                    retry_policy TEXT NOT NULL DEFAULT 'safe'
                        CHECK(retry_policy IN ('safe', 'idempotent', 'manual')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    available_at REAL NOT NULL,
                    worker_id TEXT,
                    lease_token TEXT,
                    lease_until REAL,
                    error_type TEXT,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_gateway_tasks_ready
                    ON gateway_tasks(status, available_at, sequence);

                CREATE TABLE IF NOT EXISTS session_windows (
                    session_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    lease_until REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(session_id, channel, client_id)
                );

                CREATE INDEX IF NOT EXISTS idx_session_windows_live
                    ON session_windows(session_id, lease_until, priority DESC);

                CREATE TABLE IF NOT EXISTS agent_results (
                    result_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    routing_mode TEXT NOT NULL
                        CHECK(routing_mode IN ('LIVE_BROADCAST', 'SINGLE_CONSUMER')),
                    status TEXT NOT NULL
                        CHECK(status IN ('WAITING_CONSUMER', 'ROUTED', 'DELIVERED')),
                    created_at REAL NOT NULL,
                    delivered_at REAL,
                    FOREIGN KEY(task_id) REFERENCES gateway_tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS outbound_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK(status IN ('PENDING', 'SENDING', 'SENT')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT,
                    lease_until REAL,
                    created_at REAL NOT NULL,
                    sent_at REAL,
                    UNIQUE(result_id, channel, client_id),
                    FOREIGN KEY(result_id) REFERENCES agent_results(result_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_outbound_ready
                    ON outbound_deliveries(channel, client_id, status, created_at);

                CREATE TABLE IF NOT EXISTS tool_approvals (
                    approval_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED')),
                    requested_by TEXT NOT NULL,
                    decided_by TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    requested_at REAL NOT NULL,
                    decided_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_tool_approvals_pending
                    ON tool_approvals(status, requested_at);
                """
            )

    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def enqueue(
        self,
        task: AgentTask,
        *,
        retry_policy: str = "safe",
        max_attempts: int = 3,
        available_at: float | None = None,
    ) -> bool:
        if retry_policy not in {"safe", "idempotent", "manual"}:
            raise ValueError("invalid retry policy")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        now = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO gateway_tasks(
                    task_id, session_id, source, user_id, content, reply_target,
                    metadata_json, retry_policy, max_attempts,
                    available_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.session_id,
                    task.source,
                    task.user_id,
                    task.content,
                    task.reply_target,
                    json.dumps(task.metadata, ensure_ascii=False, sort_keys=True),
                    retry_policy,
                    max_attempts,
                    now if available_at is None else available_at,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def _recover_expired_locked(self, now: float) -> None:
        self._conn.execute(
            """
            UPDATE gateway_tasks
            SET status = CASE
                    WHEN retry_policy IN ('safe', 'idempotent')
                         AND attempts < max_attempts THEN 'PENDING'
                    ELSE 'NEEDS_REVIEW'
                END,
                worker_id = NULL,
                lease_token = NULL,
                lease_until = NULL,
                available_at = ?,
                error_type = 'WorkerLeaseExpired',
                error_message = 'worker lease expired before task completion'
            WHERE status = 'RUNNING' AND lease_until < ?
            """,
            (now, now),
        )

    def claim_next(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 300,
        now: float | None = None,
    ) -> ClaimedTask | None:
        timestamp = time.time() if now is None else now
        token = uuid.uuid4().hex
        with self._lock:
            self._begin()
            try:
                self._recover_expired_locked(timestamp)
                row = self._conn.execute(
                    """
                    SELECT * FROM gateway_tasks
                    WHERE status IN ('PENDING', 'RETRY_WAIT')
                      AND available_at <= ?
                    ORDER BY sequence
                    LIMIT 1
                    """,
                    (timestamp,),
                ).fetchone()
                if row is None:
                    self._conn.commit()
                    return None
                cursor = self._conn.execute(
                    """
                    UPDATE gateway_tasks
                    SET status = 'RUNNING',
                        attempts = attempts + 1,
                        worker_id = ?,
                        lease_token = ?,
                        lease_until = ?,
                        started_at = COALESCE(started_at, ?)
                    WHERE task_id = ?
                      AND status IN ('PENDING', 'RETRY_WAIT')
                    """,
                    (
                        worker_id,
                        token,
                        timestamp + lease_seconds,
                        timestamp,
                        row["task_id"],
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return None
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        metadata = json.loads(str(row["metadata_json"]))
        task = AgentTask(
            task_id=str(row["task_id"]),
            session_id=str(row["session_id"]),
            source=str(row["source"]),
            user_id=str(row["user_id"]),
            content=str(row["content"]),
            reply_target=str(row["reply_target"]),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
        return ClaimedTask(
            sequence=int(row["sequence"]),
            task=task,
            attempts=int(row["attempts"]) + 1,
            lease_token=token,
        )

    def renew_task_lease(
        self,
        task_id: str,
        lease_token: str,
        *,
        lease_seconds: float = 300,
    ) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE gateway_tasks SET lease_until = ?
                WHERE task_id = ? AND status = 'RUNNING' AND lease_token = ?
                """,
                (time.time() + lease_seconds, task_id, lease_token),
            )
            return cursor.rowcount == 1

    def fail_task(
        self,
        task_id: str,
        lease_token: str,
        error: Exception,
        *,
        retry_delay: float = 5,
    ) -> bool:
        now = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE gateway_tasks
                SET status = CASE
                        WHEN retry_policy IN ('safe', 'idempotent')
                             AND attempts < max_attempts THEN 'RETRY_WAIT'
                        WHEN retry_policy = 'manual' THEN 'NEEDS_REVIEW'
                        ELSE 'FAILED'
                    END,
                    available_at = ?,
                    worker_id = NULL,
                    lease_token = NULL,
                    lease_until = NULL,
                    error_type = ?,
                    error_message = ?,
                    finished_at = CASE
                        WHEN attempts >= max_attempts OR retry_policy = 'manual'
                        THEN ? ELSE finished_at END
                WHERE task_id = ? AND status = 'RUNNING' AND lease_token = ?
                """,
                (
                    now + retry_delay,
                    type(error).__name__,
                    str(error),
                    now,
                    task_id,
                    lease_token,
                ),
            )
            return cursor.rowcount == 1

    def _live_windows_locked(self, session_id: str, now: float) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT * FROM session_windows
            WHERE session_id = ? AND lease_until > ?
            ORDER BY priority DESC, channel, client_id
            """,
            (session_id, now),
        ).fetchall()

    def _insert_delivery_locked(
        self,
        result_id: str,
        window: sqlite3.Row,
        now: float,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO outbound_deliveries(
                delivery_id, result_id, channel, client_id, target, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                result_id,
                window["channel"],
                window["client_id"],
                window["target"],
                now,
            ),
        )

    def complete_task(
        self,
        task_id: str,
        lease_token: str,
        content: str,
        *,
        now: float | None = None,
    ) -> str | None:
        timestamp = time.time() if now is None else now
        result_id = uuid.uuid4().hex
        with self._lock:
            self._begin()
            try:
                task = self._conn.execute(
                    """
                    SELECT session_id FROM gateway_tasks
                    WHERE task_id = ? AND status = 'RUNNING' AND lease_token = ?
                    """,
                    (task_id, lease_token),
                ).fetchone()
                if task is None:
                    self._conn.rollback()
                    return None
                windows = self._live_windows_locked(str(task["session_id"]), timestamp)
                routing_mode = "LIVE_BROADCAST" if windows else "SINGLE_CONSUMER"
                result_status = "ROUTED" if windows else "WAITING_CONSUMER"
                self._conn.execute(
                    """
                    INSERT INTO agent_results(
                        result_id, task_id, session_id, content,
                        routing_mode, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result_id,
                        task_id,
                        task["session_id"],
                        content,
                        routing_mode,
                        result_status,
                        timestamp,
                    ),
                )
                for window in windows:
                    self._insert_delivery_locked(result_id, window, timestamp)
                self._conn.execute(
                    """
                    UPDATE gateway_tasks
                    SET status = 'SUCCEEDED', worker_id = NULL,
                        lease_token = NULL, lease_until = NULL,
                        finished_at = ?, error_type = NULL, error_message = NULL
                    WHERE task_id = ?
                    """,
                    (timestamp, task_id),
                )
                self._conn.commit()
                return result_id
            except Exception:
                self._conn.rollback()
                raise

    def register_window(
        self,
        session_id: str,
        channel: str,
        client_id: str,
        target: str,
        *,
        ttl_seconds: float = 30,
        priority: int | None = None,
        route_waiting: bool = True,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        actual_priority = WINDOW_PRIORITIES.get(channel, 0) if priority is None else priority
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO session_windows(
                    session_id, channel, client_id, target,
                    priority, lease_until, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, channel, client_id) DO UPDATE SET
                    target = excluded.target,
                    priority = excluded.priority,
                    lease_until = excluded.lease_until,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    channel,
                    client_id,
                    target,
                    actual_priority,
                    timestamp + ttl_seconds,
                    timestamp,
                ),
            )
        if route_waiting:
            self.route_waiting_results(session_id, now=timestamp)

    def live_windows(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> list[tuple[str, str, str]]:
        timestamp = time.time() if now is None else now
        with self._lock:
            rows = self._live_windows_locked(session_id, timestamp)
        return [
            (str(row["channel"]), str(row["client_id"]), str(row["target"]))
            for row in rows
        ]

    def route_waiting_results(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> int:
        timestamp = time.time() if now is None else now
        with self._lock:
            self._begin()
            try:
                windows = self._live_windows_locked(session_id, timestamp)
                if not windows:
                    self._conn.commit()
                    return 0
                winner = windows[0]
                results = self._conn.execute(
                    """
                    SELECT result_id FROM agent_results
                    WHERE session_id = ? AND status = 'WAITING_CONSUMER'
                    ORDER BY created_at, result_id
                    """,
                    (session_id,),
                ).fetchall()
                for result in results:
                    self._insert_delivery_locked(str(result["result_id"]), winner, timestamp)
                    self._conn.execute(
                        """
                        UPDATE agent_results SET status = 'ROUTED'
                        WHERE result_id = ? AND status = 'WAITING_CONSUMER'
                        """,
                        (result["result_id"],),
                    )
                self._conn.commit()
                return len(results)
            except Exception:
                self._conn.rollback()
                raise

    def claim_deliveries(
        self,
        channel: str,
        client_id: str,
        *,
        limit: int = 20,
        lease_seconds: float = 60,
        now: float | None = None,
    ) -> list[OutboundDelivery]:
        timestamp = time.time() if now is None else now
        claimed: list[OutboundDelivery] = []
        with self._lock:
            self._begin()
            try:
                self._conn.execute(
                    """
                    UPDATE outbound_deliveries
                    SET status = 'PENDING', lease_token = NULL, lease_until = NULL
                    WHERE status = 'SENDING' AND lease_until < ?
                    """,
                    (timestamp,),
                )
                rows = self._conn.execute(
                    """
                    SELECT d.*, r.content FROM outbound_deliveries AS d
                    JOIN agent_results AS r ON r.result_id = d.result_id
                    WHERE d.channel = ? AND d.client_id = ? AND d.status = 'PENDING'
                    ORDER BY d.created_at, d.delivery_id
                    LIMIT ?
                    """,
                    (channel, client_id, max(1, limit)),
                ).fetchall()
                for row in rows:
                    token = uuid.uuid4().hex
                    cursor = self._conn.execute(
                        """
                        UPDATE outbound_deliveries
                        SET status = 'SENDING', attempts = attempts + 1,
                            lease_token = ?, lease_until = ?
                        WHERE delivery_id = ? AND status = 'PENDING'
                        """,
                        (token, timestamp + lease_seconds, row["delivery_id"]),
                    )
                    if cursor.rowcount == 1:
                        claimed.append(
                            OutboundDelivery(
                                delivery_id=str(row["delivery_id"]),
                                result_id=str(row["result_id"]),
                                channel=str(row["channel"]),
                                client_id=str(row["client_id"]),
                                target=str(row["target"]),
                                content=str(row["content"]),
                                attempts=int(row["attempts"]) + 1,
                                lease_token=token,
                            )
                        )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return claimed

    def mark_delivery_sent(
        self,
        delivery_id: str,
        lease_token: str,
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = time.time() if now is None else now
        with self._lock:
            self._begin()
            try:
                row = self._conn.execute(
                    """
                    SELECT result_id FROM outbound_deliveries
                    WHERE delivery_id = ? AND status = 'SENDING' AND lease_token = ?
                    """,
                    (delivery_id, lease_token),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    """
                    UPDATE outbound_deliveries
                    SET status = 'SENT', sent_at = ?, lease_token = NULL, lease_until = NULL
                    WHERE delivery_id = ?
                    """,
                    (timestamp, delivery_id),
                )
                self._conn.execute(
                    """
                    UPDATE agent_results
                    SET status = 'DELIVERED', delivered_at = ?
                    WHERE result_id = ?
                      AND (
                          routing_mode = 'SINGLE_CONSUMER'
                          OR NOT EXISTS (
                              SELECT 1 FROM outbound_deliveries
                              WHERE result_id = ? AND status != 'SENT'
                          )
                      )
                    """,
                    (timestamp, row["result_id"], row["result_id"]),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def release_delivery(self, delivery_id: str, lease_token: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE outbound_deliveries
                SET status = 'PENDING', lease_token = NULL, lease_until = NULL
                WHERE delivery_id = ? AND status = 'SENDING' AND lease_token = ?
                """,
                (delivery_id, lease_token),
            )
            return cursor.rowcount == 1

    def task_status(self, task_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM gateway_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def result_status(self, result_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM agent_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def create_approval(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object],
        requested_by: str,
        now: float | None = None,
    ) -> str:
        """Persist an approval request before the foreground prompt blocks."""
        approval_id = uuid.uuid4().hex
        timestamp = time.time() if now is None else now
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO tool_approvals(
                    approval_id, task_id, session_id, agent_id,
                    tool_call_id, tool_name, arguments_json,
                    requested_by, requested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    task_id,
                    session_id,
                    agent_id,
                    tool_call_id,
                    tool_name,
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    requested_by,
                    timestamp,
                ),
            )
        return approval_id

    def resolve_approval(
        self,
        approval_id: str,
        approved: bool,
        *,
        decided_by: str,
        error_message: str = "",
        now: float | None = None,
    ) -> bool:
        """Resolve a pending request exactly once."""
        timestamp = time.time() if now is None else now
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE tool_approvals
                SET status = ?, decided_by = ?, error_message = ?, decided_at = ?
                WHERE approval_id = ? AND status = 'PENDING'
                """,
                (
                    "APPROVED" if approved else "REJECTED",
                    decided_by,
                    error_message,
                    timestamp,
                    approval_id,
                ),
            )
            return cursor.rowcount == 1

    def get_approval(self, approval_id: str) -> StoredApproval | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tool_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredApproval(
            approval_id=str(row["approval_id"]),
            task_id=str(row["task_id"]),
            session_id=str(row["session_id"]),
            agent_id=str(row["agent_id"]),
            tool_call_id=str(row["tool_call_id"]),
            tool_name=str(row["tool_name"]),
            arguments=json.loads(str(row["arguments_json"])),
            status=str(row["status"]),
            requested_by=str(row["requested_by"]),
            decided_by=str(row["decided_by"]),
            error_message=str(row["error_message"]),
        )

    def pending_approvals(self) -> list[StoredApproval]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT approval_id FROM tool_approvals
                WHERE status = 'PENDING'
                ORDER BY requested_at, approval_id
                """
            ).fetchall()
        return [
            approval
            for row in rows
            if (approval := self.get_approval(str(row["approval_id"]))) is not None
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
