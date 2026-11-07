"""SQLite persistence for bindings, digests and Feishu event deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterable

from ..paper_digest import PaperRecommendation


@dataclass(frozen=True)
class StoredPaper:
    label: str
    topic: str
    title: str
    venue: str
    arxiv_month: str
    arxiv_id: str
    problem: str
    url: str
    is_intersection: bool


class PaperBotStore:
    """Small single-process store suitable for a personal local bot."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feishu_bindings (
                    chat_id TEXT PRIMARY KEY,
                    owner_open_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS paper_digests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    digest_date TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, digest_date)
                );

                CREATE TABLE IF NOT EXISTS paper_digest_items (
                    digest_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    title TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    arxiv_month TEXT NOT NULL,
                    arxiv_id TEXT NOT NULL DEFAULT '',
                    problem TEXT NOT NULL,
                    url TEXT NOT NULL,
                    is_intersection INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(digest_id, label),
                    FOREIGN KEY(digest_id) REFERENCES paper_digests(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversation_state (
                    chat_id TEXT PRIMARY KEY,
                    active_label TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_recent
                    ON chat_messages(chat_id, id DESC);
                """
            )

    def reserve_message(self, message_id: str) -> bool:
        """Atomically return True once for each inbound Feishu message."""
        if not message_id:
            return False
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO processed_messages(message_id) VALUES (?)",
                (message_id,),
            )
            return cursor.rowcount == 1

    def bind(self, chat_id: str, owner_open_id: str) -> tuple[bool, str]:
        """Bind the first user as owner; later users cannot take over the bot."""
        if not chat_id or not owner_open_id:
            return False, "缺少飞书会话或用户标识"
        with self._lock, self._conn:
            owner_row = self._conn.execute(
                "SELECT value FROM bot_settings WHERE key = 'owner_open_id'"
            ).fetchone()
            if owner_row is not None and owner_row["value"] != owner_open_id:
                return False, "该机器人已经绑定其他用户"
            self._conn.execute(
                "INSERT OR IGNORE INTO bot_settings(key, value) VALUES ('owner_open_id', ?)",
                (owner_open_id,),
            )
            self._conn.execute(
                """
                INSERT INTO feishu_bindings(chat_id, owner_open_id)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    owner_open_id = excluded.owner_open_id,
                    enabled = 1,
                    last_seen_at = CURRENT_TIMESTAMP
                """,
                (chat_id, owner_open_id),
            )
        return True, "绑定成功"

    def is_owner(self, owner_open_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM bot_settings WHERE key = 'owner_open_id'"
            ).fetchone()
        return row is None or row["value"] == owner_open_id

    def enabled_chats(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id FROM feishu_bindings WHERE enabled = 1 ORDER BY created_at"
            ).fetchall()
        return [str(row["chat_id"]) for row in rows]

    def is_bound(self, chat_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM feishu_bindings WHERE chat_id = ? AND enabled = 1",
                (chat_id,),
            ).fetchone()
        return row is not None

    def save_digest(
        self,
        chat_id: str,
        digest_date: date,
        recommendations: Iterable[PaperRecommendation],
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO paper_digests(chat_id, digest_date) VALUES (?, ?)",
                (chat_id, digest_date.isoformat()),
            )
            digest = self._conn.execute(
                "SELECT id FROM paper_digests WHERE chat_id = ? AND digest_date = ?",
                (chat_id, digest_date.isoformat()),
            ).fetchone()
            assert digest is not None
            digest_id = int(digest["id"])
            self._conn.execute(
                "DELETE FROM paper_digest_items WHERE digest_id = ?", (digest_id,)
            )
            for index, item in enumerate(recommendations, start=1):
                self._conn.execute(
                    """
                    INSERT INTO paper_digest_items(
                        digest_id, label, topic, title, venue, arxiv_month,
                        arxiv_id, problem, url, is_intersection
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest_id,
                        f"P{index}",
                        item.topic,
                        item.title,
                        item.venue,
                        item.arxiv_month,
                        item.arxiv_id,
                        item.problem,
                        item.url,
                        1 if item.is_intersection else 0,
                    ),
                )

    @staticmethod
    def _stored(row: sqlite3.Row) -> StoredPaper:
        return StoredPaper(
            label=str(row["label"]),
            topic=str(row["topic"]),
            title=str(row["title"]),
            venue=str(row["venue"]),
            arxiv_month=str(row["arxiv_month"]),
            arxiv_id=str(row["arxiv_id"]),
            problem=str(row["problem"]),
            url=str(row["url"]),
            is_intersection=bool(row["is_intersection"]),
        )

    def digest_for_day(self, chat_id: str, digest_date: date) -> list[StoredPaper]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.* FROM paper_digest_items AS i
                JOIN paper_digests AS d ON d.id = i.digest_id
                WHERE d.chat_id = ? AND d.digest_date = ?
                ORDER BY i.label
                """,
                (chat_id, digest_date.isoformat()),
            ).fetchall()
        return [self._stored(row) for row in rows]

    def latest_paper(self, chat_id: str, label: str = "") -> StoredPaper | None:
        normalized = label.upper().strip()
        params: list[object] = [chat_id]
        label_clause = ""
        if normalized:
            label_clause = "AND i.label = ?"
            params.append(normalized)
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT i.* FROM paper_digest_items AS i
                JOIN paper_digests AS d ON d.id = i.digest_id
                WHERE d.chat_id = ? {label_clause}
                ORDER BY d.digest_date DESC, i.label ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return self._stored(row) if row is not None else None

    def set_active_label(self, chat_id: str, label: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO conversation_state(chat_id, active_label)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    active_label = excluded.active_label,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, label.upper()),
            )

    def active_label(self, chat_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT active_label FROM conversation_state WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return str(row["active_label"]) if row is not None else ""

    def append_chat_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        keep: int = 12,
    ) -> None:
        """Persist a small rolling window for natural Feishu conversation."""
        if role not in {"user", "assistant"}:
            raise ValueError("chat role must be user or assistant")
        if not content.strip():
            return
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO chat_messages(chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content.strip()),
            )
            self._conn.execute(
                """
                DELETE FROM chat_messages
                WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM chat_messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (chat_id, chat_id, max(1, keep)),
            )

    def recent_chat_messages(
        self,
        chat_id: str,
        *,
        limit: int = 12,
    ) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM chat_messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (chat_id, max(1, limit)),
            ).fetchall()
        return [(str(row["role"]), str(row["content"])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
