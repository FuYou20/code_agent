"""Initialize and write the SQLite FTS5 decision memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..evolution.artifacts import EvolutionArtifactManager, initialize_artifact_schema


def _migrate_memory_stats_statuses(conn: sqlite3.Connection) -> None:
    """Collapse legacy Memory statuses into candidate/active/discarded."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_stats'"
    ).fetchone()
    if row is None or "'discarded'" in str(row[0]):
        return

    conn.commit()
    foreign_keys = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.executescript(
            """
            CREATE TABLE memory_stats_v2 (
                decision_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'candidate'
                    CHECK(status IN ('candidate', 'active', 'discarded')),
                scope_key TEXT NOT NULL DEFAULT '',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                recall_count INTEGER NOT NULL DEFAULT 0,
                recent_recall_count INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT,
                last_verified_at TEXT,
                last_feedback_at TEXT,
                FOREIGN KEY(decision_id) REFERENCES decisions(id) ON DELETE CASCADE
            );

            INSERT INTO memory_stats_v2(
                decision_id, status, scope_key, success_count, failure_count,
                recall_count, recent_recall_count, last_recalled_at,
                last_verified_at, last_feedback_at
            )
            SELECT
                decision_id,
                CASE status
                    WHEN 'verified' THEN 'active'
                    WHEN 'active' THEN 'active'
                    WHEN 'conflicted' THEN 'discarded'
                    WHEN 'quarantined' THEN 'discarded'
                    WHEN 'discarded' THEN 'discarded'
                    ELSE 'candidate'
                END,
                scope_key, success_count, failure_count, recall_count,
                recent_recall_count, last_recalled_at, last_verified_at,
                last_feedback_at
            FROM memory_stats;

            DROP TABLE memory_stats;
            ALTER TABLE memory_stats_v2 RENAME TO memory_stats;
            """
        )
    finally:
        conn.execute(f"PRAGMA foreign_keys = {1 if foreign_keys else 0}")


def init_db(path: Path) -> sqlite3.Connection:
    """Open a decision database and initialize its table, FTS index, and triggers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            files TEXT NOT NULL,
            outcome TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
            content,
            files,
            content='decisions',
            content_rowid='id'
        );

        CREATE TABLE IF NOT EXISTS memory_stats (
            decision_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'candidate'
                CHECK(status IN ('candidate', 'active', 'discarded')),
            scope_key TEXT NOT NULL DEFAULT '',
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            recall_count INTEGER NOT NULL DEFAULT 0,
            recent_recall_count INTEGER NOT NULL DEFAULT 0,
            last_recalled_at TEXT,
            last_verified_at TEXT,
            last_feedback_at TEXT,
            FOREIGN KEY(decision_id) REFERENCES decisions(id) ON DELETE CASCADE
        );

        CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
            INSERT INTO decisions_fts(rowid, content, files)
            VALUES (new.id, new.content, new.files);
        END;

        CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions BEGIN
            INSERT INTO decisions_fts(decisions_fts, rowid, content, files)
            VALUES ('delete', old.id, old.content, old.files);
        END;

        CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions BEGIN
            INSERT INTO decisions_fts(decisions_fts, rowid, content, files)
            VALUES ('delete', old.id, old.content, old.files);
            INSERT INTO decisions_fts(rowid, content, files)
            VALUES (new.id, new.content, new.files);
        END;

        INSERT OR IGNORE INTO memory_stats(decision_id)
        SELECT id FROM decisions;
        """
    )
    _migrate_memory_stats_statuses(conn)
    initialize_artifact_schema(conn)
    conn.commit()
    return conn


def insert_decision(
    conn: sqlite3.Connection,
    session_id,
    kind,
    content,
    files: list[str],
    outcome: str,
    scope_key: str = "",
) -> int:
    """Insert one decision and let the trigger update the FTS5 index."""
    cursor = conn.execute(
        """
        INSERT INTO decisions(session_id, kind, content, files, outcome)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(session_id),
            str(kind),
            str(content),
            json.dumps(files, ensure_ascii=False, sort_keys=True),
            str(outcome),
        ),
    )
    decision_id = int(cursor.lastrowid)
    conn.execute(
        """
        INSERT OR IGNORE INTO memory_stats(decision_id, scope_key)
        VALUES (?, ?)
        """,
        (decision_id, str(scope_key)),
    )
    if str(outcome) == "success":
        EvolutionArtifactManager(conn, initialize_schema=False).register_memory(
            decision_id,
            scope_key=str(scope_key),
            content=str(content),
        )
    conn.commit()
    return decision_id

