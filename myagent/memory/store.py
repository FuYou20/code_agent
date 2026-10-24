"""Initialize and write the SQLite FTS5 decision memory."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


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
        """
    )
    conn.commit()
    return conn


def insert_decision(
    conn: sqlite3.Connection,
    session_id,
    kind,
    content,
    files: list[str],
    outcome: str,
) -> None:
    """Insert one decision and let the trigger update the FTS5 index."""
    conn.execute(
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
    conn.commit()

