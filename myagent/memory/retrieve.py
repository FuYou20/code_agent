"""Retrieve successful decisions from the SQLite FTS5 index."""

from __future__ import annotations

import json
import re
import sqlite3


def _match_query(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    unique_tokens = list(dict.fromkeys(token for token in tokens if token))
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique_tokens)


def retrieve(conn: sqlite3.Connection, query: str, k: int = 3) -> str:
    """Return up to ``k`` successful matching decisions, or an empty string."""
    if k <= 0:
        return ""
    match_query = _match_query(query)
    if not match_query:
        return ""
    rows = conn.execute(
        """
        SELECT d.kind, d.content, d.files
        FROM decisions_fts
        JOIN decisions AS d ON d.id = decisions_fts.rowid
        WHERE decisions_fts MATCH ? AND d.outcome = 'success'
        ORDER BY bm25(decisions_fts), d.id DESC
        LIMIT ?
        """,
        (match_query, int(k)),
    ).fetchall()
    rendered: list[str] = []
    for kind, content, raw_files in rows:
        try:
            files = json.loads(raw_files)
        except (json.JSONDecodeError, TypeError):
            files = []
        file_text = ", ".join(str(path) for path in files) or "(无记录)"
        rendered.append(f"[决策 kind={kind} files={file_text}]\n{content}")
    return "\n\n".join(rendered)

