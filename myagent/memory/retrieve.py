"""Retrieve and maintain successful decisions in the SQLite FTS5 memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
import sqlite3


_BASE_TRUST = {
    "provisional": 0.40,
    "boundary-verified": 0.70,
    "verified": 1.00,
    "conflicted": 0.05,
    "quarantined": 0.00,
}
_RECENT_WINDOW = timedelta(hours=1)


@dataclass(frozen=True)
class MemoryHit:
    artifact_id: int
    decision_id: int
    kind: str
    content: str
    files: tuple[str, ...]
    status: str
    lifecycle_status: str
    scope_key: str
    success_count: int
    failure_count: int
    recall_count: int
    recent_recall_count: int
    last_recalled_at: str | None
    lexical_rank: int
    final_score: float


def _match_query(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    unique_tokens = list(dict.fromkeys(token for token in tokens if token))
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique_tokens)


def _recent_count(count: int, last_recalled_at: str | None) -> int:
    if not last_recalled_at:
        return 0
    try:
        recalled = datetime.fromisoformat(last_recalled_at).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0
    return count if datetime.now(timezone.utc) - recalled <= _RECENT_WINDOW else 0


def retrieve_hits(
    conn: sqlite3.Connection,
    query: str,
    k: int = 3,
    *,
    scope_key: str = "",
    candidate_k: int = 20,
    artifact_statuses: tuple[str, ...] | None = None,
) -> list[MemoryHit]:
    """Return candidates reranked by relevance, trust, scope, and exposure."""
    if k <= 0:
        return []
    match_query = _match_query(query)
    if not match_query:
        return []
    allowed = artifact_statuses or ("candidate", "canary", "active")
    if not allowed:
        return []
    placeholders = ",".join("?" for _ in allowed)
    rows = conn.execute(
        f"""
        SELECT
            a.id, d.id, d.kind, d.content, d.files,
            s.status, s.scope_key, s.success_count, s.failure_count,
            s.recall_count, s.recent_recall_count, s.last_recalled_at,
            a.status
        FROM decisions_fts
        JOIN decisions AS d ON d.id = decisions_fts.rowid
        JOIN memory_stats AS s ON s.decision_id = d.id
        JOIN evolution_artifacts AS a
          ON a.artifact_type = 'memory'
         AND a.subject_key = CAST(d.id AS TEXT)
         AND a.version = 1
        WHERE decisions_fts MATCH ?
          AND d.outcome = 'success'
          AND a.status IN ({placeholders})
        ORDER BY bm25(decisions_fts), d.id DESC
        LIMIT ?
        """,
        (match_query, *allowed, max(int(k), int(candidate_k))),
    ).fetchall()

    ranked: list[MemoryHit] = []
    for lexical_rank, row in enumerate(rows):
        (
            artifact_id,
            decision_id,
            kind,
            content,
            raw_files,
            status,
            memory_scope,
            success_count,
            failure_count,
            recall_count,
            recent_recall_count,
            last_recalled_at,
            lifecycle_status,
        ) = row
        try:
            loaded_files = json.loads(raw_files)
            files = tuple(str(path) for path in loaded_files) if isinstance(loaded_files, list) else ()
        except (json.JSONDecodeError, TypeError):
            files = ()

        # Concrete experiences are scope-bound. Cross-scope generalization is
        # a Skill responsibility, not a reason to inject another task's answer.
        if scope_key and memory_scope and scope_key != memory_scope:
            continue
        relevance = 1.0 / (1.0 + lexical_rank)
        scope_factor = 1.0 if scope_key and memory_scope else 0.80
        recent = _recent_count(int(recent_recall_count), last_recalled_at)
        exposure_penalty = min(recent * 0.10, 0.30)
        final_score = relevance * _BASE_TRUST.get(str(status), 0.0) * scope_factor
        final_score -= exposure_penalty
        if final_score <= 0:
            continue
        ranked.append(
            MemoryHit(
                artifact_id=int(artifact_id),
                decision_id=int(decision_id),
                kind=str(kind),
                content=str(content),
                files=files,
                status=str(status),
                lifecycle_status=str(lifecycle_status),
                scope_key=str(memory_scope),
                success_count=int(success_count),
                failure_count=int(failure_count),
                recall_count=int(recall_count),
                recent_recall_count=recent,
                last_recalled_at=last_recalled_at,
                lexical_rank=lexical_rank,
                final_score=final_score,
            )
        )

    ranked.sort(key=lambda item: (-item.final_score, item.lexical_rank, -item.decision_id))
    return ranked[:k]


def render_hits(hits: list[MemoryHit]) -> str:
    rendered: list[str] = []
    for hit in hits:
        file_text = ", ".join(hit.files) or "(无记录)"
        rendered.append(
            f"[决策 kind={hit.kind} files={file_text}]\n"
            f"[记忆 id={hit.decision_id} status={hit.lifecycle_status}]\n{hit.content}"
        )
    return "\n\n".join(rendered)


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    k: int = 3,
    *,
    scope_key: str = "",
    artifact_statuses: tuple[str, ...] | None = None,
) -> str:
    """Compatibility wrapper returning rendered successful decisions."""
    return render_hits(
        retrieve_hits(
            conn,
            query,
            k,
            scope_key=scope_key,
            artifact_statuses=artifact_statuses,
        )
    )


def mark_recalled(conn: sqlite3.Connection, decision_ids: tuple[int, ...]) -> None:
    """Record exposure without treating exposure as positive evidence."""
    if not decision_ids:
        return
    for decision_id in decision_ids:
        conn.execute(
            """
            UPDATE memory_stats
            SET recall_count = recall_count + 1,
                recent_recall_count = CASE
                    WHEN last_recalled_at >= datetime('now', '-1 hour')
                    THEN recent_recall_count + 1
                    ELSE 1
                END,
                last_recalled_at = CURRENT_TIMESTAMP
            WHERE decision_id = ?
            """,
            (int(decision_id),),
        )
    conn.commit()


def record_feedback(
    conn: sqlite3.Connection,
    decision_ids: tuple[int, ...],
    *,
    success: bool,
    decisive: bool = False,
) -> None:
    """Record correctness evidence without promoting on exposure alone.

    Token-positive A/B evidence is required for ``active``. A correctness
    failure is harmful evidence and immediately quarantines the artifact.
    """
    if not decision_ids:
        return
    for decision_id in decision_ids:
        row = conn.execute(
            """
            SELECT status, success_count, failure_count
            FROM memory_stats
            WHERE decision_id = ?
            """,
            (int(decision_id),),
        ).fetchone()
        if row is None:
            continue
        _status, success_count, failure_count = row
        from ..evolution.artifacts import EvolutionArtifactManager

        manager = EvolutionArtifactManager(conn)
        try:
            artifact = manager.get_by_key("memory", str(int(decision_id)), 1)
        except KeyError:
            continue
        if success:
            success_count = int(success_count) + 1
            conn.execute(
                """
                UPDATE memory_stats
                SET success_count = ?,
                    last_verified_at = CURRENT_TIMESTAMP,
                    last_feedback_at = CURRENT_TIMESTAMP
                WHERE decision_id = ?
                """,
                (success_count, int(decision_id)),
            )
            conn.commit()
            if artifact.status == "candidate":
                manager.start_canary(
                    artifact.artifact_id,
                    reason="independent correctness verification",
                )
        else:
            failure_count = int(failure_count) + 1
            conn.execute(
                """
                UPDATE memory_stats
                SET failure_count = ?,
                    last_feedback_at = CURRENT_TIMESTAMP
                WHERE decision_id = ?
                """,
                (failure_count, int(decision_id)),
            )
            conn.commit()
            if artifact.status in {"candidate", "canary", "active"}:
                manager.quarantine(
                    artifact.artifact_id,
                    reason=(
                        "decisive correctness failure"
                        if decisive
                        else "correctness failure after artifact exposure"
                    ),
                )
    conn.commit()


def memory_stats(conn: sqlite3.Connection, decision_id: int) -> dict[str, object]:
    """Return one lifecycle snapshot for diagnostics and tests."""
    row = conn.execute(
        """
        SELECT s.status, s.scope_key, s.success_count, s.failure_count, s.recall_count,
               recent_recall_count, last_recalled_at, last_verified_at,
               last_feedback_at, a.id, a.status
        FROM memory_stats AS s
        JOIN evolution_artifacts AS a
          ON a.artifact_type = 'memory'
         AND a.subject_key = CAST(s.decision_id AS TEXT)
         AND a.version = 1
        WHERE s.decision_id = ?
        """,
        (int(decision_id),),
    ).fetchone()
    if row is None:
        raise KeyError(decision_id)
    keys = (
        "status",
        "scope_key",
        "success_count",
        "failure_count",
        "recall_count",
        "recent_recall_count",
        "last_recalled_at",
        "last_verified_at",
        "last_feedback_at",
        "artifact_id",
        "lifecycle_status",
    )
    return dict(zip(keys, row, strict=True))
