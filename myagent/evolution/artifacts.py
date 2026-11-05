"""Unified lifecycle and token economics for Memory and Skill artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from statistics import median
from typing import Mapping


ARTIFACT_TYPES = {"memory", "skill"}
LIVE_STATUSES = {"candidate", "canary", "active"}
ARTIFACT_STATUSES = LIVE_STATUSES | {"quarantined", "retired", "purged"}
_TRANSITIONS = {
    "candidate": {"canary", "quarantined", "retired"},
    "canary": {"active", "quarantined", "retired"},
    "active": {"canary", "quarantined", "retired"},
    "quarantined": {"retired"},
    "retired": {"purged"},
    "purged": set(),
}
_LEGACY_STATUS = {
    "candidate": "provisional",
    "canary": "boundary-verified",
    "active": "verified",
    "quarantined": "quarantined",
    "retired": "quarantined",
    "purged": "quarantined",
}


@dataclass(frozen=True)
class TokenWeights:
    """Convert provider token classes into one input-token cost unit."""

    uncached_input: float = 1.0
    cached_input: float = 1.0
    output: float = 1.0

    def __post_init__(self) -> None:
        if min(self.uncached_input, self.cached_input, self.output) < 0:
            raise ValueError("token weights must be non-negative")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_mapping(cls, usage: Mapping[str, object] | None) -> "TokenUsage":
        values = usage or {}
        return cls(
            prompt_tokens=max(0, int(values.get("prompt_tokens", 0) or 0)),
            cached_tokens=max(0, int(values.get("cached_tokens", 0) or 0)),
            completion_tokens=max(
                0, int(values.get("completion_tokens", 0) or 0)
            ),
        )

    def equivalent(self, weights: TokenWeights = TokenWeights()) -> float:
        cached = min(self.cached_tokens, self.prompt_tokens)
        uncached = max(0, self.prompt_tokens - cached)
        return (
            uncached * weights.uncached_input
            + cached * weights.cached_input
            + self.completion_tokens * weights.output
        )


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: int
    artifact_type: str
    subject_key: str
    version: int
    scope_key: str
    status: str
    content_hash: str
    content_ref: str
    creation_equivalent_tokens: float
    evaluation_epoch: int
    created_at: str
    updated_at: str
    last_evaluated_at: str | None


def content_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def initialize_artifact_schema(conn: sqlite3.Connection) -> None:
    """Create the shared lifecycle tables and backfill legacy memories."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS evolution_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_type TEXT NOT NULL
                CHECK(artifact_type IN ('memory', 'skill')),
            subject_key TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version > 0),
            scope_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'candidate'
                CHECK(status IN (
                    'candidate', 'canary', 'active',
                    'quarantined', 'retired', 'purged'
                )),
            content_hash TEXT NOT NULL,
            content_ref TEXT NOT NULL DEFAULT '',
            creation_equivalent_tokens REAL NOT NULL DEFAULT 0,
            evaluation_epoch INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_evaluated_at TEXT,
            activated_at TEXT,
            quarantined_at TEXT,
            retired_at TEXT,
            purged_at TEXT,
            UNIQUE(artifact_type, subject_key, version)
        );

        CREATE TABLE IF NOT EXISTS evolution_ab_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id INTEGER NOT NULL,
            trial_key TEXT NOT NULL,
            task_key TEXT NOT NULL DEFAULT '',
            control_correct INTEGER NOT NULL CHECK(control_correct IN (0, 1)),
            treatment_correct INTEGER NOT NULL CHECK(treatment_correct IN (0, 1)),
            control_equivalent_tokens REAL NOT NULL CHECK(control_equivalent_tokens >= 0),
            treatment_equivalent_tokens REAL NOT NULL CHECK(treatment_equivalent_tokens >= 0),
            token_saving REAL NOT NULL,
            decisive_failure INTEGER NOT NULL DEFAULT 0
                CHECK(decisive_failure IN (0, 1)),
            metadata TEXT NOT NULL DEFAULT '{}',
            evaluation_epoch INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(artifact_id, trial_key),
            FOREIGN KEY(artifact_id) REFERENCES evolution_artifacts(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evolution_exposures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            variant TEXT NOT NULL CHECK(variant IN ('control', 'treatment')),
            activated INTEGER NOT NULL DEFAULT 0 CHECK(activated IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(artifact_id, run_id, variant),
            FOREIGN KEY(artifact_id) REFERENCES evolution_artifacts(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS evolution_artifact_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            from_status TEXT,
            to_status TEXT,
            reason TEXT NOT NULL DEFAULT '',
            details TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(artifact_id) REFERENCES evolution_artifacts(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS evolution_artifacts_lookup
            ON evolution_artifacts(artifact_type, subject_key, status);
        CREATE INDEX IF NOT EXISTS evolution_trials_artifact
            ON evolution_ab_trials(artifact_id, created_at);
        CREATE INDEX IF NOT EXISTS evolution_events_artifact
            ON evolution_artifact_events(artifact_id, created_at);
        """
    )
    artifact_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(evolution_artifacts)")
    }
    if "evaluation_epoch" not in artifact_columns:
        conn.execute(
            "ALTER TABLE evolution_artifacts "
            "ADD COLUMN evaluation_epoch INTEGER NOT NULL DEFAULT 0"
        )
    trial_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(evolution_ab_trials)")
    }
    if "evaluation_epoch" not in trial_columns:
        conn.execute(
            "ALTER TABLE evolution_ab_trials "
            "ADD COLUMN evaluation_epoch INTEGER NOT NULL DEFAULT 0"
        )
    # Existing memory rows remain usable after upgrading the database. Their
    # old status is mapped once; all future governance uses this shared table.
    conn.execute(
        """
        INSERT OR IGNORE INTO evolution_artifacts(
            artifact_type, subject_key, version, scope_key, status,
            content_hash, content_ref, creation_equivalent_tokens
        )
        SELECT
            'memory', CAST(d.id AS TEXT), 1, COALESCE(s.scope_key, ''),
            CASE COALESCE(s.status, 'provisional')
                WHEN 'verified' THEN 'active'
                WHEN 'boundary-verified' THEN 'canary'
                WHEN 'conflicted' THEN 'quarantined'
                WHEN 'quarantined' THEN 'quarantined'
                ELSE 'candidate'
            END,
            lower(hex(d.content)), 'decisions:' || CAST(d.id AS TEXT), 0
        FROM decisions AS d
        LEFT JOIN memory_stats AS s ON s.decision_id = d.id
        WHERE d.outcome = 'success'
        """
    )
    for artifact_id, content in conn.execute(
        """
        SELECT a.id, d.content
        FROM evolution_artifacts AS a
        JOIN decisions AS d
          ON a.artifact_type = 'memory'
         AND a.subject_key = CAST(d.id AS TEXT)
        WHERE length(a.content_hash) != 64
        """
    ).fetchall():
        conn.execute(
            "UPDATE evolution_artifacts SET content_hash = ? WHERE id = ?",
            (content_digest(str(content)), int(artifact_id)),
        )
    conn.commit()


class EvolutionArtifactManager:
    """Own lifecycle transitions, A/B token accounting, and garbage collection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        token_weights: TokenWeights = TokenWeights(),
        min_canary_trials: int = 5,
        negative_streak: int = 2,
        initialize_schema: bool = True,
    ) -> None:
        if min_canary_trials <= 0:
            raise ValueError("min_canary_trials must be positive")
        if negative_streak <= 0:
            raise ValueError("negative_streak must be positive")
        self.conn = conn
        self.token_weights = token_weights
        self.min_canary_trials = min_canary_trials
        self.negative_streak = negative_streak
        if initialize_schema:
            initialize_artifact_schema(conn)

    def register(
        self,
        *,
        artifact_type: str,
        subject_key: str,
        version: int = 1,
        scope_key: str = "",
        content_hash: str,
        content_ref: str = "",
        creation_usage: TokenUsage | None = None,
        status: str = "candidate",
    ) -> ArtifactRecord:
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unsupported artifact type: {artifact_type}")
        if status not in ARTIFACT_STATUSES:
            raise ValueError(f"unsupported artifact status: {status}")
        if not str(subject_key).strip():
            raise ValueError("subject_key must not be empty")
        creation_tokens = (creation_usage or TokenUsage()).equivalent(
            self.token_weights
        )
        tombstone = self.conn.execute(
            """
            SELECT id, status
            FROM evolution_artifacts
            WHERE artifact_type = ? AND content_hash = ?
              AND status IN ('quarantined', 'retired', 'purged')
            ORDER BY id DESC LIMIT 1
            """,
            (artifact_type, str(content_hash)),
        ).fetchone()
        effective_status = status
        if tombstone is not None and status in LIVE_STATUSES:
            effective_status = (
                "quarantined" if tombstone[1] == "quarantined" else "retired"
            )
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO evolution_artifacts(
                artifact_type, subject_key, version, scope_key, status,
                content_hash, content_ref, creation_equivalent_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_type,
                str(subject_key),
                int(version),
                str(scope_key),
                effective_status,
                str(content_hash),
                str(content_ref),
                float(creation_tokens),
            ),
        )
        record = self.get_by_key(artifact_type, str(subject_key), int(version))
        if cursor.rowcount:
            if artifact_type == "memory":
                self.conn.execute(
                    "UPDATE memory_stats SET status = ? WHERE decision_id = ?",
                    (_LEGACY_STATUS[effective_status], int(subject_key)),
                )
            if tombstone is None:
                self._event(
                    record.artifact_id,
                    "registered",
                    None,
                    effective_status,
                    "new artifact",
                )
            else:
                self._event(
                    record.artifact_id,
                    "deduplicated_tombstone",
                    None,
                    effective_status,
                    "content matches a disabled artifact",
                    {"matched_artifact_id": int(tombstone[0])},
                )
        self.conn.commit()
        return record

    def register_memory(
        self,
        decision_id: int,
        *,
        scope_key: str,
        content: str,
    ) -> ArtifactRecord:
        return self.register(
            artifact_type="memory",
            subject_key=str(int(decision_id)),
            scope_key=scope_key,
            content_hash=content_digest(content),
            content_ref=f"decisions:{int(decision_id)}",
        )

    def register_skill(
        self,
        name: str,
        *,
        version: int,
        scope_key: str,
        content: str,
        content_ref: str,
        creation_usage: TokenUsage | None = None,
    ) -> ArtifactRecord:
        return self.register(
            artifact_type="skill",
            subject_key=name,
            version=version,
            scope_key=scope_key,
            content_hash=content_digest(content),
            content_ref=content_ref,
            creation_usage=creation_usage,
        )

    @staticmethod
    def _row(row: tuple) -> ArtifactRecord:
        return ArtifactRecord(*row)

    def get(self, artifact_id: int) -> ArtifactRecord:
        row = self.conn.execute(
            """
            SELECT id, artifact_type, subject_key, version, scope_key, status,
                   content_hash, content_ref, creation_equivalent_tokens,
                   evaluation_epoch, created_at, updated_at, last_evaluated_at
            FROM evolution_artifacts WHERE id = ?
            """,
            (int(artifact_id),),
        ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return self._row(row)

    def get_by_key(
        self,
        artifact_type: str,
        subject_key: str,
        version: int | None = None,
    ) -> ArtifactRecord:
        query = """
            SELECT id, artifact_type, subject_key, version, scope_key, status,
                   content_hash, content_ref, creation_equivalent_tokens,
                   evaluation_epoch, created_at, updated_at, last_evaluated_at
            FROM evolution_artifacts
            WHERE artifact_type = ? AND subject_key = ?
        """
        params: list[object] = [artifact_type, subject_key]
        if version is not None:
            query += " AND version = ?"
            params.append(int(version))
        query += " ORDER BY version DESC LIMIT 1"
        row = self.conn.execute(query, params).fetchone()
        if row is None:
            raise KeyError((artifact_type, subject_key, version))
        return self._row(row)

    def status_for(self, artifact_type: str, subject_key: str) -> str | None:
        try:
            return self.get_by_key(artifact_type, subject_key).status
        except KeyError:
            return None

    def latest_allowed(
        self,
        artifact_type: str,
        subject_key: str,
        statuses: tuple[str, ...],
    ) -> ArtifactRecord | None:
        if not statuses:
            return None
        placeholders = ",".join("?" for _ in statuses)
        row = self.conn.execute(
            f"""
            SELECT id, artifact_type, subject_key, version, scope_key, status,
                   content_hash, content_ref, creation_equivalent_tokens,
                   evaluation_epoch, created_at, updated_at, last_evaluated_at
            FROM evolution_artifacts
            WHERE artifact_type = ? AND subject_key = ?
              AND status IN ({placeholders})
            ORDER BY version DESC LIMIT 1
            """,
            (artifact_type, subject_key, *statuses),
        ).fetchone()
        return self._row(row) if row is not None else None

    def update_content_ref(self, artifact_id: int, content_ref: str) -> ArtifactRecord:
        self.conn.execute(
            """
            UPDATE evolution_artifacts
            SET content_ref = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (str(content_ref), int(artifact_id)),
        )
        self.conn.commit()
        return self.get(artifact_id)

    def list_artifacts(self, *, include_purged: bool = False) -> list[ArtifactRecord]:
        where = "" if include_purged else "WHERE status != 'purged'"
        rows = self.conn.execute(
            f"""
            SELECT id, artifact_type, subject_key, version, scope_key, status,
                   content_hash, content_ref, creation_equivalent_tokens,
                   evaluation_epoch, created_at, updated_at, last_evaluated_at
            FROM evolution_artifacts {where}
            ORDER BY id
            """
        ).fetchall()
        return [self._row(row) for row in rows]

    def _event(
        self,
        artifact_id: int,
        event_type: str,
        from_status: str | None,
        to_status: str | None,
        reason: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO evolution_artifact_events(
                artifact_id, event_type, from_status, to_status, reason, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(artifact_id),
                event_type,
                from_status,
                to_status,
                reason,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            ),
        )

    def transition(self, artifact_id: int, to_status: str, *, reason: str) -> ArtifactRecord:
        record = self.get(artifact_id)
        if to_status == record.status:
            return record
        if to_status not in _TRANSITIONS[record.status]:
            raise ValueError(f"invalid transition: {record.status} -> {to_status}")
        timestamp_column = {
            "active": "activated_at",
            "quarantined": "quarantined_at",
            "retired": "retired_at",
            "purged": "purged_at",
        }.get(to_status)
        assignments = "status = ?, updated_at = CURRENT_TIMESTAMP"
        if record.status == "active" and to_status == "canary":
            assignments += ", evaluation_epoch = evaluation_epoch + 1"
        if timestamp_column:
            assignments += f", {timestamp_column} = CURRENT_TIMESTAMP"
        self.conn.execute(
            f"UPDATE evolution_artifacts SET {assignments} WHERE id = ?",
            (to_status, int(artifact_id)),
        )
        if record.artifact_type == "memory":
            self.conn.execute(
                "UPDATE memory_stats SET status = ? WHERE decision_id = ?",
                (_LEGACY_STATUS[to_status], int(record.subject_key)),
            )
        self._event(
            artifact_id,
            "status_changed",
            record.status,
            to_status,
            reason,
        )
        self.conn.commit()
        return self.get(artifact_id)

    def start_canary(self, artifact_id: int, *, reason: str = "sandbox approved") -> ArtifactRecord:
        return self.transition(artifact_id, "canary", reason=reason)

    def quarantine(self, artifact_id: int, *, reason: str) -> ArtifactRecord:
        record = self.get(artifact_id)
        if record.status == "quarantined":
            return record
        return self.transition(artifact_id, "quarantined", reason=reason)

    def record_exposure(
        self,
        artifact_id: int,
        *,
        run_id: str,
        variant: str,
        activated: bool,
    ) -> None:
        if variant not in {"control", "treatment"}:
            raise ValueError("variant must be control or treatment")
        self.conn.execute(
            """
            INSERT OR IGNORE INTO evolution_exposures(
                artifact_id, run_id, variant, activated
            ) VALUES (?, ?, ?, ?)
            """,
            (int(artifact_id), run_id, variant, int(activated)),
        )
        self.conn.commit()

    def record_ab_trial(
        self,
        artifact_id: int,
        *,
        trial_key: str,
        control_correct: bool,
        treatment_correct: bool,
        control_usage: TokenUsage,
        treatment_usage: TokenUsage,
        task_key: str = "",
        decisive_failure: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        record = self.get(artifact_id)
        if record.status in {"quarantined", "retired", "purged"}:
            raise ValueError(f"artifact is not testable in status {record.status}")
        control_tokens = control_usage.equivalent(self.token_weights)
        treatment_tokens = treatment_usage.equivalent(self.token_weights)
        evaluation_epoch = record.evaluation_epoch
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO evolution_ab_trials(
                artifact_id, trial_key, task_key,
                control_correct, treatment_correct,
                control_equivalent_tokens, treatment_equivalent_tokens,
                token_saving, decisive_failure, metadata, evaluation_epoch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(artifact_id),
                trial_key,
                task_key,
                int(control_correct),
                int(treatment_correct),
                float(control_tokens),
                float(treatment_tokens),
                float(control_tokens - treatment_tokens),
                int(decisive_failure),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                evaluation_epoch,
            ),
        )
        if not cursor.rowcount:
            return self.economics(artifact_id)
        self.conn.execute(
            """
            UPDATE evolution_artifacts
            SET last_evaluated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (int(artifact_id),),
        )
        self._event(
            artifact_id,
            "ab_trial_recorded",
            record.status,
            record.status,
            trial_key,
            {
                "control_correct": control_correct,
                "treatment_correct": treatment_correct,
                "token_saving": control_tokens - treatment_tokens,
            },
        )
        self.conn.commit()

        if decisive_failure or (control_correct and not treatment_correct):
            self.quarantine(
                artifact_id,
                reason="correctness regression in A/B trial",
            )
            return self.economics(artifact_id)

        current = self.get(artifact_id)
        if current.status == "candidate":
            self.start_canary(artifact_id, reason="first non-regressing A/B trial")

        metrics = self.economics(artifact_id)
        epoch_metrics = self._epoch_economics(artifact_id)
        if self.get(artifact_id).status == "active":
            if self._negative_comparable_streak(artifact_id) >= self.negative_streak:
                self.transition(
                    artifact_id,
                    "canary",
                    reason="rolling token economics regressed",
                )
            return self.economics(artifact_id)

        if int(epoch_metrics["comparable_trials"]) >= self.min_canary_trials:
            if (
                int(epoch_metrics["correctness_regressions"]) == 0
                and float(epoch_metrics["net_token_saving"]) > 0
                and float(epoch_metrics["median_token_saving"]) > 0
            ):
                self.transition(
                    artifact_id,
                    "active",
                    reason="correctness held and net token saving is positive",
                )
            else:
                self.transition(
                    artifact_id,
                    "retired",
                    reason="canary completed without positive token economics",
                )
        return self.economics(artifact_id)

    def _negative_comparable_streak(self, artifact_id: int) -> int:
        rows = self.conn.execute(
            """
            SELECT token_saving, control_correct, treatment_correct
            FROM evolution_ab_trials
            WHERE artifact_id = ?
            ORDER BY id DESC
            """,
            (int(artifact_id),),
        ).fetchall()
        streak = 0
        for saving, control_correct, treatment_correct in rows:
            if not (control_correct and treatment_correct):
                continue
            if float(saving) > 0:
                break
            streak += 1
        return streak

    def economics(self, artifact_id: int) -> dict[str, object]:
        record = self.get(artifact_id)
        rows = self.conn.execute(
            """
            SELECT control_correct, treatment_correct,
                   control_equivalent_tokens, treatment_equivalent_tokens,
                   token_saving
            FROM evolution_ab_trials
            WHERE artifact_id = ? ORDER BY id
            """,
            (int(artifact_id),),
        ).fetchall()
        comparable = [row for row in rows if row[0] and row[1]]
        savings = [float(row[4]) for row in comparable]
        gross = sum(savings)
        return {
            "artifact_id": artifact_id,
            "status": record.status,
            "trials": len(rows),
            "comparable_trials": len(comparable),
            "correctness_regressions": sum(bool(row[0] and not row[1]) for row in rows),
            "capability_gains": sum(bool(not row[0] and row[1]) for row in rows),
            "control_equivalent_tokens": sum(float(row[2]) for row in comparable),
            "treatment_equivalent_tokens": sum(float(row[3]) for row in comparable),
            "gross_token_saving": gross,
            "creation_equivalent_tokens": record.creation_equivalent_tokens,
            "net_token_saving": gross - record.creation_equivalent_tokens,
            "median_token_saving": median(savings) if savings else 0.0,
            "break_even": gross > record.creation_equivalent_tokens,
            "evaluation_epoch": record.evaluation_epoch,
            "current_epoch": self._epoch_economics(artifact_id),
        }

    def _epoch_economics(self, artifact_id: int) -> dict[str, object]:
        record = self.get(artifact_id)
        rows = self.conn.execute(
            """
            SELECT control_correct, treatment_correct, token_saving
            FROM evolution_ab_trials
            WHERE artifact_id = ? AND evaluation_epoch = ?
            ORDER BY id
            """,
            (int(artifact_id), record.evaluation_epoch),
        ).fetchall()
        comparable = [row for row in rows if row[0] and row[1]]
        savings = [float(row[2]) for row in comparable]
        gross = sum(savings)
        creation_cost = (
            record.creation_equivalent_tokens if record.evaluation_epoch == 0 else 0.0
        )
        return {
            "trials": len(rows),
            "comparable_trials": len(comparable),
            "correctness_regressions": sum(bool(row[0] and not row[1]) for row in rows),
            "gross_token_saving": gross,
            "net_token_saving": gross - creation_cost,
            "median_token_saving": median(savings) if savings else 0.0,
        }

    def garbage_collect(
        self,
        *,
        now: datetime | None = None,
        candidate_days: int = 30,
        inactive_days: int = 90,
        quarantine_days: int = 30,
        retired_days: int = 30,
        purge_content: bool = False,
        skill_store: Path | None = None,
    ) -> dict[str, int]:
        """Downgrade stale assets and optionally purge expired content safely."""
        if min(
            candidate_days,
            inactive_days,
            quarantine_days,
            retired_days,
        ) < 0:
            raise ValueError("retention days must be non-negative")
        current = now or datetime.now(timezone.utc)
        counts = {"downgraded": 0, "retired": 0, "purged": 0}
        for artifact in list(self.list_artifacts()):
            reference = artifact.last_evaluated_at or artifact.updated_at
            try:
                changed = datetime.fromisoformat(reference).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            age = current - changed
            if artifact.status == "active" and age >= timedelta(days=inactive_days):
                self.transition(
                    artifact.artifact_id,
                    "canary",
                    reason="active artifact verification expired",
                )
                counts["downgraded"] += 1
            elif artifact.status in {"candidate", "canary"} and age >= timedelta(
                days=candidate_days
            ):
                self.transition(
                    artifact.artifact_id,
                    "retired",
                    reason="candidate/canary retention expired",
                )
                counts["retired"] += 1
            elif artifact.status == "quarantined" and age >= timedelta(
                days=quarantine_days
            ):
                self.transition(
                    artifact.artifact_id,
                    "retired",
                    reason="quarantine retention expired",
                )
                counts["retired"] += 1
            elif artifact.status == "retired" and age >= timedelta(days=retired_days):
                if purge_content:
                    self._purge_content(artifact, skill_store=skill_store)
                self.transition(
                    artifact.artifact_id,
                    "purged",
                    reason="retired retention expired",
                )
                counts["purged"] += 1
        return counts

    def _purge_content(
        self,
        artifact: ArtifactRecord,
        *,
        skill_store: Path | None,
    ) -> None:
        if artifact.artifact_type == "memory":
            self.conn.execute(
                "DELETE FROM decisions WHERE id = ?",
                (int(artifact.subject_key),),
            )
            self.conn.commit()
            return
        if artifact.artifact_type != "skill" or skill_store is None:
            return
        root = skill_store.resolve()
        target = Path(artifact.content_ref).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("skill content_ref escapes configured skill store") from exc
        if target.name != "SKILL.md":
            raise ValueError("only a registered SKILL.md can be purged")
        target.unlink(missing_ok=True)
