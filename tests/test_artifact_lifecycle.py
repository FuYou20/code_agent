"""Verify the unified Memory/Skill lifecycle and token economics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from myagent.evolution.artifacts import (
    EvolutionArtifactManager,
    TokenUsage,
    TokenWeights,
)
from myagent.memory.store import init_db, insert_decision


def _usage(total: int) -> TokenUsage:
    return TokenUsage(prompt_tokens=total)


def test_token_usage_converts_provider_classes_to_cost_units() -> None:
    usage = TokenUsage(prompt_tokens=100, cached_tokens=40, completion_tokens=10)
    weights = TokenWeights(uncached_input=1.0, cached_input=0.2, output=4.0)
    assert usage.equivalent(weights) == 108.0


def test_memory_and_skill_share_candidate_registry(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "s1",
        "experience",
        "A verified repair",
        ["src/app.py"],
        "success",
        "repo-a",
    )
    manager = EvolutionArtifactManager(conn)
    memory = manager.get_by_key("memory", str(decision_id))
    skill = manager.register_skill(
        "repair-app",
        version=1,
        scope_key="repo-a",
        content="repair procedure",
        content_ref=str(tmp_path / "skills" / "repair-app" / "SKILL.md"),
        creation_usage=_usage(120),
    )

    assert memory.status == "candidate"
    assert skill.status == "candidate"
    assert skill.creation_equivalent_tokens == 120
    conn.close()


def test_three_positive_candidate_trials_promote_to_active(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, candidate_trials=3)
    artifact = manager.register_skill(
        "compact-plan",
        version=1,
        scope_key="repo-a",
        content="short plan",
        content_ref="skills/compact-plan/SKILL.md",
        creation_usage=_usage(90),
    )

    for index in range(3):
        metrics = manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"task-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(950),
        )

    assert manager.get(artifact.artifact_id).status == "active"
    assert metrics["gross_token_saving"] == 150
    assert metrics["net_token_saving"] == 60
    assert metrics["break_even"] is True
    conn.close()


def test_correct_but_token_expensive_candidate_is_discarded(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, candidate_trials=3)
    artifact = manager.register_skill(
        "verbose-plan",
        version=1,
        scope_key="repo-a",
        content="very long plan",
        content_ref="skills/verbose-plan/SKILL.md",
    )

    for index in range(3):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"task-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(1200),
        )

    assert manager.get(artifact.artifact_id).status == "discarded"
    conn.close()


def test_candidate_correctness_failure_discards_immediately(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn)
    artifact = manager.register_skill(
        "unsafe-plan",
        version=1,
        scope_key="repo-a",
        content="bad advice",
        content_ref="skills/unsafe-plan/SKILL.md",
    )

    metrics = manager.record_ab_trial(
        artifact.artifact_id,
        trial_key="regression",
        control_correct=True,
        treatment_correct=False,
        control_usage=_usage(1000),
        treatment_usage=_usage(100),
    )

    assert manager.get(artifact.artifact_id).status == "discarded"
    assert metrics["correctness_regressions"] == 1
    conn.close()


def test_capability_gain_is_not_misreported_as_token_saving(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, candidate_trials=3)
    artifact = manager.register_skill(
        "capability-only",
        version=1,
        scope_key="repo-a",
        content="procedure",
        content_ref="skills/capability-only/SKILL.md",
    )

    metrics = manager.record_ab_trial(
        artifact.artifact_id,
        trial_key="hard-task",
        control_correct=False,
        treatment_correct=True,
        control_usage=_usage(500),
        treatment_usage=_usage(1500),
    )

    assert metrics["capability_gains"] == 1
    assert metrics["comparable_trials"] == 0
    assert metrics["gross_token_saving"] == 0
    assert manager.get(artifact.artifact_id).status == "candidate"
    assert manager.get(artifact.artifact_id).evaluation_epoch == 0
    conn.close()


def test_active_artifact_does_not_run_ab_and_one_failure_discards_it(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, candidate_trials=3)
    artifact = manager.register_skill(
        "once-efficient",
        version=1,
        scope_key="repo-a",
        content="plan",
        content_ref="skills/once-efficient/SKILL.md",
    )
    for index in range(3):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"good-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(800),
        )
    assert manager.get(artifact.artifact_id).status == "active"

    try:
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key="forbidden-active-ab",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(1100),
        )
    except ValueError as exc:
        assert "only allowed for candidate" in str(exc)
    else:
        raise AssertionError("active artifacts must not run A/B")

    manager.record_active_result(
        artifact.artifact_id,
        correct=True,
        run_id="production-success",
    )
    assert manager.get(artifact.artifact_id).status == "active"
    manager.record_active_result(
        artifact.artifact_id,
        correct=False,
        run_id="hidden-test-failure",
    )
    assert manager.get(artifact.artifact_id).status == "discarded"
    conn.close()


def test_garbage_collection_keeps_tombstone_and_purges_memory_content(
    tmp_path: Path,
) -> None:
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "s1",
        "experience",
        "obsolete advice",
        ["src/old.py"],
        "success",
    )
    manager = EvolutionArtifactManager(conn)
    artifact = manager.get_by_key("memory", str(decision_id))
    manager.discard(artifact.artifact_id, reason="superseded")
    conn.execute(
        "UPDATE evolution_artifacts SET updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (artifact.artifact_id,),
    )
    conn.commit()

    result = manager.garbage_collect(
        now=datetime.now(timezone.utc) + timedelta(days=31),
        discarded_days=30,
        purge_content=True,
    )

    assert result["purged"] == 1
    assert manager.get(artifact.artifact_id).status == "discarded"
    assert manager.get(artifact.artifact_id).content_ref == ""
    assert conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE id = ?", (decision_id,)
    ).fetchone()[0] == 0
    conn.close()


def test_disabled_content_cannot_reenter_under_a_new_identity(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    first_id = insert_decision(
        conn,
        "s1",
        "experience",
        "repeatable but harmful advice",
        ["src/app.py"],
        "success",
    )
    manager = EvolutionArtifactManager(conn)
    first = manager.get_by_key("memory", str(first_id))
    manager.discard(first.artifact_id, reason="wrong patch")

    second_id = insert_decision(
        conn,
        "s2",
        "experience",
        "repeatable but harmful advice",
        ["src/app.py"],
        "success",
    )
    second = manager.get_by_key("memory", str(second_id))

    assert second.status == "discarded"
    event = conn.execute(
        """
        SELECT event_type, reason FROM evolution_artifact_events
        WHERE artifact_id = ? ORDER BY id DESC LIMIT 1
        """,
        (second.artifact_id,),
    ).fetchone()
    assert event == (
        "deduplicated_tombstone",
        "content matches a discarded artifact",
    )
    conn.close()


def test_legacy_six_state_database_migrates_to_three_states(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            files TEXT NOT NULL,
            outcome TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE memory_stats (
            decision_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'provisional'
                CHECK(status IN ('provisional', 'boundary-verified', 'verified',
                                 'conflicted', 'quarantined')),
            scope_key TEXT NOT NULL DEFAULT '',
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            recall_count INTEGER NOT NULL DEFAULT 0,
            recent_recall_count INTEGER NOT NULL DEFAULT 0,
            last_recalled_at TEXT,
            last_verified_at TEXT,
            last_feedback_at TEXT
        );
        CREATE TABLE evolution_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_type TEXT NOT NULL CHECK(artifact_type IN ('memory', 'skill')),
            subject_key TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            scope_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK(status IN (
                'candidate', 'canary', 'active', 'quarantined', 'retired', 'purged'
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
        INSERT INTO evolution_artifacts(
            artifact_type, subject_key, status, content_hash, content_ref
        ) VALUES
            ('skill', 'warming', 'canary', 'hash-1', 'warming/SKILL.md'),
            ('skill', 'unsafe', 'quarantined', 'hash-2', 'unsafe/SKILL.md');
        CREATE TABLE evolution_ab_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_id INTEGER NOT NULL,
            trial_key TEXT NOT NULL,
            task_key TEXT NOT NULL DEFAULT '',
            control_correct INTEGER NOT NULL,
            treatment_correct INTEGER NOT NULL,
            control_equivalent_tokens REAL NOT NULL,
            treatment_equivalent_tokens REAL NOT NULL,
            token_saving REAL NOT NULL,
            decisive_failure INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}',
            evaluation_epoch INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(artifact_id, trial_key),
            FOREIGN KEY(artifact_id) REFERENCES evolution_artifacts(id) ON DELETE CASCADE
        );
        INSERT INTO evolution_ab_trials(
            artifact_id, trial_key, control_correct, treatment_correct,
            control_equivalent_tokens, treatment_equivalent_tokens, token_saving
        ) VALUES (1, 'legacy-trial', 1, 1, 1000, 800, 200);
        """
    )
    legacy.close()

    conn = init_db(path)
    assert conn.execute(
        "SELECT subject_key, status FROM evolution_artifacts ORDER BY id"
    ).fetchall() == [("warming", "candidate"), ("unsafe", "discarded")]
    assert conn.execute(
        "SELECT artifact_id, trial_key FROM evolution_ab_trials"
    ).fetchall() == [(1, "legacy-trial")]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()
