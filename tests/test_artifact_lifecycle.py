"""Verify the unified Memory/Skill lifecycle and token economics."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def test_positive_ab_economics_promotes_canary_to_active(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, min_canary_trials=3)
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


def test_correct_but_token_expensive_candidate_is_retired(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, min_canary_trials=2)
    artifact = manager.register_skill(
        "verbose-plan",
        version=1,
        scope_key="repo-a",
        content="very long plan",
        content_ref="skills/verbose-plan/SKILL.md",
    )

    for index in range(2):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"task-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(1200),
        )

    assert manager.get(artifact.artifact_id).status == "retired"
    conn.close()


def test_correctness_regression_quarantines_immediately(tmp_path: Path) -> None:
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

    assert manager.get(artifact.artifact_id).status == "quarantined"
    assert metrics["correctness_regressions"] == 1
    conn.close()


def test_capability_gain_is_not_misreported_as_token_saving(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(conn, min_canary_trials=1)
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
    assert manager.get(artifact.artifact_id).status == "canary"
    assert manager.get(artifact.artifact_id).evaluation_epoch == 0
    conn.close()


def test_active_artifact_returns_to_canary_after_token_regression(tmp_path: Path) -> None:
    conn = init_db(tmp_path / "memory.db")
    manager = EvolutionArtifactManager(
        conn,
        min_canary_trials=2,
        negative_streak=2,
    )
    artifact = manager.register_skill(
        "once-efficient",
        version=1,
        scope_key="repo-a",
        content="plan",
        content_ref="skills/once-efficient/SKILL.md",
    )
    for index in range(2):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"good-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(800),
        )
    assert manager.get(artifact.artifact_id).status == "active"

    for index in range(2):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"bad-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=_usage(1000),
            treatment_usage=_usage(1100),
        )
    assert manager.get(artifact.artifact_id).status == "canary"
    assert manager.get(artifact.artifact_id).evaluation_epoch == 1

    manager.record_ab_trial(
        artifact.artifact_id,
        trial_key="revalidate-1",
        control_correct=True,
        treatment_correct=True,
        control_usage=_usage(1000),
        treatment_usage=_usage(800),
    )
    assert manager.get(artifact.artifact_id).status == "canary"
    manager.record_ab_trial(
        artifact.artifact_id,
        trial_key="revalidate-2",
        control_correct=True,
        treatment_correct=True,
        control_usage=_usage(1000),
        treatment_usage=_usage(800),
    )
    assert manager.get(artifact.artifact_id).status == "active"
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
    manager.transition(artifact.artifact_id, "retired", reason="superseded")
    conn.execute(
        "UPDATE evolution_artifacts SET updated_at = '2020-01-01 00:00:00' WHERE id = ?",
        (artifact.artifact_id,),
    )
    conn.commit()

    result = manager.garbage_collect(
        now=datetime.now(timezone.utc) + timedelta(days=31),
        retired_days=30,
        purge_content=True,
    )

    assert result["purged"] == 1
    assert manager.get(artifact.artifact_id).status == "purged"
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
    manager.quarantine(first.artifact_id, reason="wrong patch")

    second_id = insert_decision(
        conn,
        "s2",
        "experience",
        "repeatable but harmful advice",
        ["src/app.py"],
        "success",
    )
    second = manager.get_by_key("memory", str(second_id))

    assert second.status == "quarantined"
    event = conn.execute(
        """
        SELECT event_type, reason FROM evolution_artifact_events
        WHERE artifact_id = ? ORDER BY id DESC LIMIT 1
        """,
        (second.artifact_id,),
    ).fetchone()
    assert event == (
        "deduplicated_tombstone",
        "content matches a disabled artifact",
    )
    conn.close()
