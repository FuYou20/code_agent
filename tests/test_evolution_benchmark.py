"""Protect the shape and hidden-test boundary of the evolution benchmark."""

from pathlib import Path

from benchmarks.evolution import TASKS
from benchmarks.evolution.harness import grade, materialize, run_public_tests
from myagent.memory.retrieve import (
    mark_recalled,
    memory_stats,
    record_feedback,
    retrieve_hits,
)
from myagent.memory.store import init_db, insert_decision


def test_dataset_has_ten_complete_task_families() -> None:
    families: dict[str, list[str]] = {}
    for task in TASKS:
        families.setdefault(task.family, []).append(task.stage)

    assert len(TASKS) == 30
    assert len(families) == 10
    assert all(stages.count("learn") == 1 for stages in families.values())
    assert all(stages.count("transfer") == 2 for stages in families.values())


def test_materialized_workspace_does_not_contain_hidden_material(tmp_path: Path) -> None:
    task = TASKS[0]
    materialize(task, tmp_path)

    assert (tmp_path / task.filename).read_text(encoding="utf-8") == task.buggy_code
    assert (tmp_path / "test_public.py").is_file()
    assert not (tmp_path / "test_hidden.py").exists()
    assert task.fixed_code not in {
        path.read_text(encoding="utf-8") for path in tmp_path.iterdir() if path.is_file()
    }


def test_representative_bug_fails_and_reference_passes(tmp_path: Path) -> None:
    task = TASKS[0]
    buggy = tmp_path / "buggy"
    materialize(task, buggy)
    assert not run_public_tests(buggy).passed

    fixed = tmp_path / "fixed"
    materialize(task, fixed, reference=True)
    assert grade(task, fixed).passed


def test_retry_hidden_failure_quarantines_incomplete_memory_from_next_variant(
    tmp_path: Path,
) -> None:
    tasks = {task.task_id: task for task in TASKS}
    learned = tasks["retry-learn"]
    first_transfer = tasks["retry-transfer-1"]
    second_transfer = tasks["retry-transfer-2"]
    conn = init_db(tmp_path / "memory.db")
    decision_id = insert_decision(
        conn,
        "retry-learn",
        "experience",
        "Repair retry max_attempts by changing range(max_attempts - 1) to range(max_attempts)",
        [learned.filename],
        "success",
        learned.family,
    )

    hits = retrieve_hits(
        conn,
        first_transfer.prompt,
        k=1,
        scope_key=first_transfer.family,
    )
    assert [hit.decision_id for hit in hits] == [decision_id]
    mark_recalled(conn, (decision_id,))

    workspace = tmp_path / "first-transfer"
    materialize(first_transfer, workspace)
    incomplete_fix = first_transfer.buggy_code.replace(
        "range(max_attempts - 1)",
        "range(max_attempts)",
    )
    (workspace / first_transfer.filename).write_text(incomplete_fix, encoding="utf-8")
    result = grade(first_transfer, workspace)
    assert not result.passed
    assert "test_zero_attempts_is_invalid" in result.output

    record_feedback(conn, (decision_id,), success=result.passed)
    stats = memory_stats(conn, decision_id)
    assert stats["status"] == "discarded"
    assert stats["lifecycle_status"] == "discarded"
    assert stats["failure_count"] == 1

    next_hits = retrieve_hits(
        conn,
        second_transfer.prompt,
        k=1,
        scope_key=second_transfer.family,
    )
    assert next_hits == []
    conn.close()
