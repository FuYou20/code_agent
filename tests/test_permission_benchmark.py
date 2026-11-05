"""Protect the Langfuse permission benchmark catalog and harness."""

from benchmarks.permissions import CASES, run_case


def test_permission_catalog_has_twenty_one_unique_business_cases() -> None:
    assert len(CASES) == 21
    assert len({case.case_id for case in CASES}) == 21
    assert all(case.node_id.startswith("tests/") for case in CASES)
    assert all(case.scenario for case in CASES)


def test_permission_harness_executes_original_truth_source() -> None:
    case = next(case for case in CASES if case.case_id == "single-parent-atom-coverage")
    result = run_case(case, ".")
    assert result["passed"] is True
    assert result["actual_decision"] == "pass"
    assert result["returncode"] == 0
    event_names = [event["name"] for event in result["permission_events"]]
    assert "permission.child.requested" in event_names
    assert "permission.atom.coverage_checked" in event_names
    assert "permission.child.denied" in event_names
