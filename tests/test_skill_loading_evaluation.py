"""Protect the offline skill-loading benchmark from becoming self-fulfilling."""

from scripts import evaluate_skill_loading as MODULE


def test_fixture_has_100_skills_and_hard_neighbours() -> None:
    index = MODULE.build_skill_index()
    assert len(index) == 100
    assert any(item["name"] == "code-schema-validation" for item in index) is False
    assert any(item["name"] == "data-schema-validation" for item in index)
    assert any(item["name"] == "writing-technical-tutorial" for item in index)


def test_benchmark_separates_current_baseline_topk_and_warm_strategy() -> None:
    report = MODULE.evaluate()
    rows = {
        (row["scenario"], row["strategy"]): row
        for row in report["results"]
    }
    baseline = rows[("A_code_long", "full_sidecar")]
    top_k = rows[("A_code_long", "top_k")]
    warm = rows[("A_code_long", "warm_stable")]
    assert top_k["irrelevant_skill_token_proxy"] < baseline["irrelevant_skill_token_proxy"]
    assert warm["irrelevant_skill_token_proxy"] < baseline["irrelevant_skill_token_proxy"]
    assert top_k["availability_recall"] >= 0.95
    assert warm["availability_recall"] >= 0.95


def test_switch_is_visible_same_turn_and_one_off_aside_does_not_thaw() -> None:
    report = MODULE.evaluate()
    rows = {
        (row["scenario"], row["strategy"]): row
        for row in report["results"]
    }
    switched = rows[("C_code_to_writing", "warm_stable")]
    aside = rows[("E_one_turn_aside", "warm_stable")]
    assert switched["first_switch_available"] is True
    assert any(item["event"] == "thawed" for item in switched["transitions"])
    assert not any(item["event"] == "thawed" for item in aside["transitions"])


def test_retrieval_challenge_is_reported_separately_from_structural_scenarios() -> None:
    report = MODULE.evaluate()
    challenge = report["retrieval_challenge"]
    assert 0 <= challenge["candidate_recall_at_5"] <= 1
    assert 0 <= challenge["primary_accuracy"] <= 1
    assert challenge["no_skill_accuracy"] == 1
    assert any(not row["candidate_hit"] for row in challenge["cases"] if row["kind"] == "paraphrase")
