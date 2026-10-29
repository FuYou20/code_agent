"""Protect the epoch-System structural experiment and its stated boundaries."""

from scripts.evaluate_skill_epoch_cache import evaluate


def test_epoch_system_reduces_uncached_skill_proxy_without_hiding_required_skills() -> None:
    result = evaluate()
    assert result["skill_count"] == 100
    assert result["scenario_count"] == 3
    assert result["uncached_reduction"] >= 0.70
    assert all(
        row["required_skill_availability"] == 1.0
        for row in result["results"]
    )


def test_out_of_epoch_skills_use_sidecar_without_any_content_driven_rebuild() -> None:
    result = evaluate()
    by_name = {row["scenario"]: row for row in result["results"]}
    aside = by_name["one_turn_aside"]
    switched = by_name["sustained_drift"]
    assert aside["temporary_injections"] >= 1
    assert aside["system_prompt_changes"] == 1
    assert switched["temporary_injections"] >= 1
    assert switched["system_prompt_changes"] == 1
