"""Verify deterministic skill retrieval, topic freezing, drift, and hot updates."""

from myagent.skills.retriever import SkillMatch, retrieve_relevant
from myagent.skills.session_state import SkillLoadState, skill_index_generation


INDEX = [
    {
        "name": "fix-import-cycle",
        "description": "Fix Python imports and module initialization.",
        "triggers": ["circular import", "循环导入"],
        "path": "/skills/fix-import-cycle",
    },
    {
        "name": "write-import-guide",
        "description": "Write a tutorial about Python import syntax.",
        "triggers": ["import guide"],
        "path": "/skills/write-import-guide",
    },
    {
        "name": "run-tests",
        "description": "Run and explain pytest failures.",
        "triggers": ["pytest", "运行测试"],
        "path": "/skills/run-tests",
    },
]


def _matches(*names: str) -> list[SkillMatch]:
    return [SkillMatch(name=name, score=100.0 - index, reasons=("test",)) for index, name in enumerate(names)]


def test_retriever_prefers_exact_trigger_over_hard_negative() -> None:
    matches = retrieve_relevant("Fix this circular import in Python", INDEX, k=2)
    assert matches[0].name == "fix-import-cycle"
    assert matches[0].score > matches[1].score
    assert any(reason.startswith("trigger:") for reason in matches[0].reasons)


def test_retriever_is_deterministic_and_respects_k() -> None:
    left = retrieve_relevant("Python import pytest", INDEX, k=2)
    right = retrieve_relevant("Python import pytest", INDEX, k=2)
    assert left == right
    assert len(left) == 2


def test_state_freezes_after_two_stable_user_turns() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    assert state.observe(_matches("run-tests"), generation=generation, now=0) == []
    assert state.observe(_matches("run-tests"), generation=generation, now=60) == ["frozen"]
    assert state.phase == "frozen"
    assert state.frozen_names == ("run-tests",)


def test_one_turn_aside_does_not_thaw_but_sustained_drift_does() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=generation, now=0)
    state.observe(_matches("run-tests"), generation=generation, now=60)
    assert state.observe(_matches("write-import-guide"), generation=generation, now=120) == []
    assert state.phase == "frozen"
    assert state.observe(_matches("run-tests"), generation=generation, now=180) == []
    assert state.observe(_matches("write-import-guide"), generation=generation, now=240) == []
    assert state.observe(_matches("write-import-guide"), generation=generation, now=300) == ["thawed"]
    assert state.phase == "warming"
    assert state.epoch_id == 2


def test_idle_gap_allows_immediate_topic_thaw() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=generation, now=0)
    state.observe(_matches("run-tests"), generation=generation, now=60)
    events = state.observe(_matches("write-import-guide"), generation=generation, now=7 * 60 * 60)
    assert events == ["thawed"]


def test_generation_change_starts_a_new_epoch() -> None:
    state = SkillLoadState()
    first = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=first, now=0)
    changed = [*INDEX, {"name": "new", "description": "new", "triggers": [], "path": "/new"}]
    events = state.observe(_matches("run-tests"), generation=skill_index_generation(changed), now=60)
    assert events == ["version_changed"]
    assert state.epoch_id == 2
