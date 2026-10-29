"""Verify deterministic retrieval and the explicit Scene-Epoch state machine."""

from myagent.skills.retriever import SkillMatch, retrieve_relevant
from myagent.skills.session_state import SkillLoadState, skill_index_generation


INDEX = [
    {
        "name": "fix-import-cycle",
        "scene": "code-repair",
        "description": "Fix Python imports and module initialization.",
        "triggers": ["circular import", "循环导入"],
        "path": "/skills/fix-import-cycle",
    },
    {
        "name": "write-import-guide",
        "scene": "documentation",
        "description": "Write a tutorial about Python import syntax.",
        "triggers": ["import guide"],
        "path": "/skills/write-import-guide",
    },
    {
        "name": "run-tests",
        "scene": "code-repair",
        "description": "Run and explain pytest failures.",
        "triggers": ["pytest", "运行测试"],
        "path": "/skills/run-tests",
    },
]


def _matches(*names: str) -> list[SkillMatch]:
    return [
        SkillMatch(name=name, score=100.0 - index, reasons=("test",))
        for index, name in enumerate(names)
    ]


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


def test_warmup_accumulates_candidates_then_freezes_once() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    assert state.observe(
        _matches("run-tests"),
        generation=generation,
        scene="code-repair",
        now=0,
        warm_turns=3,
    ) == []
    assert state.warming_names() == ("run-tests",)
    assert state.observe(
        _matches("fix-import-cycle"),
        generation=generation,
        scene="code-repair",
        now=60,
        warm_turns=3,
    ) == []
    assert set(state.warming_names()) == {"run-tests", "fix-import-cycle"}
    assert state.observe(
        _matches("run-tests"),
        generation=generation,
        scene="code-repair",
        now=120,
        warm_turns=3,
    ) == ["frozen"]
    assert state.phase == "frozen"
    assert state.scene == "code-repair"
    assert state.frozen_names == ("fix-import-cycle", "run-tests")


def test_frozen_epoch_never_thaws_from_content_drift() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=generation, now=0, warm_turns=2)
    state.observe(_matches("run-tests"), generation=generation, now=60, warm_turns=2)
    before = state.frozen_names
    for minute in range(2, 8):
        events = state.observe(
            _matches("write-import-guide"),
            generation=generation,
            scene="documentation",
            now=minute * 60,
            warm_turns=2,
        )
        assert events == []
    assert state.phase == "frozen"
    assert state.frozen_names == before
    assert state.epoch_id == 1


def test_idle_timeout_is_the_only_automatic_reset() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=generation, now=0, warm_turns=2)
    state.observe(_matches("run-tests"), generation=generation, now=60, warm_turns=2)
    events = state.observe(
        _matches("write-import-guide"),
        generation=generation,
        scene="documentation",
        now=7 * 60 * 60,
        warm_turns=2,
    )
    assert events == ["timeout_reset"]
    assert state.phase == "warming"
    assert state.epoch_id == 2
    assert state.scene == "documentation"
    assert state.warming_names() == ("write-import-guide",)


def test_idle_expiration_clears_a_forced_scene_before_routing() -> None:
    state = SkillLoadState(
        phase="frozen",
        scene="code-repair",
        forced_scene="code-repair",
        frozen_names=("run-tests",),
        last_turn_ts=100.0,
    )

    assert state.expire_if_idle(idle_seconds=60, now=160.0) == "timeout_reset"
    assert state.phase == "warming"
    assert state.scene == ""
    assert state.forced_scene == ""


def test_skill_generation_change_does_not_mutate_frozen_epoch() -> None:
    state = SkillLoadState()
    first = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=first, now=0, warm_turns=2)
    state.observe(_matches("run-tests"), generation=first, now=60, warm_turns=2)
    changed = [
        *INDEX,
        {
            "name": "new",
            "scene": "code-repair",
            "description": "new",
            "triggers": [],
            "path": "/new",
        },
    ]
    events = state.observe(
        _matches("new"),
        generation=skill_index_generation(changed),
        now=120,
        warm_turns=2,
    )
    assert events == []
    assert state.phase == "frozen"
    assert "new" not in state.frozen_names


def test_user_reset_can_force_a_new_scene() -> None:
    state = SkillLoadState()
    generation = skill_index_generation(INDEX)
    state.observe(_matches("run-tests"), generation=generation, now=0, warm_turns=1)
    assert state.phase == "frozen"
    state.confirm_fallbacks(("write-import-guide",))
    assert state.reset(reason="user_scene", scene="documentation") == "user_scene"
    assert state.phase == "warming"
    assert state.epoch_id == 2
    assert state.scene == "documentation"
    assert state.forced_scene == "documentation"
    assert state.frozen_names == ()
    assert state.confirmed_fallback_names == ()
