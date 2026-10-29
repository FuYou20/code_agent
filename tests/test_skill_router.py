"""Verify model-assisted workset routing and runtime freezing."""

import json

from myagent.agent.loop import run_turn
from myagent.agent.state import SessionState
from myagent.context import stable
from myagent.llm.types import LLMResponse
from myagent.permissions import PermissionTree, ToolPermissionAdapter
from myagent.skills.manager import load_index
from myagent.skills.router import route_skill_workset


INDEX = [
    {
        "name": "run-tests",
        "scene": "code-repair",
        "description": "Run and explain pytest failures.",
        "triggers": ["pytest", "test failure"],
        "path": "/skills/run-tests",
    },
    {
        "name": "read-repo",
        "scene": "code-repair",
        "description": "Read source files related to a failure.",
        "triggers": ["read repository", "source file"],
        "path": "/skills/read-repo",
    },
]


class TwoResponseRouterClient:
    def __init__(self) -> None:
        self.responses = [
            LLMResponse(
                content=json.dumps(
                    {
                        "scene": "code-repair",
                        "terms": ["pytest test failure", "read repository"],
                    }
                ),
                usage={"prompt_tokens": 11},
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "scene": "code-repair",
                        "current_required": ["run-tests"],
                        "session_candidates": ["run-tests", "read-repo"],
                    }
                ),
                usage={"prompt_tokens": 23},
            ),
        ]

    def chat(self, messages, tools):
        return self.responses.pop(0)


def test_router_separates_current_and_potential_skills() -> None:
    route = route_skill_workset(
        TwoResponseRouterClient(),
        "Explain the first pytest failure",
        ("Run the test suite",),
        INDEX,
    )
    assert route.scene == "code-repair"
    assert route.current_required == ("run-tests",)
    assert route.session_candidates == ("run-tests", "read-repo")
    assert route.prompt_tokens == 34


class WarmRuntimeClient:
    def __init__(self) -> None:
        self.call_number = 0

    def chat(self, messages, tools):
        self.call_number += 1
        if self.call_number == 1:
            return LLMResponse(
                content=json.dumps(
                    {"scene": "code-repair", "terms": ["pytest", "read repository"]}
                ),
                usage={"prompt_tokens": 10},
            )
        if self.call_number == 2:
            return LLMResponse(
                content=json.dumps(
                    {
                        "scene": "code-repair",
                        "current_required": ["run-tests"],
                        "session_candidates": ["run-tests", "read-repo"],
                    }
                ),
                usage={"prompt_tokens": 20},
            )
        if self.call_number == 3:
            assert "可用技能（L0）" in messages[-1].content
            assert "固化 Skill 工作集" not in messages[0].content
        else:
            assert "固化 Skill 工作集" in messages[0].content
            assert "可用技能（L0）" not in messages[-1].content
            assert "Run tests safely." in messages[-1].content
        return LLMResponse(content="done")


def test_runtime_warms_in_sidecar_then_freezes_l0_into_system(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "stable"
    store = tmp_path / "skills"
    for name, description, triggers, body in (
        ("run-tests", "Run pytest.", "[pytest]", "Run tests safely."),
        ("read-repo", "Read source files.", "[repository]", "Read files safely."),
    ):
        directory = store / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            (
                "---\n"
                f"name: {name}\n"
                f"description: {description}\n"
                "scene: code-repair\n"
                f"triggers: {triggers}\n"
                "permission_requirements: []\n"
                "---\n"
                f"{body}\n"
            ),
            encoding="utf-8",
        )
    state = SessionState(
        session_id="warm-runtime",
        agent_id="warm-root",
        cwd=str(tmp_path),
        skill_index=load_index(store),
        skill_strategy="scene_epoch",
        skill_candidate_k=20,
        skill_max_frozen=8,
        skill_warm_turns=2,
    )
    tree = PermissionTree(tmp_path, ())
    tree.attach_agent(tree.ROOT_ID, state.agent_id)
    state.permission_tree = tree
    state.permission_adapter = ToolPermissionAdapter(tmp_path)
    client = WarmRuntimeClient()

    assert run_turn(state, "Run pytest", client, lambda _call: False) == "done"
    assert state.skill_load_state.phase == "warming"
    assert run_turn(state, "Explain pytest failure", client, lambda _call: False) == "done"
    assert state.skill_load_state.phase == "frozen"
    assert state.skill_load_state.frozen_names == ("read-repo", "run-tests")
    assert state.skill_router_prompt_tokens == 30
    assert state.skill_cache_epoch_changes == 1


class FrozenFallbackClient:
    def __init__(self) -> None:
        self.responses = [
            LLMResponse(
                content=json.dumps(
                    {"scene": "documentation", "terms": ["release notes"]}
                )
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "scene": "documentation",
                        "current_required": ["write-release-notes"],
                        "session_candidates": ["write-release-notes"],
                    }
                )
            ),
        ]

    def chat(self, messages, tools):
        if self.responses:
            return self.responses.pop(0)
        assert "run-tests: Run pytest." in messages[0].content
        assert "write-release-notes: Write release notes." in messages[-1].content
        return LLMResponse(content="done")


class ConfirmedFallbackClient:
    def chat(self, messages, tools):
        assert "run-tests: Run pytest." in messages[0].content
        assert "write-release-notes: Write release notes." in messages[-1].content
        return LLMResponse(content="done")


def test_out_of_epoch_skill_is_temporary_and_never_thaws(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "你是 myagent"
    store = tmp_path / "skills"
    for name, scene, description, trigger in (
        ("run-tests", "code-repair", "Run pytest.", "pytest"),
        ("write-release-notes", "documentation", "Write release notes.", "release notes"),
    ):
        directory = store / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            (
                "---\n"
                f"name: {name}\n"
                f"description: {description}\n"
                f"scene: {scene}\n"
                f"triggers: [{trigger}]\n"
                "permission_requirements: []\n"
                "---\n"
                f"Skill: {name}\n"
            ),
            encoding="utf-8",
        )
    state = SessionState(
        session_id="fallback",
        agent_id="fallback-root",
        cwd=str(tmp_path),
        skill_index=load_index(store),
        skill_strategy="scene_epoch",
    )
    state.skill_load_state.phase = "frozen"
    state.skill_load_state.scene = "code-repair"
    state.skill_load_state.frozen_names = ("run-tests",)
    tree = PermissionTree(tmp_path, ())
    tree.attach_agent(tree.ROOT_ID, state.agent_id)
    state.permission_tree = tree
    state.permission_adapter = ToolPermissionAdapter(tmp_path)

    client = FrozenFallbackClient()
    assert run_turn(
        state,
        "Write release notes",
        client,
        lambda _call: False,
    ) == "done"
    assert state.skill_load_state.phase == "frozen"
    assert state.skill_load_state.scene == "code-repair"
    assert state.skill_load_state.frozen_names == ("run-tests",)
    assert state.skill_cache_epoch_changes == 0
    assert state.skill_load_state.confirmed_fallback_names == (
        "write-release-notes",
    )

    assert run_turn(
        state,
        "Rewrite the release notes for end users",
        ConfirmedFallbackClient(),
        lambda _call: False,
    ) == "done"
    assert state.skill_load_state.frozen_names == ("run-tests",)
    assert state.skill_router_prompt_tokens == 0


class FrozenLocalHitClient:
    def chat(self, messages, tools):
        assert "run-tests: Run pytest." in messages[0].content
        assert "Skill: run-tests" in messages[-1].content
        return LLMResponse(content="done")


def test_frozen_local_hit_skips_model_router(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "你是 myagent"
    store = tmp_path / "skills" / "run-tests"
    store.mkdir(parents=True)
    (store / "SKILL.md").write_text(
        (
            "---\n"
            "name: run-tests\n"
            "description: Run pytest.\n"
            "scene: code-repair\n"
            "triggers: [pytest]\n"
            "permission_requirements: []\n"
            "---\n"
            "Skill: run-tests\n"
        ),
        encoding="utf-8",
    )
    state = SessionState(
        session_id="local-hit",
        agent_id="local-hit-root",
        cwd=str(tmp_path),
        skill_index=load_index(tmp_path / "skills"),
        skill_strategy="scene_epoch",
    )
    state.skill_load_state.phase = "frozen"
    state.skill_load_state.scene = "code-repair"
    state.skill_load_state.frozen_names = ("run-tests",)
    tree = PermissionTree(tmp_path, ())
    tree.attach_agent(tree.ROOT_ID, state.agent_id)
    state.permission_tree = tree
    state.permission_adapter = ToolPermissionAdapter(tmp_path)

    assert run_turn(
        state,
        "Run pytest",
        FrozenLocalHitClient(),
        lambda _call: False,
    ) == "done"
    assert state.skill_load_state.phase == "frozen"
    assert state.skill_load_state.frozen_names == ("run-tests",)
    assert state.skill_router_prompt_tokens == 0


class FrozenNoLexicalHitClient:
    def chat(self, messages, tools):
        assert "run-tests: Run pytest." in messages[0].content
        assert "可用技能（L0）" not in messages[-1].content
        return LLMResponse(content="done")


def test_frozen_epoch_trusts_workset_without_lexical_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    stable.STABLE_PREFIX = "你是 myagent"
    store = tmp_path / "skills" / "run-tests"
    store.mkdir(parents=True)
    (store / "SKILL.md").write_text(
        (
            "---\n"
            "name: run-tests\n"
            "description: Run pytest.\n"
            "scene: code-repair\n"
            "triggers: [pytest]\n"
            "permission_requirements: []\n"
            "---\n"
            "Skill: run-tests\n"
        ),
        encoding="utf-8",
    )
    state = SessionState(
        session_id="no-lexical-hit",
        agent_id="no-lexical-hit-root",
        cwd=str(tmp_path),
        skill_index=load_index(tmp_path / "skills"),
        skill_strategy="scene_epoch",
    )
    state.skill_load_state.phase = "frozen"
    state.skill_load_state.scene = "code-repair"
    state.skill_load_state.frozen_names = ("run-tests",)
    tree = PermissionTree(tmp_path, ())
    tree.attach_agent(tree.ROOT_ID, state.agent_id)
    state.permission_tree = tree
    state.permission_adapter = ToolPermissionAdapter(tmp_path)

    assert run_turn(
        state,
        "Inspect the implementation and propose the smallest fix",
        FrozenNoLexicalHitClient(),
        lambda _call: False,
    ) == "done"
    assert state.skill_router_prompt_tokens == 0
    assert state.skill_load_state.frozen_names == ("run-tests",)
