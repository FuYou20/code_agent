"""Verify L0 indexing, L1 loading, trigger matching, and bundled skills."""

from pathlib import Path

from myagent.agent.state import SessionState
from myagent.cli import _load_skill_index
from myagent.context import stable
from myagent.context.assembler import build_messages
from myagent.evolution.artifacts import EvolutionArtifactManager, TokenUsage
from myagent.memory.store import init_db
from myagent.skills.manager import find_relevant, load_body, load_index


def _write_skill(root: Path, folder: str, content: str) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return skill_dir


def test_load_index_returns_only_l0_metadata(tmp_path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "demo",
        """---
name: demo-skill
description: Demo description
triggers: [demo, 示例]
permission_requirements:
  - tool: file_read
---
Secret L1 instructions.
""",
    )
    index = load_index(tmp_path)
    assert index == [
        {
            "name": "demo-skill",
            "description": "Demo description",
            "scene": "general",
            "triggers": ["demo", "示例"],
            "always_on": False,
            "path": str(skill_dir.resolve()),
        }
    ]
    assert "Secret L1 instructions" not in str(index)


def test_load_body_accepts_skill_directory(tmp_path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "demo",
        """---
name: demo
description: Demo
triggers: []
permission_requirements: []
---
Body text.
""",
    )
    assert "Body text." in load_body(skill_dir)


def test_load_index_accepts_project_fields_inside_standard_metadata(tmp_path) -> None:
    _write_skill(
        tmp_path,
        "paper",
        """---
name: paper-demo
description: Paper demo
metadata:
  scene: paper-research
  triggers: [今日推荐]
  permission_requirements:
    - tool: paper_recommend_today
---
Use the paper tool.
""",
    )

    item = load_index(tmp_path)[0]
    assert item["scene"] == "paper-research"
    assert item["triggers"] == ["今日推荐"]


def test_find_relevant_matches_case_insensitive_triggers() -> None:
    index = [
        {
            "name": "fix-import-cycle",
            "description": "Fix imports",
            "triggers": ["circular import", "循环导入"],
            "path": "unused",
        },
        {
            "name": "run-tests",
            "description": "Run tests",
            "triggers": ["pytest"],
            "path": "unused",
        },
    ]
    assert find_relevant("Please fix this CIRCULAR IMPORT", index) == ["fix-import-cycle"]
    assert find_relevant("修一下循环导入，再跑 PYTEST", index) == [
        "fix-import-cycle",
        "run-tests",
    ]


def test_invalid_skill_is_skipped(tmp_path) -> None:
    _write_skill(tmp_path, "broken", "not frontmatter")
    assert load_index(tmp_path) == []


def test_bundled_store_contains_task_skills_and_lulu_soul() -> None:
    store = Path(__file__).resolve().parents[1] / "skills_store"
    index = load_index(store)
    assert [item["name"] for item in index] == [
        "fix-import-cycle",
        "lulu-soul",
        "paper-deep-reading",
        "paper-recommendation",
        "read-repo",
        "run-tests",
    ]
    soul = next(item for item in index if item["name"] == "lulu-soul")
    assert soul["always_on"] is True
    assert "你叫 **Lulu**" in soul["body"]


def test_skill_index_is_injected_only_into_current_sidecar(tmp_path) -> None:
    stable.STABLE_PREFIX = "stable"
    store = Path(__file__).resolve().parents[1] / "skills_store"
    index = load_index(store)
    state = SessionState(session_id="s", cwd=str(tmp_path), skill_index=index)
    messages = build_messages(state, "修一下循环导入")
    assert "fix-import-cycle" in messages[1].content
    assert state.history == []


def test_cli_startup_loads_bundled_task_skills_and_soul(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path / "home"))
    index = _load_skill_index()
    assert [item["name"] for item in index] == [
        "fix-import-cycle",
        "lulu-soul",
        "paper-deep-reading",
        "paper-recommendation",
        "read-repo",
        "run-tests",
    ]


def test_cli_registers_legacy_runtime_skill_as_candidate_not_production(
    monkeypatch, tmp_path
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("MYAGENT_HOME", str(home))
    _write_skill(
        home / "skills_store",
        "legacy",
        """---
name: legacy-runtime
description: Legacy runtime skill
triggers: [legacy]
permission_requirements: []
---
Legacy procedure.
""",
    )
    conn = init_db(home / "memory.db")
    manager = EvolutionArtifactManager(conn, candidate_trials=3)

    initial = _load_skill_index(manager)
    assert "legacy-runtime" not in {item["name"] for item in initial}
    artifact = manager.get_by_key("skill", "legacy-runtime")
    assert artifact.status == "candidate"

    for index in range(3):
        manager.record_ab_trial(
            artifact.artifact_id,
            trial_key=f"legacy-check-{index}",
            control_correct=True,
            treatment_correct=True,
            control_usage=TokenUsage(prompt_tokens=1000),
            treatment_usage=TokenUsage(prompt_tokens=500),
        )
    promoted = _load_skill_index(manager)
    assert "legacy-runtime" in {item["name"] for item in promoted}
    conn.close()
