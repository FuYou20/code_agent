"""Verify configuration merging and home-file loading semantics."""

from copy import deepcopy

from myagent.config import DEFAULT_CONFIG, deep_merge, load_config


def test_nested_override_changes_only_leaf() -> None:
    merged = deep_merge(DEFAULT_CONFIG, {"llm": {"model": "test-model"}})
    assert merged["llm"]["model"] == "test-model"
    assert merged["llm"]["provider"] == "openai"


def test_none_keeps_base_dictionary() -> None:
    merged = deep_merge(DEFAULT_CONFIG, {"llm": None})
    assert merged["llm"] == DEFAULT_CONFIG["llm"]


def test_list_is_replaced() -> None:
    merged = deep_merge(DEFAULT_CONFIG, {"tools": {"terminal_whitelist": ["^echo"]}})
    assert merged["tools"]["terminal_whitelist"] == ["^echo"]


def test_inputs_are_not_mutated() -> None:
    base = {"nested": {"items": [1, 2]}}
    override = {"nested": {"items": [3]}}
    before_base, before_override = deepcopy(base), deepcopy(override)
    merged = deep_merge(base, override)
    merged["nested"]["items"].append(4)
    assert base == before_base
    assert override == before_override


def test_load_config_uses_myagent_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MYAGENT_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  max_steps: 7\n", encoding="utf-8")
    loaded = load_config()
    assert loaded["agent"]["max_steps"] == 7
    assert loaded["llm"] == DEFAULT_CONFIG["llm"]
    assert loaded["skills"]["strategy"] == "full_system"

