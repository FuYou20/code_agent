"""Load the one-file YAML configuration with non-mutating deep merging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from .home import get_home


DEFAULT_CONFIG = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "agent": {"max_steps": 30},
    "skills": {
        "strategy": "full_sidecar",
        "top_k": 5,
        "activation_min_score": 80,
    },
    "tools": {
        "terminal_whitelist": [
            "^git (status|diff|log)",
            "^ls",
            "^cat",
            "^pytest",
        ]
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge dictionaries into a new value without mutating inputs."""
    result = deepcopy(base)
    for key, override_value in override.items():
        base_value: Any = result.get(key)
        if isinstance(base_value, dict) and override_value is None:
            continue
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)
    return result


def load_config() -> dict:
    """Load defaults and merge the optional ``config.yaml`` from agent home."""
    config = deepcopy(DEFAULT_CONFIG)
    config_path = get_home() / "config.yaml"
    if not config_path.exists():
        return config
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return config
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml 顶层必须是映射")
    return deep_merge(config, loaded)

