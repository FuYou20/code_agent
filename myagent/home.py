"""Resolve and create the agent's single runtime home directory."""

from __future__ import annotations

import os
from pathlib import Path


def get_home() -> Path:
    """Return ``MYAGENT_HOME`` or ``~/.myagent`` and create it if needed."""
    configured = os.environ.get("MYAGENT_HOME")
    home = Path(configured).expanduser() if configured else Path.home() / ".myagent"
    home.mkdir(parents=True, exist_ok=True)
    return home

