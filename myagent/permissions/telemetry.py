"""Optional JSONL permission events used by deterministic evaluations.

The production permission system has no Langfuse dependency. Events are emitted
only when ``MYAGENT_PERMISSION_TRACE_FILE`` is explicitly set by an evaluator.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from threading import RLock, get_ident
import time
from typing import Any


_LOCK = RLock()
_SEQUENCE = itertools.count(1)


def atom_payload(atom: Any) -> dict[str, object]:
    """Return a stable, non-secret representation of a permission atom."""
    return {
        "tool": str(atom.tool),
        "files": [
            {"mode": str(grant.mode), "root": str(grant.root)}
            for grant in atom.files
        ],
    }


def emit_permission_event(name: str, **data: object) -> None:
    """Append one event when permission tracing is enabled for this process."""
    target = os.environ.get("MYAGENT_PERMISSION_TRACE_FILE", "").strip()
    if not target:
        return
    with _LOCK:
        event = {
            "sequence": next(_SEQUENCE),
            "timestamp_ns": time.time_ns(),
            "thread_id": get_ident(),
            "name": name,
            "data": data,
        }
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
