"""Append and verify a newline-delimited SHA-256 audit hash chain."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from ..home import get_home

ZERO_HASH = "0" * 64


def _canonical(entry: dict) -> str:
    return json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def log(entry: dict) -> None:
    """Append an audit entry linked to the previous line's hash."""
    path = get_home() / "audit.log"
    previous = ZERO_HASH
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                previous = json.loads(lines[-1])["hash"]
            except (json.JSONDecodeError, KeyError, TypeError):
                previous = ZERO_HASH
    digest = hashlib.sha256((previous + _canonical(entry)).encode("utf-8")).hexdigest()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "prev_hash": previous,
        "entry": entry,
        "hash": digest,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical(record) + "\n")


def verify() -> bool:
    """Verify every link and digest in the current audit log."""
    path = get_home() / "audit.log"
    if not path.exists():
        return True
    previous = ZERO_HASH
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["prev_hash"] != previous:
                return False
            expected = hashlib.sha256(
                (previous + _canonical(record["entry"])).encode("utf-8")
            ).hexdigest()
            if record["hash"] != expected:
                return False
            previous = record["hash"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return True

