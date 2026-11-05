"""Execute one permission case through its existing pytest truth source."""

from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time

from .dataset import PermissionEvalCase


def run_case(
    case: PermissionEvalCase,
    repo_root: str | Path,
    *,
    python: str = sys.executable,
    timeout_seconds: float = 60,
) -> dict[str, object]:
    """Run exactly one original check and return a Langfuse-safe result row."""
    root = Path(repo_root).resolve()
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    with tempfile.TemporaryDirectory(prefix="permission-events-") as temp:
        trace_file = Path(temp) / "events.jsonl"
        environment["MYAGENT_PERMISSION_TRACE_FILE"] = str(trace_file)
        try:
            completed = subprocess.run(
                [python, "-m", "pytest", "-q", case.node_id],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            passed = completed.returncode == 0
            events = _read_events(trace_file)
            return {
                "case_id": case.case_id,
                "category": case.category,
                "passed": passed,
                "expected_decision": case.expected_decision,
                "actual_decision": "pass" if passed else "fail",
                "returncode": completed.returncode,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "failure_type": "NONE" if passed else "ASSERTION_OR_RUNTIME_FAILURE",
                "permission_event_count": len(events),
                "permission_events": events,
                "test_output": output[-8_000:],
            }
        except subprocess.TimeoutExpired as exc:
            output = "".join(
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                for value in (exc.stdout, exc.stderr)
            )
            events = _read_events(trace_file)
            return {
                "case_id": case.case_id,
                "category": case.category,
                "passed": False,
                "expected_decision": case.expected_decision,
                "actual_decision": "timeout",
                "returncode": -1,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "failure_type": "INFRASTRUCTURE_TIMEOUT",
                "permission_event_count": len(events),
                "permission_events": events,
                "test_output": output[-8_000:],
            }


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return sorted(events, key=lambda item: int(item["sequence"]))
