"""Track topic-local skill candidates and cache-stable frozen L0 metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time

from .retriever import SkillMatch


def skill_index_generation(index: list[dict]) -> str:
    """Return a stable generation id for hot-update detection."""
    projection = [
        {
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "triggers": item.get("triggers", []),
            "path": item.get("path", ""),
        }
        for item in sorted(index, key=lambda value: str(value.get("name", "")))
    ]
    raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass
class SkillLoadState:
    """State machine for one conversational skill-topic epoch."""

    phase: str = "warming"
    epoch_id: int = 1
    user_turn: int = 0
    epoch_turn: int = 0
    candidate_scores: dict[str, float] = field(default_factory=dict)
    recent_topk: list[tuple[str, ...]] = field(default_factory=list)
    frozen_names: tuple[str, ...] = ()
    stable_streak: int = 0
    drift_streak: int = 0
    last_turn_ts: float | None = None
    skill_generation: str = ""
    freeze_turn: int | None = None

    def observe(
        self,
        matches: list[SkillMatch],
        *,
        generation: str,
        now: float | None = None,
        max_frozen: int = 8,
        warm_min_turns: int = 2,
        warm_max_turns: int = 4,
        stable_overlap: float = 0.60,
        drift_overlap: float = 0.30,
        drift_turns: int = 2,
        idle_seconds: float = 6 * 60 * 60,
    ) -> list[str]:
        """Advance once per user turn and return state transition event names."""
        events: list[str] = []
        timestamp = time.time() if now is None else now
        idle = self.last_turn_ts is not None and timestamp - self.last_turn_ts >= idle_seconds
        self.last_turn_ts = timestamp
        self.user_turn += 1
        self.epoch_turn += 1
        names = tuple(match.name for match in matches)

        if self.skill_generation and generation != self.skill_generation:
            events.append("version_changed")
            self._start_new_epoch(names, matches)
        self.skill_generation = generation

        if self.phase == "frozen":
            if names:
                overlap = _jaccard(set(names), set(self.frozen_names))
                self.drift_streak = self.drift_streak + 1 if overlap < drift_overlap else 0
                required_drift = 1 if idle and overlap < drift_overlap else drift_turns
                if self.drift_streak >= required_drift:
                    self._start_new_epoch(names, matches)
                    events.append("thawed")
            return events

        previous = set(self.recent_topk[-1]) if self.recent_topk else set()
        current = set(names)
        if previous and current and _jaccard(previous, current) >= stable_overlap:
            self.stable_streak += 1
        else:
            self.stable_streak = 0
        self.recent_topk.append(names)
        self.recent_topk = self.recent_topk[-3:]
        for match in matches:
            self.candidate_scores[match.name] = (
                self.candidate_scores.get(match.name, 0.0) + match.score
            )

        stable_ready = self.epoch_turn >= warm_min_turns and self.stable_streak >= 1
        timeout_ready = self.epoch_turn >= warm_max_turns
        if self.candidate_scores and (stable_ready or timeout_ready):
            ranked = sorted(
                self.candidate_scores,
                key=lambda name: (-self.candidate_scores[name], name),
            )[:max_frozen]
            self.frozen_names = tuple(sorted(ranked))
            self.phase = "frozen"
            self.freeze_turn = self.user_turn
            self.drift_streak = 0
            events.append("frozen" if self.epoch_id == 1 else "refrozen")
        return events

    def _start_new_epoch(
        self,
        names: tuple[str, ...],
        matches: list[SkillMatch],
    ) -> None:
        self.phase = "warming"
        self.epoch_id += 1
        self.epoch_turn = 1
        self.candidate_scores = {match.name: match.score for match in matches}
        self.recent_topk = [names]
        self.frozen_names = ()
        self.stable_streak = 0
        self.drift_streak = 0
        self.freeze_turn = None
