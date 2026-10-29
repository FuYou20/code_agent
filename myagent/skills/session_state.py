"""Track one explicit scene epoch and its immutable frozen Skill workset."""

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
            "scene": item.get("scene", "general"),
            "triggers": item.get("triggers", []),
            "path": item.get("path", ""),
        }
        for item in sorted(index, key=lambda value: str(value.get("name", "")))
    ]
    raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class SkillLoadState:
    """Accumulate a scene workset, freeze it once, and reset only explicitly."""

    phase: str = "warming"
    epoch_id: int = 1
    user_turn: int = 0
    epoch_turn: int = 0
    candidate_scores: dict[str, float] = field(default_factory=dict)
    frozen_names: tuple[str, ...] = ()
    confirmed_fallback_names: tuple[str, ...] = ()
    scene: str = ""
    forced_scene: str = ""
    last_turn_ts: float | None = None
    skill_generation: str = ""
    freeze_turn: int | None = None

    def warming_names(self, max_items: int = 8) -> tuple[str, ...]:
        """Return the cumulative recall-first workset visible during warmup."""
        ranked = sorted(
            self.candidate_scores,
            key=lambda name: (-self.candidate_scores[name], name),
        )[:max_items]
        return tuple(ranked)

    def reset(self, *, reason: str, scene: str = "") -> str:
        """Start a new epoch after a user command or an idle timeout."""
        self.phase = "warming"
        self.epoch_id += 1
        self.epoch_turn = 0
        self.candidate_scores.clear()
        self.frozen_names = ()
        self.confirmed_fallback_names = ()
        self.scene = scene.strip()
        self.forced_scene = scene.strip()
        self.freeze_turn = None
        self.last_turn_ts = None
        return reason

    def confirm_fallbacks(
        self,
        names: tuple[str, ...],
        *,
        max_items: int = 8,
    ) -> None:
        """Remember model-confirmed Sidecar fallbacks without changing the workset."""
        merged = list(self.confirmed_fallback_names)
        for name in names:
            if name and name not in merged:
                merged.append(name)
        self.confirmed_fallback_names = tuple(merged[-max_items:])

    def expire_if_idle(
        self,
        *,
        idle_seconds: float,
        now: float | None = None,
    ) -> str | None:
        """Reset before routing when the current epoch has expired."""
        if idle_seconds <= 0:
            raise ValueError("idle_seconds must be greater than zero")
        timestamp = time.time() if now is None else now
        if (
            self.last_turn_ts is not None
            and timestamp - self.last_turn_ts >= idle_seconds
        ):
            return self.reset(reason="timeout_reset")
        return None

    def observe(
        self,
        matches: list[SkillMatch],
        *,
        generation: str,
        scene: str = "",
        now: float | None = None,
        max_frozen: int = 8,
        warm_turns: int = 3,
        idle_seconds: float = 6 * 60 * 60,
    ) -> list[str]:
        """Advance the epoch without any content-driven automatic thaw."""
        if warm_turns <= 0:
            raise ValueError("warm_turns must be greater than zero")
        events: list[str] = []
        timestamp = time.time() if now is None else now
        expired = self.expire_if_idle(idle_seconds=idle_seconds, now=timestamp)
        if expired:
            events.append(expired)
        self.last_turn_ts = timestamp
        self.user_turn += 1
        self.epoch_turn += 1
        self.skill_generation = generation

        if self.phase == "frozen":
            return events

        if scene and not self.forced_scene:
            self.scene = scene.strip()[:80]

        for match in matches:
            self.candidate_scores[match.name] = (
                self.candidate_scores.get(match.name, 0.0) + match.score
            )

        if self.candidate_scores and self.epoch_turn >= warm_turns:
            self.frozen_names = tuple(sorted(self.warming_names(max_frozen)))
            self.phase = "frozen"
            self.freeze_turn = self.user_turn
            events.append("frozen")
        return events
