"""Rank L0 skill metadata deterministically without calling an LLM."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TERM_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u3400-\u9fff]+")
_STOP_TERMS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "use",
    "with",
    "使用",
    "处理",
    "用于",
}


@dataclass(frozen=True)
class SkillMatch:
    """One explainable L0 retrieval result."""

    name: str
    score: float
    reasons: tuple[str, ...]


def _normalise(value: object) -> str:
    return " ".join(str(value).casefold().replace("_", "-").split())


def _terms(value: object) -> set[str]:
    return {
        term
        for term in _TERM_RE.findall(_normalise(value))
        if len(term) > 1 and term not in _STOP_TERMS
    }


def retrieve_relevant(query: str, index: list[dict], k: int = 5) -> list[SkillMatch]:
    """Return the best L0 matches using triggers first and metadata overlap second.

    The scorer is deliberately deterministic and explainable.  It is a local
    candidate generator, not a claim that a model has successfully applied a
    skill body.
    """
    if k <= 0 or not query.strip():
        return []

    normalised_query = _normalise(query)
    query_terms = _terms(query)
    ranked: list[SkillMatch] = []

    for item in index:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        score = 0.0
        reasons: list[str] = []

        normalised_name = _normalise(name).replace("-", " ")
        if normalised_name and normalised_name in normalised_query:
            score += 80.0
            reasons.append("name-exact")

        for trigger_value in item.get("triggers", []):
            trigger = _normalise(trigger_value)
            if trigger and trigger in normalised_query:
                score += 100.0 + min(len(trigger), 40) / 10
                reasons.append(f"trigger:{trigger_value}")

        name_overlap = query_terms & _terms(name.replace("-", " "))
        trigger_overlap = query_terms & _terms(" ".join(map(str, item.get("triggers", []))))
        description_overlap = query_terms & _terms(item.get("description", ""))
        if name_overlap:
            score += 14.0 * len(name_overlap)
            reasons.append("name-terms:" + ",".join(sorted(name_overlap)))
        if trigger_overlap:
            score += 9.0 * len(trigger_overlap)
            reasons.append("trigger-terms:" + ",".join(sorted(trigger_overlap)))
        if description_overlap:
            score += 3.0 * len(description_overlap)
            reasons.append("description-terms:" + ",".join(sorted(description_overlap)))

        if score > 0:
            ranked.append(SkillMatch(name=name, score=score, reasons=tuple(reasons)))

    ranked.sort(key=lambda match: (-match.score, match.name))
    return ranked[:k]
