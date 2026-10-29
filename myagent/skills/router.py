"""Route a user turn into immediate and topic-potential Skill selections."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from ..llm.types import Message
from .retriever import SkillMatch, retrieve_relevant


@dataclass(frozen=True)
class SkillRoute:
    """One model-assisted routing decision for a conversational scene epoch."""

    scene: str
    current_required: tuple[str, ...]
    session_candidates: tuple[str, ...]
    matches: tuple[SkillMatch, ...]
    prompt_tokens: int = 0

    @property
    def topic(self) -> str:
        """Compatibility alias for older audit and evaluation consumers."""
        return self.scene


def _json_object(content: str) -> dict:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Skill router response must be a JSON object")
    return value


def _unique_allowed(values: object, allowed: set[str], limit: int) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    selected: list[str] = []
    for value in values:
        name = str(value)
        if name in allowed and name not in selected:
            selected.append(name)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _render_candidates(candidates: list[dict]) -> str:
    return "\n".join(
        json.dumps(
            {
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "scene": item.get("scene", "general"),
                "triggers": item.get("triggers", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in candidates
    )


def route_skill_workset(
    client,
    query: str,
    recent_user_inputs: tuple[str, ...],
    index: list[dict],
    *,
    candidate_k: int = 20,
    max_current: int = 2,
    max_session_candidates: int = 8,
    preferred_scene: str = "",
) -> SkillRoute:
    """Classify a scene, retrieve globally, and plan one session workset."""
    history = "\n".join(f"- {item}" for item in recent_user_inputs[-4:]) or "(无)"
    scenes = sorted(
        {str(item.get("scene", "general")).strip() or "general" for item in index}
    )
    scene_text = ", ".join(scenes)
    forced_text = preferred_scene.strip() or "(未指定)"
    expansion = client.chat(
        [
            Message(
                role="system",
                content=(
                    "判断这段会话所属场景，并把当前请求和最近用户意图改写成用于 Skill 检索的短语。"
                    f"scene 只能从这些值选择：{scene_text}。用户指定场景优先。"
                    "输出严格 JSON：{\"scene\":\"scene-name\",\"terms\":[\"term 1\"]}。"
                    "最多8个中英文技术短语；"
                    "覆盖当前动作及同一任务链中可能紧接着发生的动作，不得猜测 Skill 名称。"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"用户指定场景：{forced_text}\n\n最近用户意图：\n{history}"
                    f"\n\n当前请求：\n{query}"
                ),
            ),
        ],
        tools=[],
    )
    payload = _json_object(expansion.content or "")
    raw_terms = payload.get("terms", [])
    if not isinstance(raw_terms, list):
        raise ValueError("Skill query expansion terms must be a list")
    proposed_scene = str(payload.get("scene", "")).strip()
    selected_scene = preferred_scene.strip() or (
        proposed_scene if proposed_scene in scenes else ""
    )
    expanded_query = " ".join([query, *(str(term) for term in raw_terms[:8])])
    global_matches = retrieve_relevant(expanded_query, index, k=candidate_k)
    expansion_tokens = int(expansion.usage.get("prompt_tokens", 0))

    by_name = {str(item.get("name", "")): item for item in index}
    candidate_names: list[str] = []
    if selected_scene:
        scene_index = [
            item for item in index if str(item.get("scene", "general")) == selected_scene
        ]
        for match in retrieve_relevant(expanded_query, scene_index, k=candidate_k):
            if match.name not in candidate_names:
                candidate_names.append(match.name)
        for item in scene_index:
            name = str(item.get("name", ""))
            if name and name not in candidate_names:
                candidate_names.append(name)
    for match in global_matches:
        if match.name not in candidate_names:
            candidate_names.append(match.name)
    candidate_names = candidate_names[:candidate_k]
    if not candidate_names:
        return SkillRoute(selected_scene, (), (), (), expansion_tokens)

    candidates = [by_name[name] for name in candidate_names if name in by_name]
    allowed = {str(item.get("name", "")) for item in candidates}
    reranked = client.chat(
        [
            Message(
                role="system",
                content=(
                    "你是会话场景的 Skill 工作集规划器。根据场景、当前请求、最近用户意图和候选 Skill，"
                    f"分别选择：current_required（当前任务立即必需，最多{max_current}个）和 "
                    f"session_candidates（同一话题未来几轮较可能使用，最多{max_session_candidates}个）。"
                    "session_candidates 应以召回为先，覆盖该场景未来多轮可能发生的自然步骤；"
                    "它只用于能力目录，不能扩大权限。闲聊或都不适用时返回空数组。"
                    "只能使用候选 name。输出严格 JSON："
                    "{\"scene\":\"scene-name\",\"current_required\":[],"
                    "\"session_candidates\":[]}。"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"当前场景：{selected_scene or '(待确认)'}"
                    f"\n\n最近用户意图：\n{history}\n\n当前请求：\n{query}"
                    f"\n\n候选 Skill：\n{_render_candidates(candidates)}"
                ),
            ),
        ],
        tools=[],
    )
    routed = _json_object(reranked.content or "")
    current = _unique_allowed(routed.get("current_required"), allowed, max_current)
    potential = list(
        _unique_allowed(
            routed.get("session_candidates"),
            allowed,
            max_session_candidates,
        )
    )
    for name in current:
        if name not in potential:
            potential.insert(0, name)
    potential = potential[:max_session_candidates]
    routed_scene = str(routed.get("scene", "")).strip()
    final_scene = selected_scene or (routed_scene if routed_scene in scenes else "")
    match_by_name = {match.name: match for match in global_matches}
    matches = tuple(
        match_by_name.get(
            name,
            SkillMatch(name=name, score=1.0, reasons=(f"scene:{final_scene}",)),
        )
        for name in candidate_names
    )
    return SkillRoute(
        scene=final_scene,
        current_required=current,
        session_candidates=tuple(potential),
        matches=matches,
        prompt_tokens=(
            expansion_tokens + int(reranked.usage.get("prompt_tokens", 0))
        ),
    )
