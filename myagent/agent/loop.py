"""Run one synchronous turn through context, permission, and evolution engines."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time
from typing import Callable

from ..audit.log import log as audit_log
from ..context.assembler import build_messages
from ..context.compressor import compress_if_needed
from ..evolution import ToolObservation, evaluate_turn, record_experience
from ..llm.types import Message, ToolCall
from ..memory.retrieve import mark_recalled, render_hits, retrieve_hits
from ..permissions import advertised_tools, check_and_execute
from ..skills.manager import load_skill
from ..skills.retriever import SkillMatch, retrieve_relevant
from ..skills.router import SkillRoute, route_skill_workset
from ..skills.session_state import skill_index_generation
from ..tools import registry
from .state import SessionState


def _skill_items_for_names(state: SessionState, names) -> list[dict]:
    by_name = {str(item.get("name", "")): item for item in state.skill_index}
    return [by_name[name] for name in names if name in by_name]


def _activate_primary_skill(
    state: SessionState,
    primary_name: str | None,
) -> str:
    if not primary_name:
        return ""
    by_name = {str(item.get("name", "")): item for item in state.skill_index}
    item = by_name.get(primary_name)
    try:
        if item is None or not item.get("path"):
            raise ValueError("skill index entry has no trusted path")
        skill = load_skill(Path(str(item["path"])))
        if skill.frontmatter.name != primary_name:
            raise ValueError("skill name changed after index loading")
    except (OSError, ValueError) as exc:
        audit_log(
            {
                "event": "skill_denied",
                "session_id": state.session_id,
                "skill": primary_name,
                "reason": type(exc).__name__,
                "message": str(exc),
            }
        )
        return ""

    required_tools = sorted(
        {item.tool for item in skill.frontmatter.permission_requirements}
    )
    available = (
        advertised_tools(state.permission_tree, state.agent_id)
        if state.permission_tree is not None and state.agent_id
        else set()
    )

    audit_log(
        {
            "event": "skill_activated",
            "session_id": state.session_id,
            "skill": skill.frontmatter.name,
            "required_tools": required_tools,
            "missing_tools": sorted(set(required_tools) - available),
            "note": "skill requirements do not grant or narrow authority",
        }
    )
    state.last_activated_skill_name = skill.frontmatter.name
    state.last_activated_skill_version = int(item.get("version", skill.version))
    artifact_id = item.get("artifact_id") if item is not None else None
    if state.artifact_manager is not None and artifact_id is not None:
        state.artifact_manager.record_exposure(
            int(artifact_id),
            run_id=(
                f"{state.session_id}:skill:{len(state.history)}:"
                f"{skill.frontmatter.name}"
            ),
            variant="treatment",
            activated=True,
        )
    return skill.body


def _prepare_legacy_skill_turn(
    state: SessionState,
    user_input: str,
) -> tuple[list[dict], str]:
    matches = retrieve_relevant(user_input, state.skill_index, k=state.skill_top_k)
    by_name = {str(item.get("name")): item for item in state.skill_index}
    if state.skill_strategy == "full_sidecar":
        selected = [item for item in state.skill_index if not item.get("always_on")]
    elif state.skill_strategy == "full_system":
        selected = []
    else:
        selected = [by_name[match.name] for match in matches if match.name in by_name]
    audit_log(
        {
            "event": "skill_retrieved",
            "session_id": state.session_id,
            "query_sha256": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "query_chars": len(user_input),
            "matches": [
                {
                    "name": match.name,
                    "score": match.score,
                    "reasons": list(match.reasons),
                }
                for match in matches
            ],
        }
    )

    if not matches or matches[0].score < state.skill_activation_min_score:
        return selected, ""
    return selected, _activate_primary_skill(state, matches[0].name)


def _weighted_workset_matches(route: SkillRoute) -> list[SkillMatch]:
    by_name = {match.name: match for match in route.matches}
    current = set(route.current_required)
    weighted: list[SkillMatch] = []
    for name in route.session_candidates:
        original = by_name.get(name)
        base_score = original.score if original is not None else 1.0
        weight = 1000.0 if name in current else 100.0
        weighted.append(
            SkillMatch(
                name=name,
                score=base_score + weight,
                reasons=("current-required" if name in current else "session-potential",),
            )
        )
    return weighted


def _fallback_route(state: SessionState, matches: list[SkillMatch]) -> SkillRoute:
    current = (
        (matches[0].name,)
        if matches and matches[0].score >= state.skill_activation_min_score
        else ()
    )
    candidates = tuple(match.name for match in matches[: state.skill_max_frozen])
    return SkillRoute("", current, candidates, tuple(matches), 0)


def _prepare_scene_epoch_skill_turn(
    state: SessionState,
    user_input: str,
    client,
) -> tuple[list[dict], str]:
    load_state = state.skill_load_state
    initial_system_state = (load_state.phase, load_state.frozen_names)
    now = time.time()
    transitions: list[str] = []
    expired = load_state.expire_if_idle(
        idle_seconds=state.skill_idle_seconds,
        now=now,
    )
    if expired:
        transitions.append(expired)
    local_matches = retrieve_relevant(
        user_input,
        state.skill_index,
        k=state.skill_candidate_k,
    )
    active_names = (
        load_state.frozen_names
        if load_state.phase == "frozen"
        else load_state.warming_names(state.skill_max_frozen)
    )
    active_set = set(active_names)
    local_active = next(
        (match for match in local_matches if match.name in active_set),
        None,
    )
    local_escape = next(
        (match for match in local_matches if match.name not in active_set),
        None,
    )
    confirmed_escape = bool(
        local_escape
        and local_escape.score >= state.skill_activation_min_score
        and local_escape.name in load_state.confirmed_fallback_names
    )
    needs_model_route = not active_names or bool(
        local_escape
        and local_escape.score >= state.skill_activation_min_score
        and not confirmed_escape
    )

    if not needs_model_route:
        current_match = local_escape if confirmed_escape else local_active
        route = SkillRoute(
            scene=load_state.scene,
            current_required=((current_match.name,) if current_match else ()),
            session_candidates=active_names,
            matches=tuple(local_matches),
        )
        route_mode = "confirmed_fallback" if confirmed_escape else "shadow_local"
    else:
        recent_inputs = tuple(
            message.content
            for message in state.history
            if message.role == "user" and message.content.strip()
        )[-4:]
        try:
            route = route_skill_workset(
                client,
                user_input,
                recent_inputs,
                state.skill_index,
                candidate_k=state.skill_candidate_k,
                max_current=1,
                max_session_candidates=state.skill_max_frozen,
                preferred_scene=load_state.forced_scene,
            )
            route_mode = "model_rerank"
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            route = _fallback_route(state, local_matches)
            route_mode = "local_fallback"
            audit_log(
                {
                    "event": "skill_router_failed",
                    "session_id": state.session_id,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )

    state.skill_router_prompt_tokens += route.prompt_tokens
    weighted = _weighted_workset_matches(route)
    transitions.extend(load_state.observe(
        weighted,
        generation=skill_index_generation(state.skill_index),
        scene=route.scene,
        now=now,
        max_frozen=state.skill_max_frozen,
        warm_turns=state.skill_warm_turns,
        idle_seconds=state.skill_idle_seconds,
    ))
    after_system_state = (load_state.phase, load_state.frozen_names)
    if initial_system_state != after_system_state:
        state.skill_cache_epoch_changes += 1
    if load_state.phase == "frozen" and route_mode == "model_rerank":
        frozen = set(load_state.frozen_names)
        load_state.confirm_fallbacks(
            tuple(name for name in route.current_required if name not in frozen),
            max_items=state.skill_max_frozen,
        )

    audit_log(
        {
            "event": "skill_workset_routed",
            "session_id": state.session_id,
            "query_sha256": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "query_chars": len(user_input),
            "mode": route_mode,
            "scene": load_state.scene or route.scene,
            "current_required": list(route.current_required),
            "session_candidates": list(route.session_candidates),
            "phase": load_state.phase,
            "epoch_id": load_state.epoch_id,
            "transitions": transitions,
            "router_prompt_tokens": route.prompt_tokens,
        }
    )

    if load_state.phase == "warming":
        sidecar_names = load_state.warming_names(state.skill_max_frozen)
    else:
        frozen = set(load_state.frozen_names)
        sidecar_names = tuple(
            name for name in route.current_required if name not in frozen
        )
    skill_items = _skill_items_for_names(state, sidecar_names)
    primary_name = route.current_required[0] if route.current_required else None
    skill_body = _activate_primary_skill(state, primary_name)
    return skill_items, skill_body


def _prepare_skill_turn(
    state: SessionState,
    user_input: str,
    client,
) -> tuple[list[dict], str]:
    """Retrieve L0 candidates and activate at most one L1 Skill."""
    if not user_input.strip() or not state.skill_index:
        return [], ""
    if state.skill_strategy in {"warm_stable", "scene_epoch"}:
        return _prepare_scene_epoch_skill_turn(state, user_input, client)
    return _prepare_legacy_skill_turn(state, user_input)


def run_turn(
    state: SessionState,
    user_input: str,
    client,
    approve_fn: Callable[[ToolCall], bool],
) -> str:
    """Run model/tool steps synchronously until a final answer or step limit.
    state 是当前 Agent 会话的运行上下文
    state.history                 # 历史对话
    state.permission_tree         # 权限树
    state.agent_id                # 当前 Agent 身份
    state.memory_conn             # Memory 的 SQLite 连接
    state.artifact_manager        # Memory/Skill 生命周期管理器
    state.skill_index             # 当前可检索的 Skill 元数据
    state.skill_load_state        # Skill 预热、固化状态
    state.last_recalled_memory_ids
    state.last_recorded_memory_id
    state.step_count
    user_input 用户当前轮次输入
    client 大模型调用客户端
    """
    if state.permission_tree is None or state.permission_adapter is None or not state.agent_id:
        raise RuntimeError("session has no atomic permission-tree binding")
    max_steps = state.max_steps
    state.compression_client = client # 上下文压缩 智能体
    original_input = user_input
    state.last_activated_skill_name = "" # 清空上一轮skill 名称
    state.last_activated_skill_version = None
    # The complete Sidecar lives only in this transient per-turn prefix.  It is
    # sent once, reused as an earlier (cacheable) message during tool steps, and
    # never committed to durable conversation history.
    transient_turn: list[Message] = []  # 创建当前轮的临时消息列表。
    durable_turn: list[Message] = [] # 创建当前轮需要持久保留的消息列表。剥离只对当前轮有效的 Memory、Skill 和检索信息，避免这些动态内容进入长期历史并反复污染后续上下文。
    recorded_input = False
    #  每轮任务开始时，系统先根据用户问题和当前会话工作集执行 Skill 路由。
    skill_items, skill_body = _prepare_skill_turn(
        state,
        original_input,
        client,
    )
    # Reuse exactly one durable channel per turn.  A high-confidence L1 Skill
    # already contains the generalized procedure, so adding historical answers
    # would duplicate dynamic Sidecar text.  Without an activated Skill, recall
    # at most one concrete experience and suppress unrelated L0 rows.
    state.last_recalled_memory_ids = ()
    state.last_recalled_memory_scope = ""
    state.last_recorded_memory_id = None
    memory_hits = (
        retrieve_hits(
            state.memory_conn, # state.memory_conn 数据库连接
            original_input, # 用户原始输入
            k=1,
            scope_key=state.memory_scope,# 只允许召回当前项目作用域内的具体经验。
            artifact_statuses=state.evolution_artifact_statuses, # 只允许召回当前项目作用域内的具体经验。
        )
        if not skill_body and state.memory_conn is not None and original_input.strip()
        else []
    )
    retrieved = render_hits(memory_hits) # 将记忆转换为可以召回的字符串
    if memory_hits and state.memory_conn is not None:
        state.last_recalled_memory_ids = tuple(hit.decision_id for hit in memory_hits)
        state.last_recalled_memory_scope = memory_hits[0].scope_key
        mark_recalled(state.memory_conn, state.last_recalled_memory_ids)  # 数据库中登记“它这次被用过”
        if state.artifact_manager is not None:
            for hit in memory_hits:
                state.artifact_manager.record_exposure(
                    hit.artifact_id,
                    run_id=(
                        f"{state.session_id}:memory:{len(state.history)}:"
                        f"{hit.decision_id}"
                    ),
                    variant="treatment",
                    activated=True,
                )
    if retrieved:
        skill_items = []
    observations: list[ToolObservation] = []

    try:
        for _ in range(max_steps):
            state.step_count += 1
            compress_if_needed(state)
            messages = build_messages(
                state,
                user_input,
                retrieved,
                skill_items=skill_items,
                skill_body=skill_body,
                turn_messages=transient_turn,
            )
            response = client.chat(
                messages,
                tools=registry.schemas(advertised_tools(state.permission_tree, state.agent_id)),
                # advertised_tools是模型调用前的“工具可见性门禁”
            )

            if not recorded_input:
                # The actual Sidecar is retained only while this turn executes;
                # the durable history keeps the original user text instead.
                transient_turn.append(messages[-1])
                durable_turn.append(Message(role="user", content=original_input))
                recorded_input = True

            assistant = Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls or None,
                reasoning_content=response.reasoning_content,
            )
            transient_turn.append(assistant)
            durable_turn.append(assistant)

            if not response.tool_calls:
                decision = evaluate_turn(response.content or "", observations)
                # 根据当前轮真实工具轨迹，判断该任务是否有资格沉淀为 Memory，或者进一步用于 Skill 蒸馏；如果没有资格，就返回具体原因。
                state.last_evolution_eligible = decision.eligible
                state.last_evolution_reasons = decision.reasons
                state.last_evolution_history = (
                    list(durable_turn) if decision.eligible else []
                )
                # 用户任务、相关文件、最终解决结论和验证证据；
                state.last_recorded_memory_id = record_experience(
                    state.memory_conn,
                    state.session_id,
                    original_input, # 用户原始输入，进入task 字段
                    response.content or "", # 最终结论 lesson 字段
                    observations,
                    decision,
                    state.memory_scope,
                )
                return response.content or ""

            for call in response.tool_calls:
                result = check_and_execute(
                    call,
                    state.permission_tree,
                    state.agent_id,
                    state.permission_adapter,
                    approve_fn,
                )
                observations.append(ToolObservation(call, result))
                tool_message = Message(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False, sort_keys=True),
                    tool_call_id=call.id,
                )
                transient_turn.append(tool_message)
                durable_turn.append(tool_message)

            # The initial Sidecar already lives in transient_turn.  Reusing that
            # exact prefix avoids appending the same dynamic memory after every
            # tool result.
            user_input = ""
            retrieved = ""
            skill_items = []
            skill_body = ""

        state.last_evolution_eligible = False
        state.last_evolution_reasons = ("step_limit_reached",)
        state.last_evolution_history = []
        return "达到步数上限"
    finally:
        state.history.extend(durable_turn)

