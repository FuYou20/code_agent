"""Verify permission-bound child creation, execution, budgeting, and unloading."""

from threading import Event

import pytest

from myagent.permissions import PermissionAtom, PermissionTree
from myagent.subagents import (
    AgentOutput,
    AgentStatus,
    BudgetExceeded,
    CreateAgentRequest,
    GlobalBudgetLedger,
    SkillDescription,
    ToolDescription,
)
from myagent.subagents.manager import AgentManager, ContextVisibilityDenied


@pytest.fixture
def manager(tmp_path):
    root_atoms = (
        PermissionAtom.read("file_read", "."),
        PermissionAtom.write("file_write", "generated"),
        PermissionAtom.read_write(
            "test_run",
            read=(".",),
            write=(".pytest_cache",),
        ),
    )
    tree = PermissionTree(tmp_path, root_atoms)
    budget = GlobalBudgetLedger(
        input_tokens=10_000,
        output_tokens=2_000,
        tool_calls=20,
    )
    value = AgentManager(tree, budget, max_workers=2)
    value.register_root_agent(
        "root-agent",
        prompt="parent system prompt",
        tools=(
            ToolDescription("file_read", "Read a file"),
            ToolDescription("file_write", "Write a file"),
            ToolDescription("test_run", "Run tests"),
        ),
        skills=(
            SkillDescription(
                "read-repo",
                "Read repository files",
                (PermissionAtom.read("file_read", "."),),
            ),
            SkillDescription(
                "write-report",
                "Write generated reports",
                (PermissionAtom.write("file_write", "generated"),),
            ),
        ),
    )
    yield value
    value.shutdown()


def test_child_context_inherits_prompt_and_filters_tools_and_skills(manager) -> None:
    child = manager.create_agent(
        CreateAgentRequest(
            parent_agent_id="root-agent",
            agent_id="reader",
            task="inspect src/auth",
            parent_context_summary="login tests are failing",
            requested_atoms=(PermissionAtom.read("file_read", "src/auth"),),
            requested_skills=("read-repo",),
        )
    )
    context = manager.context(child)
    assert context.inherited_prompt == "parent system prompt"
    assert context.task_prompt == "inspect src/auth"
    assert context.parent_context_summary == "login tests are failing"
    assert [tool.name for tool in context.tools] == ["file_read"]
    assert [skill.name for skill in context.skills] == ["read-repo"]
    assert manager.allows(
        child,
        PermissionAtom.read("file_read", "src/auth/token.py"),
    )
    assert not manager.allows(
        child,
        PermissionAtom.write("file_write", "generated/report.md"),
    )


def test_incompatible_skill_is_not_delegated(manager) -> None:
    with pytest.raises(ContextVisibilityDenied):
        manager.create_agent(
            CreateAgentRequest(
                parent_agent_id="root-agent",
                agent_id="bad-skill-child",
                task="read only",
                requested_atoms=(PermissionAtom.read("file_read", "src"),),
                requested_skills=("write-report",),
            )
        )
    with pytest.raises(KeyError):
        manager.context("bad-skill-child")


def test_unspecified_skills_are_filtered_to_child_policy(manager) -> None:
    child = manager.create_agent(
        CreateAgentRequest(
            parent_agent_id="root-agent",
            agent_id="auto-filtered-child",
            task="read only",
            requested_atoms=(PermissionAtom.read("file_read", "src"),),
        )
    )
    assert [skill.name for skill in manager.context(child).skills] == ["read-repo"]


def test_equal_child_policies_share_node_and_unload_after_results(manager) -> None:
    request = lambda agent_id: CreateAgentRequest(
        parent_agent_id="root-agent",
        agent_id=agent_id,
        task="inspect",
        requested_atoms=(PermissionAtom.read("file_read", "src"),),
        requested_skills=("read-repo",),
    )
    first = manager.create_agent(request("reader-a"))
    second = manager.create_agent(request("reader-b"))
    assert manager._record(first).node_id == manager._record(second).node_id
    shared_node = manager._record(first).node_id

    first_result = manager.run_agent(first, lambda execution: "first complete")
    assert first_result.status == AgentStatus.SUCCEEDED
    assert manager.permission_tree.node_exists(shared_node)

    second_result = manager.run_agent(
        second,
        lambda execution: AgentOutput(
            summary="second complete",
            data={"files": ["src/main.py"]},
        ),
    )
    assert second_result.data == {"files": ["src/main.py"]}
    assert not manager.permission_tree.node_exists(shared_node)
    delivered = manager.drain_results("root-agent")
    assert [result.agent_id for result in delivered] == ["reader-a", "reader-b"]


def test_global_budget_is_shared_and_unused_reservation_is_returned(manager) -> None:
    first = manager.budget.reserve("agent-a", input_tokens=8_000, tool_calls=10)
    with pytest.raises(BudgetExceeded):
        manager.budget.reserve("agent-b", input_tokens=3_000)
    manager.budget.settle(
        first.reservation_id,
        input_tokens=2_000,
        output_tokens=0,
        tool_calls=2,
    )
    snapshot = manager.budget.snapshot()
    assert snapshot.input_tokens == 8_000
    assert snapshot.tool_calls == 18


def test_async_timeout_revokes_access_and_preserves_timeout_result(manager) -> None:
    child = manager.create_agent(
        CreateAgentRequest(
            parent_agent_id="root-agent",
            agent_id="slow-reader",
            task="wait for cancellation",
            requested_atoms=(PermissionAtom.read("file_read", "src"),),
        )
    )
    worker_stopped = Event()

    def worker(execution):
        execution.reserve_budget(input_tokens=100, tool_calls=1)
        execution.wait_cancelled(2)
        worker_stopped.set()
        return "late result"

    manager.start_agent(child, worker)
    result = manager.wait(child, timeout=0.01)
    assert result.status == AgentStatus.TIMED_OUT
    assert worker_stopped.wait(1)
    assert not manager.allows(
        child,
        PermissionAtom.read("file_read", "src/main.py"),
    )
    assert manager.wait(child).status == AgentStatus.TIMED_OUT
    snapshot = manager.budget.snapshot()
    assert snapshot.input_tokens == 10_000
    assert snapshot.tool_calls == 20


def test_revoking_permission_subtree_cancels_all_mounted_agents(manager) -> None:
    parent = manager.create_agent(
        CreateAgentRequest(
            parent_agent_id="root-agent",
            agent_id="parent-child",
            task="parent task",
            requested_atoms=(PermissionAtom.read("file_read", "src"),),
        )
    )
    grandchild = manager.create_agent(
        CreateAgentRequest(
            parent_agent_id=parent,
            agent_id="grandchild",
            task="nested task",
            requested_atoms=(PermissionAtom.read("file_read", "src/auth"),),
        )
    )
    node_id = manager._record(parent).node_id

    results = manager.revoke_permission_subtree(node_id)
    assert {result.agent_id for result in results} == {parent, grandchild}
    assert manager.status(parent) == AgentStatus.CANCELLED
    assert manager.status(grandchild) == AgentStatus.CANCELLED
    assert not manager.allows(
        grandchild,
        PermissionAtom.read("file_read", "src/auth/token.py"),
    )
