"""Business-facing catalog for the 21 permission-governance checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionEvalCase:
    case_id: str
    node_id: str
    category: str
    scenario: str
    expected_decision: str = "pass"


def _case(
    case_id: str,
    node_id: str,
    category: str,
    scenario: str,
) -> PermissionEvalCase:
    return PermissionEvalCase(case_id, node_id, category, scenario)


CASES: tuple[PermissionEvalCase, ...] = (
    _case(
        "atomic-no-cross-tool-path",
        "tests/test_permission_tree.py::test_tool_and_file_permissions_cannot_cross_atoms",
        "atomicity",
        "A child cannot combine a tool with a file scope granted by another atom.",
    ),
    _case(
        "single-parent-atom-coverage",
        "tests/test_permission_tree.py::test_one_parent_atom_must_cover_the_complete_child_atom",
        "atomicity",
        "One and only one parent atom must cover the complete delegated atom.",
    ),
    _case(
        "file-tool-cannot-become-fileless",
        "tests/test_permission_tree.py::test_file_bound_tool_cannot_be_redeclared_as_fileless",
        "atomicity",
        "A file-bound tool cannot be redeclared as an unrestricted file-less capability.",
    ),
    _case(
        "equal-policy-node-reuse",
        "tests/test_permission_tree.py::test_equal_policy_reuses_parent_and_equal_child_policy_reuses_node",
        "policy-reuse",
        "Equivalent policies reuse the same node without duplicating authority.",
    ),
    _case(
        "single-agent-single-node",
        "tests/test_permission_tree.py::test_agent_cannot_be_attached_to_two_nodes",
        "node-isolation",
        "One Agent cannot be mounted on two permission nodes simultaneously.",
    ),
    _case(
        "sibling-node-isolation",
        "tests/test_permission_tree.py::test_sibling_agents_see_and_use_only_their_bound_atoms",
        "node-isolation",
        "Sibling Agents can see and execute only the atoms on their own nodes.",
    ),
    _case(
        "detach-prunes-empty-leaves",
        "tests/test_permission_tree.py::test_detach_prunes_only_empty_leaf_nodes",
        "lifecycle",
        "Detaching Agents prunes only newly empty leaf nodes and preserves live branches.",
    ),
    _case(
        "subtree-revocation-detaches-descendants",
        "tests/test_permission_tree.py::test_revoke_subtree_detaches_active_descendant_agents",
        "revocation",
        "Revoking a subtree removes descendant bindings and immediately denies later access.",
    ),
    _case(
        "root-revocation-denied",
        "tests/test_permission_tree.py::test_root_cannot_be_revoked",
        "revocation",
        "The permanent root permission node cannot be revoked.",
    ),
    _case(
        "workspace-path-containment",
        "tests/test_permission_tree.py::test_paths_cannot_escape_workspace",
        "path-security",
        "Relative traversal, absolute paths, and escaping symlinks cannot leave the workspace.",
    ),
    _case(
        "bound-node-controls-visible-and-executable-tools",
        "tests/test_permission_broker.py::test_bound_node_alone_controls_advertised_and_executable_tools",
        "runtime-authorization",
        "The bound node alone controls both model-visible tools and executable calls.",
    ),
    _case(
        "path-escape-denied-before-execution",
        "tests/test_permission_broker.py::test_path_escape_is_denied_before_tool_execution",
        "runtime-authorization",
        "A path-escape request is denied before the underlying tool function runs.",
    ),
    _case(
        "approval-does-not-grant-authority",
        "tests/test_permission_broker.py::test_high_risk_approval_is_separate_from_authority",
        "runtime-authorization",
        "Human approval is an additional gate and never substitutes for permission authority.",
    ),
    _case(
        "child-context-is-permission-filtered",
        "tests/test_agent_manager.py::test_child_context_inherits_prompt_and_filters_tools_and_skills",
        "delegation",
        "A child inherits task context while tools and Skills are filtered to its requested subset.",
    ),
    _case(
        "incompatible-skill-delegation-denied",
        "tests/test_agent_manager.py::test_incompatible_skill_is_not_delegated",
        "delegation",
        "A Skill whose required atoms exceed the child policy cannot be delegated.",
    ),
    _case(
        "implicit-skill-list-is-policy-filtered",
        "tests/test_agent_manager.py::test_unspecified_skills_are_filtered_to_child_policy",
        "delegation",
        "Automatically inherited Skills are still filtered by the child permission node.",
    ),
    _case(
        "shared-policy-unloads-after-last-agent",
        "tests/test_agent_manager.py::test_equal_child_policies_share_node_and_unload_after_results",
        "policy-reuse",
        "Equal child policies share a node which is removed only after the last Agent finishes.",
    ),
    _case(
        "global-budget-reservation-returned",
        "tests/test_agent_manager.py::test_global_budget_is_shared_and_unused_reservation_is_returned",
        "shared-budget",
        "Agents share one global budget and unused reservations are returned after settlement.",
    ),
    _case(
        "async-timeout-revokes-access",
        "tests/test_agent_manager.py::test_async_timeout_revokes_access_and_preserves_timeout_result",
        "async-revocation",
        "An asynchronous timeout revokes access, releases budget, and preserves the timeout result.",
    ),
    _case(
        "manager-cascade-revocation",
        "tests/test_agent_manager.py::test_revoking_permission_subtree_cancels_all_mounted_agents",
        "async-revocation",
        "Manager-level subtree revocation cancels every mounted descendant Agent.",
    ),
    _case(
        "distillation-cannot-invent-permissions",
        "tests/test_distiller.py::test_distill_removes_all_permissions_when_trajectory_has_no_tools",
        "distillation-governance",
        "A trajectory with no real tool calls must distill to a Skill with no permissions.",
    ),
)


if len(CASES) != 21 or len({case.case_id for case in CASES}) != 21:
    raise RuntimeError("permission benchmark must contain 21 unique cases")
