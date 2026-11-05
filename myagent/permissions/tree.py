"""Manage a monotonically attenuating tree of atomic tool-file permissions."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
import uuid

from .model import FileGrant, PermissionAtom
from .telemetry import atom_payload, emit_permission_event


class PermissionEscalationDenied(Exception):
    """Raised when requested child atoms are not covered by the parent node."""


class UnknownNode(KeyError):
    """Raised when a permission node does not exist or is no longer active."""


class UnknownAgent(KeyError):
    """Raised when an agent has no active permission-tree binding."""


class AgentAlreadyAttached(Exception):
    """Raised when an agent is already attached to a different node."""


class RootRevocationDenied(Exception):
    """Raised when code attempts to delete the permanent root node."""


@dataclass
class _PermissionNode:
    node_id: str
    parent_id: str | None
    atoms: tuple[PermissionAtom, ...]
    policy_hash: str
    attached_agents: set[str] = field(default_factory=set)
    child_nodes: set[str] = field(default_factory=set)
    state: str = "active"


@dataclass(frozen=True)
class PermissionNodeView:
    """A safe manager-facing snapshot without mutable tree collections."""

    node_id: str
    parent_id: str | None
    atoms: tuple[PermissionAtom, ...]
    policy_hash: str
    attached_agent_count: int
    child_node_count: int
    state: str


def _atom_key(atom: PermissionAtom) -> tuple:
    return (
        atom.tool,
        tuple((grant.mode, grant.root) for grant in atom.files),
    )


def _canonical_atoms(atoms) -> tuple[PermissionAtom, ...]:
    values = tuple(atoms)
    if any(not isinstance(atom, PermissionAtom) for atom in values):
        raise TypeError("permission nodes accept only PermissionAtom values")
    return tuple(sorted(set(values), key=_atom_key))


def _policy_hash(atoms: tuple[PermissionAtom, ...]) -> str:
    payload = [
        {
            "tool": atom.tool,
            "files": [
                {"mode": grant.mode, "root": grant.root}
                for grant in atom.files
            ],
        }
        for atom in atoms
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PermissionTree:
    """Bind agents to reusable nodes whose atoms can only narrow by delegation."""

    ROOT_ID = "root"

    def __init__(self, workspace_root: str | Path, root_atoms) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        atoms = _canonical_atoms(root_atoms)
        self._validate_atoms(atoms)
        root = _PermissionNode(
            node_id=self.ROOT_ID,
            parent_id=None,
            atoms=atoms,
            policy_hash=_policy_hash(atoms),
        )
        self._nodes: dict[str, _PermissionNode] = {self.ROOT_ID: root}
        self._agent_nodes: dict[str, str] = {}
        self._lock = RLock()
        emit_permission_event(
            "permission.tree.initialized",
            root_id=self.ROOT_ID,
            atom_count=len(atoms),
            policy_hash=root.policy_hash,
        )

    def _resolve_root(self, logical_root: str) -> Path:
        relative = Path(*logical_root.split("/")) if logical_root != "." else Path()
        resolved = (self.workspace_root / relative).resolve(strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"permission path resolves outside workspace: {logical_root!r}"
            ) from exc
        return resolved

    @staticmethod
    def _path_contains(parent: Path, child: Path) -> bool:
        parent_text = os.path.normcase(str(parent))
        child_text = os.path.normcase(str(child))
        try:
            Path(child_text).relative_to(Path(parent_text))
        except ValueError:
            return False
        return True

    def _grant_covers(self, parent: FileGrant, child: FileGrant) -> bool:
        if parent.mode != child.mode:
            return False
        return self._path_contains(
            self._resolve_root(parent.root),
            self._resolve_root(child.root),
        )

    def atom_covers(self, parent: PermissionAtom, child: PermissionAtom) -> bool:
        """Return whether one parent atom wholly covers one child/request atom."""
        if parent.tool != child.tool:
            return False
        if not child.files:
            return not parent.files
        return all(
            any(self._grant_covers(parent_grant, child_grant) for parent_grant in parent.files)
            for child_grant in child.files
        )

    def atom_intersects(self, left: PermissionAtom, right: PermissionAtom) -> bool:
        """Return whether two atoms have a usable common tool/file scope.

        This is intentionally weaker than :meth:`atom_covers`.  It is used only
        to decide whether a Skill description remains useful in an already
        attenuated child context; runtime authorization still uses ``allows``.
        """
        if left.tool != right.tool:
            return False
        if not left.files or not right.files:
            return not left.files and not right.files
        return all(
            any(
                left_grant.mode == right_grant.mode
                and (
                    self._grant_covers(left_grant, right_grant)
                    or self._grant_covers(right_grant, left_grant)
                )
                for left_grant in left.files
            )
            for right_grant in right.files
        )

    def _validate_atoms(self, atoms: tuple[PermissionAtom, ...]) -> None:
        for atom in atoms:
            for grant in atom.files:
                self._resolve_root(grant.root)

    def _active_node(self, node_id: str) -> _PermissionNode:
        node = self._nodes.get(node_id)
        if node is None or node.state != "active":
            raise UnknownNode(node_id)
        return node

    @staticmethod
    def _view(node: _PermissionNode) -> PermissionNodeView:
        return PermissionNodeView(
            node_id=node.node_id,
            parent_id=node.parent_id,
            atoms=node.atoms,
            policy_hash=node.policy_hash,
            attached_agent_count=len(node.attached_agents),
            child_node_count=len(node.child_nodes),
            state=node.state,
        )

    def node(self, node_id: str) -> PermissionNodeView:
        with self._lock:
            return self._view(self._active_node(node_id))

    def node_exists(self, node_id: str) -> bool:
        with self._lock:
            node = self._nodes.get(node_id)
            return node is not None and node.state == "active"

    def get_or_create_child(self, parent_id: str, requested_atoms) -> str:
        """Reuse an equal policy node or create a strictly attenuated child."""
        atoms = _canonical_atoms(requested_atoms)
        self._validate_atoms(atoms)
        emit_permission_event(
            "permission.child.requested",
            parent_node_id=parent_id,
            requested_atoms=[atom_payload(atom) for atom in atoms],
        )
        with self._lock:
            parent = self._active_node(parent_id)
            emit_permission_event(
                "permission.node.loaded",
                node_id=parent.node_id,
                state=parent.state,
                atom_count=len(parent.atoms),
                policy_hash=parent.policy_hash,
            )
            uncovered = []
            for atom in atoms:
                covering = [
                    parent_atom
                    for parent_atom in parent.atoms
                    if self.atom_covers(parent_atom, atom)
                ]
                emit_permission_event(
                    "permission.atom.coverage_checked",
                    operation="delegate",
                    requested_atom=atom_payload(atom),
                    covering_atom_count=len(covering),
                    covered=bool(covering),
                )
                if not covering:
                    uncovered.append(atom)
            if uncovered:
                emit_permission_event(
                    "permission.child.denied",
                    parent_node_id=parent_id,
                    uncovered_atoms=[atom_payload(atom) for atom in uncovered],
                )
                names = ", ".join(atom.tool for atom in uncovered)
                raise PermissionEscalationDenied(
                    f"requested atoms are not covered by parent {parent_id}: {names}"
                )

            if atoms == parent.atoms:
                emit_permission_event(
                    "permission.node.reused",
                    node_id=parent.node_id,
                    reuse_kind="parent",
                )
                return parent.node_id

            digest = _policy_hash(atoms)
            for child_id in parent.child_nodes:
                child = self._nodes[child_id]
                if child.state == "active" and child.policy_hash == digest and child.atoms == atoms:
                    emit_permission_event(
                        "permission.node.reused",
                        node_id=child.node_id,
                        reuse_kind="sibling-policy",
                    )
                    return child.node_id

            node_id = f"perm-{uuid.uuid4().hex}"
            child = _PermissionNode(
                node_id=node_id,
                parent_id=parent.node_id,
                atoms=atoms,
                policy_hash=digest,
            )
            self._nodes[node_id] = child
            parent.child_nodes.add(node_id)
            emit_permission_event(
                "permission.node.created",
                node_id=node_id,
                parent_node_id=parent.node_id,
                atom_count=len(atoms),
                policy_hash=digest,
            )
            return node_id

    def attach_agent(self, node_id: str, agent_id: str) -> None:
        """Attach one agent to an active node without exposing the tree to it."""
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        with self._lock:
            node = self._active_node(node_id)
            current = self._agent_nodes.get(agent_id)
            if current is not None and current != node_id:
                emit_permission_event(
                    "permission.agent.attach_denied",
                    agent_id=agent_id,
                    requested_node_id=node_id,
                    current_node_id=current,
                )
                raise AgentAlreadyAttached(
                    f"agent {agent_id!r} is already attached to {current}"
                )
            node.attached_agents.add(agent_id)
            self._agent_nodes[agent_id] = node_id
            emit_permission_event(
                "permission.agent.attached",
                agent_id=agent_id,
                node_id=node_id,
                attached_agent_count=len(node.attached_agents),
            )

    def policy_for_agent(self, agent_id: str) -> tuple[PermissionAtom, ...]:
        """Return only the bound node's atoms, never ancestors or siblings."""
        with self._lock:
            node_id = self._agent_nodes.get(agent_id)
            if node_id is None:
                raise UnknownAgent(agent_id)
            return self._active_node(node_id).atoms

    def allows(self, agent_id: str, requested_atom: PermissionAtom) -> bool:
        """Authorize only when one bound atom covers the complete request atom."""
        with self._lock:
            node_id = self._agent_nodes.get(agent_id)
            if node_id is None:
                emit_permission_event(
                    "permission.agent.node_lookup",
                    agent_id=agent_id,
                    found=False,
                )
                emit_permission_event(
                    "permission.authorization.decided",
                    agent_id=agent_id,
                    requested_atom=atom_payload(requested_atom),
                    allowed=False,
                    reason="agent_not_bound",
                )
                return False
            node = self._nodes.get(node_id)
            if node is None or node.state != "active":
                emit_permission_event(
                    "permission.agent.node_lookup",
                    agent_id=agent_id,
                    node_id=node_id,
                    found=False,
                    state=None if node is None else node.state,
                )
                emit_permission_event(
                    "permission.authorization.decided",
                    agent_id=agent_id,
                    requested_atom=atom_payload(requested_atom),
                    allowed=False,
                    reason="node_inactive",
                )
                return False
            emit_permission_event(
                "permission.agent.node_lookup",
                agent_id=agent_id,
                node_id=node_id,
                found=True,
                state=node.state,
                atom_count=len(node.atoms),
            )
            try:
                self._validate_atoms((requested_atom,))
                covering = [
                    atom for atom in node.atoms if self.atom_covers(atom, requested_atom)
                ]
                allowed = bool(covering)
                emit_permission_event(
                    "permission.atom.coverage_checked",
                    operation="execute",
                    requested_atom=atom_payload(requested_atom),
                    covering_atom_count=len(covering),
                    covered=allowed,
                )
                emit_permission_event(
                    "permission.authorization.decided",
                    agent_id=agent_id,
                    node_id=node_id,
                    requested_atom=atom_payload(requested_atom),
                    allowed=allowed,
                    reason="covered" if allowed else "not_covered",
                )
                return allowed
            except (TypeError, ValueError, OSError) as exc:
                emit_permission_event(
                    "permission.authorization.decided",
                    agent_id=agent_id,
                    node_id=node_id,
                    requested_atom=atom_payload(requested_atom),
                    allowed=False,
                    reason="invalid_atom",
                    error_type=type(exc).__name__,
                )
                return False

    def detach_agent(self, agent_id: str) -> tuple[str, ...]:
        """Detach an agent and prune newly empty leaf nodes towards the root."""
        with self._lock:
            node_id = self._agent_nodes.pop(agent_id, None)
            if node_id is None:
                raise UnknownAgent(agent_id)
            node = self._active_node(node_id)
            node.attached_agents.discard(agent_id)
            removed = self._prune_empty_ancestors(node.node_id)
            emit_permission_event(
                "permission.agent.detached",
                agent_id=agent_id,
                node_id=node_id,
                pruned_node_ids=list(removed),
            )
            return removed

    def _prune_empty_ancestors(self, node_id: str) -> tuple[str, ...]:
        removed: list[str] = []
        current_id = node_id
        while current_id != self.ROOT_ID:
            node = self._nodes[current_id]
            if node.attached_agents or node.child_nodes:
                break
            parent = self._nodes[node.parent_id]  # type: ignore[index]
            parent.child_nodes.discard(current_id)
            node.state = "deleted"
            del self._nodes[current_id]
            removed.append(current_id)
            current_id = parent.node_id
        return tuple(removed)

    def revoke_subtree(self, node_id: str) -> tuple[str, ...]:
        """Atomically revoke a non-root node, descendants, and attached agents."""
        with self._lock:
            if node_id == self.ROOT_ID:
                emit_permission_event(
                    "permission.subtree.revocation_denied",
                    node_id=node_id,
                    reason="root_is_permanent",
                )
                raise RootRevocationDenied("the root permission node is permanent")
            root = self._active_node(node_id)
            ordered: list[_PermissionNode] = []

            def visit(current: _PermissionNode) -> None:
                current.state = "revoking"
                for child_id in sorted(current.child_nodes):
                    visit(self._nodes[child_id])
                ordered.append(current)

            visit(root)
            affected_agents = sorted(
                agent_id
                for node in ordered
                for agent_id in node.attached_agents
            )
            for agent_id in affected_agents:
                self._agent_nodes.pop(agent_id, None)

            parent = self._nodes[root.parent_id]  # type: ignore[index]
            parent.child_nodes.discard(root.node_id)
            for node in ordered:
                node.attached_agents.clear()
                node.child_nodes.clear()
                node.state = "deleted"
                self._nodes.pop(node.node_id, None)
            self._prune_empty_ancestors(parent.node_id)
            emit_permission_event(
                "permission.subtree.revoked",
                node_id=node_id,
                affected_agents=affected_agents,
                removed_node_count=len(ordered),
            )
            return tuple(affected_agents)
