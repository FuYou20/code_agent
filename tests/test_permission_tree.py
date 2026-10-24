"""Verify atomic tool-file delegation, node reuse, and lifecycle cleanup."""

import pytest

from myagent.permissions import (
    AgentAlreadyAttached,
    FileGrant,
    PermissionAtom,
    PermissionEscalationDenied,
    PermissionTree,
    RootRevocationDenied,
    UnknownAgent,
)


def _root_atoms():
    return (
        PermissionAtom.read("file_read", "."),
        PermissionAtom.write("file_write", "generated"),
        PermissionAtom.read_write(
            "test_run",
            read=(".",),
            write=(".pytest_cache",),
        ),
    )


def test_tool_and_file_permissions_cannot_cross_atoms(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())

    with pytest.raises(PermissionEscalationDenied):
        tree.get_or_create_child(
            tree.ROOT_ID,
            [PermissionAtom.write("file_write", "src")],
        )

    child = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.write("file_write", "generated/reports")],
    )
    tree.attach_agent(child, "writer")
    assert tree.allows(
        "writer",
        PermissionAtom.write("file_write", "generated/reports/result.txt"),
    )
    assert not tree.allows(
        "writer",
        PermissionAtom.read("file_read", "generated/reports/result.txt"),
    )


def test_one_parent_atom_must_cover_the_complete_child_atom(tmp_path) -> None:
    root = (
        PermissionAtom.read("file_read", "src"),
        PermissionAtom.read("file_read", "tests"),
    )
    tree = PermissionTree(tmp_path, root)
    combined = PermissionAtom(
        "file_read",
        (
            FileGrant("read", "src/auth"),
            FileGrant("read", "tests/auth"),
        ),
    )
    with pytest.raises(PermissionEscalationDenied):
        tree.get_or_create_child(tree.ROOT_ID, [combined])


def test_file_bound_tool_cannot_be_redeclared_as_fileless(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    with pytest.raises(PermissionEscalationDenied):
        tree.get_or_create_child(
            tree.ROOT_ID,
            [PermissionAtom.without_files("file_read")],
        )


def test_equal_policy_reuses_parent_and_equal_child_policy_reuses_node(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    assert tree.get_or_create_child(tree.ROOT_ID, _root_atoms()) == tree.ROOT_ID

    requested = [PermissionAtom.read("file_read", "src")]
    first = tree.get_or_create_child(tree.ROOT_ID, requested)
    second = tree.get_or_create_child(tree.ROOT_ID, requested)
    assert first == second

    tree.attach_agent(first, "reader-a")
    tree.attach_agent(second, "reader-b")
    assert tree.node(first).attached_agent_count == 2
    assert tree.policy_for_agent("reader-a") == tree.policy_for_agent("reader-b")


def test_agent_cannot_be_attached_to_two_nodes(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    read_node = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.read("file_read", "src")],
    )
    write_node = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.write("file_write", "generated")],
    )
    tree.attach_agent(read_node, "agent")
    with pytest.raises(AgentAlreadyAttached):
        tree.attach_agent(write_node, "agent")


def test_sibling_agents_see_and_use_only_their_bound_atoms(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    read_node = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.read("file_read", "src")],
    )
    write_node = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.write("file_write", "generated")],
    )
    tree.attach_agent(read_node, "reader")
    tree.attach_agent(write_node, "writer")

    assert tree.policy_for_agent("reader") == (
        PermissionAtom.read("file_read", "src"),
    )
    assert tree.policy_for_agent("writer") == (
        PermissionAtom.write("file_write", "generated"),
    )
    assert not tree.allows(
        "reader",
        PermissionAtom.write("file_write", "generated/result.txt"),
    )
    assert not tree.allows(
        "writer",
        PermissionAtom.read("file_read", "src/main.py"),
    )


def test_detach_prunes_only_empty_leaf_nodes(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    parent = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.read("file_read", "src")],
    )
    child = tree.get_or_create_child(
        parent,
        [PermissionAtom.read("file_read", "src/auth")],
    )
    tree.attach_agent(parent, "parent-agent")
    tree.attach_agent(child, "child-agent")

    assert tree.detach_agent("parent-agent") == ()
    assert tree.node_exists(parent)
    assert tree.node_exists(child)

    removed = tree.detach_agent("child-agent")
    assert removed == (child, parent)
    assert not tree.node_exists(child)
    assert not tree.node_exists(parent)
    assert tree.node_exists(tree.ROOT_ID)


def test_revoke_subtree_detaches_active_descendant_agents(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    parent = tree.get_or_create_child(
        tree.ROOT_ID,
        [PermissionAtom.read("file_read", "src")],
    )
    child = tree.get_or_create_child(
        parent,
        [PermissionAtom.read("file_read", "src/auth")],
    )
    tree.attach_agent(parent, "parent-agent")
    tree.attach_agent(child, "child-agent")

    affected = tree.revoke_subtree(parent)
    assert affected == ("child-agent", "parent-agent")
    assert not tree.node_exists(parent)
    assert not tree.node_exists(child)
    assert not tree.allows(
        "child-agent",
        PermissionAtom.read("file_read", "src/auth/token.py"),
    )
    with pytest.raises(UnknownAgent):
        tree.policy_for_agent("parent-agent")


def test_root_cannot_be_revoked(tmp_path) -> None:
    tree = PermissionTree(tmp_path, _root_atoms())
    with pytest.raises(RootRevocationDenied):
        tree.revoke_subtree(tree.ROOT_ID)


def test_paths_cannot_escape_workspace(tmp_path) -> None:
    with pytest.raises(ValueError):
        PermissionAtom.read("file_read", "../secret")
    with pytest.raises(ValueError):
        PermissionAtom.write("file_write", "C:/secret")

    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError):
        PermissionTree(
            tmp_path,
            [PermissionAtom.read("file_read", "link")],
        )
