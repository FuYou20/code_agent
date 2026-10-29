<!-- Design contract for atomic tool-file permissions used by all agents. -->
# Atomic permission tree

The permission tree is the sole runtime authority. The interactive root Agent is
attached to the root node; the child Agent Manager uses the same tree for delegation.

Each node contains only indivisible `PermissionAtom` values. An atom binds one
tool to its complete file-access footprint; tool names and paths are never
stored or combined as independent permission lists.

```python
PermissionAtom.read("file_read", "src")
PermissionAtom.write("file_write", "generated")
PermissionAtom.read_write(
    "test_run",
    read=(".",),
    write=(".pytest_cache",),
)
```

Delegation obeys these invariants:

1. Every child atom must be wholly covered by one parent atom.
2. Tools, access modes, and paths cannot be recombined across atoms.
3. Equal policies under one parent reuse a node; equal-to-parent policies reuse
   the parent node itself.
4. Agents receive only the atoms of their bound node, not ancestors or siblings.
5. Detaching an agent prunes empty leaf nodes, never nodes with active children.
6. Explicit subtree revocation immediately removes every descendant binding and
   returns the affected agent IDs for an Agent Manager to cancel.
7. The root node is permanent.

Policy paths are workspace-relative subtree roots. Construction resolves them
against the workspace and rejects absolute paths, `..` escapes, and existing
symlinks that resolve outside it.

The trusted tool adapter must convert a concrete tool call into a request atom
that describes every file effect. Model-visible tool schemas are filtered by the
bound node, and every concrete call is checked again before execution. Skill
requirements never grant or narrow authority. Generic shell execution still needs
an operating-system filesystem sandbox for strong process-level isolation.
