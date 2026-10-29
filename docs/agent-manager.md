<!-- Lifecycle contract for permission-bound child agents. -->
# Agent Manager MVP

The Agent Manager builds on the atomic permission tree and is not yet connected
to the interactive CLI. It provides the lifecycle required before child agents
can safely execute the existing model/tool loop.

## Creation

`CreateAgentRequest` contains a parent agent, task, requested permission atoms,
an optional parent-context summary, and optional Skill names. The manager:

1. verifies that the parent is live;
2. derives or reuses a permission node under the parent;
3. mounts the new agent;
4. filters the parent's tool descriptions by the child atom set;
5. exposes only useful Skill descriptions whose declared requirements intersect
   the child policy; this is context filtering, not authorization;
6. creates an immutable `ChildContext` containing the inherited parent prompt,
   task, summary, filtered tools, filtered Skills, and permission atoms.

## Execution and delivery

Workers receive an `AgentExecution`, not the manager or permission tree. It can
read its immutable context, check cancellation, request atom authorization, and
reserve or settle shared budget. Workers can run synchronously or in the shared
thread pool.

The manager persists a structured `ChildResult` to the parent's mailbox before
detaching the child. Detach prunes only empty permission leaves. A parent that
finishes while descendants are active cancels those descendants first.

## Cancellation and revocation

Waiting with a timeout marks the child `timed_out`, signals cooperative
cancellation, revokes its permission binding, and preserves the timeout result.
Explicit permission-subtree revocation first removes every binding in the tree,
then marks all affected managed agents cancelled. Late worker results cannot
overwrite an existing terminal result.

Thread workers cannot be forcibly killed by Python. Security therefore depends
on every model/tool operation going through the manager and permission broker;
after cancellation, authorization fails. A future hard-kill implementation
should use isolated worker processes.

## Global budget

All agents share one concurrency-safe `GlobalBudgetLedger`. Operations reserve a
maximum before starting and settle actual usage afterward, returning unused
capacity. Cancellation and unloading return all outstanding reservations, and
a terminal agent cannot reserve more. The ledger does not allocate independent
per-agent budgets.
