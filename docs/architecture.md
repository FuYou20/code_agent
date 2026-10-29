# MyAgent architecture

## Design objective

MyAgent keeps one small runtime and three explicit engines. New features must belong to
one of these engines or remain outside the core. This prevents memory, Skills, security,
and sub-agent orchestration from creating parallel control paths.

| Layer | Owns | Must not own |
|---|---|---|
| Runtime | model loop, tool protocol, CLI, child lifecycle | authorization policy, durable learning |
| Context Engine | prompt placement, compression, Skill routing, cache epochs | tool authorization |
| Permission Tree | tool-and-file authority, delegation, revocation | Skill selection, prompt content |
| Evolution Engine | success gate, experience memory, Skill distillation | live execution authority |

## Runtime request flow

```text
User input
  → Context Engine assembles stable prefix, history, and Sidecar
  → Model receives only tools advertised by the bound permission node
  → Trusted adapter maps a tool call to one complete PermissionAtom
  → Permission Tree authorizes or returns permission_denied
  → High-risk approval and tool-local constraints run
  → Tool result returns to the model and audit chain
  → Evolution Engine evaluates the completed trace
```

There is no Effective Scope. Skill metadata can state requirements for planning and
diagnostics, but only the bound permission node can authorize execution.

## Context lifecycle

The stable prefix changes rarely. Retrieved memory and L1 procedures remain ephemeral.
During Skill warmup, scene-filtered and globally retrieved L0 candidates accumulate in
the Sidecar instead of being replaced each turn. After the configured warmup, the
deterministic workset enters the System Prompt and remains immutable for that epoch.
Out-of-workset Skills are temporary Sidecar overlays. A model-confirmed fallback can be
reused within the epoch when local retrieval finds the same high-confidence Skill again;
it never joins the frozen workset. Only an idle timeout or explicit `/skill-reset` or
`/scene` command starts another epoch.

## Evolution lifecycle

The shared gate rejects conversation-only, failed-tool, unverified-test, and empty-final
traces. An eligible trace may create a compact FTS5 experience immediately. Explicit
`/learn` may also turn the verified tool trajectory into a versioned procedural Skill.
Neither output receives authority: future retrieval only supplies context.

## Sub-agent boundary

The Agent Manager belongs to Runtime. It creates a child, requests a strict subset of the
parent permission atoms, mounts the child, filters visible tools, runs synchronously or
asynchronously, delivers a structured result, and finally unmounts it. All agents share
one global budget ledger. The manager does not implement a second permission system or a
second memory store.
