# Unified evolution artifact lifecycle

Long-term experiences and procedural Skills are different payloads governed by the
same control plane. Both are registered as versioned evolution artifacts with candidate
A/B evidence, production correctness signals, exposure logs, lifecycle events, and
retention.

## State machine

```text
candidate --3 comparable correct trials + cumulative token saving > 0--> active
candidate --any Treatment failure or no cumulative token saving--> discarded
active --one decisive correctness failure--> discarded
discarded --retention expires--> content purged; tombstone retained
```

Production retrieval accepts only `active`. Benchmark and sandbox callers must
explicitly opt in to `candidate`. Lifecycle status controls visibility, never tool or
file authority; the atomic permission tree remains the sole authorization source.

## Correctness gate and token objective

Correctness is a hard gate. Token is the only economic optimization objective. Tool
calls, files read, and latency remain diagnostic fields and cannot promote an artifact.

Provider usage is normalized into configurable equivalent input-token units:

```text
equivalent tokens =
    uncached input × uncached_input_weight
  + cached input   × cached_input_weight
  + output         × output_weight
```

The default weights are all `1.0` until the operator configures ratios matching the
provider's bill. Every comparable trial requires a correct Control and Treatment. The
candidate promotion balance is:

```text
gross token saving = Σ(control - treatment)
```

A candidate becomes `active` after three comparable trials when every Treatment is
correct and cumulative gross token saving is positive. Any Treatment failure discards
the candidate immediately. A Treatment success when Control fails is recorded as a
capability gain but does not count as a comparable token trial.

Creation or distillation cost is still reported as `net_token_saving` for diagnostics,
but does not complicate the three-trial promotion rule. Once active, the asset never
runs automatic A/B. `record_active_result` records correctness only, and one decisive
failure discards it.

## Unified storage

- `evolution_artifacts`: Memory/Skill identity, version, state, content hash/reference,
  creation cost, and lifecycle timestamps.
- `evolution_ab_trials`: candidate paired correctness and equivalent-token measurements.
- `evolution_exposures`: exact artifact version exposed or activated in each run.
- `evolution_artifact_events`: append-only registration, trial, transition, correctness,
  and deletion audit trail.

Memory content remains in `decisions` with FTS5 retrieval. Skill content remains in
versioned `SKILL.md` files. The shared registry owns whether either payload is visible.
When a new Skill version is a candidate, the previous active version continues serving
from `.history/vN.md`; a candidate cannot silently replace production.

## Retention and deletion

Garbage collection runs on CLI startup and every 24 hours by default, and can be
invoked immediately with `/evolution-gc`:

- expired candidates become `discarded`;
- expired discarded Memory rows and Skill files are physically removed;
- active assets are not duplicated for periodic A/B and remain active until a decisive
  correctness failure is recorded;
- the artifact row, content hash, failure reason, trial evidence, and event history remain
  as a tombstone so the same bad asset is not silently relearned.

All destructive Skill cleanup resolves the registered path under the configured runtime
Skill store and only unlinks a registered Skill Markdown file.

## Runtime and sandbox

The normal CLI constructs an `EvolutionArtifactManager`, migrates the former six-state
schema, backfills legacy Memory rows, registers pre-lifecycle runtime Skills as
candidates, and loads only `active` runtime artifacts. Bundled project Skills are
trusted static assets and remain outside the runtime evolution registry.

The evaluation runners opt into candidate visibility, pair Fixed and Evolved
task results, and submit provider-reported usage to `record_ab_trial`. `/learn` now
creates a candidate Skill and reports its artifact ID instead of activating it directly.

Use `/evolution-status` to inspect every live artifact and its lifetime economics. Use
`/evolution-gc` to apply the configured retention policy immediately.
