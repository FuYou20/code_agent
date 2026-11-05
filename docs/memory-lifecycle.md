# Unified evolution artifact lifecycle

Long-term experiences and procedural Skills are different payloads governed by the
same control plane. Both are stored as versioned evolution artifacts with correctness
evidence, paired A/B token evidence, exposure logs, lifecycle events, and retention.

## State machine

```text
candidate --first non-regressing A/B--> canary
canary --correctness holds + net token saving > 0--> active

candidate/canary --correct but token-expensive--> retired
active --two consecutive token-negative paired trials--> canary
candidate/canary/active --correctness regression--> quarantined
quarantined --retention expires--> retired --retention expires--> purged
```

Production retrieval accepts only `active`. Benchmark/sandbox callers must explicitly
opt in to `candidate` and `canary`. Lifecycle status controls visibility, never tool or
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
artifact's economic balance is:

```text
net token saving = Σ(control - treatment) - creation/distillation tokens
```

An artifact becomes `active` only after the configured minimum number of comparable
trials, no correctness regression, positive cumulative net saving, and positive median
saving. A Treatment failure when Control passes quarantines the artifact immediately.
A Treatment success when Control fails is recorded as a capability gain but is not
misreported as token saving.

## Unified storage

- `evolution_artifacts`: Memory/Skill identity, version, state, content hash/reference,
  creation cost, and lifecycle timestamps.
- `evolution_ab_trials`: paired correctness and equivalent-token measurements.
- `evolution_exposures`: exact artifact version exposed or activated in each run.
- `evolution_artifact_events`: append-only registration, trial, transition, and deletion
  audit trail.

Memory content remains in `decisions` with FTS5 retrieval. Skill content remains in
versioned `SKILL.md` files. The shared registry owns whether either payload is visible.
When a new Skill version is a candidate, the previous active version continues serving
from `.history/vN.md`; a candidate cannot silently replace production.

## Retention and deletion

Garbage collection runs on CLI startup and every 24 hours by default, and can be
invoked immediately with `/evolution-gc`:

- stale active assets return to `canary` for revalidation;
- expired candidates/canaries and quarantined assets become `retired`;
- expired retired Memory rows and Skill files are physically removed;
- the artifact row, content hash, failure reason, trial evidence, and event history remain
  as a tombstone so the same bad asset is not silently relearned.

All destructive Skill cleanup resolves the registered path under the configured runtime
Skill store and only unlinks a registered `SKILL.md`.

## Runtime and sandbox

The normal CLI constructs an `EvolutionArtifactManager`, backfills legacy Memory rows,
registers pre-lifecycle runtime Skills as candidates, and loads only `active` runtime
artifacts. Bundled project Skills are trusted static assets and remain outside the
runtime evolution registry.

The evaluation runners opt into candidate/canary visibility, pair Fixed and Evolved
task results, and submit provider-reported usage to `record_ab_trial`. `/learn` now
creates a candidate Skill and reports its artifact ID instead of activating it directly.

Use `/evolution-status` to inspect every live artifact and its lifetime economics. Use
`/evolution-gc` to apply the configured retention policy immediately.
