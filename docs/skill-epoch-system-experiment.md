# Warm-Stable Skill architecture A/B

> Historical result: this automatic content-drift strategy was superseded by
> Scene-Epoch loading. Frozen Scene-Epoch worksets no longer thaw from conversation content.

Date: 2026-10-29

## Question

Does the current warmup/freeze/shadow/thaw architecture reduce repeated Skill
context without hiding the Skill required by the main Agent?

The old 85.15% and 13/14 experiment only evaluated the retrieval-and-reranking
component. It must not be presented as an end-to-end result for the current
cache-epoch architecture.

## Two arms

| Arm | Main-agent prompt behavior |
|---|---|
| Full-System | Put all 100 Skill L0 rows in the stable System prefix before history |
| Warm-Stable | Route during warmup, freeze stable L0 into System, shadow-retrieve while frozen, inject one-off Skills into Sidecar, thaw after sustained drift |

The Warm-Stable total includes query-expansion and reranking input. Skill
`permission_requirements` never participate in authorization.

## Dataset

The catalog contains 100 synthetic but closely related Skill L0 records. The test
uses two six-turn conversations and one seven-turn conversation, for 19 grounded turns:

1. `stable_topic`: six consecutive pytest-failure turns;
2. `one_turn_aside`: four pytest turns, one release-note aside, then one pytest turn;
3. `sustained_drift`: four pytest turns followed by three release-note turns.

Every turn declares one required Skill. This makes the correctness metric auditable
without asking the evaluation model to grade its own answer.

## Headline metrics

1. **Required-Skill visibility**: before the main Agent answers, the required
   Skill name must occur in the System workset, current Sidecar, or active L1.
2. **Total provider prompt tokens**: includes main-agent, query-expansion, and
   reranking calls.

Provider cached tokens, uncached tokens, router tokens, and System-prompt changes
are supporting diagnostics.

## Archived offline structural result

The table below was produced by the superseded Warm-Stable implementation. The current
`evaluate_skill_epoch_cache.py` evaluates Scene-Epoch and does not reproduce this table.

| Metric | Full-System | Warm-Stable |
|---|---:|---:|
| Conversations / turns | 3 / 19 | 3 / 19 |
| Required-Skill visibility | 19/19 | 19/19 |
| Uncached Skill-token proxy | 13,599 | 2,884 |
| Structural reduction | - | 78.79% |

Behavioral checks:

- stable topic: one System rebuild;
- one-turn aside: one temporary Sidecar injection and no additional rebuild;
- sustained drift: deterministic planner thaws and refreezes, with three
  System-state changes in total.

This is a deterministic structural result. It assumes a correct topic-workset
planner, excludes router input, and is not a provider billing-token result.

## Archived provider-reported run

The historical invocation used the then-current online runner. The current
`evaluate_skill_epoch_live.py` has been updated to compare Full-System with Scene-Epoch.

The online runner executes the same 19 turns in both arms. It reports:

- required-Skill visibility;
- all prompt tokens, including router overhead;
- main versus router prompt tokens;
- provider-reported cached and uncached tokens;
- cache ratio and System-prompt changes.

## Provider result

DeepSeek Flash was run once with a unique evaluation prefix to prevent previous
main-Agent prompts from contaminating the cache comparison:

| Metric | Full-System | Warm-Stable |
|---|---:|---:|
| Main Agent turns | 19 | 19 |
| Total API calls | 19 | 52 |
| Required-Skill visibility | 19/19 (100%) | 15/19 (78.95%) |
| Total prompt tokens | 76,913 | 49,846 |
| Router prompt tokens | 0 | 19,873 |
| Uncached prompt tokens | 21,617 | 30,774 |
| Main-Agent cache ratio | 71.89% | 47.83% |
| System-state changes | 0 | 4 |

Warm-Stable reduced total provider input by **35.19%**, including all query
expansion and reranking calls. However, uncached input increased by **42.36%**,
main-Agent cache ratio fell by 24.06 percentage points, and required-Skill
visibility fell below the acceptance gate. Stable-topic and one-turn-aside
visibility were each 4/6; sustained-drift visibility was 7/7.

The online sustained-drift session thawed after the topic switch and ended in a
new warming phase; it did not refreeze within the remaining three writing turns.
This is an observed model-routing result, unlike the deterministic structural
planner above. The complete machine-readable result is stored in
`output/skill-architecture-live.json`.

## Decision

Do not promote Warm-Stable. It is retained only as historical evidence. Use Full-System
as the default because it is the strong cache-aware baseline and guarantees Skill
visibility. Scene-Epoch is the current canary; its first online A/B is documented in
[`scene-epoch-ab-report.md`](scene-epoch-ab-report.md). Its optimized paired run reduces
both total and uncached input, but still requires repetition and broader workloads before
promotion.
