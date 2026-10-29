# Scene-Epoch online A/B report

Date: 2026-10-29

## Setup

- Provider/model: DeepSeek / `deepseek-flash`
- Catalog: 100 synthetic Skill L0 records
- Workload: 3 independent conversations, 19 grounded turns in total
- Baseline: all Skill L0 records fixed in the System prompt (`full_system`)
- Candidate: scene classification, retrieval, reranking, three-turn warmup, one System
  freeze, local shadow routing, and model-confirmed Sidecar fallback (`scene_epoch`)
- Accounting: every provider prompt token is counted, including query expansion and
  reranking
- Correctness gate: the required Skill name must be visible to the main Agent before it
  answers

The final machine-readable result is stored in
[`../output/scene-epoch-ab-fallback-cache-20261029.json`](../output/scene-epoch-ab-fallback-cache-20261029.json).

## Optimization iterations

All iterations use the same 100-Skill, 19-turn workload and the same correctness gate.
Each percentage is paired with the Full-System arm from the same provider run.

| Scene-Epoch version | Visibility | API calls | Router tokens | Total-token change | Uncached-token change |
|---|---:|---:|---:|---:|---:|
| Route every turn | 19/19 | 57 | 29,317 | -20.50% | +91.21% |
| Route on initialization/escape | 19/19 | 33 | 10,655 | -37.31% | +9.19% |
| Reuse confirmed fallback | 19/19 | 29 | 7,598 | -48.83% | -9.47% |

## Aggregate result

| Metric | Full-System | Scene-Epoch | Change |
|---|---:|---:|---:|
| Required-Skill visibility | 19/19 | 19/19 | equal |
| API calls | 19 | 29 | +10 router calls |
| Total prompt tokens | 77,426 | 39,619 | -48.83% |
| Main-Agent prompt tokens | 77,426 | 32,021 | -58.64% |
| Router prompt tokens | 0 | 7,598 | +7,598 |
| Uncached prompt tokens | 22,130 | 20,035 | -9.47% |
| Main-Agent cache ratio | 71.42% | 47.57% | -23.85 pp |
| Overall cache ratio | 71.42% | 49.43% | -21.99 pp |
| System-prompt changes | 0 | 3 | one per conversation |

## Per-conversation result

| Conversation | Turns | Full tokens | Scene-Epoch tokens | Required visible | System changes | Temporary fallback |
|---|---:|---:|---:|---:|---:|---:|
| Stable topic | 6 | 24,390 | 11,491 | 6/6 | 1 | 0 |
| One-turn aside | 6 | 24,424 | 13,149 | 6/6 | 1 | 1 |
| Sustained drift | 7 | 28,612 | 14,979 | 7/7 | 1 | 3 |

The state machine behaved as designed: each conversation froze exactly once, the
one-turn aside stayed temporary, and sustained topic drift did not thaw the frozen
workset.

## Interpretation

The initial implementation treated weak or absent lexical evidence as a reason to call
the model router. This caused two extra calls on every turn. The optimized implementation
instead trusts the active workset by default: it routes with the model only when the epoch
has no workset or local retrieval finds high-confidence evidence for an outside Skill.

After the model confirms an outside Skill, that routing decision is cached for the epoch.
Later high-confidence matches can reuse it as a per-turn Sidecar overlay. The confirmed
fallback never enters the frozen System workset and is cleared by timeout or explicit
epoch reset.

This reduces router calls from 38 to 10 and router input from 29,317 to 7,598 tokens.
Although the cache ratio remains lower because redundant cacheable Skill text has been
removed, the absolute uncached input is now 9.47% below Full-System. The sustained-drift
conversation alone still has 4.05% more uncached input than its baseline; the aggregate
win comes from the stable and one-turn-aside conversations.

## Decision

Scene-Epoch passes this experiment's visibility, total-input, and uncached-input gates.
Keep `full_system` as the default until the result is repeated and the workload is expanded
beyond one synthetic catalog and one provider run. The next evaluation should add longer
stable sessions and multiple true scene switches rather than tune thresholds against this
single dataset.
