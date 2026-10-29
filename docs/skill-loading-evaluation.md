# Skill loading evaluation history

## 1. Lexical retrieval checkpoint

The first deterministic checkpoint used keyword retrieval only. On 14 positive
challenge tasks it reached 42.9% candidate recall@5, so keyword Top-K was rejected.
This result explains why the runtime does not rely on literal user wording.

Reproduce:

```powershell
python scripts/evaluate_skill_loading.py
```

## 2. Semantic routing component

The next experiment compared reranking all 100 Skill rows with:

```text
main-model query expansion → local candidate recall@20 → main-model reranking
```

On the 14 positive tasks, the component reduced selection-stage provider input
from 70,057 to 10,401 tokens (85.15%) and kept the required Skill visible in
13/14 tasks. The miss was scanned-PDF extraction: the candidate set contained the
required Skill, but reranking selected OCR correction only.

Reproduce with the configured provider credential:

```powershell
python scripts/evaluate_skill_rag.py --json
```

This is a routing-component result. It does not measure System-prompt freezing,
provider cache reuse, temporary Sidecar injection, or drift recovery.

## 3. Superseded automatic-drift experiment

The earlier `warm_stable` canary used:

```text
warmup routing → System L0 freeze → shadow retrieval → sustained-drift thaw
```

That evaluation exposed unstable warmup recall and cache invalidation. Its results remain
as historical evidence, but the content-driven thaw strategy has been removed.

## 4. Scene-Epoch strategy

The current canary classifies a stable task scene, accumulates recall-first candidates
during warmup, freezes the workset once, and handles out-of-workset Skills as temporary
Sidecar overlays. Frozen epochs reset only after an idle timeout or explicit user command.

The deterministic structural check now reports 19/19 required-Skill visibility,
one System change per conversation, and a 75.87% reduction in the uncached Skill-token
proxy across the same 100-Skill, 19-turn workload. It validates placement and state
transitions under a successful planner; it deliberately excludes retrieval quality,
router input, and provider billing/cache behavior.

The first provider-reported implementation reached 19/19 required-Skill visibility but
increased uncached input by 91.21% because two model-router calls occurred on every turn.
Local escape gating and epoch-scoped fallback confirmation reduced router calls from 38
to 10. The final paired run kept 19/19 visibility, reduced total prompt input by 48.83%,
and reduced uncached input by 9.47%. See
[`scene-epoch-ab-report.md`](scene-epoch-ab-report.md) for the current result and
[`skill-epoch-system-experiment.md`](skill-epoch-system-experiment.md) for the superseded
strategy's historical report.

Authorization is outside this comparison. Skill requirements are context metadata;
the atomic permission tree remains the only runtime authority.
