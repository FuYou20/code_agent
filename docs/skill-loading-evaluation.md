<!-- Reproducible status report for the skill-loading redesign. -->
# Skill loading: first evaluation checkpoint

Date: 2026-10-23

## Decision

**No-go for making Top-K or Warm-Stable the default.** The structural cost
experiment passes, but the independent retrieval challenge fails the 95%
availability gate. `full_sidecar` therefore remains the default; `top_k` is an
explicit canary option only.

## Reproduce

```powershell
python -m pytest -q
python scripts/evaluate_skill_loading.py
python scripts/evaluate_skill_loading.py --json
```

The benchmark creates exactly 100 L0 records and evaluates five strategies over
eight session shapes. Its token number is `ceil(UTF-8 bytes / 4)`, a deterministic
relative proxy rather than a provider billing-token claim.

## Results

Across all eight structural scenarios:

| Strategy | Skill token proxy | Irrelevant token proxy | L0 availability |
|---|---:|---:|---:|
| Full-Sidecar (current baseline) | 344,508 | 341,306 | 100% |
| Full-System | 344,508 | 341,306 | 100% |
| Top-K | 15,868 | 12,664 | 100% |
| Warm-Stable | 21,419 | 18,210 | 100% |
| Oracle | 3,202 | 0 | 100% |

On exact-trigger structural traffic, Top-K reduces total Skill context by 95.4%
and irrelevant Skill context by 96.3%. It is also smaller than Warm-Stable, so a
freeze mechanism is not justified until retrieval quality itself is solved.

The separate challenge set is deliberately not generated from the Skill
metadata and includes paraphrases, compound intentions, hard negatives, and
no-Skill chat:

| Metric | Result | Gate |
|---|---:|---:|
| Relevant Skill visible in Top-5 | 42.9% | >=95% |
| Correct primary Skill | 28.6% | diagnostic |
| No-Skill chat correctly returns empty | 100% | 100% |

The failure is mostly semantic paraphrasing. Increasing `k` cannot recover a
Skill that lexical retrieval never finds, and therefore cannot honestly close
the gap.

## Runtime changes already safe to keep

- Deterministic, explainable L0 retrieval with scored reasons.
- At most one high-confidence L1 Skill is loaded per user turn.
- The L1 manifest is intersected with the parent Scope; it cannot add tools.
- API tool schemas are filtered to the effective Scope, and the guard repeats
  the same check before execution.
- L0/L1 sidecars remain ephemeral and are not written into chat history.
- Retrieval, activation, denial, and tool denial are auditable. Raw user text is
  represented by a SHA-256 digest and length in Skill-retrieval audit entries.
- Prompt order is chronological: stable system, history, current sidecar.

## Next gate

Before changing the default, compare these alternatives on at least 30 labeled
tasks with three real-model repetitions each:

1. richer author-provided triggers plus lexical Top-K;
2. confidence-gated fallback to Full-Sidecar;
3. a semantic/router step that selects a Skill before operational tools are
   exposed.

Measure relevant L0 visibility, correct L1 activation, end-task success, input
tokens, provider-reported cached tokens, switch recovery, and Scope denials.
Only promote a strategy if L0 availability reaches 95%, task success is within
10 percentage points of Full-Sidecar, irrelevant Skill tokens fall at least 70%,
and no Scope escalation occurs.
