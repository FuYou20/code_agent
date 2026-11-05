# Evolution A/B benchmark

This benchmark measures cross-session reuse rather than one-shot coding ability.
It contains ten repair families. Each family has one learning task and two held-out
transfer tasks, for 30 tasks in total.

The two arms receive the same initial bundled Skills, model, permissions, task order,
and grading tests:

- `fixed`: does not persist memory or distill new Skills;
- `evolved`: persists successful experience and distills one Skill after each learning
  task that passes the evaluator-only tests and the local evolution gate.

Every Agent workspace contains only the buggy source, the natural-language task, and a
small public test. After the Agent returns, the harness injects evaluator-only tests,
runs the complete grade, and immediately removes those hidden tests. A task passes only
when both public and hidden tests pass.

Validate all task fixtures offline:

```powershell
python scripts/validate_evolution_benchmark.py
```

Run a cheap smoke A/B on one family before the complete experiment:

```powershell
python scripts/evaluate_evolution.py --families 1 --output output/evolution-smoke.json
```

Run the full 10-family A/B:

```powershell
python scripts/evaluate_evolution.py --output output/evolution-ab.json
```

Optionally report each task, nested model call, and deterministic score to a
Langfuse project. The benchmark remains the execution and grading authority;
Langfuse stores the traces and experiment comparison data.

```powershell
python -m pip install -e ".[observability]"
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_BASE_URL = "http://localhost:3000"
python scripts/evaluate_evolution.py --families 1 --langfuse `
  --output output/evolution-langfuse-smoke.json
```

Prompt and source contents are hashed by default. Set
`MYAGENT_LANGFUSE_CAPTURE_CONTENT=true` only when the target project is approved
to store repository content.

Run the targeted Retry memory-conflict regression:

```powershell
python scripts/evaluate_evolution.py --family retry-budget --output output/evolution-retry-feedback.json
```

Primary metrics are transfer-task hidden-test success rate, provider-reported prompt
tokens, tool calls, and wall time. Distillation calls are retained in the arm-level
provider totals so that the learning cost cannot be hidden, while transfer-only metrics
show downstream reuse efficiency.

When the provider reports cached tokens, compare both total prompt tokens and
`prompt_tokens - cached_tokens`. Total prompt tokens measure context volume even when a
prefix is reused; uncached tokens measure how much input had to be recomputed and are the
direct signal for the transient-prefix optimization.

This is a controlled project benchmark, not a public general-purpose leaderboard. A
later external-validity pass should sample executable tasks from SWE-smith or BugsInPy.

## Long-task procedural Skill reuse

`scripts/evaluate_long_skill_reuse.py` creates a Langfuse Dataset and two comparable
Dataset Runs. It uses fresh sessions and isolated multi-file repositories, disables
experience-memory retrieval in both arms, and lets only the evolved arm distill and
reuse a procedural Skill. Planning cost is measured as provider calls and prompt tokens
before the first `file_write`; hidden-test success, exploration, total tokens, tool
calls, and latency are recorded alongside it.

```powershell
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_BASE_URL = "http://localhost:3000"
python scripts/evaluate_long_skill_reuse.py
```
