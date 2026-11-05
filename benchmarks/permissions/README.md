# Permission governance evaluation

This benchmark mirrors the 21 deterministic permission checks into a Langfuse
Dataset without replacing their original pytest assertions.

The catalog covers atomic delegation, node isolation, path containment, runtime
authorization, policy reuse, Agent lifecycle, asynchronous revocation, shared
budgeting, and permission-safe Skill distillation. It does not call an LLM.

```powershell
.\.venv-langfuse\Scripts\python.exe -m scripts.evaluate_permissions_langfuse `
  --env-file ..\agent-trace-eval\.env `
  --output output\permission-langfuse-run.json
```

Use `--dataset-only` to synchronize the Dataset without starting a run. Each
case records `permission_check_passed`, `decision_matches_expected`,
`duration_seconds`, and `permission_event_count` scores. The local JSON report
includes the Dataset Run URL.

During an evaluation only, the runner enables the dependency-free permission
event sink. Real tree and broker operations then become child spans such as
`permission.child.requested`, `permission.atom.coverage_checked`,
`permission.authorization.decided`, `tool.approval.checked`, and
`tool.execution.finished`. Normal MyAgent runs do not enable this sink.
