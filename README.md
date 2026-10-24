<!-- Project overview and quick-start instructions. -->
# myagent

A small, single-process Python coding agent with six built-in tools, scope checks,
cache-stable prompt assembly, and a tamper-evident audit log.

Skill loading uses a safe baseline by default: all L0 metadata is placed in the
ephemeral sidecar. High-confidence trigger matches additionally load one L1 body,
and that skill's manifest narrows both the advertised tools and the execution
Scope for the turn.

## Install

```powershell
python -m pip install -e ".[test]"
$env:OPENAI_API_KEY = "..."
```

## Run

```powershell
python -m myagent "读一下 README.md，然后告诉我内容"
python -m myagent
```

## Measure prompt caching

Run five turns in one session and print each turn's cached-token ratio:

```powershell
python scripts/measure_cache.py
```

OpenAI reports cache hits only for eligible models and sufficiently long shared
prefixes, so the first turn normally has no cached tokens.

For DeepSeek Flash, put this non-secret configuration in the agent home
`config.yaml`, and keep the key only in the named environment variable:

```yaml
llm:
  provider: deepseek
  model: deepseek-flash
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
```

Configuration is read from `%MYAGENT_HOME%\config.yaml`; when `MYAGENT_HOME` is
unset, the default directory is `~/.myagent`.

## Skill-loading experiment

The primary comparison uses the configured main model for both arms and reports
only input tokens and required-Skill hits:

```powershell
python scripts/evaluate_skill_rag.py
python scripts/evaluate_skill_rag.py --json
```

The RAG arm counts both query-expansion and reranking input tokens. Set the API
key named by `llm.api_key_env` before running it.

The older deterministic 100-skill structural diagnostic can still be run with:

```powershell
python scripts/evaluate_skill_loading.py
python scripts/evaluate_skill_loading.py --json
```

The experimental Top-K path is feature-gated because the current lexical
retriever has not passed the paraphrase-recall gate. To enable it for a canary
session, add:

```yaml
skills:
  strategy: top_k
  top_k: 5
  activation_min_score: 80
```

Use `full_sidecar` to retain the safe baseline. See
[`docs/skill-loading-evaluation.md`](docs/skill-loading-evaluation.md) for the
latest result and go/no-go decision.

