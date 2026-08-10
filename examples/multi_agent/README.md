# Multi-agent Earth-spin example

Round-robin `GroupChat`: **DataGuy → PhysicsGuy → Explainer**.  
Each agent can use a **different model id** on the same API base URL.

```bash
# from repo root — secrets in .env, models in agent.yaml
python examples/multi_agent/team_example.py

cat-agent serve --factory team_example:registry
cat-agent deploy --dir examples/multi_agent
```

## `.env` vs `agent.yaml`

| Concern | Where | Examples |
| --- | --- | --- |
| **Secrets** (API keys) | **`.env` only** — never in `agent.yaml` | `OLLAMA_API_KEY`, `OPENAI_API_KEY`, `CAT_AGENT_LLM_API_KEY` |
| **API base URL** | **`.env`** (local) / gateway on deploy | `OLLAMA_API_BASE`, `CAT_AGENT_LLM_BASE_URL`, `OPENAI_BASE_URL` |
| **Which model** | **`agent.yaml`** | `model.alias`, `env.CAT_AGENT_LLM_MODEL_*` |

### Secrets and URL → `.env` (repo root)

```bash
# .env
OLLAMA_API_BASE=https://ollama.com/v1
OLLAMA_API_KEY=...
# optional overrides:
# CAT_AGENT_LLM_BASE_URL=...
# CAT_AGENT_LLM_API_KEY=...
# CAT_AGENT_LLM_MODEL=minimax-m3   # beats agent.yaml model.alias if set
```

`cat_agent` loads this `.env` on import. Shell exports still win over the file.

### Models → `agent.yaml`

```yaml
model:
  type: api
  alias: minimax-m3          # Explainer (+ default CAT_AGENT_LLM_MODEL on deploy)

env:
  CAT_AGENT_LLM_MODEL_DATAGUY: gemma4:cloud
  CAT_AGENT_LLM_MODEL_PHYSICSGUY: gpt-oss:20b
  LOG_LEVEL: INFO
```

| Agent | Model source |
| --- | --- |
| Explainer | `model.alias` (or `.env` `CAT_AGENT_LLM_MODEL` if set) |
| DataGuy | `env.CAT_AGENT_LLM_MODEL_DATAGUY` |
| PhysicsGuy | `env.CAT_AGENT_LLM_MODEL_PHYSICSGUY` |

Local `python team_example.py` copies `agent.yaml` `env:` into the process when those keys are unset (so you do not need to `export` them).  
On **Nomad deploy**, the platform injects `model.alias` as `CAT_AGENT_LLM_MODEL` and merges `env:` into the job — still **no API keys** in the yaml; the job uses the team Vault key + gateway.

### Do not put in `agent.yaml`

- `OLLAMA_API_KEY` / `OPENAI_API_KEY` / `CAT_AGENT_LLM_API_KEY`
- `OPENAI_BASE_URL` / `CAT_AGENT_LLM_BASE_URL` (reserved on deploy — gateway owns routing)

## Layout

| File | Role |
| --- | --- |
| `team_example.py` | `EarthSpinTeam` + `registry()` for serve/deploy |
| `agent.yaml` | Deploy manifest + per-agent model ids |
| `group_chat_example.py` / `router_example.py` | Smaller demos (no deploy yaml required) |
