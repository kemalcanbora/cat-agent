# Structured execution traces

Cat-Agent can record a **machine-readable** execution trace for every agent run.
Logging (Loguru) remains for humans; traces are for cost accounting, evaluation,
debugging, and failure analysis.

References: Yehudai et al. (2025/2026) [arXiv:2503.16416](https://arxiv.org/abs/2503.16416);
[OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/);
Anthropic (2025) [How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system).

## Enable

Off by default (library-friendly):

```bash
export CAT_AGENT_TRACE=1
export CAT_AGENT_TRACE_FILE=/tmp/cat-agent-traces.jsonl   # optional; else in-memory
export CAT_AGENT_TRACE_PRICE_TABLE='{"stub-model":{"input_per_1m":0.5,"output_per_1m":1.5}}'
```

Or pass `trace=True` / `trace_store=...` / `run_limits=RunLimits(...)` into `agent.run(...)`.

## Schema (v1.0)

A `Run` contains ordered `Step`s. Each step is flushed as it completes so a crashed
process still leaves a partial JSONL file.

| Field | Notes |
| --- | --- |
| `schema_version` | `"1.0"` |
| `run_id` | uuid4 |
| `status` | `running` \| `completed` \| `failed` \| `terminated` |
| `termination_reason` | e.g. `goal_reached`, `max_steps`, `max_tokens`, `wall_clock` |
| `llm_config` | redacted — never persists API keys |
| `steps[].kind` | `llm_call` \| `tool_call` \| `handoff` \| `context_op` \| `user_input` \| `error` |
| `steps[].parent_step_id` | set for nested agents / handoffs |

LLM payloads also emit OTel-compatible keys (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`, …) for a future exporter without schema migration.
No OpenTelemetry dependency is required.

## RunLimits

```python
from cat_agent.trace import RunLimits
from cat_agent.agents import Assistant

bot = Assistant(
    llm={...},
    run_limits=RunLimits(max_steps=30, max_total_tokens=50_000, max_wall_clock_seconds=120),
)
```

When a limit is hit the run stops cleanly with `status="terminated"`.

## Example JSONL records

```json
{"record_type":"run_header","schema_version":"1.0","run_id":"...","agent_name":"math","status":"running","steps":[],"...":"..."}
{"record_type":"step","run_id":"...","step_index":0,"kind":"llm_call","payload":{"model":"gpt-4o-mini","gen_ai.request.model":"gpt-4o-mini","prompt_tokens":120,"completion_tokens":18,"gen_ai.usage.input_tokens":120,"gen_ai.usage.output_tokens":18}}
{"record_type":"step","run_id":"...","step_index":1,"kind":"tool_call","payload":{"tool_name":"sum_two_number","arguments":{"a":2,"b":3},"succeeded":true}}
{"record_type":"run_final","schema_version":"1.0","run_id":"...","status":"completed","termination_reason":"goal_reached","totals":{"steps":2,"total_tokens":138}}
```

See also: [failure-analysis.md](failure-analysis.md), [context.md](context.md).
