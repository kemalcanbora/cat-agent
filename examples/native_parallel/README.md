# Live parallel native tool-call check

Point any OpenAI-compatible endpoint (LiteLLM, OpenAI, vLLM, …) at this script.
It does **not** require [cat-agent-stack](https://github.com/kemalcanbora/cat-agent-stack).

```bash
export CAT_AGENT_GATEWAY_URL=http://127.0.0.1:4000/v1   # or OPENAI_BASE_URL
export OPENAI_API_KEY=sk-...                            # or CAT_AGENT_GATEWAY_KEY
export CAT_AGENT_GATEWAY_MODEL=gpt-4o-mini              # optional

python3.10 examples/native_parallel/run_parallel_tools.py
```

If the URL is unset or unreachable, the script exits 0 with `SKIP: …`.
On success it prints the outbound `tools` payload and the returned `tool_calls`
array (with distinct ids), then the cat_agent internal + wire conversion.
