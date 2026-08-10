# API-backed calculator for ``cat-agent serve`` / Nomad deploy.

Local llama.cpp variant: [`../llama_cpp_math_guy/`](../llama_cpp_math_guy/)
(local-only — no `agent.yaml`).

```bash
pip install 'cat-agent[serve,platform]'
export PYTHONPATH=examples/serve_fastapi
python examples/serve_fastapi/serve_math_guy.py

# Deploy to local Nomad (sibling cat-agent-stack repo):
cd ../cat-agent-stack
export CAT_AGENT_STACK_DIR=$PWD
export CAT_AGENT_CONFIG=$PWD/cat-agent.config.toml
cat-agent stack bootstrap   # once
cd ../cat-agent
cat-agent deploy --dir examples/serve_fastapi
```
