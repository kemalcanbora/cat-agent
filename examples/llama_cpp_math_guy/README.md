# llama.cpp math guy (local-only)

Local GGUF calculator demo. Exposes `registry()` for `cat-agent serve`, but
**has no `agent.yaml`** — local models are not Nomad-deployable. Use
`examples/serve_fastapi/` for the API-backed deployable variant.
