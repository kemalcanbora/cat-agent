# Synthesis from a JSON/YAML ToolSpec

Advanced path: you already have a structured `ToolSpec` and want ToolSmith to
synthesise a sandboxed tool without the Markdown interview.

```
spec.json  →  ToolSmith  →  generated_tools/<name>/
```

Prefer [`../from_draft/`](../from_draft/) for business drafts.

## Setup

```bash
python3.10 -m pip install "cat-agent[wasm,synthesis]"
```

Configure `.env` at the repo root (see `.env.example`).

## Run

```bash
python3.10 examples/synthesis/from_spec/run_synthesis.py
python3.10 examples/synthesis/from_spec/run_synthesis.py \
  --spec examples/synthesis/from_spec/kdv_spec.json
```

## Artifacts

```
<workspace>/generated_tools/<function_name>/
  impl.py              # validated body for WASM
  tool.py              # sandboxed @tool proxy (load_generated_tools)
  <function_name>.py   # Assistant-ready @tool (logic inline)
  spec.json
  tests.json
  manifest.json
```
