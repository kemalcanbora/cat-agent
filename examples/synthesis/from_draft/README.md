# Synthesis from a Markdown draft

Primary path for business users. You write a draft in your own words; the model
interviews you about gaps, confirms understanding in plain language, then
compiles an internal `ToolSpec` and synthesises a sandboxed `@tool`. You never
author JSON by hand.

```
draft.md  →  interview  →  confirmation  →  ToolSpec  →  ToolSmith
```

## Setup

```bash
python3.10 -m pip install "cat-agent[wasm]"
```

Configure the model in the repo `.env` (see `.env.example`):

```
CAT_AGENT_OFFLINE=0
OLLAMA_API_KEY=...
LLM_MODEL=minimax-m2.5:cloud
OLLAMA_API_BASE=https://ollama.com/v1
# optional: stronger / more multilingual model for intake
# INTAKE_LLM_MODEL=...
```

Ollama is reached via `model_type=oai` (same pattern as
`examples/multi_agent/team_example.py`).

## Start from a draft

Write a blank template (English by default; also `de`, `fr`, `es`, `it`, `nl`, `tr`):

```bash
python3.10 -m cat_agent.cli synth init my_tool --lang en
python3.10 -m cat_agent.cli synth init mein_werkzeug --lang de
```

Interactive VAT example (rounding deliberately left unspecified so the interview
asks about it):

```bash
python3.10 examples/synthesis/from_draft/run_from_draft.py
python3.10 examples/synthesis/from_draft/run_from_draft.py --draft examples/synthesis/from_draft/vat_draft_de.md
```

Or via the CLI:

```bash
python3.10 -m cat_agent.cli synth run examples/synthesis/from_draft/vat_draft.md
```

If no API key is configured, the script exits with a short message — never a raw
traceback.

## Artifacts

Successful runs write under the workspace:

```
<workspace>/generated_tools/<function_name>/
  draft.md         # your original text, verbatim
  interview.json   # questions, answers, confirmation
  impl.py          # model code for WASM (never imported by the host)
  tool.py          # sandboxed @tool proxy (load_generated_tools)
  <function_name>.py  # Assistant-ready @tool with logic inline
  spec.json        # compiled ToolSpec (internal)
  tests.json       # work / holdout split
  manifest.json    # includes draft_sha256, interview_sha256, draft_lang, locale
```

The task-named file (e.g. `vat_split.py`) is a normal `@tool` with the validated
logic inlined — copy it into an agent or import it directly. `tool.py` remains
the sandboxed path used by `load_generated_tools()`.

Generated tools are **opt-in**:

```python
from cat_agent.synthesis import load_generated_tools
from cat_agent.tools.base import enable_optional_tools

load_generated_tools()
enable_optional_tools('generated_split_vat_inclusive')
```

## `holdout_failed`

If synthesised code passes every work example but fails an unseen holdout case,
intake re-opens the interview on that case (up to 2 rounds), adds your answer as
a new example, and re-runs synthesis. After the cap, everything collected is
handed back for you to review. Holdout values are never fed into the code model
as repair feedback.

For hand-authored JSON specs (tests / CI / other systems), see
[../from_spec/](../from_spec/).

For group-scoped promote / demote / share → adopt (no LLM), see
[../deploy/](../deploy/).
