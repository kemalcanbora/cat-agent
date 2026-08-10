# Synthesis promote: groups, promote, share → adopt

Demonstrates the **content-addressed promote flow** and **cross-group sharing**
without calling an LLM. Artifacts are written with a fixed validated `impl.py`
(same shape ToolSmith would leave after synthesis).

See also: [`docs/synthesis-promote.md`](../../../docs/synthesis-promote.md).

```
write_artifacts → staging.json pointer
       ↓ promote (promoter)
   active.json  →  load_generated_tools  →  enable_optional_tools
       ↓ share (sharer in finance)
   shares.json offer
       ↓ adopt --version <sha> (sharer in ops)
   ops active.json pin  (owner-qualified name kept)
```

## Setup

```bash
python3.10 -m pip install -e ".[wasm]"   # or cat-agent[wasm] from a wheel
```

No API keys required for these scripts.

## Run

```bash
# In-group staging → promote → principal-scoped tools → demote
python3.10 examples/synthesis/promote/run_promote_demote.py

# Finance shares; ops adopts; pin survives re-promote; unshare fails loudly
python3.10 examples/synthesis/promote/run_share_adopt.py
```

Both scripts use a temp workspace under the example dir (or `--workspace`) and
the bundled [`groups.json`](groups.json) via `CAT_AGENT_MEMBERSHIP_PATH` /
`--membership`.

## CLI equivalents

With the same membership file and a real workspace:

```bash
export CAT_AGENT_MEMBERSHIP_PATH=$PWD/examples/synthesis/promote/groups.json
export CAT_AGENT_USER=lead

cat-agent synth list --group finance --workspace /path/to/ws
cat-agent synth promote validate_iban --group finance --workspace /path/to/ws --yes
cat-agent synth share validate_iban --with ops --group finance --workspace /path/to/ws
# as ops sharer:
cat-agent synth adopt finance/validate_iban --version <sha> --group ops \
  --workspace /path/to/ws --yes
```
