# Cat-Agent

<div align="center">

<img src="https://i.ibb.co/gZJj7LTC/Chat-GPT-Image-Feb-7-2026-02-04-10-PM-removebg-preview.png" width="120" alt="Cat-Agent" />

**On-premise, sandboxed AI agent platform for regulated sectors** — public sector, finance, and healthcare deployments where data must not leave your infrastructure.

[![PyPI](https://img.shields.io/badge/PyPI-cat--agent-blue)](https://pypi.org/project/cat-agent/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Overview

**Cat-Agent** is a lightweight Python framework for building LLM-powered agents that run fully on-premise. It provides pluggable tools, multi-agent workflows, WASM-sandboxed code execution, native RAG, and air-gap controls for environments that cannot call cloud APIs.

Set `CAT_AGENT_OFFLINE=1` to disable network-dependent tools at registration time and block outbound HTTP/socket calls with an explicit `OfflineViolationError`.

### What's new in 0.7.0

On-prem security platform for regulated sectors:

- **Air-gap mode** — `CAT_AGENT_OFFLINE=1` kill-switch; opt-in network tools; SearxNG web search
- **Encrypted storage** — AES-GCM for doc caches, agent memory, and RAG indexes
- **Audit trail** — hash-chained JSONL with `audit-verify` / `audit-export` CLI
- **PII redaction** — offline regex (and optional Presidio) at RAG, prompt, and audit layers
- **Deployment package** — `deploy/docker-compose.yml` for air-gapped Docker installs
- **SBOM** — CycloneDX bill of materials generated on each release
- **CLI** — `cat-agent offline-check`, `synth`, `encrypt-storage`, `fetch-runtime`, and more

### What's new in 0.8.1

- **Tool retry** — opt-in per-tool retries with shared exponential backoff; attempts stay out of message history
- **Timeouts** — `attempt_timeout` (async) and `run_timeout`; sync path warns instead of faking interrupts
- **Rate limiting** — shareable token-bucket + concurrency cap for LLM and tools (no process global)
- **code_interpreter `timeout` cfg** — `{'name': 'code_interpreter', 'timeout': 10}` now arms the kernel timer
- **Example** — `examples/tool_resilience/` (local llama_cpp; requires model download)

### Security & on-prem controls

| Control | Purpose |
|---|---|
| `CAT_AGENT_OFFLINE=1` | Air-gap kill-switch: disable network tools, block outbound requests |
| `CAT_AGENT_OFFLINE_ALLOW_HOSTS` | Comma-separated allowlist for on-prem endpoints (hostnames, IPs, CIDR) |
| `.env` / `CAT_AGENT_ENV_FILE` | Load on-prem settings from a file (no manual `export` needed) |
| `cat-agent offline-check` | Readiness report (WASM runtime, disabled tools, issues) |
| `cat-agent fetch-runtime --output <dir>` | Copy bundled WASM assets for offline transfer |
| Opt-in network tools | `web_search`, `image_search`, `web_extractor` are not in the default registry |
| Self-hosted web search | Point `CAT_AGENT_SEARXNG_URL` at your SearxNG instance (Serper is legacy opt-in only) |
| `pip install 'cat-agent[wasm-bundled]'` | WASM CPython runtime shipped in the wheel (no download on first use) |
| Encrypted doc cache | Doc-parser cache encrypted at rest (AES-GCM); keys via OS keyring or env |
| `CAT_AGENT_ENCRYPT_AT_REST=1` | Encrypt agent memory, RAG indexes, and all SQLite caches |
| `cat-agent encrypt-storage` | Migrate plaintext workspace data to encrypted format |
| `deploy/docker-compose.yml` | One-command on-prem deployment package |
| `CAT_AGENT_AUDIT=1` | Tamper-evident audit trail (hash-chained JSONL) |
| PII redaction | Offline regex redaction for RAG, prompts, and audit logs |

```bash
cp .env.example .env
# edit .env — set OPENAI_BASE_URL, CAT_AGENT_ENCRYPTION_KEY, allowlist, etc.
cat-agent offline-check
```

Cat-Agent loads `.env` from the current working directory on `import cat_agent` and when using the `cat-agent` CLI. Set `CAT_AGENT_ENV_FILE` to use another path. Shell exports still work and take precedence over `.env`.

`OPENAI_BASE_URL` (or `CAT_AGENT_LLM_BASE_URL`) is automatically added to the allowlist so agents can reach your on-prem model gateway while outbound internet access stays blocked.

### Encrypted local storage

All sensitive local data is encrypted at rest by default (`CAT_AGENT_ENCRYPT_AT_REST=1`):

| Data | Location |
|---|---|
| Doc-parser cache | `workspace/tools/doc_parser/` |
| Parsed document cache | `workspace/tools/simple_doc_parser/` |
| Agent memory (Storage tool) | `workspace/tools/storage/` |
| BM25 / vector RAG indexes | `workspace/storage/keyword_indexes/`, `vector_indexes/` |

Uses AES-GCM with the same key management as the doc cache (see below). RAG index
files are stored as `*.enc` blobs; metadata JSON is encrypted in place.

**Migrate existing plaintext data:**

```bash
cat-agent encrypt-storage --workspace ./workspace
```

**Strict mode** — refuse encrypted components if plaintext remains:

```bash
# .env
CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE=1
```

Disable encryption only for local development:

```bash
# .env
CAT_AGENT_ENCRYPT_AT_REST=0
```

### Encrypted document cache

Parsed and chunked documents are cached under `workspace/tools/doc_parser/`.
Encryption is controlled by `CAT_AGENT_ENCRYPT_AT_REST=1` (on by default).
Cache keys are SHA-256 hashes — document paths are not written to disk or logs.

**Key management** (first match wins):

1. `CAT_AGENT_ENCRYPTION_KEY` — base64-encoded 32-byte AES key (recommended for
   air-gapped servers and CI)
2. OS keyring (`cat-agent` / `encryption-key`) — auto-generated on first use when
   available

```bash
# Generate a key for offline transfer to the target host
python - <<'PY'
import base64, secrets
print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())
PY

# .env
CAT_AGENT_ENCRYPTION_KEY=<paste-key-here>
```

**Migrate an existing plaintext cache:**

```bash
cat-agent encrypt-storage --workspace ./workspace
# or a single cache directory:
cat-agent encrypt-cache --path ./workspace/tools/doc_parser
```

**Strict mode** — refuse to start if any plaintext storage remains:

```bash
# .env
CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE=1
```

Disable encryption only for local development:

```bash
# .env
CAT_AGENT_ENCRYPT_AT_REST=0
```

### Air-gapped deployment package

Build and transfer a self-contained Docker image for regulated networks:

```bash
cp deploy/.env.example deploy/.env   # set CAT_AGENT_ENCRYPTION_KEY
docker compose -f deploy/docker-compose.yml build
docker save cat-agent:on-prem | gzip > cat-agent-on-prem.tar.gz
```

On the target host (no internet):

```bash
docker load < cat-agent-on-prem.tar.gz
docker compose -f deploy/docker-compose.yml up
```

See [deploy/README.md](deploy/README.md) for volume layout (workspace, models, audit).

Release artifacts include a CycloneDX SBOM (`sbom-cat-agent.cdx.json`) attached to the
GitHub release (not uploaded to PyPI). Generate locally:

```bash
./scripts/generate_sbom.sh sbom/
```

### Tamper-evident audit trail (AI Act-ready)

Enable hash-chained audit logging for prompts, model outputs, tool calls, and
file access. Each record embeds the hash of the previous record; optional HMAC
signatures use your encryption key.

```bash
# .env
CAT_AGENT_AUDIT=1
CAT_AGENT_AUDIT_PATH=./workspace/storage/audit/audit.jsonl

# Verify chain integrity (for auditors / compliance reviews)
cat-agent audit-verify --path ./workspace/storage/audit/audit.jsonl

# Export records for external review
cat-agent audit-export --path ./workspace/storage/audit/audit.jsonl --output ./audit-export.jsonl
```

Audit records are written automatically on every agent run when
`CAT_AGENT_AUDIT=1`. File paths in audit logs use SHA-256 hashes, not plaintext
paths.

### PII detection & redaction (GDPR data minimization)

Fully offline regex-based PII redaction is enabled by default at three points:

| Interception point | Env var | Default |
|---|---|---|
| RAG document ingestion | `CAT_AGENT_PII_REDACT_RAG` | on |
| Prompts sent to the LLM | `CAT_AGENT_PII_REDACT_PROMPTS` | on |
| Audit log records | `CAT_AGENT_PII_REDACT_AUDIT` | on |

Detected patterns include emails, phone numbers, IBANs, credit-card-like
sequences, and Turkish TC kimlik numbers (with checksum validation). Redacted
values are replaced with `[PII]`.

```bash
# .env — disable all PII redaction (dev only)
CAT_AGENT_PII_REDACT=0

# Disable a single interception point
CAT_AGENT_PII_REDACT_AUDIT=0
```

Optional NER-based redaction via Presidio (still fully offline):

```bash
pip install 'cat-agent[pii]'
```

Cloud-backed tools (`image_search` via SerpAPI, legacy `WEB_SEARCH_BACKEND=serper`) require explicit opt-in via `cat_agent.tools.enable_optional_tools(...)` and are blocked when offline mode is on.

### Long-term agent memory

Give any `Assistant` persistent, cross-session memory with `memory_cfg`:

```python
from cat_agent.agents import Assistant

agent = Assistant(
    llm=llm_cfg,
    memory_cfg={
        'scope': 'user:alice',            # cross-session namespace
        'top_k': 5,                       # memories recalled per query
        'auto_record': True,              # store each completed exchange
        'auto_summarize': True,           # compact long histories via the LLM
        'session_window_tokens': 8000,    # dialogue token budget before compaction
    },
)
```

What it does on every run:

1. **Recall** — the last user query is embedded (offline hash embeddings + native
   HNSW index) and the top-k relevant memories are injected into the system prompt.
2. **Compaction** — when history exceeds the token budget, older turns are
   summarized by the LLM into a `summary` memory; recent turns stay verbatim.
3. **Auto-record** — completed user/assistant exchanges are stored as `episode`
   memories, so the next session can recall them.

Programmatic access via `MemoryManager` / `MemoryStore`:

```python
from cat_agent.memory import MemoryManager

memory = MemoryManager(cfg={'scope': 'user:alice'})
memory.remember('Alice prefers reports in Turkish', kind='fact')
memory.recall('what language should the report use?')
memory.forget(memory_id)
```

Memory records live in `workspace/memory/memories.sqlite` and are encrypted at
rest (AES-GCM) with the same key management as all other Cat-Agent storage.
Everything runs fully offline — recall uses hash embeddings by default, or set
`embedding_backend: 'onnx'` with a local model for semantic vectors.

### CLI (`cat-agent`)

| Command | Purpose |
|---|---|
| `synth init <name> [--lang en]` | Write a blank Markdown draft template |
| `synth run <draft.md>` | Interview + synthesise a sandboxed tool from a draft |
| `offline-check [--strict]` | Air-gap readiness report |
| `fetch-runtime --output <dir>` | Copy WASM assets for offline transfer |
| `encrypt-storage [--workspace <dir>]` | Encrypt plaintext caches and RAG indexes |
| `encrypt-cache --path <dir>` | Encrypt a single SQLite cache directory |
| `audit-verify --path <file>` | Verify tamper-evident audit chain |
| `audit-export --path <file> --output <file>` | Export audit records for reviewers |

### Native Rust extension (`cat_agent._native`)

Platform wheels ship a PyO3 extension that powers the performance-critical RAG
and tokenization paths. Python remains the public API; there are **no fallbacks**
for these components:

| Module | Python entry points |
|---|---|
| BM25 index | `KeywordSearch`, `RagIndex` |
| HNSW vector index | `VectorSearch`, `VectorIndex` |
| Hash embeddings | `HashEmbedder`, `native.hash_embed` |
| Tokenizer / truncation | `count_tokens`, `truncate_messages` |
| Document chunking | `DocParser.split_doc_to_chunk` |
| PDF text extraction | `.pdf` ingestion in `DocParser` |

```python
import cat_agent._native as native
print(native.__version__)
```

Source installs build the extension via maturin; published wheels do not require
a local Rust toolchain.

### Features

- **Tool synthesis** — Markdown draft → interview → sandboxed `@tool` (`cat_agent.synthesis`; needs `[wasm]`)
- **Agent workflows** — `Agent`, `Assistant`, `ReActChat`, `FnCallAgent`, `DocQAAgent`, `GroupChat`, `Router`, and more
- **Async runs** — `arun` / `arun_nonstream`; parallel tool calls via `asyncio.gather` (does not stream tokens)
- **`@tool` decorator** — register plain sync/async functions as OpenAI-compatible tools
- **Multi-agent collaboration** — blackboard artifacts, handoff, and ask-agent tools (`cat_agent.multi_agent`)
- **Graph workflows (DAG)** — Compose agents and tools into branching/looping graphs with `StateGraph`; a compiled graph is itself an `Agent`
- **Function calling** — Native tool/function support for LLMs
- **RAG** — Native keyword (BM25), vector (HNSW), and hybrid search; no LangChain/FAISS/LEANN
- **Long-term memory** — Encrypted cross-session memory with vector recall and automatic history compaction (`memory_cfg`)
- **Code interpreter** — Safe Python execution via Docker or WASM sandbox (no Docker required)
- **Rich tool set** — Doc parsing, RAG, MCP, storage, WASM sandbox; network tools (web search, image search) are opt-in
- **Multiple LLM backends** — OpenAI-compatible APIs (base install); Transformers, LlamaCpp, MLX-LM, OpenVINO via optional extras
- **Structured logging** — Loguru-powered logging with coloured console, JSON, and file rotation support
- **Observability hooks** — Structured run/node/LLM/tool events with pluggable handlers (callbacks, print, loguru, Mermaid, OpenTelemetry, Langfuse)

## Requirements

- **Python 3.10+** (use `python3.10` or later to run examples and tests)

## Installation

Requires **Python 3.10+**. The **base install is lightweight** (OpenAI-compatible
API client, tools, agents, and the native Rust extension). Heavy local-model
backends are optional extras. On zsh, quote extras:

```bash
  pip install cat-agent
  pip install 'cat-agent[rag]'
  pip install 'cat-agent[transformers]'   # HuggingFace local/GPU models
  pip install 'cat-agent[llama]'          # llama-cpp-python GGUF models
  pip install 'cat-agent[wasm]'           # WASM code interpreter (downloads runtime if not bundled)
  pip install 'cat-agent[wasm-bundled]' # WASM runtime included in wheel (air-gap friendly)
  pip install 'cat-agent[local]'          # transformers + llama + wasm
```

**Optional extras:**

```bash
  pip install 'cat-agent[rag]'                # RAG doc parsing and retrieval
  pip install 'cat-agent[transformers]'       # Transformers / PyTorch backend
  pip install 'cat-agent[llama]'              # LlamaCpp (+ vision) backend
  pip install 'cat-agent[wasm]'               # WASM sandboxed code interpreter
  pip install 'cat-agent[wasm-bundled]'       # WASM runtime bundled in package (offline installs)
  pip install 'cat-agent[local]'              # All local backends above
  pip install 'cat-agent[mlx]'                # MLX-LM backend (Apple silicon)
  pip install 'cat-agent[mcp]'                # MCP (Model Context Protocol)
  pip install 'cat-agent[pii]'                # Optional Presidio NER redaction
  pip install 'cat-agent[python_executor]'    # Unsafe in-process Python executor
  pip install 'cat-agent[code_interpreter]'   # Docker/Jupyter code interpreter server
  pip install 'cat-agent[otel]'                 # OpenTelemetry export for graph/agent traces
  pip install 'cat-agent[synthesis]'            # ToolSpec YAML helpers for tool synthesis
```

### Local test with the same install path as PyPI

Before publishing, install from a built wheel exactly like an end user:

```bash
  ./scripts/install_consumer.sh rag examples/rag_keyword/rust_keyword_search_demo.py
```

This builds the native wheel, runs `pip install cat-agent[rag]`, syncs the native
extension into `./cat_agent/` (so imports work when you run from the repo root),
and optionally runs an example. CI uses the same script in the `consumer-install` job.

### Rust RAG engine

Released platform wheels include the native Rust stack used by RAG:

- **BM25 index** for `KeywordSearch`
- **HNSW vector index** for `VectorSearch` (usearch; native hash or ONNX embeddings)
- **Keyword tokenization** (English stemming + Chinese segmentation via `jieba-rs`)
- **Qwen token counting / truncation / document chunking**
- **LLM input truncation** before each model call
- **PDF text extraction** for `.pdf` ingestion

There are no Python fallbacks for these paths. Installing a published wheel does
not require a local Rust toolchain; source installs require Rust because maturin
builds the native extension during install.

The public `KeywordSearch`, `Retrieval`, `Record`, and `Chunk` APIs do not
change. The Rust implementation caches one index per unchanged corpus instead
of rebuilding and re-tokenizing it for every query. The index is persisted under
`workspace/storage/keyword_indexes/`; pass
`rebuild_rag=True` to rebuild it or `keyword_index_path` to override its path.

The Rust PDF parser is text-only. It does not preserve tables, images, or
layout metadata the way the old Python pdfminer/pdfplumber stack did.

### RAG search backends

Configured via `rag_searchers` (default: `keyword_search` + `front_page_search`):

| Searcher | Backend | Notes |
|---|---|---|
| `keyword_search` | Rust BM25 | Default; persistent index on disk |
| `vector_search` | Rust HNSW (usearch) | Hash embeddings by default; optional ONNX via `[rag]` |
| `front_page_search` | Heuristic | Boosts first chunks when the doc fits in context |
| `hybrid_search` | Fusion | Used automatically when multiple searchers are configured |

Removed backends: **LEANN**, LangChain, FAISS, and OpenAI embedding APIs are no longer used.

## Logging

Cat-Agent uses [Loguru](https://github.com/Delgan/loguru) for structured, coloured logging. By default the logger is **silent** (library-friendly). Activate it with a single environment variable:

```bash
# Pretty coloured output
CAT_AGENT_LOG_LEVEL=INFO python my_script.py

# Full debug verbosity
CAT_AGENT_LOG_LEVEL=DEBUG python my_script.py

# Structured JSON logs (for log aggregation pipelines)
CAT_AGENT_LOG_LEVEL=INFO CAT_AGENT_LOG_FORMAT=json python my_script.py

# Also write to a rotating log file
CAT_AGENT_LOG_LEVEL=DEBUG CAT_AGENT_LOG_FILE=agent.log python my_script.py
```

Or configure programmatically:

```python
from cat_agent.log import logger, setup_logger

setup_logger(level="DEBUG")                         # coloured stderr
setup_logger(level="INFO", log_file="/tmp/cat.log") # + rotating file
setup_logger(level="DEBUG", fmt="json")             # structured JSON

logger.info("Agent started")
logger.debug("Processing query: {}", query)
```

| Env Variable | Values | Default |
|---|---|---|
| `CAT_AGENT_LOG_LEVEL` | `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | *(silent)* |
| `CAT_AGENT_LOG_FILE` | file path | *(none)* |
| `CAT_AGENT_LOG_FORMAT` | `pretty`, `json` | `pretty` |

## Defining tools with `@tool`

Prefer a plain function when the tool is a thin wrapper around typed arguments.
Schemas are derived from type hints and the docstring (`docstring_parser` supports
Google, NumPy, and reST). Class-based `BaseTool` tools keep working unchanged.

**Before** (hand-written class):

```python
from cat_agent.tools.base import BaseTool, register_tool

@register_tool('sum_two_number')
class SumTwoNumber(BaseTool):
    description = 'Adds two numbers.'
    parameters = {
        'type': 'object',
        'properties': {
            'a': {'description': 'First number', 'type': 'number'},
            'b': {'description': 'Second number', 'type': 'number'},
        },
        'required': ['a', 'b'],
    }

    def call(self, params, **kwargs):
        params = self._verify_json_format_args(params)
        return str(float(params['a']) + float(params['b']))
```

**After** (`@tool`):

```python
from cat_agent.tools import tool

@tool
def sum_two_number(a: float, b: float) -> float:
    """Adds two numbers.

    Args:
        a: First number
        b: Second number
    """
    return a + b

# Still a normal function:
assert sum_two_number(2, 3) == 5.0

# And a registered BaseTool for agents:
# Assistant(..., function_list=['sum_two_number'])
# Assistant(..., function_list=[sum_two_number])
```

See `examples/tool_decorator/sum_two_number.py`.

## Async API (`arun`)

Cat-Agent provides a native async entry point alongside sync `run` / `run_nonstream`:

| Sync | Async |
|---|---|
| `run` | `arun` |
| `run_nonstream` | `arun_nonstream` |

**The async path does not stream tokens.** `arun` collects each model turn fully and yields complete message lists (not token deltas). Prefer `arun_nonstream` when you only need the final response.

```python
import asyncio
from cat_agent.agents import Assistant

async def main():
    agent = Assistant(llm={'model': 'qwen2.5', 'model_server': 'http://localhost:8000/v1'})
    result = await agent.arun_nonstream([{'role': 'user', 'content': 'Hello'}])
    print(result)
    await agent.aclose()

asyncio.run(main())
```

When the model returns **multiple tool calls in one turn**, `FnCallAgent.arun` runs them concurrently (`asyncio.gather`). Sync tools are offloaded with `asyncio.to_thread`; `@tool` async functions and MCP tools use native awaitables (`acall`).

Calling sync `run()` from inside a running event loop still works but **blocks** that loop and emits a one-time `RuntimeWarning` — use `arun()` from FastAPI/Jupyter instead.

### Which LLM backends benefit?

| Backend | Async benefit |
|---|---|
| `oai` (OpenAI-compatible HTTP) | Real — reused `AsyncOpenAI` client, non-blocking HTTP |
| `llama_cpp`, `transformers`, `mlx_lm`, `openvino` | Loop stays responsive via thread offload only; **no** inference concurrency gain on one model instance |

Optional lifecycle: `async with agent:` calls `aclose()` (closes the reused async HTTP client). Do not call `arun` after `aclose`.

See `examples/async_agent/physics_guy.py` for a real Assistant that uses ``arun``
with parallel physics tools (kinetic energy, potential energy, equatorial speed).

## Graph Workflows (DAG)

Compose agents and tools into a graph of nodes and edges instead of a fixed loop. Build a `StateGraph`, wire nodes with static or conditional edges (loops allowed, bounded by `max_steps`), then `compile()` it. The compiled `GraphAgent` **is an `Agent`**, so it streams, emits observability events, and composes with `Router`/`GroupChat`.

```python
from cat_agent.agents import Assistant
from cat_agent.graph import StateGraph, AgentNode, FunctionNode, GraphState, END

math_guy = Assistant(llm=llm_cfg, name="math_guy", function_list=["sum_numbers"])
chat = Assistant(llm=llm_cfg, name="chat")

def classify(state: GraphState) -> GraphState:
    text = state.last_message.content or ""
    state.scratch["is_math"] = any(c.isdigit() for c in text)
    return state

app = (
    StateGraph()
    .add_node(FunctionNode("classify", classify))
    .add_node(AgentNode("math_guy", math_guy))
    .add_node(AgentNode("chat", chat))
    .set_entry("classify")
    .add_conditional_edges("classify", lambda s: "math_guy" if s.scratch["is_math"] else "chat")
    .add_edge("math_guy", END)
    .add_edge("chat", END)
    .compile(name="MathGuyGraph")
)

for chunk in app.run([{"role": "user", "content": "What is 1+2+3+4+5?"}]):
    print(chunk[-1]["content"])
```

**Node types:** `AgentNode` (wrap any agent or sub-graph), `FunctionNode` (arbitrary Python / routing flags), `ToolNode` (invoke a registered tool).

### Visualizing the DAG

The graph engine emits `node.start` / `node.end` events (each `node.end` records the `next` edge taken). Two handlers turn them into a diagram:

```python
from cat_agent.observability import MermaidExporter, OpenTelemetryHandler

# 1) Mermaid diagram (dependency-free) — writes graph_dag.mmd on run.end
exporter = MermaidExporter(path="graph_dag.mmd")
app = graph.compile(name="MathGuyGraph", handlers=[exporter])
# ... after a run: print(exporter.to_mermaid()) or paste the .mmd into https://mermaid.live

# 2) OpenTelemetry spans (pip install cat-agent[otel]) — view in Jaeger, Tempo,
#    or as an agent graph in Arize Phoenix. Configure your OTel exporter, then:
app = graph.compile(name="MathGuyGraph", handlers=[OpenTelemetryHandler()])
```

### Example

```bash
  python examples/graph/math_guy.py            # run the graph
  GRAPH_TRACE=1 python examples/graph/math_guy.py  # + print node trace and write graph_dag.mmd
```

## Observability

Cat-Agent emits structured events for agent runs, graph nodes, LLM calls, and tool execution. Handlers are **opt-in** — when none are registered, behavior and performance are unchanged.

### Quick start

```python
from cat_agent.agents import Assistant
from cat_agent.llm.schema import USER, Message
from cat_agent.observability import CallbackHandler, PrintHandler

# Option 1: callback (no manual event parsing — use event.summary())
def on_event(event):
    print(event.summary())

bot = Assistant(llm=..., handlers=[CallbackHandler(on_event)])

# Option 2: print directly
bot = Assistant(llm=..., handlers=[PrintHandler()])

list(bot.run([Message(role=USER, content="Hello")]))
```

### Environment variables

```bash
# Enable default loguru trace output
CAT_AGENT_TRACE=1 python my_script.py

# Optional trace log level (default: INFO)
CAT_AGENT_TRACE_LEVEL=DEBUG python my_script.py
```

### Event types

| Event | When |
|---|---|
| `run.start` / `run.end` / `run.error` | Agent `run()` lifecycle |
| `node.start` / `node.end` | Each graph node (`node.end` records the `next` edge) |
| `llm.start` / `llm.end` / `llm.chunk` | Each LLM call (chunks optional) |
| `tool.start` / `tool.end` / `tool.error` | Each tool invocation |

Each event includes `trace_id`, `run_id`, `span_id`, agent name/class, and a typed `payload` dict. Use `event.to_dict()` for JSON export.

### Example

```bash
  python examples/observability/observability_example.py
```

### Tool synthesis (draft → sandboxed `@tool`)

Business users write a Markdown draft; Cat-Agent interviews for gaps, confirms in
plain language, compiles an internal `ToolSpec`, and synthesises code that is
**validated inside WASM**. Successful runs also write a task-named Assistant-ready
`@tool` module with the logic inlined.

```bash
pip install 'cat-agent[wasm,synthesis]'
# configure OLLAMA_API_KEY / LLM_MODEL in .env — see .env.example
python3.10 -m cat_agent.cli synth init my_tool --lang en
python3.10 -m cat_agent.cli synth run my_tool_draft.md
# or:
python3.10 examples/synthesis/from_draft/run_from_draft.py
```

Artifacts land under `workspace/generated_tools/<name>/` (`impl.py`, sandboxed
`tool.py`, Assistant-ready `<name>.py`, `spec.json`, provenance). Load the
sandboxed proxy with `load_generated_tools()` + `enable_optional_tools(...)`.

Advanced / JSON path: `examples/synthesis/from_spec/`.

### Langfuse (local Docker)

Use `@with_langfuse` — it loads `LANGFUSE_*` / `OTEL_*` from `.env`, configures
OTLP export, and flushes on exit. Attach `OpenTelemetryHandler` to the agent.

```bash
cd examples/langfuse && docker compose up -d
cp examples/langfuse/.env.example examples/langfuse/.env
pip install 'cat-agent[otel]'
python examples/langfuse/langfuse_example.py
# UI: demo@example.com / password1
```

## LLM Backends

| Backend | `model_type` | Description |
|---|---|---|
| OpenAI-compatible | `oai` | Any OpenAI-compatible API (default) |
| Transformers | `transformers` | HuggingFace Transformers models (`pip install 'cat-agent[transformers]'`) |
| MLX-LM | `mlx_lm` | Apple silicon local models via mlx-lm (`pip install 'cat-agent[mlx]'`) |
| OpenVINO | `openvino` | Optimised inference on Intel hardware |
| LlamaCpp | `llama_cpp` | Local GGUF models via llama-cpp-python (`pip install 'cat-agent[llama]'`) |
| LlamaCpp Vision | `llama_cpp_vision` | Multimodal GGUF models (`pip install 'cat-agent[llama]'`) |

```python
from cat_agent.agents import Assistant

bot = Assistant(
    llm={"model_type": "llama_cpp", "repo_id": "Salesforce/xLAM-2-3b-fc-r-gguf", "filename": "xLAM-2-3B-fc-r-F16.gguf"},
    name="MyAgent",
    function_list=["my_tool"],
)
```

## Project Structure

| Component | Description |
|---|---|
| `cat_agent.agent` | Base `Agent` class (`run` / `arun`) |
| `cat_agent.agents` | Assistant, ReActChat, FnCallAgent, DocQA, GroupChat, Router |
| `cat_agent.multi_agent` | Blackboard, handoff, ask-agent, artifact tools for team workflows |
| `cat_agent.graph` | `StateGraph` / `GraphAgent` DAG engine with Agent/Function/Tool nodes |
| `cat_agent.llm` | Chat model backends (OAI, LlamaCpp, LlamaCpp Vision, OpenVINO, Transformers, MLX) |
| `cat_agent.tools` | `@tool`, CodeInterpreter, WASMCodeInterpreter, Retrieval, DocParser, Storage, MCP, and more |
| `cat_agent.memory` | Memory, RAG, and context utilities |
| `cat_agent.log` | Loguru-based structured logging |
| `cat_agent.observability` | Run/node/LLM/tool event hooks and handlers (incl. Mermaid, OpenTelemetry, Langfuse) |
| `cat_agent.synthesis` | ToolSpec → ToolSmith → sandboxed `@tool`; Markdown intake interview |
| `cat_agent.settings` | Configuration via environment variables |
| `cat_agent._native` | Rust extension: BM25, HNSW, PDF parser, Qwen tokenizer, chunking, truncation |
| `native` | Rust source (maturin/PyO3); builds `cat_agent._native` |
| `examples` | Runnable demos — see list below |
| `benchmarks` | Repeatable RAG / vector / truncation micro-benchmarks |

### Examples (in-repo)

| Path | Topic |
|---|---|
| `examples/tool_decorator/` | `@tool` function decorator |
| `examples/tool_resilience/` | Tool retry, attempt_timeout, rate limiting (llama_cpp; needs model) |
| `examples/async_agent/` | `arun` + parallel tools (PhysicsGuy / llama_cpp) |
| `examples/transformers_math_guy/` | Transformers + function calling |
| `examples/llama_cpp_math_guy/` | LlamaCpp + tools |
| `examples/mlx_lm_math_guy/` | MLX-LM (Apple silicon) |
| `examples/llama_cpp_vision/` | Multimodal LlamaCpp |
| `examples/doc_parser_agent/` | Document Q&A |
| `examples/multi_agent/` | GroupChat, Router, and team blackboard demo |
| `examples/mcp_service/` | Expose an agent as an MCP server |
| `examples/graph/` | DAG workflow (`StateGraph`) |
| `examples/observability/` | Trace handlers |
| `examples/langfuse/` | OpenTelemetry → local Langfuse UI |
| `examples/synthesis/from_draft/` | Markdown draft → interview → sandboxed tool |
| `examples/synthesis/from_spec/` | JSON/YAML ToolSpec → ToolSmith |
| `examples/rag_keyword/` | Rust BM25 keyword search |
| `examples/rag_vector/` | Native HNSW vector search |
| `examples/rag_native/` | Chunking + vector + truncation pipeline |
| `examples/long_term_memory/` | Cross-session memory, recall, and compaction |
| `examples/wasm_code_interpreter/` | WASM sandbox execution |
| `examples/logging_demo/` | Loguru configuration |

Run any example from the repo root after installing the matching extras, e.g.
`python examples/rag_vector/native_vector_search_demo.py`.

## Testing

- **Test count:** 890+ test functions (`pytest`)
- **Run tests:** `pytest` (install with `pip install -e ".[test,local,rag,otel,synthesis]"`)
- **Report coverage:** `pytest --cov=cat_agent --cov-report=term`
- **Native checks:** `cargo test --manifest-path native/Cargo.toml --no-default-features`
- **BM25 benchmark:** `python benchmarks/benchmark_rag.py --chunks 1000 --queries 25`
- **PDF benchmark:** `python benchmarks/benchmark_pdf_parser.py --pages 10 --repeats 3`
- **Chunking benchmark:** `python benchmarks/benchmark_native_chunking.py --pages 20 --paragraphs 10`
- **Vector benchmark:** `python benchmarks/benchmark_native_vector.py --chunks 2000 --queries 25`
- **Truncation benchmark:** `python benchmarks/benchmark_native_truncation.py --turns 40 --max-tokens 2048`

## Versioning

Release wheels are built for **abi3 Python 3.10+** on:

- Linux x86_64 and aarch64 (`manylinux`)
- macOS arm64
- Windows amd64

```bash
    chmod +x release.sh        # one time
    ./release.sh 0.1.2         # or any new X.Y.Z version
```

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

**Kemalcan Bora** — [kemalcanbora@gmail.com](mailto:kemalcanbora@gmail.com)
GitHub: [kemalcanbora/cat-agent](https://github.com/kemalcanbora/cat-agent)
