# Cat-Agent

<div align="center">

<img src="https://i.ibb.co/gZJj7LTC/Chat-GPT-Image-Feb-7-2026-02-04-10-PM-removebg-preview.png" width="120" alt="Cat-Agent" />

**Enhancing LLMs with Agent Workflows, RAG, Function Calling, and Code Interpreter.**

[![PyPI](https://img.shields.io/badge/PyPI-cat--agent-blue)](https://pypi.org/project/cat-agent/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Overview

**Cat-Agent** is a Python framework for building LLM-powered agents with pluggable tools, multi-agent workflows, and production-ready features. Use it to add function calling, RAG, code execution, and custom tools to your chat or automation pipelines.

### Features

- **Agent workflows** — `Agent`, `Assistant`, `ReActChat`, `FnCallAgent`, `DocQAAgent`, `GroupChat`, `Router`, and more
- **Graph workflows (DAG)** — Compose agents and tools into branching/looping graphs with `StateGraph`; a compiled graph is itself an `Agent`
- **Function calling** — Native tool/function support for LLMs
- **RAG** — Native keyword (BM25), vector (HNSW), and hybrid search; no LangChain/FAISS/LEANN
- **Code interpreter** — Safe Python execution via Docker or WASM sandbox (no Docker required)
- **Rich tool set** — Web search, doc parsing, image generation, MCP, storage, and extensible custom tools
- **Multiple LLM backends** — Transformers (default local/GPU), OpenAI-compatible APIs, LlamaCpp (+ vision), OpenVINO; MLX-LM optional on Apple silicon
- **Structured logging** — Loguru-powered logging with coloured console, JSON, and file rotation support
- **Observability hooks** — Structured run/node/LLM/tool events with pluggable handlers (callbacks, print, loguru, Mermaid, OpenTelemetry)

## Requirements

- **Python 3.10+** (use `python3.10` or later to run examples and tests)

## Installation

Requires **Python 3.10+**. Base install includes **Transformers**, **PyTorch**, and **Accelerate** for the default local/GPU backend. On zsh, quote extras:

```bash
  pip install cat-agent
  pip install 'cat-agent[rag]'
```

**Optional extras:**

```bash
  pip install 'cat-agent[rag]'              # RAG (retrieval, doc parsing, etc.)
  pip install 'cat-agent[mlx]'              # MLX-LM backend (Apple silicon only)
  pip install 'cat-agent[mcp]'              # MCP (Model Context Protocol)
  pip install 'cat-agent[python_executor]'    # Python executor (math, sympy, etc.)
  pip install 'cat-agent[code_interpreter]' # Code interpreter server (Jupyter, FastAPI)
  pip install 'cat-agent[otel]'               # OpenTelemetry export for graph/agent traces
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
- **HNSW vector index** for `VectorSearch` (usearch; hash or ONNX embeddings)
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

## LLM Backends

| Backend | `model_type` | Description |
|---|---|---|
| OpenAI-compatible | `oai` | Any OpenAI-compatible API (default) |
| LlamaCpp | `llama_cpp` | Local GGUF models via llama-cpp-python |
| LlamaCpp Vision | `llama_cpp_vision` | Multimodal GGUF models (Qwen2-VL, LLaVA, etc.) |
| Transformers | `transformers` | HuggingFace Transformers models (included in base install) |
| MLX-LM | `mlx_lm` | Apple silicon local models via mlx-lm (`pip install 'cat-agent[mlx]'`) |
| OpenVINO | `openvino` | Optimised inference on Intel hardware |

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
| `cat_agent.agent` | Base `Agent` class |
| `cat_agent.agents` | Assistant, ReActChat, FnCallAgent, DocQA, GroupChat, Router |
| `cat_agent.graph` | `StateGraph` / `GraphAgent` DAG engine with Agent/Function/Tool nodes |
| `cat_agent.llm` | Chat model backends (OAI, LlamaCpp, LlamaCpp Vision, OpenVINO, Transformers) |
| `cat_agent.tools` | CodeInterpreter, WASMCodeInterpreter, Retrieval, DocParser, Storage, MCP, and more |
| `cat_agent.memory` | Memory, RAG, and context utilities |
| `cat_agent.log` | Loguru-based structured logging |
| `cat_agent.observability` | Run/node/LLM/tool event hooks and handlers (incl. Mermaid, OpenTelemetry) |
| `cat_agent.settings` | Configuration via environment variables |
| `native` | PyO3 persistent BM25 index and experimental PDF text parser; Python remains the public API |
| `examples` | Runnable demos (agents, RAG, graph workflows, observability, code interpreter) |
| `benchmarks` | Repeatable RAG index and token-accounting micro-benchmarks |

## Testing

- **Test count:** 230+ tests including observability coverage in `tests/test_observability.py`.
- **Test coverage:** **59%** (6,038 lines total).
- **Run tests:** `pytest` (install with `pip install -e ".[test]"`).
- **Report coverage:** `pytest --cov=cat_agent --cov-report=term`
- **Native checks:** `cargo test --manifest-path native/Cargo.toml --no-default-features`
- **BM25 benchmark:** `python benchmarks/benchmark_rag.py --chunks 1000 --queries 25`
- **PDF benchmark:** `python benchmarks/benchmark_pdf_parser.py --pages 10 --repeats 3`
- **Chunking benchmark:** `python benchmarks/benchmark_native_chunking.py --pages 20 --paragraphs 10`
- **Vector benchmark:** `python benchmarks/benchmark_native_vector.py --chunks 2000 --queries 25`
- **Truncation benchmark:** `python benchmarks/benchmark_native_truncation.py --turns 40 --max-tokens 2048`

## Versioning

```bash
    chmod +x release.sh        # one time
    ./release.sh 0.1.2         # or any new X.Y.Z version
```

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

**Kemalcan Bora** — [kemalcanbora@gmail.com](mailto:kemalcanbora@gmail.com)
GitHub: [kemalcanbora/cat-agent](https://github.com/kemalcanbora/cat-agent)
