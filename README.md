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
- **Function calling** — Native tool/function support for LLMs
- **RAG** — Retrieval-augmented generation with vector, keyword, and hybrid search
- **Code interpreter** — Safe Python execution via Docker or WASM sandbox (no Docker required)
- **Rich tool set** — Web search, doc parsing, image generation, MCP, storage, and extensible custom tools
- **Multiple LLM backends** — OpenAI-compatible APIs, LlamaCpp (+ vision), OpenVINO, Transformers
- **Structured logging** — Loguru-powered logging with coloured console, JSON, and file rotation support

## Installation

```bash
  pip install cat-agent
```

**Optional extras:**

```bash
  pip install cat-agent[rag]              # RAG (retrieval, doc parsing, etc.)
  pip install cat-agent[mcp]              # MCP (Model Context Protocol)
  pip install cat-agent[python_executor]  # Python executor (math, sympy, etc.)
  pip install cat-agent[code_interpreter] # Code interpreter server (Jupyter, FastAPI)
```

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

## Examples

### Math tool with LlamaCpp

Registers a custom `sum_two_number` tool and uses a local GGUF model:

```bash
  python examples/llama_cpp_math_guy/llama_cpp_example.py
```

### Math tool with Transformers

Same concept using the HuggingFace Transformers backend (Qwen3-1.7B):

```bash
  python examples/transformers_math_guy/math_guy.py
```

### Vision with LlamaCpp

Analyse images from URLs using a multimodal GGUF model (Qwen2-VL):

```bash
  python examples/llama_cpp_vision/llama_cpp_vision_example.py
```

### Document parsing agent

Parse CSV/PDF/DOCX files and ask questions about their contents:

```bash
  python examples/doc_parser_agent/doc_parser_example.py
```

### Multi-agent: GroupChat

Two agents (Alice and Bob) converse in round-robin to plan a weekend trip:

```bash
  python examples/multi_agent/group_chat_example.py
```

### Multi-agent: Router

Intelligently route queries to specialised agents (MathExpert vs GeneralAssistant):

```bash
  python examples/multi_agent/router_example.py
```

### RAG with LEANN retriever

Retrieval-augmented generation using LEANN semantic search:

```bash
  pip install cat-agent[rag]
  python examples/rag_leann/leann_qwen3_demo.py
```

Minimal RAG usage in code:

```python
from pathlib import Path
from cat_agent.llm.schema import Message, USER
from cat_agent.memory import Memory
import torch

llm_cfg = {
    "model": "Qwen/Qwen3-1.7B",
    "model_type": "transformers",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

mem = Memory(llm=llm_cfg, files=["doc.txt"], rag_cfg={"enable_leann": True, "rag_searchers": ["leann_search"]})
messages = [Message(role=USER, content="How much storage does LEANN save?")]
responses = mem.run_nonstream(messages, force_search=True)
print(responses[-1].content)
```

### WASM code interpreter

Secure Python code execution in a WebAssembly sandbox (no Docker or Node.js needed):

```bash
  python examples/wasm_code_interpreter/wasm_code_interpreter_example.py
```

### Logging demo

Demonstrates coloured console logs, JSON output, and file logging alongside an agent:

```bash
  python examples/logging_demo/logging_example.py

  # Or with env-var driven config:
  CAT_AGENT_LOG_LEVEL=DEBUG python examples/logging_demo/logging_example.py
```

## LLM Backends

| Backend | `model_type` | Description |
|---|---|---|
| OpenAI-compatible | `oai` | Any OpenAI-compatible API (default) |
| LlamaCpp | `llama_cpp` | Local GGUF models via llama-cpp-python |
| LlamaCpp Vision | `llama_cpp_vision` | Multimodal GGUF models (Qwen2-VL, LLaVA, etc.) |
| Transformers | `transformers` | HuggingFace Transformers models |
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
| `cat_agent.llm` | Chat model backends (OAI, LlamaCpp, LlamaCpp Vision, OpenVINO, Transformers) |
| `cat_agent.tools` | CodeInterpreter, WASMCodeInterpreter, Retrieval, DocParser, Storage, MCP, and more |
| `cat_agent.memory` | Memory, RAG, and context utilities |
| `cat_agent.log` | Loguru-based structured logging |
| `cat_agent.settings` | Configuration via environment variables |

## Testing

- **Test count:** 222+ tests across `tests/test_agent.py`, `tests/test_agents.py`, `tests/test_llm.py`, `tests/test_memory.py`, `tests/test_tools.py`, and `tests/test_utils.py`.
- **Test coverage:** **59%** (6,038 lines total).
- **Run tests:** `pytest` (install with `pip install -e ".[test]"`).
- **Report coverage:** `pytest --cov=cat_agent --cov-report=term`

## Versioning
    chmod +x release.sh        # one time
    ./release.sh 0.1.2         # or any new X.Y.Z version

## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

**Kemalcan Bora** — [kemalcanbora@gmail.com](mailto:kemalcanbora@gmail.com)
GitHub: [kemalcanbora/cat-agent](https://github.com/kemalcanbora/cat-agent)
