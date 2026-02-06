# Cat-Agent

<div align="center">

<img src="https://pngimg.com/uploads/cat/cat_PNG50523.png" width="120" alt="Cat-Agent" />

**Enhancing LLMs with Agent Workflows, RAG, Function Calling, and Code Interpreter.**

[![PyPI](https://img.shields.io/badge/PyPI-cat--agent-blue)](https://pypi.org/project/cat-agent/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## Overview

**Cat-Agent** is a Python framework for building LLM-powered agents with pluggable tools, multi-agent workflows, and production-ready features. Use it to add function calling, RAG, code execution, and custom tools to your chat or automation pipelines.

### Features

- **Agent workflows** — `Agent`, `Assistant`, `ReActChat`, `FnCallAgent`, `DocQAAgent`, `GroupChat`, and more
- **Function calling** — Native tool/function support for LLMs
- **RAG** — Retrieval-augmented generation with vector, keyword, and hybrid search
- **Code interpreter** — Safe Python execution for math and code tasks
- **Rich tool set** — Web search, doc parsing, image generation, MCP, storage, and extensible custom tools
- **Multiple LLM backends** — DashScope (Qwen), OpenAI-compatible APIs, LlamaCpp, OpenVINO, Transformers

## Installation

``` bash
    pip install cat-agent
```

**Optional extras:**

```  
    pip install cat-agent[rag] # RAG (retrieval, doc parsing, etc.) 
    pip install cat-agent[mcp] # MCP (Model Context Protocol) 
    pip install cat-agent[python_executor] # Python executor (math, sympy, etc.)
``` 
 
## Examples

- **Math tool with `Assistant`**

  Run the example that registers a custom `sum_numbers` tool and uses the `Assistant` agent:

  ```bash
  python examples/math_guy/math_guy.py
  ```
  
  Example file location: `examples/math_guy/math_guy.py`

- **RAG with LEANN retriever**

  Run a Retrieval-Augmented Generation (RAG) example that uses LEANN as the semantic retriever via `Memory`:

  ```bash
  python examples/rag_leann/leann_qwen3_demo.py
  ```

  Minimal RAG usage in code:

  ```python
  from pathlib import Path
  from cat_agent.llm.schema import Message, USER
  from cat_agent.memory import Memory
  import torch

  examples_dir = Path(__file__).parent
  doc_path = examples_dir / "leann_demo_doc.txt"

  llm_cfg = {
      "model": "Qwen/Qwen3-1.7B",
      "model_type": "transformers",
      "device": "cuda:0" if torch.cuda.is_available() else "mps",
  }

  rag_cfg = {
      "enable_leann": True,
      "rag_searchers": ["leann_search"],
  }

  mem = Memory(llm=llm_cfg, files=[str(doc_path)], rag_cfg=rag_cfg)
  messages = [Message(role=USER, content="How much storage does LEANN save?")]
  responses = mem.run_nonstream(messages, force_search=True)
  print(responses[-1].content)
  ```

  This script:
  - Creates or loads `examples/rag_leann/leann_demo_doc.txt`
  - Builds RAG over the document using LEANN (`leann_search`)
  - Asks a question and prints the retrieved answer

## Project structure

| Component   | Description |
|------------|-------------|
| `cat_agent.agent` | Base agent and `BasicAgent` |
| `cat_agent.agents` | Assistant, ReActChat, FnCallAgent, DocQA, GroupChat, Router, etc. |
| `cat_agent.llm`   | Chat models (DashScope, OAI, LlamaCpp, OpenVINO, Transformers) |
| `cat_agent.tools` | CodeInterpreter, Retrieval, DocParser, Storage, WebSearch, MCP, and more |
| `cat_agent.memory`| Memory and context utilities |
 
## License

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Author

**Kemalcan Bora** — [kemalcanbora@gmail.com](mailto:kemalcanbora@gmail.com)  
GitHub: [kemalcanbora/cat-agent](https://github.com/kemalcanbora/cat-agent)
