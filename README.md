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

```bash
pip install cat-agent
```

**Optional extras:**

```bash
# RAG (retrieval, doc parsing, etc.)
pip install cat-agent[rag]

# MCP (Model Context Protocol)
pip install cat-agent[mcp]

# Python executor (math, sympy, etc.)
pip install cat-agent[python_executor]

# Code interpreter (Jupyter kernel)
pip install cat-agent[code_interpreter]

# Gradio GUI
pip install cat-agent[gui]
```

## Quick Start

```python
from cat_agent import Agent

agent = Agent(
    llm={'model': 'qwen-plus', 'model_server': 'dashscope'},
    function_list=['code_interpreter'],
)

messages = [{'role': 'user', 'content': 'What is 2^10?'}]
for response in agent.run(messages):
    if response:
        print(response)
```

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
