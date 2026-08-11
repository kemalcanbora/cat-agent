# ParallelDocQA example

Exhaustive per-chunk document Q&A over two small text files in this directory.

## When to use this vs `Assistant` + `retrieval`

| | `Assistant(function_list=['retrieval'])` | `ParallelDocQA` |
| --- | --- | --- |
| Strategy | Agent calls retrieval; top-k chunks by keyword/searchers | LLM-scans **every** chunk (map), then GenKeyword + retrieval + summary (reduce) |
| Recall | Good when search hits | Higher recall on long / noisy corpora |
| Cost | Usually a few LLM calls | **One member LLM call per chunk** (+2), capped by `max_chunks` (default 32) |

Reach for ParallelDocQA when missing a buried sentence is worse than spending tokens.
Prefer Assistant + retrieval for routine Q&A and anything behind a tight team budget.

## Run

```bash
# Point at any OpenAI-compatible server (Ollama example):
export OPENAI_API_BASE=http://127.0.0.1:11434/v1
export CAT_AGENT_MODEL=qwen3:1.7b

python examples/parallel_doc_qa/run_parallel_doc_qa.py
```

The script prints a **cost estimate** (`estimate_member_calls`) before running.
If chunk count exceeds `max_chunks`, ParallelDocQA raises with the counts instead of
issuing unbounded member calls.
