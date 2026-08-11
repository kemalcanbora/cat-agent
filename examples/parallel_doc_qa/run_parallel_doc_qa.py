"""ParallelDocQA over two small text files.

Unlike Assistant + retrieval (top-k keyword/vector recall), ParallelDocQA
LLM-scans every chunk, then runs GenKeyword + retrieval + a summary pass.
Cost scales with chunk count; default max_chunks=32 refuses larger sets.

Requires an OpenAI-compatible endpoint (or set CAT_AGENT_MODEL / API env vars).
"""

from __future__ import annotations

import os
from pathlib import Path

from cat_agent.agents import ParallelDocQA
from cat_agent.llm.schema import USER, ContentItem, Message


def main() -> None:
    here = Path(__file__).resolve().parent
    files = [str(here / 'refunds.txt'), str(here / 'shipping.txt')]

    llm = {
        'model': os.getenv('CAT_AGENT_MODEL', 'qwen3:1.7b'),
        'model_type': 'oai',
        'api_key': os.getenv('OPENAI_API_KEY', 'EMPTY'),
        'api_base': os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:11434/v1'),
    }

    bot = ParallelDocQA(llm=llm, use_polars=False, max_chunks=32)

    messages = [Message(USER, [
        ContentItem(text='What is the refund window, and how long does shipping take?'),
        ContentItem(file=files[0]),
        ContentItem(file=files[1]),
    ])]

    estimate = bot.estimate_member_calls(messages)
    print(f'Cost estimate: {estimate} member LLM calls '
          f'(plus GenKeyword + summary ≈ {estimate + 2} total)')
    print(f'max_chunks={bot.max_chunks}\n')

    for rsp in bot.run(messages):
        last = rsp[-1]
        role = last.get('role') if isinstance(last, dict) else last.role
        content = last.get('content') if isinstance(last, dict) else last.content
        if role == 'assistant' and content:
            print(content)


if __name__ == '__main__':
    main()
