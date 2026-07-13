"""Benchmark native LLM input message truncation.

Run after installing cat-agent:

    python benchmarks/benchmark_native_truncation.py --turns 40 --max-tokens 2048

Build the native extension with ``maturin develop`` before running.
"""

from __future__ import annotations

import argparse
import statistics
import time

from cat_agent.llm.base.truncation import truncate_input_messages_roughly
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer


def _conversation(turns: int) -> list[Message]:
    messages = [Message(role=SYSTEM, content='You are a helpful assistant.')]
    for index in range(turns):
        messages.append(
            Message(
                role=USER,
                content=f'Question {index}: ' + ('explain RAG pipelines and retrieval ' * 30),
            )
        )
        messages.append(
            Message(
                role=ASSISTANT,
                content=f'Answer {index}: ' + ('retrieval augments generation with context ' * 40),
            )
        )
    return messages


def _measure(fn, repeats: int) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--turns', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()

    try:
        from cat_agent._native import truncate_messages
    except ImportError:
        raise SystemExit('Rust extension unavailable; run `maturin develop`')
    _ = truncate_messages

    ensure_qwen_tokenizer()
    messages = _conversation(args.turns)
    input_tokens = sum(count_tokens(msg.content if isinstance(msg.content, str) else str(msg.content)) for msg in messages)

    mean_ms, median_ms = _measure(
        lambda: truncate_input_messages_roughly(messages, args.max_tokens),
        args.repeats,
    )
    truncated = truncate_input_messages_roughly(messages, args.max_tokens)
    output_tokens = sum(
        count_tokens(msg.content if isinstance(msg.content, str) else str(msg.content))
        for msg in truncated
    )

    print(f'Input messages: {len(messages)} ({args.turns} user turns)')
    print(f'Input tokens (rough): {input_tokens}')
    print(f'Max token budget: {args.max_tokens}')
    print(f'Output messages: {len(truncated)}')
    print(f'Output tokens (rough): {output_tokens}')
    print(
        f'Truncation time: mean={mean_ms:.2f} ms median={median_ms:.2f} ms '
        f'over {args.repeats} runs'
    )


if __name__ == '__main__':
    main()
