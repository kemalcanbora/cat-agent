"""Benchmark native document chunking."""

from __future__ import annotations

import argparse
import statistics
import time

from cat_agent.tools.doc_parser import DocParser
from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer


def _build_doc(pages: int, paragraphs_per_page: int) -> list:
    doc = []
    for page_num in range(1, pages + 1):
        content = []
        for index in range(paragraphs_per_page):
            text = (
                f"Page {page_num} paragraph {index}: retrieval systems chunk long documents "
                f"into token-budgeted segments for RAG pipelines. Category {index % 17}."
            )
            content.append({'text': text, 'token': count_tokens(text)})
        doc.append({'page_num': page_num, 'content': content})
    return doc


def _measure(fn, repeats: int) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=20)
    parser.add_argument('--paragraphs', type=int, default=10)
    parser.add_argument('--parser-page-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    ensure_qwen_tokenizer()
    doc = _build_doc(args.pages, args.paragraphs)
    tool = DocParser({'parser_page_size': args.parser_page_size})

    def run_chunking():
        tool.split_doc_to_chunk(
            doc,
            path='benchmark://chunking',
            title='Chunking benchmark',
            parser_page_size=args.parser_page_size,
        )

    mean_ms, median_ms = _measure(run_chunking, args.repeats)
    chunks = tool.split_doc_to_chunk(
        doc,
        path='benchmark://chunking',
        title='Chunking benchmark',
        parser_page_size=args.parser_page_size,
    )

    print(
        f"Native chunking: pages={args.pages} paragraphs/page={args.paragraphs} "
        f"parser_page_size={args.parser_page_size}"
    )
    print(f"Output chunks: {len(chunks)}")
    print(f"Chunking time: mean={mean_ms:.2f} ms median={median_ms:.2f} ms over {args.repeats} runs")


if __name__ == '__main__':
    main()
