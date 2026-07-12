"""Micro-benchmark repeated BM25 searches and Qwen token accounting.

Run after installing the RAG dependencies:

    python benchmarks/benchmark_rag.py --chunks 1000 --queries 25

Build the native extension with ``maturin develop`` before running. If the
legacy ``rank_bm25`` package is installed, the script also reports it as a
comparison; production ``KeywordSearch`` uses only the Rust index.
"""

from __future__ import annotations

import argparse
import statistics
import time

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.keyword_search import split_text_into_keywords
from cat_agent.utils.tokenization_qwen import count_tokens, tokenizer


def _corpus(size: int) -> Record:
    chunks = []
    for index in range(size):
        text = (
            f"Document chunk {index} discusses machine learning retrieval "
            f"and persistent search index category {index % 37}."
        )
        chunks.append(
            Chunk(
                content=text,
                metadata={"source": "benchmark", "chunk_id": index},
                token=count_tokens(text),
            )
        )
    return Record(url="benchmark", raw=chunks, title="Synthetic benchmark")


def _measure(fn, repeats: int) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=25)
    args = parser.parse_args()

    record = _corpus(args.chunks)
    query = "machine learning category 11"
    tokenized = [split_text_into_keywords(chunk.content) for chunk in record.raw]
    query_tokens = split_text_into_keywords(query)
    print(f"Corpus chunks: {args.chunks}")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Legacy rank_bm25 comparison: unavailable")
    else:
        build_started = time.perf_counter()
        python_index = BM25Okapi(tokenized)
        python_build_ms = (time.perf_counter() - build_started) * 1000
        python_mean, python_median = _measure(
            lambda: python_index.get_scores(query_tokens), args.queries
        )
        print(f"Legacy Python index build: {python_build_ms:.2f} ms")
        print(
            f"Legacy Python repeated query: mean={python_mean:.2f} ms "
            f"median={python_median:.2f} ms"
        )

    try:
        from cat_agent._native import RagIndex
    except ImportError:
        raise SystemExit("Rust extension unavailable; run `maturin develop`")
    else:
        build_started = time.perf_counter()
        index = RagIndex(tokenized)
        rust_build_ms = (time.perf_counter() - build_started) * 1000
        rust_mean, rust_median = _measure(
            lambda: index.scores(query_tokens), args.queries
        )
        print(f"Rust index build: {rust_build_ms:.2f} ms")
        print(
            f"Rust repeated query: mean={rust_mean:.2f} ms "
            f"median={rust_median:.2f} ms"
        )

    sample = "Token accounting should stay on tiktoken's native encode path. " * 100
    count_mean, count_median = _measure(lambda: count_tokens(sample), args.queries)
    surface_mean, surface_median = _measure(
        lambda: len(tokenizer.tokenize(sample)), args.queries
    )
    print(
        f"Direct token count: mean={count_mean:.2f} ms median={count_median:.2f} ms"
    )
    print(
        f"Token surface conversion: mean={surface_mean:.2f} ms "
        f"median={surface_median:.2f} ms"
    )


if __name__ == "__main__":
    main()
