"""Benchmark native HNSW vector search (usearch + hash embeddings).

Run after installing the RAG dependencies:

    python benchmarks/benchmark_native_vector.py --chunks 2000 --queries 25

Build the native extension with ``maturin develop`` before running.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.embedding import HashEmbedder
from cat_agent.tools.search_tools.vector_search import VectorSearch
from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer


def _corpus(size: int) -> Record:
    chunks = []
    for index in range(size):
        text = (
            f"Document chunk {index} discusses machine learning retrieval "
            f"and vector embeddings category {index % 37}."
        )
        chunks.append(
            Chunk(
                content=text,
                metadata={'source': 'benchmark', 'chunk_id': index},
                token=count_tokens(text),
            )
        )
    return Record(url='benchmark', raw=chunks, title='Synthetic benchmark')


def _measure(fn, repeats: int) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.mean(samples), statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', type=int, default=2000)
    parser.add_argument('--queries', type=int, default=25)
    parser.add_argument('--dimensions', type=int, default=384)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    try:
        from cat_agent._native import VectorIndex
    except ImportError:
        raise SystemExit('Rust extension unavailable; run `maturin develop`')

    ensure_qwen_tokenizer()
    record = _corpus(args.chunks)
    texts = [chunk.content for chunk in record.raw]
    query = 'machine learning category 11'
    embedder = HashEmbedder(dimensions=args.dimensions)

    embed_started = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_ms = (time.perf_counter() - embed_started) * 1000

    build_started = time.perf_counter()
    index = VectorIndex(args.dimensions, 'cos')
    index.add(list(range(len(vectors))), vectors)
    build_ms = (time.perf_counter() - build_started) * 1000

    query_vector = embedder.embed([query])[0]
    search_mean, search_median = _measure(
        lambda: index.search(query_vector, args.top_k),
        args.queries,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = f'{tmpdir}/vector.usearch'
        save_started = time.perf_counter()
        index.save(index_path)
        save_ms = (time.perf_counter() - save_started) * 1000
        load_started = time.perf_counter()
        loaded = VectorIndex.load(index_path, args.dimensions, 'cos')
        load_ms = (time.perf_counter() - load_started) * 1000
        loaded_mean, loaded_median = _measure(
            lambda: loaded.search(query_vector, args.top_k),
            args.queries,
        )

    print(f'Corpus chunks: {args.chunks}')
    print(f'Embedding dimensions: {args.dimensions}')
    print(f'Hash embedding (all chunks): {embed_ms:.2f} ms')
    print(f'HNSW index build: {build_ms:.2f} ms')
    print(
        f'HNSW search (top-{args.top_k}): mean={search_mean:.2f} ms '
        f'median={search_median:.2f} ms over {args.queries} runs'
    )
    print(f'Index save: {save_ms:.2f} ms')
    print(f'Index load: {load_ms:.2f} ms')
    print(
        f'Loaded index search: mean={loaded_mean:.2f} ms '
        f'median={loaded_median:.2f} ms'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        search = VectorSearch({
            'rebuild_rag': False,
            'embedding_backend': 'hash',
            'vector_index_path': f'{tmpdir}/vector.usearch',
            'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
        })
        search.sort_by_scores(query, [record])
        tool_mean, tool_median = _measure(
            lambda: search.sort_by_scores(query, [record]),
            max(3, args.queries // 5),
        )
    print(
        f'VectorSearch.sort_by_scores (cached): mean={tool_mean:.2f} ms '
        f'median={tool_median:.2f} ms'
    )


if __name__ == '__main__':
    main()
