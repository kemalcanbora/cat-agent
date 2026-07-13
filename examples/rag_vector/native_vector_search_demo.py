"""Direct example of the native HNSW vector index (usearch)."""

import tempfile

from cat_agent._native import __version__ as native_version
from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.vector_search import VectorSearch
from cat_agent.utils.tokenization_qwen import count_tokens


def main() -> None:
    texts = [
        "Rust provides the persistent HNSW index used by Cat-Agent vector retrieval.",
        "Keyword search complements vector retrieval for exact term matching.",
        "Python remains the public API for agents, tools, and document records.",
    ]
    chunks = [
        Chunk(
            content=text,
            metadata={"source": "demo", "chunk_id": index},
            token=count_tokens(text),
        )
        for index, text in enumerate(texts)
    ]
    record = Record(url="demo", raw=chunks, title="Native vector-search demo")

    with tempfile.TemporaryDirectory() as tmpdir:
        search = VectorSearch({
            'rebuild_rag': False,
            'embedding_backend': 'hash',
            'vector_index_path': f'{tmpdir}/vector.usearch',
            'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
        })
        ranked = search.sort_by_scores("HNSW vector retrieval", [record])

    print(f"cat_agent._native version: {native_version}")
    print(f"Embedding backend: hash (default)")
    print(f"Persistent index: {search.index_path}")
    print("Ranked results:")
    for source, chunk_id, score in ranked:
        print(f"  score={score:.4f} source={source} chunk={chunk_id}")
        print(f"    {record.raw[chunk_id].content}")


if __name__ == "__main__":
    main()
