"""Direct example of the mandatory Rust-backed KeywordSearch index."""

from cat_agent._native import __version__ as native_version
from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.keyword_search import KeywordSearch
from cat_agent.utils.tokenization_qwen import count_tokens


def main() -> None:
    texts = [
        "Rust provides the persistent BM25 index used by Cat-Agent keyword retrieval.",
        "Vector search can complement keyword retrieval for semantic matching.",
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
    record = Record(url="demo", raw=chunks, title="Rust keyword-search demo")

    search = KeywordSearch({"rebuild_rag": False})
    ranked = search.sort_by_scores("Rust BM25 keyword retrieval", [record])

    print(f"cat_agent._native version: {native_version}")
    print(f"Persistent index: {search.index_path}")
    print("Ranked results:")
    for source, chunk_id, score in ranked:
        print(f"  score={score:.4f} source={source} chunk={chunk_id}")
        print(f"    {record.raw[chunk_id].content}")


if __name__ == "__main__":
    main()
