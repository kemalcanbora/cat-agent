"""Demonstrate native document chunking, vector search, and message truncation."""

import tempfile

from cat_agent.llm.base import truncate_input_messages_roughly
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.tools.doc_parser import DocParser, Record
from cat_agent.tools.search_tools.vector_search import VectorSearch
from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer


def _sample_doc():
    pages = []
    for page_num in range(1, 4):
        paragraphs = []
        for index in range(5):
            text = (
                f"Page {page_num}, paragraph {index}: "
                f"native chunking packs paragraphs into token-budgeted chunks. "
                f"Topic category {index % 3}."
            )
            paragraphs.append({'text': text, 'token': count_tokens(text)})
        pages.append({'page_num': page_num, 'content': paragraphs})
    return pages


def main() -> None:
    ensure_qwen_tokenizer()
    doc = _sample_doc()

    parser = DocParser({'parser_page_size': 120})
    chunks = parser.split_doc_to_chunk(doc, path='demo://native-rag', title='Native RAG pipeline')
    print(f"Chunked {len(doc)} pages into {len(chunks)} chunks (parser_page_size=120)")
    for chunk in chunks[:3]:
        print(f"  chunk_id={chunk.metadata['chunk_id']} tokens={chunk.token}")
        print(f"    {chunk.content[:120]}...")

    record = Record(
        url='demo://native-rag',
        raw=chunks,
        title='Native RAG pipeline',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        search = VectorSearch({
            'embedding_backend': 'hash',
            'vector_index_path': f'{tmpdir}/vector.usearch',
            'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
        })
        ranked = search.sort_by_scores('native chunking token budget', [record])
    print("\nVector search top result:")
    top = ranked[0]
    print(f"  chunk={top[1]} score={top[2]:.4f}")
    print(f"  {record.raw[top[1]].content[:160]}...")

    messages = [
        Message(role=SYSTEM, content='You are helpful.'),
        Message(role=USER, content='Summarize the native RAG pipeline.'),
        Message(role=ASSISTANT, content='It chunks documents and searches them with HNSW.'),
        Message(role=USER, content='word ' * 2000),
    ]
    truncated = truncate_input_messages_roughly(messages, max_tokens=256)
    print(f"\nTruncated conversation from {len(messages)} to {len(truncated)} messages (max_tokens=256)")
    for msg in truncated:
        preview = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"  {msg.role}: {preview[:80]}...")


if __name__ == "__main__":
    main()
