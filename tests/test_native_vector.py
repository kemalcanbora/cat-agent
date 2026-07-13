"""Native vector index, chunking, and truncation tests."""

import tempfile

import pytest

from cat_agent.llm.base import truncate_input_messages_roughly
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, Message
from cat_agent.tools.doc_parser import DocParser
from cat_agent.tools.search_tools.vector_search import VectorSearch
from cat_agent.utils.tokenization_qwen import (
    count_tokens,
    ensure_qwen_tokenizer,
    tokenizer,
    truncate_tokens,
)


def _native():
    pytest.importorskip('cat_agent._native')
    from cat_agent import _native as native

    return native


class TestNativeVectorIndex:

    def test_add_search_save_load_roundtrip(self, tmp_path):
        native = _native()
        index = native.VectorIndex(4, 'cos')
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ]
        index.add([0, 1, 2], vectors)
        matches = index.search([1.0, 0.0, 0.0, 0.0], 2)
        assert matches[0][0] in (0, 2)

        path = tmp_path / 'vector.usearch'
        index.save(str(path))
        loaded = native.VectorIndex.load(str(path), 4, 'cos')
        loaded_matches = loaded.search([0.0, 1.0, 0.0, 0.0], 1)
        assert loaded_matches[0][0] == 1

    def test_dimension_mismatch_raises(self):
        native = _native()
        index = native.VectorIndex(3, 'cos')
        with pytest.raises(Exception):
            index.add([0], [[1.0, 0.0]])


class TestNativeDocChunker:

    def test_split_doc_to_chunks_single_page(self):
        native = _native()
        ensure_qwen_tokenizer()
        doc = [{'page_num': 1, 'content': [{'text': 'Short paragraph.', 'token': count_tokens('Short paragraph.')}]}]
        chunks = native.split_doc_to_chunks(doc, 'file:///x.pdf', 'T', 1000, '\n')
        assert len(chunks) == 1
        assert 'Short paragraph' in chunks[0]['content']

    def test_split_doc_to_chunks_multiple_pages(self):
        native = _native()
        ensure_qwen_tokenizer()
        doc = [
            {'page_num': 1, 'content': [{'text': 'Page one text.', 'token': count_tokens('Page one text.')}]},
            {'page_num': 2, 'content': [{'text': 'Page two text.', 'token': count_tokens('Page two text.')}]},
        ]
        chunks = native.split_doc_to_chunks(doc, 'doc', 'T', 50, '\n')
        assert len(chunks) >= 1
        joined = '\n'.join(chunk['content'] for chunk in chunks)
        assert 'Page one' in joined
        assert 'Page two' in joined

    def test_split_doc_to_chunks_splits_long_paragraph(self):
        native = _native()
        ensure_qwen_tokenizer()
        long_text = 'Sentence one. ' * 80
        doc = [{'page_num': 1, 'content': [{'text': long_text, 'token': count_tokens(long_text)}]}]
        chunks = native.split_doc_to_chunks(doc, 'doc', 'T', 64, '\n')
        assert len(chunks) >= 2


class TestDocParserNativeIntegration:

    def test_split_doc_to_chunk_uses_native_path(self, tmp_path):
        _native()
        ensure_qwen_tokenizer()
        doc = [
            {'page_num': 1, 'content': [{'text': 'Alpha beta gamma.', 'token': count_tokens('Alpha beta gamma.')}]},
            {'page_num': 2, 'content': [{'text': 'Delta epsilon zeta.', 'token': count_tokens('Delta epsilon zeta.')}]},
        ]
        parser = DocParser({'path': str(tmp_path), 'parser_page_size': 40})
        chunks = parser.split_doc_to_chunk(doc, path='demo', title='T', parser_page_size=40)
        assert len(chunks) >= 1
        assert chunks[0].metadata['chunk_id'] == 0


class TestNativeTruncation:

    def test_truncate_messages_keeps_system_and_user(self):
        native = _native()
        ensure_qwen_tokenizer()
        messages = [
            {'role': 'system', 'text': 'You are helpful.'},
            {'role': 'user', 'text': 'Hello'},
            {'role': 'assistant', 'text': 'Hi there'},
        ]
        out = native.truncate_messages(messages, 1_000_000)
        assert len(out) == 3
        assert out[0]['role'] == 'system'

    def test_truncate_messages_reduces_long_history(self):
        native = _native()
        ensure_qwen_tokenizer()
        messages = [
            {'role': 'system', 'text': 'System prompt.'},
            {'role': 'user', 'text': 'short'},
            {'role': 'assistant', 'text': 'reply'},
            {'role': 'user', 'text': 'word ' * 3000},
        ]
        out = native.truncate_messages(messages, 128)
        total = sum(count_tokens(item['text']) for item in out)
        assert total <= 128

    def test_truncate_input_messages_preserves_function_name(self):
        _native()
        ensure_qwen_tokenizer()
        messages = [
            Message(role=USER, content='hi'),
            Message(role=ASSISTANT, content='', function_call=None),
            Message(
                role=FUNCTION,
                content=[ContentItem(text='tool output ' * 500)],
                name='my_tool',
            ),
        ]
        result = truncate_input_messages_roughly(messages, max_tokens=64)
        fn_msgs = [msg for msg in result if msg.role == FUNCTION]
        if fn_msgs:
            assert fn_msgs[0].name == 'my_tool'


class TestQwenTokenizerExtensions:

    def test_encode_decode_roundtrip(self):
        native = _native()
        ensure_qwen_tokenizer()
        sample = 'Native tokenizer encode/decode path.'
        token_ids = native.encode_qwen_tokens(sample)
        restored = native.decode_qwen_tokens(token_ids)
        assert count_tokens(restored) == len(token_ids)

    def test_batch_count_matches_individual_counts(self):
        native = _native()
        ensure_qwen_tokenizer()
        texts = ['hello world', 'native chunking', 'vector search']
        batch = native.batch_count_qwen_tokens(texts)
        assert batch == [count_tokens(text) for text in texts]

    def test_truncate_tokens_respects_budget(self):
        native = _native()
        ensure_qwen_tokenizer()
        sample = 'hello world ' * 20
        truncated = truncate_tokens(sample, 10, keep_both_sides=True)
        assert count_tokens(truncated) <= 10

    def test_truncate_tokens_matches_python_tiktoken_baseline(self):
        _native()
        ensure_qwen_tokenizer()
        sample = 'Token budget checks should stay aligned. ' * 15
        rust_truncated = truncate_tokens(sample, 24, keep_both_sides=False)
        python_truncated = tokenizer.truncate(sample, max_token=24, keep_both_sides=False)
        assert count_tokens(rust_truncated) <= 24
        assert count_tokens(python_truncated) <= 24


class TestVectorSearchIntegration:

    def test_end_to_end_ranking_with_hash_embeddings(self):
        _native()
        from cat_agent.tools.doc_parser import Chunk, Record

        chunk_a = Chunk(content='rust hnsw vector index', metadata={'source': 'u', 'chunk_id': 0}, token=5)
        chunk_b = Chunk(content='unrelated cooking recipe', metadata={'source': 'u', 'chunk_id': 1}, token=5)
        record = Record(url='u', raw=[chunk_a, chunk_b], title='T')
        with tempfile.TemporaryDirectory() as tmpdir:
            search = VectorSearch({
                'vector_index_path': f'{tmpdir}/vector.usearch',
                'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
            })
            ranked = search.sort_by_scores('rust vector index', [record])
        scores = {chunk_id: score for _, chunk_id, score in ranked}
        assert scores[0] > scores[1]
