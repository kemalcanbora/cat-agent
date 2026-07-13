"""Parity tests for the optional PyO3 RAG index."""

import pytest


native = pytest.importorskip("cat_agent._native")


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode())
        data.extend(body)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(data)


def test_rust_scores_match_rank_bm25():
    rank_bm25 = pytest.importorskip("rank_bm25")
    corpus = [
        ["machine", "learning", "retrieval"],
        ["python", "programming"],
        ["machine", "vision"],
    ]
    query = ["machine", "retrieval"]

    expected = rank_bm25.BM25Okapi(corpus).get_scores(query)
    actual = native.RagIndex(corpus).scores(query)

    assert actual == pytest.approx(expected.tolist(), rel=1e-12, abs=1e-12)


def test_rust_index_round_trip(tmp_path):
    corpus = [["one", "two"], ["two", "three"], ["four"]]
    query = ["two"]
    path = tmp_path / "rag-index.json"
    index = native.RagIndex(corpus)

    index.save(str(path))
    restored = native.RagIndex.load(str(path))

    assert len(restored) == len(corpus)
    assert restored.scores(query) == pytest.approx(index.scores(query))


def test_rust_tokenizer_filters_stop_words():
    assert "the" not in native.split_text_into_keywords("the quick brown")
    assert "quick" in native.split_text_into_keywords("the quick brown")


def test_rust_qwen_token_count_matches_python_baseline():
    from pathlib import Path

    from cat_agent.utils.tokenization_qwen import tokenizer

    vocab = str(Path(__file__).resolve().parents[1] / "cat_agent/utils/qwen.tiktoken")
    native.init_qwen_tokenizer(vocab)
    sample = "Token accounting should stay consistent across Rust and Python."
    assert native.count_qwen_tokens(sample) == len(tokenizer.encode(sample))


def test_rust_qwen_encode_decode_roundtrip():
    from pathlib import Path

    vocab = str(Path(__file__).resolve().parents[1] / "cat_agent/utils/qwen.tiktoken")
    native.init_qwen_tokenizer(vocab)
    sample = "Chunking and truncation share the same native tokenizer."
    token_ids = native.encode_qwen_tokens(sample)
    restored = native.decode_qwen_tokens(token_ids)
    assert native.count_qwen_tokens(restored) == len(token_ids)


def test_rust_doc_chunker_splits_pages(tmp_path):
    from pathlib import Path

    from cat_agent.utils.tokenization_qwen import count_tokens

    vocab = str(Path(__file__).resolve().parents[1] / "cat_agent/utils/qwen.tiktoken")
    native.init_qwen_tokenizer(vocab)
    text_a = "First page paragraph."
    text_b = "Second page paragraph."
    doc = [
        {"page_num": 1, "content": [{"text": text_a, "token": count_tokens(text_a)}]},
        {"page_num": 2, "content": [{"text": text_b, "token": count_tokens(text_b)}]},
    ]
    chunks = native.split_doc_to_chunks(doc, "demo", "T", 32, "\n")
    assert len(chunks) >= 1
    joined = "\n".join(chunk["content"] for chunk in chunks)
    assert "First page" in joined
    assert "Second page" in joined


def test_rust_pdf_extracts_page_text(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(_minimal_pdf("Hello native PDF"))

    pages = native.parse_pdf_text(str(path))

    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "Hello native PDF" in pages[0][1]
