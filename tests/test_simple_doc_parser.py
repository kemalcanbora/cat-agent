"""Tests for cat_agent.tools.simple_doc_parser."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.storage import KeyNotExistsError
from cat_agent.tools.simple_doc_parser import (
    PARAGRAPH_SPLIT_SYMBOL,
    PARSER_SUPPORTED_FILE_TYPES,
    clean_paragraph,
    df_to_md,
    get_plain_doc,
    SimpleDocParser,
    DocParserError,
    parse_txt,
    parse_word,
    parse_ppt,
    parse_excel,
    parse_csv,
    parse_tsv,
    parse_html_bs,
)


class TestSimpleDocParserHelpers:

    def test_get_plain_doc(self):
        doc = [
            {"page_num": 1, "content": [{"text": "Hello"}, {"table": "|a|b|"}]},
            {"page_num": 2, "content": [{"text": "World"}]},
        ]
        out = get_plain_doc(doc)
        assert "Hello" in out
        assert "|a|b|" in out
        assert "World" in out
        assert PARAGRAPH_SPLIT_SYMBOL in out

    def test_get_plain_doc_includes_image_key(self):
        doc = [{"page_num": 1, "content": [{"text": "x"}, {"image": "base64data"}]}]
        out = get_plain_doc(doc)
        assert "x" in out
        assert "base64data" in out

    def test_parser_supported_file_types(self):
        assert "pdf" in PARSER_SUPPORTED_FILE_TYPES
        assert "txt" in PARSER_SUPPORTED_FILE_TYPES
        assert "docx" in PARSER_SUPPORTED_FILE_TYPES
        assert "html" in PARSER_SUPPORTED_FILE_TYPES
        assert "csv" in PARSER_SUPPORTED_FILE_TYPES

    def test_clean_paragraph_removes_cid(self):
        out = clean_paragraph("Hello (cid:123) world")
        assert "(cid:123)" not in out

    def test_clean_paragraph_removes_hexadecimal(self):
        # rm_hexadecimal removes 21+ consecutive hex chars; use non-hex 'x','y' so they are not removed
        out = clean_paragraph("x" + "0" * 21 + "y")
        assert "x" in out
        assert "y" in out
        assert "0" * 21 not in out

    def test_clean_paragraph_collapses_placeholders(self):
        out = clean_paragraph("a\n\n\n\n\nb")
        assert out.count("\n") < 5


class TestDocParserError:

    def test_init_with_exception(self):
        exc = ValueError("bad")
        e = DocParserError(exception=exc)
        assert str(e) == "bad"
        assert e.exception is exc

    def test_init_with_code_and_message(self):
        e = DocParserError(code="500", message="Server error")
        assert "500" in str(e)
        assert "Server error" in str(e)


class TestParseTxt:

    def test_parse_txt_returns_one_page_with_paragraphs(self):
        with patch("cat_agent.tools.simple_doc_parser.read_text_from_file", return_value="Line one\nLine two"):
            doc = parse_txt("/fake/path.txt")
        assert len(doc) == 1
        assert doc[0]["page_num"] == 1
        assert len(doc[0]["content"]) == 2
        assert doc[0]["content"][0]["text"] == "Line one"
        assert doc[0]["content"][1]["text"] == "Line two"


class TestParseWord:

    def test_parse_word_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_word("/fake/path.docx", extract_image=True)

    def test_parse_word_with_mock_document(self):
        # Document is imported inside parse_word as "from docx import Document"
        fake_para = MagicMock()
        fake_para.text = "Paragraph text"
        cell_a, cell_b = MagicMock(), MagicMock()
        cell_a.text, cell_b.text = "A", "B"
        fake_table_row = MagicMock()
        fake_table_row.cells = [cell_a, cell_b]
        fake_table = MagicMock()
        fake_table.rows = [fake_table_row]
        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para]
        fake_doc.tables = [fake_table]
        # parse_word does "from docx import Document" inside the function
        with patch("docx.Document", return_value=fake_doc):
            doc = parse_word("/fake/path.docx")
        assert len(doc) == 1
        assert doc[0]["page_num"] == 1
        assert len(doc[0]["content"]) >= 1
        assert doc[0]["content"][0]["text"] == "Paragraph text"
        assert any("table" in c for c in doc[0]["content"]) or any("|" in str(c.get("table", "")) for c in doc[0]["content"])


class TestDfToMd:

    def test_df_to_md_basic(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = df_to_md(df)
        assert "a" in out
        assert "b" in out
        assert "1" in out and "2" in out


class TestParsePpt:

    def test_parse_ppt_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_ppt("/fake/path.pptx", extract_image=True)


class TestParseExcel:

    def test_parse_excel_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_excel("/fake/path.xlsx", extract_image=True)


class TestParseCsv:

    def test_parse_csv_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_csv("/fake/path.csv", extract_image=True)


class TestParseTsv:

    def test_parse_tsv_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_tsv("/fake/path.tsv", extract_image=True)


class TestParseHtmlBs:

    def test_parse_html_bs_extract_image_raises(self):
        with pytest.raises(ValueError, match="extracting images"):
            parse_html_bs("/fake/path.html", extract_image=True)

    def test_parse_html_bs_basic(self):
        pytest.importorskip("bs4")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><head><title>Hi</title></head><body><p>Hello</p></body></html>")
            path = f.name
        try:
            doc = parse_html_bs(path)
            assert len(doc) == 1
            assert doc[0]["page_num"] == 1
            assert "Hello" in str(doc[0]["content"])
            assert doc[0].get("title") == "Hi"
        finally:
            import os
            os.unlink(path)


class TestSimpleDocParser:

    @pytest.fixture
    def parser_path(self):
        return tempfile.mkdtemp()

    def test_init_and_schema(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        assert p.name == "simple_doc_parser"
        assert "url" in p.parameters["properties"]

    def test_init_structured_doc_and_extract_image(self, parser_path):
        p = SimpleDocParser({"path": parser_path, "structured_doc": True, "extract_image": False})
        assert p.structured_doc is True
        assert p.extract_image is False

    def test_call_requires_url(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        with pytest.raises(Exception):  # ValidationError or ValueError
            p.call("{}")

    def test_call_unsupported_type_fails(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        with patch("cat_agent.tools.simple_doc_parser.get_file_type", return_value="xyz"):
            with pytest.raises(Exception):  # NameError (unset parsed_file) or DocParserError
                p.call('{"url": "file.xyz"}')

    def test_call_cache_hit_returns_cached(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        with patch.object(p.db, "get", return_value='[{"page_num": 1, "content": [{"text": "cached"}]}]'):
            out = p.call('{"url": "http://example.com/doc.txt"}')
        assert "cached" in out

    def test_call_parse_txt_integration(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        with patch("cat_agent.tools.simple_doc_parser.get_file_type", return_value="txt"):
            with patch("cat_agent.tools.simple_doc_parser.read_text_from_file", return_value="Hello\nWorld"):
                with patch.object(p.db, "get", side_effect=KeyNotExistsError("key")):
                    with patch.object(p.db, "put"):
                        out = p.call('{"url": "/local/file.txt"}')
        assert "Hello" in out
        assert "World" in out

    def test_call_parse_docx_mocked(self, parser_path):
        p = SimpleDocParser({"path": parser_path})
        with patch("cat_agent.tools.simple_doc_parser.get_file_type", return_value="docx"):
            with patch("cat_agent.tools.simple_doc_parser.parse_word", return_value=[{"page_num": 1, "content": [{"text": "docx"}]}]):
                with patch.object(p.db, "get", side_effect=KeyNotExistsError("key")):
                    with patch.object(p.db, "put"):
                        out = p.call('{"url": "/local/file.docx"}')
        assert "docx" in out
