"""Tests for cat_agent.utils.utils."""

from unittest.mock import patch
from cat_agent.llm.schema import SYSTEM, USER, ContentItem, Message
from cat_agent.utils import utils as utils_module
from cat_agent.utils.utils import (
    extract_code,
    extract_files_from_messages,
    extract_images_from_messages,
    extract_markdown_urls,
    extract_text_from_message,
    extract_urls,
    get_basename_from_url,
    get_last_usr_msg_idx,
    hash_sha256,
    has_chinese_chars,
    has_chinese_messages,
    is_http_url,
    is_image,
    json_loads,
    merge_generate_cfgs,
    rm_default_system,
)


class TestHashSha256:

    def test_deterministic(self):
        assert hash_sha256("hello") == hash_sha256("hello")
        assert hash_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_different_inputs_different_hashes(self):
        assert hash_sha256("a") != hash_sha256("b")


class TestHasChineseChars:

    def test_no_chinese(self):
        assert has_chinese_chars("hello world") is False
        assert has_chinese_chars("123") is False

    def test_has_chinese(self):
        assert has_chinese_chars("你好") is True
        assert has_chinese_chars("hello 世界") is True

    def test_converts_to_string(self):
        assert has_chinese_chars(123) is False
        assert has_chinese_chars(["你好"]) is True  # f'{data}' -> "['你好']" contains 你


class TestHasChineseMessages:

    def test_empty(self):
        assert has_chinese_messages([]) is False

    def test_no_chinese(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert has_chinese_messages(msgs) is False

    def test_chinese_in_user(self):
        msgs = [{"role": "user", "content": "你好"}]
        assert has_chinese_messages(msgs) is True

    def test_chinese_in_system(self):
        msgs = [{"role": "system", "content": "你是助手"}]
        assert has_chinese_messages(msgs) is True

    def test_assistant_ignored_by_default(self):
        msgs = [{"role": "assistant", "content": "你好"}]
        assert has_chinese_messages(msgs) is False


class TestGetBasenameFromUrl:

    def test_http_url_with_path(self):
        assert get_basename_from_url("https://example.com/path/to/file.pdf") == "file.pdf"

    def test_http_url_trailing_slash(self):
        assert get_basename_from_url("https://github.com/") in ("", "github.com")

    def test_local_unix_path(self):
        assert get_basename_from_url("/mnt/a/b/c") == "c"

    def test_url_decoded(self):
        assert " " in get_basename_from_url("https://x.com/foo%20bar") or get_basename_from_url("https://x.com/foo%20bar") == "foo bar"


class TestIsHttpUrl:

    def test_https(self):
        assert is_http_url("https://example.com") is True

    def test_http(self):
        assert is_http_url("http://example.com") is True

    def test_file_path(self):
        assert is_http_url("/local/path") is False
        assert is_http_url("file:///local") is False


class TestIsImage:

    def test_image_extensions(self):
        assert is_image("https://x.com/photo.jpg") is True
        assert is_image("https://x.com/photo.jpeg") is True
        assert is_image("https://x.com/photo.png") is True
        assert is_image("https://x.com/photo.webp") is True

    def test_non_image(self):
        assert is_image("https://x.com/doc.pdf") is False


class TestExtractUrls:

    def test_finds_http_urls(self):
        text = "See https://a.com and http://b.com/path"
        urls = extract_urls(text)
        assert "https://a.com" in urls
        assert "http://b.com/path" in urls

    def test_empty(self):
        assert extract_urls("no url here") == []


class TestExtractMarkdownUrls:

    def test_link_syntax(self):
        text = "Click [here](https://example.com) or ![img](https://img.com/x.png)"
        urls = extract_markdown_urls(text)
        assert "https://example.com" in urls
        assert "https://img.com/x.png" in urls

    def test_empty(self):
        assert extract_markdown_urls("plain text") == []


class TestExtractCode:

    def test_triple_backtick_block(self):
        text = "```python\nx = 1\n```"
        assert extract_code(text).strip() == "x = 1"

    def test_no_block_returns_original(self):
        text = "just text"
        assert extract_code(text) == "just text"


class TestJsonLoads:

    def test_plain_json(self):
        assert json_loads('{"a": 1}') == {"a": 1}

    def test_json5_trailing_comma(self):
        out = json_loads('{"a": 1,}')
        assert out == {"a": 1}

    def test_strips_whitespace(self):
        assert json_loads('  {"b": 2}  ') == {"b": 2}


class TestMergeGenerateCfgs:

    def test_empty_base(self):
        assert merge_generate_cfgs(None, {"a": 1}) == {"a": 1}

    def test_empty_new(self):
        assert merge_generate_cfgs({"a": 1}, None) == {"a": 1}

    def test_stop_merged_not_duplicated(self):
        base = {"stop": ["a"]}
        new = {"stop": ["a", "b"]}
        out = merge_generate_cfgs(base, new)
        assert out["stop"] == ["a", "b"]

    def test_other_keys_overwritten(self):
        out = merge_generate_cfgs({"a": 1, "b": 2}, {"b": 3})
        assert out["a"] == 1
        assert out["b"] == 3


class TestExtractTextFromMessage:

    def test_str_content(self):
        msg = Message(role=USER, content="Hello")
        assert extract_text_from_message(msg, add_upload_info=False) == "Hello"

    def test_list_content(self):
        msg = Message(role=USER, content=[ContentItem(text="Hi")])
        assert extract_text_from_message(msg, add_upload_info=False) == "Hi"


class TestExtractFilesFromMessages:

    def test_empty(self):
        assert extract_files_from_messages([], include_images=False) == []

    def test_file_item(self):
        msg = Message(role=USER, content=[ContentItem(file="/path/to/doc.pdf")])
        assert extract_files_from_messages([msg], include_images=False) == ["/path/to/doc.pdf"]

    def test_include_images(self):
        msg = Message(role=USER, content=[ContentItem(image="https://img.png")])
        assert extract_files_from_messages([msg], include_images=True) == ["https://img.png"]
        assert extract_files_from_messages([msg], include_images=False) == []


class TestExtractImagesFromMessages:

    def test_image_item(self):
        msg = Message(role=USER, content=[ContentItem(image="https://x.png")])
        assert extract_images_from_messages([msg]) == ["https://x.png"]

    def test_no_images(self):
        msg = Message(role=USER, content=[ContentItem(file="/f.pdf")])
        assert extract_images_from_messages([msg]) == []


class TestGetLastUsrMsgIdx:

    def test_last_is_user(self):
        msgs = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
        assert get_last_usr_msg_idx(msgs) == 1

    def test_only_user(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert get_last_usr_msg_idx(msgs) == 0


class TestRmDefaultSystem:

    def test_first_is_default_system_str_removed(self):
        from cat_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE

        msgs = [
            Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE),
            Message(role=USER, content="Hi"),
        ]
        out = rm_default_system(msgs)
        assert len(out) == 1
        assert out[0].role == USER

    def test_first_not_default_unchanged(self):
        msgs = [
            Message(role=SYSTEM, content="Custom system"),
            Message(role=USER, content="Hi"),
        ]
        out = rm_default_system(msgs)
        assert len(out) == 2

    def test_empty_or_single_unchanged(self):
        assert rm_default_system([]) == []
        one = [Message(role=USER, content="x")]
        assert rm_default_system(one) == one


class TestGetFileType:

    def test_known_extension_pdf(self):
        assert utils_module.get_file_type("https://x.com/doc.pdf") == "pdf"
        assert utils_module.get_file_type("/local/doc.pdf") == "pdf"

    def test_known_extension_docx(self):
        assert utils_module.get_file_type("file.docx") == "docx"

    def test_txt_via_mock_read(self):
        with patch("cat_agent.utils.utils.read_text_from_file", return_value="<p>html</p>"):
            out = utils_module.get_file_type("/nonexistent/doc.txt")
            assert out == "html"
        with patch("cat_agent.utils.utils.read_text_from_file", return_value="plain text"):
            out = utils_module.get_file_type("/nonexistent/doc.txt")
            assert out == "txt"
