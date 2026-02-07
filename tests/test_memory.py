"""Tests for cat_agent.memory.memory.Memory"""

import json
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, USER, ContentItem, Message
from cat_agent.memory.memory import Memory
from cat_agent.settings import (
    DEFAULT_MAX_REF_TOKEN,
    DEFAULT_PARSER_PAGE_SIZE,
    DEFAULT_RAG_KEYGEN_STRATEGY,
    DEFAULT_RAG_SEARCHERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm():
    """Return a lightweight mock that satisfies BaseChatModel checks."""
    llm = MagicMock()
    llm.chat = MagicMock(return_value=iter([]))
    return llm


def _msg(role: str, content, **kwargs):
    return Message(role=role, content=content, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm():
    return _make_mock_llm()


@pytest.fixture()
def memory_defaults(mock_llm):
    """Memory with all defaults and a mock LLM."""
    with patch("cat_agent.memory.memory.Agent.__init__", return_value=None):
        mem = Memory(llm=mock_llm)
        # Agent.__init__ was mocked out, so manually set what it would have set
        mem.llm = mock_llm
        mem.function_map = {}
    return mem


# ---------------------------------------------------------------------------
# Tests: __init__ / configuration
# ---------------------------------------------------------------------------

class TestMemoryInit:
    """Verify that __init__ correctly processes rag_cfg and defaults."""

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_defaults_with_llm(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)

        assert mem.max_ref_token == DEFAULT_MAX_REF_TOKEN
        assert mem.parser_page_size == DEFAULT_PARSER_PAGE_SIZE
        assert mem.rag_searchers == DEFAULT_RAG_SEARCHERS
        assert mem.rag_keygen_strategy == DEFAULT_RAG_KEYGEN_STRATEGY
        assert mem.rebuild_rag is None
        assert mem.system_files == []

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_defaults_without_llm_sets_keygen_none(self, _agent_init):
        """When no LLM is provided, keygen strategy must fall back to 'none'."""
        mem = Memory(llm=None)

        assert mem.rag_keygen_strategy == "none"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_custom_rag_cfg(self, _agent_init, mock_llm):
        cfg = {
            "max_ref_token": 8000,
            "parser_page_size": 1000,
            "rag_searchers": ["keyword_search"],
            "rag_keygen_strategy": "GenKeyword",
            "rebuild_rag": True,
        }
        mem = Memory(llm=mock_llm, rag_cfg=cfg)

        assert mem.max_ref_token == 8000
        assert mem.parser_page_size == 1000
        assert mem.rag_searchers == ["keyword_search"]
        assert mem.rag_keygen_strategy == "GenKeyword"
        assert mem.rebuild_rag is True

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_system_files_stored(self, _agent_init, mock_llm):
        files = ["/path/to/doc.pdf", "/path/to/report.docx"]
        mem = Memory(llm=mock_llm, files=files)

        assert mem.system_files == files

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_system_files_default_empty(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)

        assert mem.system_files == []

    # -- LEANN toggle --

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_enable_leann_adds_searcher(self, _agent_init, mock_llm):
        cfg = {
            "rag_searchers": ["keyword_search", "front_page_search"],
            "enable_leann": True,
        }
        mem = Memory(llm=mock_llm, rag_cfg=cfg)

        assert "leann_search" in mem.rag_searchers

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_enable_leann_does_not_duplicate(self, _agent_init, mock_llm):
        cfg = {
            "rag_searchers": ["keyword_search", "leann_search"],
            "enable_leann": True,
        }
        mem = Memory(llm=mock_llm, rag_cfg=cfg)

        assert mem.rag_searchers.count("leann_search") == 1

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_disable_leann_removes_searcher(self, _agent_init, mock_llm):
        cfg = {
            "rag_searchers": ["keyword_search", "leann_search", "front_page_search"],
            "enable_leann": False,
        }
        mem = Memory(llm=mock_llm, rag_cfg=cfg)

        assert "leann_search" not in mem.rag_searchers
        assert "keyword_search" in mem.rag_searchers
        assert "front_page_search" in mem.rag_searchers

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_leann_none_leaves_searchers_unchanged(self, _agent_init, mock_llm):
        searchers = ["keyword_search", "front_page_search"]
        cfg = {"rag_searchers": searchers}
        mem = Memory(llm=mock_llm, rag_cfg=cfg)

        assert mem.rag_searchers == searchers

    # -- Agent.__init__ call --

    def test_agent_init_called_with_retrieval_and_doc_parser(self, mock_llm):
        """Verify Agent.__init__ receives the retrieval + doc_parser tools."""
        with patch("cat_agent.memory.memory.Agent.__init__", return_value=None) as agent_init:
            Memory(llm=mock_llm)

            agent_init.assert_called_once()
            call_kwargs = agent_init.call_args
            fn_list = call_kwargs.kwargs.get("function_list") or call_kwargs[1].get("function_list")
            if fn_list is None:
                # positional
                fn_list = call_kwargs[0][0] if call_kwargs[0] else None

            tool_names = [t["name"] for t in fn_list if isinstance(t, dict)]
            assert "retrieval" in tool_names
            assert "doc_parser" in tool_names


# ---------------------------------------------------------------------------
# Tests: get_rag_files
# ---------------------------------------------------------------------------

class TestGetRagFiles:

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_empty_messages(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        result = mem.get_rag_files([])

        assert result == []

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_supported_files_from_messages(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        messages = [
            _msg(USER, [ContentItem(file="/docs/readme.pdf")]),
            _msg(USER, [ContentItem(file="/docs/report.docx")]),
        ]

        result = mem.get_rag_files(messages)

        assert "/docs/readme.pdf" in result
        assert "/docs/report.docx" in result

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_unsupported_files_filtered_out(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        messages = [
            _msg(USER, [ContentItem(file="/docs/photo.jpg")]),
        ]

        result = mem.get_rag_files(messages)

        assert result == []

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_system_files_included(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm, files=["/system/manual.pdf"])

        result = mem.get_rag_files([])

        assert "/system/manual.pdf" in result

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_deduplication(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm, files=["/docs/readme.pdf"])

        messages = [
            _msg(USER, [ContentItem(file="/docs/readme.pdf")]),
        ]

        result = mem.get_rag_files(messages)

        assert result.count("/docs/readme.pdf") == 1

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_mixed_supported_and_unsupported(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        messages = [
            _msg(USER, [ContentItem(file="/docs/report.pdf")]),
            _msg(USER, [ContentItem(file="/docs/image.png")]),
            _msg(USER, [ContentItem(file="/docs/data.csv")]),
        ]

        result = mem.get_rag_files(messages)

        assert "/docs/report.pdf" in result
        assert "/docs/data.csv" in result
        assert "/docs/image.png" not in result

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_all_supported_types(self, _agent_init, mock_llm):
        """Ensure each supported extension is accepted."""
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        supported = ["pdf", "docx", "pptx", "txt", "html", "csv", "tsv", "xlsx", "xls"]
        messages = [
            _msg(USER, [ContentItem(file=f"/docs/file.{ext}")])
            for ext in supported
        ]

        # get_file_type reads local files for txt/html detection, so mock it
        def fake_get_file_type(path):
            return path.rsplit(".", 1)[-1].lower()

        with patch("cat_agent.memory.memory.get_file_type", side_effect=fake_get_file_type):
            result = mem.get_rag_files(messages)

        for ext in supported:
            assert f"/docs/file.{ext}" in result

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_text_messages_ignored(self, _agent_init, mock_llm):
        """Plain text messages (no file attachments) produce no rag files."""
        mem = Memory(llm=mock_llm)
        mem.system_files = []

        messages = [_msg(USER, "Hello, how are you?")]

        result = mem.get_rag_files(messages)

        assert result == []


# ---------------------------------------------------------------------------
# Tests: _run
# ---------------------------------------------------------------------------

class TestMemoryRun:

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_no_files_yields_empty_memory_message(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = []
        mem.function_map = {}
        mem.rag_keygen_strategy = "none"

        messages = [_msg(USER, "What is AI?")]
        results = list(mem._run(messages))

        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].role == ASSISTANT
        assert results[0][0].content == ""
        assert results[0][0].name == "memory"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_with_files_calls_retrieval(self, _agent_init, mock_llm):
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "none"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "Retrieved content about AI from manual."
        mem.function_map = {"retrieval": mock_retrieval}

        messages = [_msg(USER, "What is AI?")]
        results = list(mem._run(messages))

        mock_retrieval.call.assert_called_once()
        call_args = mock_retrieval.call.call_args[0][0]
        assert call_args["query"] == "What is AI?"
        assert "/docs/manual.pdf" in call_args["files"]

        assert len(results) == 1
        assert results[0][0].role == ASSISTANT
        assert results[0][0].content == "Retrieved content about AI from manual."
        assert results[0][0].name == "memory"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_retrieval_result_dict_serialized_to_json(self, _agent_init, mock_llm):
        """When retrieval returns a non-string, it should be JSON-serialized."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/data.csv"]
        mem.rag_keygen_strategy = "none"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = {"key": "value", "items": [1, 2, 3]}
        mem.function_map = {"retrieval": mock_retrieval}

        messages = [_msg(USER, "Show me the data")]
        results = list(mem._run(messages))

        content = results[0][0].content
        parsed = json.loads(content)
        assert parsed["key"] == "value"
        assert parsed["items"] == [1, 2, 3]

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_no_user_message_sends_empty_query(self, _agent_init, mock_llm):
        """If the last message is not from the user, query should be empty."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "none"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "Some content"
        mem.function_map = {"retrieval": mock_retrieval}

        messages = [_msg(ASSISTANT, "Here is my reply")]
        list(mem._run(messages))

        call_args = mock_retrieval.call.call_args[0][0]
        assert call_args["query"] == ""

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_strategy_invoked_when_not_none(self, _agent_init, mock_llm):
        """When keygen_strategy is not 'none', keyword generation should run."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "Retrieved via keywords"
        mem.function_map = {"retrieval": mock_retrieval}

        # Mock the keygen module import
        mock_keygen_instance = MagicMock()
        keyword_response = json.dumps({"keywords": ["AI", "machine learning"], "text": "What is AI?"})
        mock_keygen_instance.run.return_value = iter([
            [_msg(ASSISTANT, keyword_response)]
        ])

        mock_keygen_cls = MagicMock(return_value=mock_keygen_instance)
        mock_module = MagicMock()
        mock_module.GenKeyword = mock_keygen_cls

        with patch("cat_agent.memory.memory.import_module", return_value=mock_module):
            messages = [_msg(USER, "What is AI?")]
            results = list(mem._run(messages))

        # Keygen class should have been instantiated and run
        mock_keygen_cls.assert_called_once_with(llm=mock_llm)
        mock_keygen_instance.run.assert_called_once()

        # Retrieval should still be called
        mock_retrieval.call.assert_called_once()
        assert results[0][0].content == "Retrieved via keywords"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_json_code_block_stripped(self, _agent_init, mock_llm):
        """Keygen output wrapped in ```json ... ``` should be unwrapped."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "content"
        mem.function_map = {"retrieval": mock_retrieval}

        inner_json = json.dumps({"keywords": ["test"], "text": "query"})
        keyword_output = f"```json{inner_json}```"

        mock_keygen_instance = MagicMock()
        mock_keygen_instance.run.return_value = iter([
            [_msg(ASSISTANT, keyword_output)]
        ])

        mock_keygen_cls = MagicMock(return_value=mock_keygen_instance)
        mock_module = MagicMock()
        mock_module.GenKeyword = mock_keygen_cls

        with patch("cat_agent.memory.memory.import_module", return_value=mock_module):
            messages = [_msg(USER, "query")]
            list(mem._run(messages))

        # The query sent to retrieval should be the parsed JSON (with 'text' key)
        call_args = mock_retrieval.call.call_args[0][0]
        parsed_query = json.loads(call_args["query"])
        assert "keywords" in parsed_query
        assert parsed_query["text"] == "query"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_fallback_on_invalid_json(self, _agent_init, mock_llm):
        """When keygen produces invalid JSON, the original query is preserved."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "content"
        mem.function_map = {"retrieval": mock_retrieval}

        mock_keygen_instance = MagicMock()
        mock_keygen_instance.run.return_value = iter([
            [_msg(ASSISTANT, "not valid json at all")]
        ])

        mock_keygen_cls = MagicMock(return_value=mock_keygen_instance)
        mock_module = MagicMock()
        mock_module.GenKeyword = mock_keygen_cls

        with patch("cat_agent.memory.memory.import_module", return_value=mock_module):
            messages = [_msg(USER, "What is AI?")]
            list(mem._run(messages))

        call_args = mock_retrieval.call.call_args[0][0]
        assert call_args["query"] == "What is AI?"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_adds_text_key_when_missing(self, _agent_init, mock_llm):
        """When keygen JSON lacks 'text', the original query is injected."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "content"
        mem.function_map = {"retrieval": mock_retrieval}

        keyword_json = json.dumps({"keywords": ["AI"]})  # no 'text' key

        mock_keygen_instance = MagicMock()
        mock_keygen_instance.run.return_value = iter([
            [_msg(ASSISTANT, keyword_json)]
        ])

        mock_keygen_cls = MagicMock(return_value=mock_keygen_instance)
        mock_module = MagicMock()
        mock_module.GenKeyword = mock_keygen_cls

        with patch("cat_agent.memory.memory.import_module", return_value=mock_module):
            messages = [_msg(USER, "What is AI?")]
            list(mem._run(messages))

        call_args = mock_retrieval.call.call_args[0][0]
        parsed_query = json.loads(call_args["query"])
        assert parsed_query["text"] == "What is AI?"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_empty_response_uses_original_query(self, _agent_init, mock_llm):
        """When keygen returns no responses, fall back to the original query."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "content"
        mem.function_map = {"retrieval": mock_retrieval}

        mock_keygen_instance = MagicMock()
        # Empty iterator – no responses
        mock_keygen_instance.run.return_value = iter([])

        mock_keygen_cls = MagicMock(return_value=mock_keygen_instance)
        mock_module = MagicMock()
        mock_module.GenKeyword = mock_keygen_cls

        with patch("cat_agent.memory.memory.import_module", return_value=mock_module):
            messages = [_msg(USER, "What is AI?")]
            list(mem._run(messages))

        call_args = mock_retrieval.call.call_args[0][0]
        # With empty keygen response, keyword is '' and json parse fails,
        # so query stays the original
        assert call_args["query"] == "What is AI?"

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_files_from_messages_with_content_items(self, _agent_init, mock_llm):
        """Files embedded as ContentItem in messages are picked up for RAG."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = []
        mem.rag_keygen_strategy = "none"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "file content"
        mem.function_map = {"retrieval": mock_retrieval}

        messages = [
            _msg(USER, [
                ContentItem(file="/docs/guide.pdf"),
                ContentItem(text="Tell me about this document"),
            ]),
        ]

        list(mem._run(messages))

        call_args = mock_retrieval.call.call_args[0][0]
        assert "/docs/guide.pdf" in call_args["files"]

    @patch("cat_agent.memory.memory.Agent.__init__", return_value=None)
    def test_keygen_not_called_with_empty_query(self, _agent_init, mock_llm):
        """When there is no user query, keygen should be skipped even if strategy is set."""
        mem = Memory(llm=mock_llm)
        mem.llm = mock_llm
        mem.system_files = ["/docs/manual.pdf"]
        mem.rag_keygen_strategy = "GenKeyword"

        mock_retrieval = MagicMock()
        mock_retrieval.call.return_value = "content"
        mem.function_map = {"retrieval": mock_retrieval}

        # Last message is from assistant, so query will be empty
        messages = [_msg(ASSISTANT, "I can help with that")]

        with patch("cat_agent.memory.memory.import_module") as mock_import:
            list(mem._run(messages))
            # import_module should NOT be called because query is empty
            mock_import.assert_not_called()
