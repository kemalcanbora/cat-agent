"""Tests for cat_agent.llm.llama_cpp_vision."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, USER, ContentItem, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create a LlamaCppVision instance with __init__ bypassed."""
    from cat_agent.llm.llama_cpp_vision import LlamaCppVision
    with patch.object(LlamaCppVision, "__init__", lambda self, cfg=None: None):
        model = LlamaCppVision.__new__(LlamaCppVision)
    return model


# ---------------------------------------------------------------------------
# _resolve_mmproj_path
# ---------------------------------------------------------------------------

class TestResolveMmprojPath:

    def test_returns_none_when_no_mmproj_keys(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _resolve_mmproj_path
        assert _resolve_mmproj_path({}) is None

    def test_returns_local_path_directly(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _resolve_mmproj_path
        assert _resolve_mmproj_path({"mmproj_path": "/some/local.gguf"}) == "/some/local.gguf"

    def test_raises_when_filename_without_repo(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _resolve_mmproj_path
        with pytest.raises(ValueError, match="mmproj_repo_id"):
            _resolve_mmproj_path({"mmproj_filename": "clip.gguf"})

    def test_falls_back_to_repo_id(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _resolve_mmproj_path
        with patch("huggingface_hub.hf_hub_download", return_value="/cached/clip.gguf") as mock_dl:
            result = _resolve_mmproj_path({
                "repo_id": "org/model-GGUF",
                "mmproj_filename": "clip.gguf",
            })
        mock_dl.assert_called_once_with(repo_id="org/model-GGUF", filename="clip.gguf")
        assert result == "/cached/clip.gguf"

    def test_uses_explicit_mmproj_repo_id(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _resolve_mmproj_path
        with patch("huggingface_hub.hf_hub_download", return_value="/cached/clip.gguf") as mock_dl:
            _resolve_mmproj_path({
                "repo_id": "org/model-GGUF",
                "mmproj_repo_id": "org/other-repo",
                "mmproj_filename": "clip.gguf",
            })
        mock_dl.assert_called_once_with(repo_id="org/other-repo", filename="clip.gguf")


# ---------------------------------------------------------------------------
# _build_chat_handler
# ---------------------------------------------------------------------------

class TestBuildChatHandler:

    def test_returns_none_when_no_mmproj(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _build_chat_handler
        assert _build_chat_handler({}, None) is None

    def test_explicit_handler_name(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _build_chat_handler

        mock_handler_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.MyHandler = mock_handler_cls

        with patch("llama_cpp.llama_chat_format", mock_module):
            result = _build_chat_handler({"chat_handler_name": "MyHandler"}, "/path/mmproj.gguf")

        mock_handler_cls.assert_called_once_with(clip_model_path="/path/mmproj.gguf")
        assert result == mock_handler_cls.return_value

    def test_explicit_handler_name_not_found_raises(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _build_chat_handler

        mock_module = MagicMock(spec=[])  # empty spec → getattr returns None

        with patch("llama_cpp.llama_chat_format", mock_module):
            with pytest.raises(ValueError, match="not found"):
                _build_chat_handler({"chat_handler_name": "NoSuchHandler"}, "/path/mmproj.gguf")

    def test_auto_detect_falls_through_on_exception(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import _build_chat_handler

        mock_module = MagicMock()
        # Both handlers raise → should return None
        mock_module.Qwen2VLChatHandler.side_effect = RuntimeError("boom")
        mock_module.Llava15ChatHandler.side_effect = RuntimeError("boom")

        with patch("llama_cpp.llama_chat_format", mock_module):
            assert _build_chat_handler({}, "/path/mmproj.gguf") is None


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestLlamaCppVisionInit:

    def test_init_requires_model_path_or_repo(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision

        with pytest.raises(ValueError, match="model_path|repo_id|filename"):
            LlamaCppVision({})
        with pytest.raises(ValueError, match="model_path|repo_id|filename"):
            LlamaCppVision({"repo_id": "only_repo"})


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_support_multimodal_input_is_true(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        assert model.support_multimodal_input is True

    def test_support_audio_input_is_false(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        assert model.support_audio_input is False


# ---------------------------------------------------------------------------
# _resolve_image_value
# ---------------------------------------------------------------------------

class TestResolveImageValue:

    def test_http_url_passthrough(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        url = "http://example.com/photo.jpg"
        assert LlamaCppVision._resolve_image_value(url) == url

    def test_https_url_passthrough(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        url = "https://cdn.example.com/img.png"
        assert LlamaCppVision._resolve_image_value(url) == url

    def test_data_uri_passthrough(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        uri = "data:image/jpeg;base64,/9j/4AAQ..."
        assert LlamaCppVision._resolve_image_value(uri) == uri

    def test_file_uri_strips_prefix(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        with patch("os.path.exists", return_value=True), \
             patch("cat_agent.llm.llama_cpp_vision.encode_image_as_base64", return_value="data:image/jpeg;base64,abc"):
            result = LlamaCppVision._resolve_image_value("file:///tmp/photo.jpg")
        assert result == "data:image/jpeg;base64,abc"

    def test_local_file_encoded_as_base64(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        with patch("os.path.exists", return_value=True), \
             patch("cat_agent.llm.llama_cpp_vision.encode_image_as_base64", return_value="data:image/jpeg;base64,xyz") as mock_enc:
            result = LlamaCppVision._resolve_image_value("/tmp/photo.jpg")
        mock_enc.assert_called_once_with("/tmp/photo.jpg", max_short_side_length=1080)
        assert result.startswith("data:")

    def test_missing_local_file_raises(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp_vision import LlamaCppVision
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                LlamaCppVision._resolve_image_value("/no/such/file.jpg")


# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------

class TestConvertMessages:

    def test_string_content(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._convert_messages([Message(USER, "Hello")])
        assert out == [{"role": "user", "content": "Hello"}]

    def test_text_content_items(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        msg = Message(USER, [ContentItem(text="A"), ContentItem(text="B")])
        out = model._convert_messages([msg])
        assert out[0]["role"] == "user"
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "A"}
        assert content[1] == {"type": "text", "text": "B"}

    def test_image_url_content_item(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        url = "https://example.com/cat.jpg"
        msg = Message(USER, [ContentItem(image=url)])
        out = model._convert_messages([msg])
        content = out[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == url

    def test_mixed_text_and_image(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        msg = Message(USER, [
            ContentItem(image="https://example.com/cat.jpg"),
            ContentItem(text="What is this?"),
        ])
        out = model._convert_messages([msg])
        content = out[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1] == {"type": "text", "text": "What is this?"}

    def test_dict_input_with_text(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._convert_messages([{"role": "user", "content": "Hi"}])
        assert out == [{"role": "user", "content": "Hi"}]

    def test_dict_content_items(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._convert_messages([{
            "role": "user",
            "content": [{"text": "describe"}, {"image": "https://x.com/a.jpg"}],
        }])
        content = out[0]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://x.com/a.jpg"

    def test_empty_content_list_becomes_empty_string(self):
        """An empty content list should become an empty string, not an empty list."""
        pytest.importorskip("llama_cpp")
        model = _make_model()
        # Audio ContentItem is silently skipped, leaving new_content empty
        msg = Message(USER, [ContentItem(audio="some_audio.wav")])
        out = model._convert_messages([msg])
        assert out[0]["content"] == ''

    def test_multiple_messages(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        msgs = [
            Message("system", "You are a vision assistant."),
            Message(USER, [ContentItem(image="https://x.com/a.jpg"), ContentItem(text="Describe.")]),
            Message(ASSISTANT, "It shows a cat."),
        ]
        out = model._convert_messages(msgs)
        assert len(out) == 3
        assert out[0] == {"role": "system", "content": "You are a vision assistant."}
        assert out[2] == {"role": "assistant", "content": "It shows a cat."}

    def test_non_string_non_list_content_becomes_str(self):
        """Fallback: unexpected content type is str()-ified."""
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._convert_messages([{"role": "user", "content": 42}])
        assert out[0]["content"] == "42"


# ---------------------------------------------------------------------------
# _prepare_generate_kwargs
# ---------------------------------------------------------------------------

class TestPrepareGenerateKwargs:

    def test_defaults(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._prepare_generate_kwargs({})
        assert out["temperature"] == 0.7
        assert out["top_p"] == 0.9
        assert out["max_tokens"] == 1024
        assert out["stop"] is None

    def test_custom_values(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._prepare_generate_kwargs({
            "temperature": 0.3,
            "top_p": 0.5,
            "max_tokens": 256,
            "stop": ["<|end|>"],
        })
        assert out["temperature"] == 0.3
        assert out["top_p"] == 0.5
        assert out["max_tokens"] == 256
        assert out["stop"] == ["<|end|>"]

    def test_max_new_tokens_fallback(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._prepare_generate_kwargs({"max_new_tokens": 512})
        assert out["max_tokens"] == 512

    def test_extra_keys_passed_through(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        out = model._prepare_generate_kwargs({"repeat_penalty": 1.1})
        assert out["repeat_penalty"] == 1.1


# ---------------------------------------------------------------------------
# _chat_no_stream
# ---------------------------------------------------------------------------

class TestChatNoStream:

    def test_returns_assistant_message(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        model.llm.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "The Statue of Liberty."}}]
        }
        result = model._chat_no_stream([Message(USER, "Describe.")], generate_cfg={})
        assert len(result) == 1
        assert result[0].role == ASSISTANT
        assert result[0].content == "The Statue of Liberty."

    def test_empty_response_returns_empty_content(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        model.llm.create_chat_completion.return_value = {"choices": []}
        result = model._chat_no_stream([Message(USER, "Describe.")], generate_cfg={})
        assert result[0].content == ''

    def test_passes_converted_messages_to_llama(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        model.llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        msg = Message(USER, [ContentItem(image="https://x.com/a.jpg"), ContentItem(text="Go")])
        model._chat_no_stream([msg], generate_cfg={})

        call_args = model.llm.create_chat_completion.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert messages_sent[0]["content"][0]["type"] == "image_url"
        assert messages_sent[0]["content"][1] == {"type": "text", "text": "Go"}


# ---------------------------------------------------------------------------
# _chat_stream
# ---------------------------------------------------------------------------

class TestChatStream:

    def test_stream_accumulates_tokens(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        model.llm.create_chat_completion.return_value = iter(chunks)

        outputs = list(model._chat_stream([Message(USER, "Hi")], delta_stream=False, generate_cfg={}))
        assert len(outputs) == 2
        assert outputs[0][0].content == "Hello"
        assert outputs[1][0].content == "Hello world"

    def test_stream_delta_mode(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        chunks = [
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
        ]
        model.llm.create_chat_completion.return_value = iter(chunks)

        outputs = list(model._chat_stream([Message(USER, "Hi")], delta_stream=True, generate_cfg={}))
        assert outputs[0][0].content == "A"
        assert outputs[1][0].content == "B"

    def test_stream_skips_empty_chunks(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        chunks = [
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": [{"delta": {"content": "ok"}}]},
        ]
        model.llm.create_chat_completion.return_value = iter(chunks)

        outputs = list(model._chat_stream([Message(USER, "Hi")], generate_cfg={}))
        assert len(outputs) == 1
        assert outputs[0][0].content == "ok"

    def test_stream_handles_malformed_chunks(self):
        pytest.importorskip("llama_cpp")
        model = _make_model()
        model.llm = MagicMock()
        chunks = [
            {"choices": []},           # IndexError path
            {"bad_key": "value"},       # KeyError path
            None,                       # TypeError path
            {"choices": [{"delta": {"content": "ok"}}]},
        ]
        model.llm.create_chat_completion.return_value = iter(chunks)

        outputs = list(model._chat_stream([Message(USER, "Hi")], generate_cfg={}))
        assert len(outputs) == 1
        assert outputs[0][0].content == "ok"
