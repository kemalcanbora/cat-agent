"""llama.cpp vision backend for multimodal GGUF models (Qwen2-VL, Qwen3-VL, LLaVA, etc.)."""

import copy
import os
from typing import Dict, Iterator, List, Optional, Union

from cat_agent.llm.base import register_llm
from cat_agent.llm.function_calling import BaseFnCallModel
from cat_agent.llm.schema import ASSISTANT, ContentItem, Message
from cat_agent.log import logger
from cat_agent.utils.utils import encode_image_as_base64

try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "llama-cpp-python is required to use the llama_cpp_vision backend.\n"
        "Install it with: pip install llama-cpp-python\n"
        "(add --extra-index-url for GPU/metal/...)"
    ) from e


def _resolve_mmproj_path(cfg: dict) -> Optional[str]:
    """Resolve the vision-projection (mmproj / clip) model path.

    Supports:
      - ``mmproj_path``  → local file
      - ``mmproj_repo_id`` + ``mmproj_filename`` → download from HuggingFace
      - ``mmproj_filename`` alone → uses the main ``repo_id``
    """
    mmproj_path = cfg.get('mmproj_path')
    if mmproj_path:
        return mmproj_path

    mmproj_filename = cfg.get('mmproj_filename')
    if not mmproj_filename:
        return None

    mmproj_repo_id = cfg.get('mmproj_repo_id', cfg.get('repo_id'))
    if not mmproj_repo_id:
        raise ValueError(
            "When 'mmproj_filename' is set you must also provide "
            "'mmproj_repo_id' (or 'repo_id') so the file can be downloaded."
        )

    from huggingface_hub import hf_hub_download
    logger.info(f"Downloading mmproj from HuggingFace: {mmproj_repo_id} / {mmproj_filename}")
    return hf_hub_download(repo_id=mmproj_repo_id, filename=mmproj_filename)


def _build_chat_handler(cfg: dict, mmproj_path: Optional[str]):
    """Build a llama-cpp-python chat handler for vision models.

    Strategy:
      1. If ``chat_handler_name`` is set explicitly, use that class.
      2. Otherwise try ``Qwen2VLChatHandler`` (covers Qwen2-VL / Qwen3-VL).
      3. Fall back to ``Llava15ChatHandler`` (generic vision handler).
      4. If nothing works, return *None* and let llama-cpp figure it out.
    """
    if mmproj_path is None:
        return None

    handler_name = cfg.get('chat_handler_name')

    if handler_name:
        from llama_cpp import llama_chat_format
        handler_cls = getattr(llama_chat_format, handler_name, None)
        if handler_cls is None:
            raise ValueError(f"Chat handler '{handler_name}' not found in llama_cpp.llama_chat_format")
        logger.info(f"Using explicitly requested chat handler: {handler_name}")
        return handler_cls(clip_model_path=mmproj_path)

    # Auto-detect: prefer Qwen2VL handler, then Llava15
    for name in ('Qwen2VLChatHandler', 'Llava15ChatHandler'):
        try:
            from llama_cpp import llama_chat_format
            handler_cls = getattr(llama_chat_format, name, None)
            if handler_cls is not None:
                handler = handler_cls(clip_model_path=mmproj_path)
                logger.info(f"Auto-selected chat handler: {name}")
                return handler
        except Exception:
            continue

    logger.warning(
        "No vision chat handler found in llama-cpp-python. "
        "Multimodal messages will be passed as-is to create_chat_completion."
    )
    return None


@register_llm('llama_cpp_vision')
class LlamaCppVision(BaseFnCallModel):
    """llama.cpp vision backend – extends llama-cpp-python with multimodal support.

    Config example::

        llm_cfg = {
            'model_type':       'llama_cpp_vision',

            # ── Main model (local path OR HuggingFace) ──
            'repo_id':          'Qwen/Qwen3-VL-2B-Instruct-GGUF',
            'filename':         'Qwen3VL-2B-Instruct-F16.gguf',
            # 'model_path':     '/local/path/to/model.gguf',

            # ── Vision projection / clip model ──
            'mmproj_filename':  'mmproj-Qwen3VL-2B-Instruct-F16.gguf',
            # 'mmproj_repo_id': 'Qwen/Qwen3-VL-2B-Instruct-GGUF',  # defaults to repo_id
            # 'mmproj_path':    '/local/path/to/mmproj.gguf',


    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        super().__init__(cfg)

        model_path = cfg.get('model_path')
        repo_id = cfg.get('repo_id')
        filename = cfg.get('filename')

        if not (model_path or (repo_id and filename)):
            raise ValueError(
                "llama_cpp_vision backend requires either 'model_path' "
                "or both 'repo_id' and 'filename'."
            )

        # ── Vision projection model ──
        mmproj_path = _resolve_mmproj_path(cfg)
        chat_handler = _build_chat_handler(cfg, mmproj_path)

        # ── Llama constructor kwargs ──
        llama_kwargs = {
            'n_ctx': cfg.get('n_ctx', 8192),
            'n_gpu_layers': cfg.get('n_gpu_layers', -1),
            'n_threads': cfg.get('n_threads'),
            'n_batch': cfg.get('n_batch', 512),
            'verbose': cfg.get('verbose', False),
        }
        if chat_handler is not None:
            llama_kwargs['chat_handler'] = chat_handler

        # ── Load model ──
        if model_path:
            logger.info(f"Loading llama.cpp vision model from: {model_path}")
            self.llm = Llama(model_path=model_path, **llama_kwargs)
        else:
            logger.info(f"Downloading/loading vision model: {repo_id} / {filename}")
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cfg.get('cache_dir'),
                **llama_kwargs,
            )

        self._supports_function_calling = True

    # ------------------------------------------------------------------
    # Multimodal flags
    # ------------------------------------------------------------------

    @property
    def support_multimodal_input(self) -> bool:
        return True

    @property
    def support_audio_input(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Message conversion  (ContentItem → OpenAI multimodal format)
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: List[Union[Message, Dict]]) -> List[Dict]:
        """Convert cat-agent Message objects to OpenAI-compatible dicts.

        Handles ``ContentItem`` objects that contain text and/or images,
        producing the ``image_url`` content blocks expected by
        llama-cpp-python vision chat handlers.
        """
        result = []

        for msg in messages:
            if isinstance(msg, Message):
                role = msg.role
                content = msg.content
            else:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

            if isinstance(content, str):
                result.append({'role': role, 'content': content})
            elif isinstance(content, (list, tuple)):
                new_content = []
                for item in content:
                    if isinstance(item, ContentItem):
                        t, v = item.get_type_and_value()
                        if t == 'text' and v:
                            new_content.append({'type': 'text', 'text': v})
                        elif t == 'image':
                            url = self._resolve_image_value(v)
                            new_content.append({'type': 'image_url', 'image_url': {'url': url}})
                        # Silently skip unsupported types (audio, video, file)
                    elif isinstance(item, dict):
                        if 'text' in item:
                            new_content.append({'type': 'text', 'text': item['text']})
                        elif 'image' in item:
                            url = self._resolve_image_value(item['image'])
                            new_content.append({'type': 'image_url', 'image_url': {'url': url}})
                    else:
                        new_content.append({'type': 'text', 'text': str(item)})

                if not new_content:
                    new_content = ''
                result.append({'role': role, 'content': new_content})
            else:
                result.append({'role': role, 'content': str(content)})

        return result

    @staticmethod
    def _resolve_image_value(v: str) -> str:
        """Turn an image reference into a URL or ``data:`` URI.

        * HTTP(S) URLs and ``data:`` URIs are returned as-is.
        * ``file://`` prefixes are stripped, then the local path is encoded.
        * Local file paths are base64-encoded as JPEG data URIs.
        """
        if v.startswith('file://'):
            v = v[len('file://'):]
        if v.startswith(('http://', 'https://', 'data:')):
            return v
        if os.path.exists(v):
            return encode_image_as_base64(v, max_short_side_length=1080)
        raise FileNotFoundError(f'Local image file "{v}" does not exist.')

    # ------------------------------------------------------------------
    # Generation helpers (reused from LlamaCpp pattern)
    # ------------------------------------------------------------------

    def _prepare_generate_kwargs(self, generate_cfg: Dict) -> Dict:
        cfg = copy.deepcopy(generate_cfg or {})
        return {
            'temperature': cfg.pop('temperature', 0.7),
            'top_p': cfg.pop('top_p', 0.9),
            'max_tokens': cfg.pop('max_tokens', cfg.pop('max_new_tokens', 1024)),
            'stop': cfg.pop('stop', None),
            **cfg,
        }

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool = False,
        generate_cfg: Optional[Dict] = None,
    ) -> Iterator[List[Message]]:
        llama_messages = self._convert_messages(messages)
        gen_kwargs = self._prepare_generate_kwargs(generate_cfg or {})

        accumulated = ""
        for chunk in self.llm.create_chat_completion(
            messages=llama_messages,
            stream=True,
            **gen_kwargs,
        ):
            try:
                delta = chunk['choices'][0]['delta']
                token = delta.get('content', '')
            except (KeyError, IndexError, TypeError):
                token = ''
            if token:
                accumulated += token
                if delta_stream:
                    yield [Message(ASSISTANT, token)]
                else:
                    yield [Message(ASSISTANT, accumulated)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: Optional[Dict] = None,
    ) -> List[Message]:
        llama_messages = self._convert_messages(messages)
        gen_kwargs = self._prepare_generate_kwargs(generate_cfg or {})

        response = self.llm.create_chat_completion(
            messages=llama_messages,
            stream=False,
            **gen_kwargs,
        )

        try:
            content = response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            content = ''

        return [Message(ASSISTANT, content)]
