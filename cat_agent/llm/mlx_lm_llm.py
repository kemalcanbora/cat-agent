import copy
from typing import Dict, Iterator, List, Optional, Union

from cat_agent.llm.base import ModelServiceError, register_llm
from cat_agent.llm.function_calling import BaseFnCallModel
from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.log import logger

# Note: this is an optional dependency; we *do not* hard-fail at import time
# so that users/CI can run without installing MLX. We only error when someone
# actually tries to instantiate the MLXLm backend.
try:  # pragma: no cover - optional dependency path
    from mlx_lm import generate, load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    _HAS_MLX_LM = True
except Exception:  # ImportError or runtime errors if MLX bindings are missing
    generate = load = stream_generate = make_sampler = None  # type: ignore
    _HAS_MLX_LM = False


@register_llm("mlx_lm")
class MLXLm(BaseFnCallModel):
    """
    Apple MLX-based local LLM backend via `mlx-lm`.

    Minimal config example:

        llm_cfg = {
            "model_type": "mlx_lm",
            "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "generate_cfg": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        }
    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        super().__init__(cfg)

        if not _HAS_MLX_LM:
            # Match other backends: raise when the backend is *used*, not imported.
            raise ModelServiceError(
                code="MissingDependency",
                message=(
                    "mlx-lm backend requested but `mlx-lm` (and its MLX dependencies) "
                    "are not installed.\n"
                    "Install it with: pip install mlx-lm==0.31.1"
                ),
            )

        model_id = cfg.get("model") or cfg.get("model_id")
        if not model_id:
            raise ValueError('mlx_lm backend requires `model` (HuggingFace repo id or local path).')

        logger.info(f"Loading mlx_lm model: {model_id}")
        # mlx_lm.load returns (model, tokenizer)
        # Important: keep BaseChatModel.model as a string model-id for downstream logic.
        self.mlx_model, self.mlx_tokenizer = load(model_id)

        # mlx-lm is text-only today
        self._supports_function_calling = True
        self._support_multimodal_input = False

    @property
    def support_multimodal_input(self) -> bool:
        return self._support_multimodal_input

    @property
    def support_audio_input(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: List[Union[Message, Dict]]) -> List[Dict]:
        """Convert internal Message objects to OpenAI-style dicts for chat templates."""
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            if isinstance(content, str):
                text = content
            elif isinstance(content, (list, tuple)):
                parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(str(item))
                text = "".join(parts)
            else:
                text = str(content)

            result.append({"role": role, "content": text})
        return result

    def _prepare_generate_kwargs(self, generate_cfg: Dict) -> Dict:
        cfg = copy.deepcopy(generate_cfg or {})
        # BaseChatModel always injects a `seed`, but mlx-lm generation APIs do not accept it.
        cfg.pop("seed", None)
        # Align naming with transformers/llama_cpp configs used in this repo
        max_new_tokens = cfg.pop("max_new_tokens", cfg.pop("max_tokens", 1024))
        temperature = cfg.pop("temperature", None)
        top_p = cfg.pop("top_p", None)
        stop = cfg.pop("stop", None)

        gen_kwargs: Dict = {"max_tokens": max_new_tokens}
        if stop:
            gen_kwargs["stop"] = stop

        # mlx-lm (>=0.31.x) uses a `sampler` callable instead of passing temp/top_p directly.
        # Only create a sampler if the user provided sampling params.
        if (temperature is not None) or (top_p is not None) or ("top_k" in cfg) or ("min_p" in cfg):
            sampler_kwargs = {
                "top_k": cfg.pop("top_k", None),
                "xtc_probability": cfg.pop("xtc_probability", None),
                "xtc_threshold": cfg.pop("xtc_threshold", None),
            }
            # remove None keys
            sampler_kwargs = {k: v for k, v in sampler_kwargs.items() if v is not None}

            gen_kwargs["sampler"] = make_sampler(
                temperature if temperature is not None else 0.0,
                top_p if top_p is not None else 1.0,
                cfg.pop("min_p", 0.0),
                cfg.pop("min_tokens_to_keep", 1),
                **sampler_kwargs,
            )
        # Any remaining keys are passed through to mlx_lm.generate/stream_generate
        gen_kwargs.update(cfg)
        return gen_kwargs

    def _build_prompt(self, messages: List[Union[Message, Dict]]) -> str:
        mlx_messages = self._convert_messages(messages)
        # Use the tokenizer's chat template if available; fall back to a simple
        # concatenation otherwise.
        if hasattr(self.mlx_tokenizer, "apply_chat_template"):
            return self.mlx_tokenizer.apply_chat_template(
                mlx_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        # Fallback prompt building (very simple, but works reasonably well)
        lines = []
        for m in mlx_messages:
            lines.append(f"{m['role'].upper()}: {m['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core chat implementations
    # ------------------------------------------------------------------

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool = False,
        generate_cfg: Optional[Dict] = None,
    ) -> Iterator[List[Message]]:
        prompt = self._build_prompt(messages)
        gen_kwargs = self._prepare_generate_kwargs(generate_cfg or {})

        accumulated = ""
        # mlx_lm.stream_generate yields chunks of text
        for chunk in stream_generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt, **gen_kwargs):
            text = getattr(chunk, "text", None)
            if text is None:
                # Newer mlx-lm versions often yield plain strings
                text = str(chunk)
            if not text:
                continue

            accumulated += text
            if delta_stream:
                yield [Message(ASSISTANT, text)]
            else:
                yield [Message(ASSISTANT, accumulated)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: Optional[Dict] = None,
    ) -> List[Message]:
        prompt = self._build_prompt(messages)
        gen_kwargs = self._prepare_generate_kwargs(generate_cfg or {})

        # mlx_lm.generate returns the full text
        text = generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt, **gen_kwargs)
        # Some versions return an object with .text, others a plain string
        if not isinstance(text, str):
            text = getattr(text, "text", str(text))
        return [Message(ASSISTANT, text)]

