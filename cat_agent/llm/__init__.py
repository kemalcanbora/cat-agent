from importlib import import_module
from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI

_BACKEND_MODULES = {
    'openvino': 'cat_agent.llm.openvino',
    'transformers': 'cat_agent.llm.transformers_llm',
    'llama_cpp': 'cat_agent.llm.llama_cpp',
    'llama_cpp_vision': 'cat_agent.llm.llama_cpp_vision',
    'mlx_lm': 'cat_agent.llm.mlx_lm_llm',
}


def _ensure_backend(model_type: str) -> None:
    module_path = _BACKEND_MODULES.get(model_type)
    if module_path is not None:
        import_module(module_path)


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        _ensure_backend(model_type)
        if model_type in LLM_REGISTRY:
            return LLM_REGISTRY[model_type](cfg)
        raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    cfg['model_type'] = 'oai'
    return LLM_REGISTRY['oai'](cfg)


def __getattr__(name: str):
    lazy_exports = {
        'OpenVINO': ('cat_agent.llm.openvino', 'OpenVINO'),
        'Transformers': ('cat_agent.llm.transformers_llm', 'Transformers'),
        'LlamaCpp': ('cat_agent.llm.llama_cpp', 'LlamaCpp'),
        'LlamaCppVision': ('cat_agent.llm.llama_cpp_vision', 'LlamaCppVision'),
        'MLXLm': ('cat_agent.llm.mlx_lm_llm', 'MLXLm'),
    }
    if name in lazy_exports:
        module_path, attr_name = lazy_exports[name]
        return getattr(import_module(module_path), attr_name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'BaseChatModel',
    'TextChatAtOAI',
    'OpenVINO',
    'Transformers',
    'MLXLm',
    'get_chat_model',
    'ModelServiceError',
    'LlamaCpp',
    'LlamaCppVision',
]
