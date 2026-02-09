from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .transformers_llm import Transformers
from .llama_cpp import LlamaCpp
from .llama_cpp_vision import LlamaCppVision


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            return LLM_REGISTRY[model_type](cfg)
        raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Default to OpenAI-compatible API when model_type is not specified
    cfg['model_type'] = 'oai'
    return LLM_REGISTRY['oai'](cfg)


__all__ = [
    'BaseChatModel',
    'TextChatAtOAI',
    'OpenVINO',
    'Transformers',
    'get_chat_model',
    'ModelServiceError',
    'LlamaCpp',
    'LlamaCppVision',
]
