# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .transformers_llm import Transformers
from .qwenomni_oai import QwenOmniChatAtOAI
from .qwenvl_oai import QwenVLChatAtOAI
from .llama_cpp import LlamaCpp


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            return LLM_REGISTRY[model_type](cfg)
        raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided:

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_oai'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'qwen' in model.lower():
        model_type = 'oai'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')


__all__ = [
    'BaseChatModel',
    'TextChatAtOAI',
    'QwenVLChatAtOAI',
    'QwenOmniChatAtOAI',
    'OpenVINO',
    'Transformers',
    'get_chat_model',
    'ModelServiceError',
    'LlamaCpp',
]
