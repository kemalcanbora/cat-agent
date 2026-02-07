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

import logging
import os


def setup_logger(level=None):
    _logger = logging.getLogger('cat_agent_logger')
    # Disable all logging output for this project.
    _logger.handlers.clear()
    _logger.addHandler(logging.NullHandler())
    _logger.propagate = False
    _logger.disabled = True
    return _logger


logger = setup_logger()
