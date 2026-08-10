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

"""On-demand FastAPI serving for named long-lived agents."""

from __future__ import annotations

from cat_agent.serve.app import create_app
from cat_agent.serve.factory import load_registry
from cat_agent.serve.registry import AgentInfo, AgentRegistry, AgentState, CapacityFull
from cat_agent.serve.server import run_app

__all__ = [
    'AgentInfo',
    'AgentRegistry',
    'AgentState',
    'CapacityFull',
    'create_app',
    'load_registry',
    'run_app',
]
