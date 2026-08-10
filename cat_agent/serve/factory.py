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

"""Load an AgentRegistry from a ``module:callable`` factory string."""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Union

from cat_agent.agent import Agent
from cat_agent.serve.registry import AgentRegistry


def load_registry(factory: str) -> AgentRegistry:
    """Import ``module:attr``, call it if callable, and coerce to :class:`AgentRegistry`.

    Accepted return values:
      * :class:`AgentRegistry`
      * ``dict[str, Agent]`` — keys become registry names
      * a single :class:`Agent` — registered under ``agent.name``
    """
    spec = (factory or '').strip()
    if ':' not in spec:
        raise ValueError(
            f'Invalid factory {factory!r}; expected "module.path:callable"'
        )
    module_name, _, attr_path = spec.partition(':')
    module_name = module_name.strip()
    attr_path = attr_path.strip()
    if not module_name or not attr_path:
        raise ValueError(
            f'Invalid factory {factory!r}; expected "module.path:callable"'
        )

    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_path.split('.'):
        obj = getattr(obj, part)

    result = obj() if callable(obj) else obj
    return coerce_registry(result)


def coerce_registry(result: Union[AgentRegistry, Agent, Mapping[str, Agent]]) -> AgentRegistry:
    if isinstance(result, AgentRegistry):
        if len(result) == 0:
            raise ValueError('Factory returned an empty AgentRegistry')
        return result

    if isinstance(result, Agent):
        registry = AgentRegistry()
        registry.register(result)
        return registry

    if isinstance(result, Mapping):
        registry = AgentRegistry()
        if not result:
            raise ValueError('Factory returned an empty agent mapping')
        for name, agent in result.items():
            if not isinstance(agent, Agent):
                raise TypeError(
                    f'Factory mapping value for {name!r} is not an Agent: {type(agent)!r}'
                )
            registry.register(agent, name=str(name))
        return registry

    raise TypeError(
        'Factory must return AgentRegistry, Agent, or Mapping[str, Agent]; '
        f'got {type(result)!r}'
    )
