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

"""Agent deployment platform (Nomad). Import via CLI after ``pip install 'cat-agent[platform]'``."""

from __future__ import annotations

from typing import Any

__all__ = [
    'AgentManifest',
    'ManifestError',
    'PlatformConfig',
    'load_manifest',
    'load_platform_config',
]


def __getattr__(name: str) -> Any:
    if name in ('AgentManifest', 'ManifestError', 'load_manifest'):
        from cat_agent.platform import manifest as mod

        return getattr(mod, name)
    if name in ('PlatformConfig', 'load_platform_config'):
        from cat_agent.platform import config as mod

        return getattr(mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
