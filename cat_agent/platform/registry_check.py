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

"""Deploy-time check: manifest.name vs AgentRegistry HTTP route keys."""

from __future__ import annotations

from typing import Sequence


class RegistryNameError(ValueError):
    """manifest.name does not line up with the entrypoint registry."""


def validate_manifest_registry_names(
    manifest_name: str,
    registry_names: Sequence[str],
) -> None:
    """Ensure deployment name and ``/agents/{name}`` routes stay linked.

    * Exactly one registered agent whose key ≠ ``manifest.name`` → error naming both.
    * Several agents → ``manifest.name`` must be one of them (the job identity);
      otherwise error lists the routes and notes they all share the deployment host.
    """
    name = (manifest_name or '').strip()
    names = [str(n).strip() for n in registry_names if str(n).strip()]
    if not names:
        raise RegistryNameError('entrypoint returned an empty AgentRegistry')

    if len(names) == 1:
        only = names[0]
        if only != name:
            raise RegistryNameError(
                f'manifest.name is {name!r} but the entrypoint registry exposes a '
                f'single agent {only!r}; HTTP clients would call '
                f'/agents/{only}/run under this deployment, not /agents/{name}/run. '
                f'Register the agent as {name!r} or set manifest.name to {only!r}.'
            )
        return

    if name not in names:
        raise RegistryNameError(
            f'manifest.name {name!r} is not among the entrypoint registry agents '
            f'{names}. Multi-agent registries expose all of those routes under this '
            f'deployment\'s host (e.g. /agents/<name>/run for each); set '
            f'manifest.name to one of {names} so the deployment name matches a route.'
        )
