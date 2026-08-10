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

"""Scheduler driver factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cat_agent.scheduling.store import JobStore


def get_driver(name: str, store: 'JobStore', **kwargs):
    """Return a driver instance by name (``apscheduler`` | ``oneshot``)."""
    key = (name or '').strip().lower()
    if key in ('apscheduler', 'apsched', 'async'):
        from cat_agent.scheduling.drivers.apscheduler_driver import APSchedulerDriver

        return APSchedulerDriver(store, **kwargs)
    if key in ('oneshot', 'k8s', 'cronjob'):
        # oneshot is a process entry point, not a long-lived object
        from cat_agent.scheduling.drivers import oneshot as oneshot_mod

        return oneshot_mod
    raise ValueError(f'Unknown scheduler driver: {name!r}')


__all__ = ['get_driver']
