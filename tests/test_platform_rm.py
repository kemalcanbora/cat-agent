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

"""rm semantics tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.platform import commands as platform_commands
from cat_agent.platform.commands import CommandError


def _job(jid, team, agent, jobs_mode='inline'):
    return {
        'ID': jid,
        'Meta': {
            'managed_by': 'cat-agent',
            'team': team,
            'agent': agent,
            'jobs_mode': jobs_mode,
        },
        'Status': 'running',
    }


def test_ambiguous_name_requires_team():
    client = MagicMock()
    client.list_agents.return_value = [
        _job('agent-a-calc', 'a', 'calc'),
        _job('agent-b-calc', 'b', 'calc'),
    ]
    args = SimpleNamespace(
        config=None,
        nomad_addr=None,
        registry=None,
        name='calc',
        team=None,
        yes=True,
        force=False,
    )
    with patch.object(platform_commands, '_load_cfg', return_value=MagicMock()):
        with patch.object(platform_commands, '_client', return_value=client):
            with pytest.raises(CommandError, match='ambiguous'):
                platform_commands.cmd_rm(args)


def test_dispatch_stops_service_before_task():
    service = _job('agent-growth-scout', 'growth', 'scout', 'dispatch')
    task = _job('agent-growth-scout-task', 'growth', 'scout', 'dispatch')
    client = MagicMock()
    client.list_agents.return_value = [service]
    client.get_job.return_value = task
    client.allocations.return_value = []
    order = []

    def stop(jid, purge=True):
        order.append(jid)
        return {}

    client.stop.side_effect = stop
    args = SimpleNamespace(
        config=None,
        nomad_addr=None,
        registry=None,
        name='scout',
        team='growth',
        yes=True,
        force=False,
    )
    with patch.object(platform_commands, '_load_cfg', return_value=MagicMock()):
        with patch.object(platform_commands, '_client', return_value=client):
            assert platform_commands.cmd_rm(args) == 0
    assert order[0] == 'agent-growth-scout'
    assert order[1] == 'agent-growth-scout-task'


def test_running_tasks_block_without_force():
    service = _job('agent-growth-scout', 'growth', 'scout', 'dispatch')
    task = _job('agent-growth-scout-task', 'growth', 'scout', 'dispatch')
    client = MagicMock()
    client.list_agents.return_value = [service]
    client.get_job.return_value = task
    client.allocations.return_value = [{'ClientStatus': 'running'}]
    args = SimpleNamespace(
        config=None,
        nomad_addr=None,
        registry=None,
        name='scout',
        team='growth',
        yes=True,
        force=False,
    )
    with patch.object(platform_commands, '_load_cfg', return_value=MagicMock()):
        with patch.object(platform_commands, '_client', return_value=client):
            with pytest.raises(CommandError, match='still running'):
                platform_commands.cmd_rm(args)
