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

"""Tests for deploy-time manifest.name vs AgentRegistry route keys."""

from __future__ import annotations

import pytest

from cat_agent.platform.registry_check import (
    RegistryNameError,
    validate_manifest_registry_names,
)


def test_single_agent_matching_ok():
    validate_manifest_registry_names('calculator', ['calculator'])


def test_single_agent_mismatch_names_both():
    with pytest.raises(RegistryNameError, match=r"manifest\.name is 'arith'.*'calculator'"):
        validate_manifest_registry_names('arith', ['calculator'])


def test_multi_agent_manifest_must_be_one_of():
    with pytest.raises(
        RegistryNameError,
        match=r"Multi-agent registries expose all of those routes",
    ):
        validate_manifest_registry_names('missing', ['alpha', 'beta'])


def test_multi_agent_ok_when_manifest_matches_one():
    validate_manifest_registry_names('beta', ['alpha', 'beta'])


def test_empty_registry():
    with pytest.raises(RegistryNameError, match='empty'):
        validate_manifest_registry_names('x', [])
