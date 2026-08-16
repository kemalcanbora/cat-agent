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

"""Tests for cat_agent.observability.emitter."""

from cat_agent.observability.emitter import (
    clear_handlers,
    emit,
    get_default_handlers,
    register_handler,
    resolve_handlers,
)
from cat_agent.observability.events import EventEnvelope
from cat_agent.observability.handlers.callback import CallbackHandler


def setup_function():
    clear_handlers()


def teardown_function():
    clear_handlers()


def test_register_and_clear_handlers():
    seen = []
    h = CallbackHandler(lambda e: seen.append(e.event_type))
    register_handler(h)
    assert len(get_default_handlers()) == 1
    clear_handlers()
    assert get_default_handlers() == []


def test_resolve_handlers_prefers_run_over_agent():
    a = CallbackHandler(lambda e: None)
    b = CallbackHandler(lambda e: None)
    assert resolve_handlers([a], [b]) == [b]
    assert resolve_handlers([a], None) == [a]


def test_emit_noop_without_run_context():
    # No RunContext → emit is a no-op (must not raise).
    emit(EventEnvelope(
        event_type='run.start',
        timestamp=0.0,
        trace_id='t',
        run_id='r',
        span_id='s',
        parent_span_id=None,
        agent_name=None,
        agent_class='A',
        payload={},
    ))
