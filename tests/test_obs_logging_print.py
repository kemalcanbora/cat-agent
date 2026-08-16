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

"""Tests for observability LoggingHandler and PrintHandler extras."""

from unittest.mock import patch

from cat_agent.observability.events import EventEnvelope
from cat_agent.observability.handlers.logging import LoggingHandler
from cat_agent.observability.handlers.print_handler import PrintHandler


def _event(etype: str, **payload) -> EventEnvelope:
    return EventEnvelope(
        event_type=etype,
        timestamp=0.0,
        trace_id='t',
        run_id='r',
        span_id='s',
        parent_span_id=None,
        agent_name='bot',
        agent_class='Assistant',
        payload=payload,
    )


def test_logging_handler_run_start_and_end():
    h = LoggingHandler(level='INFO')
    with patch('cat_agent.observability.handlers.logging.logger') as log:
        h.on_event(_event('run.start', message_count=2, lang='en'))
        h.on_event(_event('run.end', duration_ms=10, yield_count=1))
        h.on_event(_event('llm.start', model='m', message_count=1, tool_count=0))
        assert log.log.call_count >= 3


def test_print_handler_run_end_with_metrics(capsys):
    h = PrintHandler()
    h.on_event(_event(
        'run.end',
        duration_ms=100,
        yield_count=2,
        metrics={
            'llm_calls': 1,
            'tool_calls': 0,
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'llm_ms': 50,
            'max_context_ratio': 0.2,
        },
    ))
    out = capsys.readouterr().out
    assert 'run.end' in out
    assert 'tok=10/5' in out
