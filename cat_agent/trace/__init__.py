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

"""Machine-readable execution traces for cat-agent runs.

Enable with ``CAT_AGENT_TRACE=1`` and optionally ``CAT_AGENT_TRACE_FILE=path.jsonl``.
Tracing is off by default and independent of Loguru logging.

References: Yehudai et al. arXiv:2503.16416; OpenTelemetry GenAI semantic conventions.
"""

from cat_agent.trace.recorder import (
    TraceRecorder,
    default_trace_store,
    get_trace_recorder,
    is_trace_enabled,
    trace_run,
)
from cat_agent.trace.schema import (
    SCHEMA_VERSION,
    ContextOpPayload,
    ErrorPayload,
    HandoffPayload,
    LLMCallPayload,
    Run,
    RunLimits,
    RunTotals,
    Step,
    ToolCallPayload,
)
from cat_agent.trace.store import (
    InMemoryTraceStore,
    JSONLTraceStore,
    TraceStore,
    load_runs_from_jsonl,
    parse_partial_jsonl,
)

__all__ = [
    'SCHEMA_VERSION',
    'ContextOpPayload',
    'ErrorPayload',
    'HandoffPayload',
    'InMemoryTraceStore',
    'JSONLTraceStore',
    'LLMCallPayload',
    'Run',
    'RunLimits',
    'RunTotals',
    'Step',
    'ToolCallPayload',
    'TraceRecorder',
    'TraceStore',
    'default_trace_store',
    'get_trace_recorder',
    'is_trace_enabled',
    'load_runs_from_jsonl',
    'parse_partial_jsonl',
    'trace_run',
]
