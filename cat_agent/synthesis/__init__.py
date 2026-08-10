"""Tool synthesis: specify by example, generate a sandboxed ``@tool``."""

from __future__ import annotations

from cat_agent.synthesis.artifacts import generated_tools_root, tool_artifact_dir
from cat_agent.synthesis.executors import (
    ExecResult,
    SandboxExecutor,
    WasmExecutor,
    get_executor,
)
from cat_agent.synthesis.intake import (
    Draft,
    IntakeResult,
    SpecInterviewer,
    synthesize_from_draft,
    write_template,
)
from cat_agent.synthesis.registry import (
    AdoptedToolError,
    load_generated_tools,
    list_generated_tool_names,
    tools_for_principal,
)
from cat_agent.synthesis.smith import (
    AttemptRecord,
    Status,
    SynthesisResult,
    ToolSmith,
)
from cat_agent.synthesis.spec import (
    Example,
    ParameterSpec,
    ToolSpec,
    load_tool_spec,
    tool_spec_from_dict,
)

__all__ = [
    'AdoptedToolError',
    'AttemptRecord',
    'Draft',
    'Example',
    'ExecResult',
    'IntakeResult',
    'ParameterSpec',
    'SandboxExecutor',
    'SpecInterviewer',
    'Status',
    'SynthesisResult',
    'ToolSmith',
    'ToolSpec',
    'WasmExecutor',
    'demote',
    'generated_tools_root',
    'get_executor',
    'list_generated_tool_names',
    'load_generated_tools',
    'tools_for_principal',
    'load_tool_spec',
    'promote',
    'synthesize_from_draft',
    'tool_artifact_dir',
    'tool_spec_from_dict',
    'write_template',
]


def __getattr__(name: str):
    if name == 'promote':
        from cat_agent.synthesis.promote import promote as _promote
        return _promote
    if name == 'demote':
        from cat_agent.synthesis.promote import demote as _demote
        return _demote
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
