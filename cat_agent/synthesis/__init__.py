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
from cat_agent.synthesis.registry import load_generated_tools, list_generated_tool_names
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
    'generated_tools_root',
    'get_executor',
    'list_generated_tool_names',
    'load_generated_tools',
    'load_tool_spec',
    'synthesize_from_draft',
    'tool_artifact_dir',
    'tool_spec_from_dict',
    'write_template',
]
