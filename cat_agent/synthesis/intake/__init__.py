"""Intake: Markdown draft → interview → ToolSpec → ToolSmith."""

from __future__ import annotations

from cat_agent.synthesis.intake.compile import CompileResult, compile_to_spec, sanitise_name
from cat_agent.synthesis.intake.draft import Draft, OpenQuestion
from cat_agent.synthesis.intake.interview import (
    Phase,
    InterviewState,
    Question,
    SpecInterviewer,
    holdout_question,
    insensitivity_question,
    is_affirmative,
    is_deferral,
    sanitize_messages_for_llm,
)
from cat_agent.synthesis.intake.pipeline import IntakeResult, synthesize_from_draft
from cat_agent.synthesis.intake.template import TEMPLATES, get_template, write_template
from cat_agent.synthesis.spec_quality import SpecWarning, lint_spec

__all__ = [
    'CompileResult',
    'Draft',
    'IntakeResult',
    'InterviewState',
    'OpenQuestion',
    'Phase',
    'Question',
    'SpecInterviewer',
    'SpecWarning',
    'TEMPLATES',
    'compile_to_spec',
    'get_template',
    'holdout_question',
    'insensitivity_question',
    'is_affirmative',
    'is_deferral',
    'lint_spec',
    'sanitize_messages_for_llm',
    'sanitise_name',
    'synthesize_from_draft',
    'write_template',
]
