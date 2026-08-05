"""Ensure synthesised code exposes the expected entry-point name."""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple


def extract_impl_code(raw: str, function_name: str) -> str:
    """Pull the implementation from an LLM reply.

    Prefers a fenced block that defines ``function_name``, then any ``def``,
    then the last fence, then a bare ``def`` match, then :func:`extract_code`.
    """
    from cat_agent.synthesis.llm_text import strip_thinking_markup
    from cat_agent.utils.utils import extract_code

    text = strip_thinking_markup(raw or '')
    fences = re.findall(
        r'```(?:python|py)?\s*\n?(.*?)```',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fences:
        named = [
            b for b in fences
            if re.search(rf'\bdef\s+{re.escape(function_name)}\b', b)
        ]
        if named:
            return named[-1].strip()
        with_def = [b for b in fences if re.search(r'\b(?:async\s+)?def\s+\w+', b)]
        if with_def:
            return with_def[-1].strip()
        return fences[-1].strip()

    # Unfenced reply that is still just a function body.
    bare = re.search(
        rf'((?:async\s+)?def\s+{re.escape(function_name)}\b[\s\S]+)',
        text,
    )
    if bare:
        return bare.group(1).strip()
    bare_any = re.search(r'((?:async\s+)?def\s+\w+\b[\s\S]+)', text)
    if bare_any and 'def ' in text:
        return bare_any.group(1).strip()

    return (extract_code(text) or '').strip()


def top_level_function_names(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def ensure_entry_point(code: str, function_name: str) -> Tuple[str, Optional[str]]:
    """Return ``(code, error)``.

    If the code defines exactly one top-level function under a different name,
    rename it to *function_name*. If the expected name is already present,
    return the code unchanged. Otherwise return a clear error for the retry loop.
    """
    text = (code or '').strip()
    if not text:
        return text, (
            f'No code produced. Define exactly `def {function_name}(...):` '
            f'in a single python markdown fence.'
        )
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return text, f'Syntax error in generated code: {exc}'

    names = top_level_function_names(text)
    if function_name in names:
        return text, None
    if len(names) == 1:
        old = names[0]
        renamed = re.sub(
            rf'^([ \t]*)((?:async[ \t]+)?def)[ \t]+{re.escape(old)}\b',
            rf'\1\2 {function_name}',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if function_name in top_level_function_names(renamed):
            return renamed, None
        return text, (
            f'Could not rename `{old}` to `{function_name}`. '
            f'Rewrite the function as `def {function_name}(...):`.'
        )
    if not names:
        return text, (
            f'No top-level function found. Define exactly '
            f'`def {function_name}(...):`.'
        )
    return text, (
        f'Function must be named `{function_name}`. Found: {names}. '
        f'Rename your function to `{function_name}` and remove extras.'
    )


def simplify_name_error(
    *,
    function_name: str,
    error: Optional[str],
    stderr: Optional[str],
) -> Optional[str]:
    """Return a short, actionable message when the entry point is missing."""
    blob = f'{error or ""}\n{stderr or ""}'
    needle = f"name '{function_name}' is not defined"
    if needle in blob or (
        'NameError' in blob and function_name in blob
    ):
        return (
            f'NameError: `{function_name}` is not defined at call time. '
            f'Your next reply must define exactly `def {function_name}(...):` '
            f'with that spelling — do not use a different name.'
        )
    return None
