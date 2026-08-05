"""Persist synthesised tools under the workspace."""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from cat_agent import __version__ as CAT_AGENT_VERSION
from cat_agent.log import logger
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.synthesis.spec import Example, ParameterSpec, ToolSpec

_TYPE_MAP = {
    'string': 'str',
    'str': 'str',
    'integer': 'int',
    'int': 'int',
    'number': 'float',
    'float': 'float',
    'boolean': 'bool',
    'bool': 'bool',
    'object': 'dict',
    'dict': 'dict',
    'array': 'list',
    'list': 'list',
    'any': 'Any',
}


def generated_tools_root(base: Optional[str] = None) -> Path:
    root = Path(base or DEFAULT_WORKSPACE) / 'generated_tools'
    return root


def tool_artifact_dir(spec: ToolSpec, base: Optional[str] = None) -> Path:
    return generated_tools_root(base) / spec.function_name


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def spec_hash(spec: ToolSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    return sha256_text(payload)


def _python_type(type_str: str) -> str:
    """Map a ToolSpec type string to a safe Python annotation token.

    Model prose such as ``"object with two numeric fields, net and tax..."``
    must not be pasted into a signature (that yields SyntaxError).
    """
    key = (type_str or 'Any').strip()
    if not key:
        return 'Any'
    mapped = _TYPE_MAP.get(key.lower())
    if mapped:
        return mapped
    if re_fullmatch_identifierish(key):
        return key
    # Prose → leading JSON-schema-ish token ("object with ..." → dict).
    leading = re.match(r'[A-Za-z_][A-Za-z0-9_]*', key)
    if leading:
        mapped = _TYPE_MAP.get(leading.group(0).lower())
        if mapped:
            return mapped
    return 'Any'


_PROSE_STOPWORDS = frozenset({
    'with', 'and', 'each', 'the', 'for', 'as', 'to', 'of', 'or', 'a', 'an',
    'rounded', 'fields', 'numeric', 'amount', 'value', 'two', 'into',
})


def re_fullmatch_identifierish(value: str) -> bool:
    """True for compact type expressions only — not English prose."""
    if not value:
        return False
    # Reject sentences that smuggle English between type-looking characters.
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', value)
    if any(t.lower() in _PROSE_STOPWORDS for t in tokens):
        return False
    return bool(
        re.fullmatch(
            r'[A-Za-z_][A-Za-z0-9_]*'
            r'(\[[A-Za-z0-9_\[\],\|\.\s]+\])?'
            r'(\s*\|\s*[A-Za-z_][A-Za-z0-9_]*'
            r'(\[[A-Za-z0-9_\[\],\|\.\s]+\])?)*',
            value,
        )
    )


def coerce_spec_type(type_str: str) -> str:
    """Normalise a returns/parameter type to a short schema token for ToolSpec."""
    py = _python_type(type_str)
    reverse = {
        'dict': 'object',
        'list': 'array',
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'Any': 'Any',
    }
    return reverse.get(py, py)


def _param_type(param: ParameterSpec | str) -> str:
    if isinstance(param, ParameterSpec):
        return param.type
    return str(param)


def _param_doc(name: str, param: ParameterSpec | str) -> str:
    if isinstance(param, ParameterSpec) and (param.description or '').strip():
        return f'        {name}: {param.description.strip()}'
    return f'        {name}: Parameter of type {_param_type(param)}.'


def build_proxy_source(spec: ToolSpec, *, executor_name: str = 'wasm') -> str:
    """Render the fixed proxy ``tool.py`` template (never model-generated)."""
    fn = spec.function_name
    reg = spec.registered_name
    params_sig = ', '.join(
        f'{name}: {_python_type(_param_type(param))}'
        for name, param in spec.parameters.items()
    )
    args_docs = '\n'.join(
        _param_doc(name, param) for name, param in spec.parameters.items()
    )
    call_dict = ', '.join(f'{name!r}: {name}' for name in spec.parameters)
    returns = _python_type(spec.returns)
    type_names = [_python_type(_param_type(p)) for p in spec.parameters.values()]
    needs_any = 'Any' in (returns, *type_names)
    typing_import = 'from typing import Any\n' if needs_any else ''
    network = 'True' if spec.requires_network else 'False'
    description = spec.description.strip().replace('"""', "'''")
    # Keep docstring body indented even when description has blank lines —
    # textwrap.dedent over an interpolated multiline description would otherwise
    # leave the whole module unexpectedly indented (IndentationError).
    desc_lines = description.splitlines() or ['']
    if len(desc_lines) == 1:
        doc_body = desc_lines[0]
    else:
        doc_body = desc_lines[0] + '\n' + textwrap.indent(
            '\n'.join(desc_lines[1:]), '    '
        )

    # Emit at column 0 — do not wrap in textwrap.dedent.
    source = (
        f'"""Auto-generated proxy for synthesised tool ``{reg}``.\n'
        f'\n'
        f'The host NEVER imports ``impl.py``. Arguments are serialised and executed\n'
        f'inside a :class:`~cat_agent.synthesis.executors.base.SandboxExecutor`.\n'
        f'"""\n'
        f'from __future__ import annotations\n'
        f'\n'
        f'from pathlib import Path\n'
        f'\n'
        f'{typing_import}'
        f'from cat_agent.synthesis.executors import get_executor\n'
        f'from cat_agent.tools import tool\n'
        f'from cat_agent.tools.base import ToolExecutionError\n'
        f'\n'
        f"_IMPL_PATH = Path(__file__).with_name('impl.py')\n"
        f'_EXECUTOR = get_executor({executor_name!r})\n'
        f'_TOOL_NAME = {reg!r}\n'
        f'\n'
        f'\n'
        f'@tool(\n'
        f'    name={reg!r},\n'
        f'    register_by_default=False,\n'
        f'    requires_network={network},\n'
        f'    allow_overwrite=True,\n'
        f')\n'
        f'def {fn}({params_sig}) -> {returns}:\n'
        f'    """{doc_body}\n'
        f'\n'
        f'    Args:\n'
        f'{args_docs}\n'
        f'    """\n'
        f"    code = _IMPL_PATH.read_text(encoding='utf-8')\n"
        f'    result = _EXECUTOR.run(\n'
        f'        code,\n'
        f'        {{{call_dict}}},\n'
        f'        function_name={fn!r},\n'
        f'    )\n'
        f'    if not result.ok:\n'
        f'        raise ToolExecutionError(\n'
        f"            _TOOL_NAME,\n"
        f"            result.error or 'sandbox execution failed',\n"
        f'        )\n'
        f'    return result.returned\n'
    )
    return format_python_source(source)


def assistant_tool_filename(spec: ToolSpec) -> str:
    """Task-related filename for the standalone Assistant ``@tool`` module."""
    return f'{spec.function_name}.py'


def build_assistant_tool_source(spec: ToolSpec, code: str) -> str:
    """Wrap validated function body as a self-contained ``@tool`` module.

    Logic lives inline under the decorated function — no ``impl.py`` import,
    no executor proxy. Intended for Assistant use after ToolSmith validation.
    """
    from cat_agent.synthesis.entry_point import ensure_entry_point

    body, err = ensure_entry_point((code or '').strip(), spec.function_name)
    if err:
        raise ValueError(f'Cannot build assistant tool source: {err}')

    # Drop a model-added @tool if present; we attach our own.
    body = re.sub(r'(?m)^[ \t]*@tool(\s*\([\s\S]*?\))?[ \t]*\n', '', body)
    body = body.strip() + '\n'

    # Preserve model imports that are not our decorator import.
    import_lines: List[str] = []
    rest_lines: List[str] = []
    saw_def = False
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        if not saw_def and (
            stripped.startswith('import ')
            or stripped.startswith('from ')
        ):
            if re.search(r'from\s+cat_agent\.tools\s+import\b.*\btool\b', stripped):
                continue
            if stripped.startswith('from cat_agent.tools import tool'):
                continue
            import_lines.append(line if line.endswith('\n') else line + '\n')
            continue
        if stripped.startswith('def ') or stripped.startswith('async def '):
            saw_def = True
        rest_lines.append(line if line.endswith('\n') else line + '\n')

    rest_text = ''.join(rest_lines)
    needs_any = (
        re.search(r'\bAny\b', rest_text) is not None
        or _python_type(spec.returns) == 'Any'
        or any(
            _python_type(_param_type(p)) == 'Any'
            for p in spec.parameters.values()
        )
    )
    header_imports = [
        'from __future__ import annotations\n',
        '\n',
    ]
    if needs_any and not any('Any' in ln for ln in import_lines):
        header_imports.append('from typing import Any\n')
    header_imports.append('from cat_agent.tools import tool\n')
    header_imports.append('\n')

    network = 'True' if spec.requires_network else 'False'
    reg = spec.registered_name
    decorator = (
        f'@tool(\n'
        f'    name={reg!r},\n'
        f'    register_by_default=False,\n'
        f'    requires_network={network},\n'
        f'    allow_overwrite=True,\n'
        f')\n'
    )

    module_doc = (
        f'"""Assistant-ready tool ``{spec.function_name}`` '
        f'(validated by ToolSmith).\n\n'
        f'Logic is inline under the ``@tool`` function — import or copy into an agent.\n'
        f'"""\n'
    )

    parts = [module_doc, *header_imports]
    if import_lines:
        parts.extend(import_lines)
        parts.append('\n')
    parts.append(decorator)
    parts.extend(rest_lines)
    text = ''.join(parts)
    if not text.endswith('\n'):
        text += '\n'
    return format_python_source(text)


def format_python_source(source: str) -> str:
    """Format with black then ruff when available; otherwise return unchanged."""
    text = source
    text = _run_formatter(
        ['python3.10', '-m', 'black', '-q', '-'],
        text,
        label='black',
    )
    text = _run_formatter(
        ['python3.10', '-m', 'ruff', 'format', '-'],
        text,
        label='ruff format',
    )
    return text if text.endswith('\n') else text + '\n'


def _run_formatter(
    argv: List[str],
    source: str,
    *,
    label: str,
    accept_codes: tuple = (0,),
) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            argv,
            input=source,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.debug('Skipping {}: {}', label, exc)
        return source
    if proc.returncode not in accept_codes:
        logger.debug(
            'Skipping {} (exit {}): {}',
            label,
            proc.returncode,
            (proc.stderr or '')[:200],
        )
        return source
    out = proc.stdout
    return out if out else source


def write_artifacts(
    *,
    spec: ToolSpec,
    code: str,
    executor_name: str,
    model_name: Optional[str],
    attempt_count: int,
    example_results: List[Dict[str, Any]],
    work: Sequence[Example],
    holdout: Sequence[Example],
    base: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write impl/tool/assistant/spec/manifest under ``generated_tools/<name>/``.

    When *provenance* is provided (intake pipeline), also writes ``draft.md`` and
    ``interview.json`` as UTF-8 and records their hashes in the manifest.
    """
    out_dir = tool_artifact_dir(spec, base)
    out_dir.mkdir(parents=True, exist_ok=True)

    impl_text = code if code.endswith('\n') else code + '\n'
    (out_dir / 'impl.py').write_text(impl_text, encoding='utf-8')

    proxy = build_proxy_source(spec, executor_name=executor_name)
    (out_dir / 'tool.py').write_text(proxy, encoding='utf-8')

    assistant_name = assistant_tool_filename(spec)
    assistant_text = build_assistant_tool_source(spec, impl_text)
    (out_dir / assistant_name).write_text(assistant_text, encoding='utf-8')

    (out_dir / 'spec.json').write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    tests_payload = {
        'work': [_example_dict(ex) for ex in work],
        'holdout': [_example_dict(ex) for ex in holdout],
    }
    (out_dir / 'tests.json').write_text(
        json.dumps(tests_payload, indent=2, ensure_ascii=False, default=str) + '\n',
        encoding='utf-8',
    )

    manifest: Dict[str, Any] = {
        'spec_hash': spec_hash(spec),
        'impl_sha256': sha256_text(impl_text),
        'assistant_tool_file': assistant_name,
        'assistant_tool_sha256': sha256_text(assistant_text),
        'model': model_name,
        'backend': executor_name,
        'attempt_count': attempt_count,
        'example_results': example_results,
        'executor': executor_name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cat_agent_version': CAT_AGENT_VERSION,
        'registered_name': spec.registered_name,
        'function_name': spec.function_name,
        'requires_network': spec.requires_network,
        'spec_file': 'spec.json',
    }

    if provenance:
        draft_md = provenance.get('draft_markdown')
        if isinstance(draft_md, str):
            (out_dir / 'draft.md').write_text(draft_md, encoding='utf-8')
            manifest['draft_sha256'] = sha256_text(draft_md)
            manifest['draft_file'] = 'draft.md'
        interview_obj = provenance.get('interview')
        if interview_obj is not None:
            interview_text = json.dumps(
                interview_obj, indent=2, ensure_ascii=False, default=str
            ) + '\n'
            (out_dir / 'interview.json').write_text(interview_text, encoding='utf-8')
            manifest['interview_sha256'] = sha256_text(interview_text)
            manifest['interview_file'] = 'interview.json'
        if provenance.get('draft_lang'):
            manifest['draft_lang'] = provenance['draft_lang']
        if provenance.get('locale') is not None:
            manifest['locale'] = provenance['locale']
        _audit_intake_completed(spec, out_dir, manifest)

    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + '\n',
        encoding='utf-8',
    )
    return out_dir


def _audit_intake_completed(
    spec: ToolSpec,
    out_dir: Path,
    manifest: Dict[str, Any],
) -> None:
    from cat_agent.security.audit import append_audit_record, is_audit_enabled

    if not is_audit_enabled():
        return
    append_audit_record(
        'synthesis.intake_completed',
        {
            'name': spec.function_name,
            'registered_name': spec.registered_name,
            'artifact_dir': str(out_dir),
            'draft_sha256': manifest.get('draft_sha256'),
            'interview_sha256': manifest.get('interview_sha256'),
            'draft_lang': manifest.get('draft_lang'),
            'locale': manifest.get('locale'),
        },
    )

def _example_dict(example: Example) -> Dict[str, Any]:
    return {
        'inputs': example.inputs,
        'expected': example.expected,
        'note': example.note,
    }


def read_manifest(tool_dir: Path) -> Dict[str, Any]:
    return json.loads((tool_dir / 'manifest.json').read_text(encoding='utf-8'))


def verify_impl_hash(tool_dir: Path, manifest: Optional[Dict[str, Any]] = None) -> bool:
    manifest = manifest or read_manifest(tool_dir)
    impl = (tool_dir / 'impl.py').read_text(encoding='utf-8')
    return sha256_text(impl) == manifest.get('impl_sha256')
