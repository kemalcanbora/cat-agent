"""Tests for synthesised tool artifacts and loader."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from cat_agent.synthesis.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    build_assistant_tool_source,
    build_proxy_source,
    normalize_manifest,
    read_manifest,
    sha256_text,
    write_artifacts,
)
from cat_agent.synthesis.registry import load_generated_tools
from cat_agent.synthesis.spec import Example, ParameterSpec, ToolSpec
from cat_agent.tools.base import (
    OPTIONAL_TOOL_REGISTRY,
    TOOL_REGISTRY,
    ToolExecutionError,
    enable_optional_tools,
)


def _spec(name: str = 'artifact_add_one') -> ToolSpec:
    return ToolSpec(
        name=name,
        description='Return x + 1',
        parameters={'x': 'integer'},
        returns='integer',
        examples=[
            Example(inputs={'x': 1}, expected=2),
            Example(inputs={'x': 2}, expected=3),
            Example(inputs={'x': 3}, expected=4),
        ],
    )


_IMPL = '''\
def artifact_add_one(x: int) -> int:
    """Return x + 1.

    Args:
        x: Integer value.
    """
    return x + 1
'''


class TestArtifacts:

    def test_file_layout_and_manifest(self, tmp_path: Path):
        spec = _spec('artifact_layout_tool')
        impl = _IMPL.replace('artifact_add_one', spec.function_name)
        work, holdout = spec.split_examples()
        out = write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='test-model',
            attempt_count=2,
            example_results=[{'ok': True}],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        assert (out / 'impl.py').is_file()
        assert (out / 'tool.py').is_file()
        assert (out / f'{spec.function_name}.py').is_file()
        assert (out / 'spec.json').is_file()
        assert not (out / 'spec.yaml').exists()
        assert (out / 'manifest.json').is_file()
        assert (out / 'tests.json').is_file()
        text = (out / 'tool.py').read_text(encoding='utf-8')
        assert 'import impl' not in text
        assert 'register_by_default=False' in text
        assert 'requires_network=False' in text
        assert '_IMPL_PATH.read_text' in text
        assert 'ToolExecutionError' in text
        assistant = (out / f'{spec.function_name}.py').read_text(encoding='utf-8')
        assert '@tool(' in assistant
        assert f'def {spec.function_name}' in assistant
        assert 'return x + 1' in assistant
        assert '_IMPL_PATH' not in assistant
        assert 'get_executor' not in assistant
        compile(assistant, f'{spec.function_name}.py', 'exec')
        manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['spec_file'] == 'spec.json'
        assert 'impl_sha256' in manifest
        assert manifest['assistant_tool_file'] == f'{spec.function_name}.py'
        assert 'assistant_tool_sha256' in manifest
        assert 'test-model' in manifest.values() or manifest.get('model') == 'test-model'
        assert sha256_text(impl if impl.endswith('\n') else impl + '\n') == manifest['impl_sha256']
        payload = json.loads((out / 'spec.json').read_text(encoding='utf-8'))
        assert payload['name'] == spec.name

    def test_verification_block_and_v1_back_compat(self, tmp_path: Path):
        spec = _spec('artifact_verify_tool')
        impl = _IMPL.replace('artifact_add_one', spec.function_name)
        work, holdout = spec.split_examples()
        verification = {
            'code_mutation': {'killed': 10, 'total': 12, 'threshold': 0.8},
            'input_sensitivity': [
                {'param': 'iban', 'changed': 0, 'variants': 198},
                {'param': 'iban', 'changed': 0, 'variants': 198},
            ],
            'spec_warnings': [
                {'code': 'negatives_far_from_positives', 'severity': 'warn'},
            ],
            'warnings_overridden': False,
            'holdout_size': len(holdout),
        }
        out = write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='test-model',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
            provenance={'verification': verification},
        )
        manifest = read_manifest(out)
        assert manifest['schema_version'] == MANIFEST_SCHEMA_VERSION
        assert manifest['verification'] == verification

        # v1 manifests without verification still load.
        legacy = {
            'spec_hash': 'x',
            'impl_sha256': 'y',
            'registered_name': spec.registered_name,
        }
        normalised = normalize_manifest(legacy)
        assert normalised['schema_version'] == 1
        assert normalised['verification'] is None

    def test_writes_spec_json_without_pyyaml(self, tmp_path: Path, monkeypatch):
        real_import = builtins.__import__

        def _no_yaml(name, *args, **kwargs):
            if name == 'yaml' or (isinstance(name, str) and name.startswith('yaml.')):
                raise ImportError('PyYAML intentionally unavailable in this test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _no_yaml)
        spec = _spec('artifact_no_yaml_tool')
        impl = _IMPL.replace('artifact_add_one', spec.function_name)
        work, holdout = spec.split_examples()
        out = write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        data = json.loads((out / 'spec.json').read_text(encoding='utf-8'))
        assert data['name'] == spec.name
        assert (out / 'spec.json').is_file()
        assert not (out / 'spec.yaml').exists()

    def test_writes_spec_json_with_pyyaml(self, tmp_path: Path):
        pytest.importorskip('yaml')
        spec = _spec('artifact_with_yaml_tool')
        impl = _IMPL.replace('artifact_add_one', spec.function_name)
        work, holdout = spec.split_examples()
        out = write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        data = json.loads((out / 'spec.json').read_text(encoding='utf-8'))
        assert data['name'] == spec.name

    def test_proxy_has_no_impl_import(self):
        spec = _spec('artifact_proxy_tool')
        src = build_proxy_source(spec)
        assert 'import impl' not in src
        assert 'from impl' not in src
        assert 'ToolExecutionError' in src
        assert not src.startswith(' '), 'proxy must start at column 0'
        compile(src, '<proxy>', 'exec')

    def test_proxy_survives_multiline_description(self):
        """Regression: description newlines used to break textwrap.dedent."""
        spec = ToolSpec(
            name='proxy_multiline_desc',
            description='Split VAT.\n\nResolved decisions:\n- half up',
            parameters={
                'gross': ParameterSpec(type='number', description='Gross'),
                'rate': ParameterSpec(type='number', description='Rate'),
            },
            returns='object',
            examples=[
                Example(inputs={'gross': 1.0, 'rate': 0.2}, expected={'net': 0.83, 'tax': 0.17}),
                Example(inputs={'gross': 2.0, 'rate': 0.0}, expected={'net': 2.0, 'tax': 0.0}),
                Example(inputs={'gross': 3.0, 'rate': 0.1}, expected={'net': 2.73, 'tax': 0.27}),
            ],
        )
        src = build_proxy_source(spec)
        assert not src.startswith(' ')
        compile(src, 'tool.py', 'exec')
        assert 'Resolved decisions' in src
        assert 'gross: Gross' in src
        assert 'rate: Rate' in src

    def test_prose_returns_becomes_dict_annotation(self):
        """Model often writes returns as a sentence; signature must stay valid."""
        from cat_agent.synthesis.artifacts import _python_type, coerce_spec_type

        prose = (
            'object with two numeric fields, net and tax, '
            'each rounded to two decimal places.'
        )
        assert _python_type(prose) == 'dict'
        assert coerce_spec_type(prose) == 'object'

        spec = ToolSpec(
            name='vat_split_prose_ret',
            description='Split VAT',
            parameters={
                'gross': ParameterSpec(type='number', description='Gross'),
                'rate': ParameterSpec(type='number', description='Rate'),
            },
            returns=prose,
            examples=[
                Example(inputs={'gross': 1.0, 'rate': 0.2}, expected={'net': 0.83, 'tax': 0.17}),
                Example(inputs={'gross': 2.0, 'rate': 0.0}, expected={'net': 2.0, 'tax': 0.0}),
                Example(inputs={'gross': 3.0, 'rate': 0.1}, expected={'net': 2.73, 'tax': 0.27}),
            ],
        )
        src = build_proxy_source(spec)
        sig = next(line for line in src.splitlines() if line.startswith('def '))
        assert '-> dict:' in sig
        assert 'numeric' not in sig
        compile(src, 'tool.py', 'exec')

    def test_assistant_tool_is_self_contained(self):
        spec = _spec('sum_two_number')
        impl = '''\
def sum_two_number(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: The first number
        b: The second number
    """
    return f'The sum of {a} and {b} is {a + b}.'
'''
        src = build_assistant_tool_source(spec, impl)
        assert src.startswith('"""') or 'from cat_agent.tools import tool' in src
        assert '@tool(' in src
        assert 'def sum_two_number' in src
        assert 'return f' in src or 'a + b' in src
        assert 'impl.py' not in src
        assert 'get_executor' not in src
        assert '_IMPL_PATH' not in src
        compile(src, 'sum_two_number.py', 'exec')

    def test_param_descriptions_reach_function_schema(self, tmp_path: Path):
        pytest.importorskip('wasmtime')
        spec = ToolSpec(
            name='artifact_desc_tool',
            description='Add one to amount',
            parameters={
                'amount': ParameterSpec(
                    type='number',
                    description='Gross amount including tax',
                ),
            },
            returns='number',
            examples=[
                Example(inputs={'amount': 1.0}, expected=2.0),
                Example(inputs={'amount': 2.0}, expected=3.0),
                Example(inputs={'amount': 3.0}, expected=4.0),
            ],
        )
        impl = f'''\
def {spec.function_name}(amount: float) -> float:
    """Add one to amount.

    Args:
        amount: Gross amount including tax
    """
    return amount + 1.0
'''
        work, holdout = spec.split_examples()
        write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
        TOOL_REGISTRY.pop(spec.registered_name, None)
        loaded = load_generated_tools(path=str(tmp_path / 'generated_tools'))
        tool = loaded[spec.registered_name]
        props = tool.function['parameters']['properties']
        assert props['amount']['description'] == 'Gross amount including tax'

    def test_sandbox_failure_raises_tool_execution_error(self, tmp_path: Path):
        pytest.importorskip('wasmtime')
        spec = _spec('artifact_fail_tool')
        impl = f'''\
def {spec.function_name}(x: int) -> int:
    """Always boom.

    Args:
        x: Integer value.
    """
    raise RuntimeError("sandbox boom")
'''
        work, holdout = spec.split_examples()
        write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
        TOOL_REGISTRY.pop(spec.registered_name, None)
        loaded = load_generated_tools(path=str(tmp_path / 'generated_tools'))
        tool = loaded[spec.registered_name]
        with pytest.raises(ToolExecutionError) as exc_info:
            tool(1)
        assert exc_info.value.tool_name == spec.registered_name
        assert 'boom' in str(exc_info.value).lower() or 'error' in str(exc_info.value).lower()

    def test_missing_runtime_dir_readable(self, tmp_path: Path):
        from cat_agent.tools.wasm_code_interpreter import WasmPythonRuntime

        runtime = WasmPythonRuntime(str(tmp_path / 'empty_runtime'))
        with pytest.raises(FileNotFoundError, match='python.*\\.wasm|No python'):
            runtime.execute('print(1)')

    def test_missing_wasmtime_readable(self, monkeypatch):
        import cat_agent.tools.wasm_code_interpreter as wci

        def _boom():
            raise ImportError(
                'The wasmtime package is required for the WASM code interpreter. '
                'Install it with: pip install "cat-agent[wasm]"'
            )

        monkeypatch.setattr(wci, '_check_wasmtime_available', _boom)
        with pytest.raises(ImportError, match='wasmtime'):
            wci.WasmCodeInterpreter({})

    def test_hash_mismatch_refuses_load(self, tmp_path: Path):
        spec = _spec('artifact_hash_tool')
        impl = _IMPL.replace('artifact_add_one', spec.function_name)
        work, holdout = spec.split_examples()
        out = write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        # Tamper with impl.py
        (out / 'impl.py').write_text(impl + '\n# tampered\n', encoding='utf-8')
        loaded = load_generated_tools(path=str(tmp_path / 'generated_tools'))
        assert spec.registered_name not in loaded

    def test_register_by_default_false(self, tmp_path: Path):
        pytest.importorskip('wasmtime')
        spec = _spec('artifact_opt_in_tool')
        impl = f'''\
def {spec.function_name}(x: int) -> int:
    """Return x + 1.

    Args:
        x: Integer value.
    """
    return x + 1
'''
        work, holdout = spec.split_examples()
        write_artifacts(
            spec=spec,
            code=impl,
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
        )
        # Ensure clean slate
        OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
        TOOL_REGISTRY.pop(spec.registered_name, None)

        loaded = load_generated_tools(path=str(tmp_path / 'generated_tools'))
        assert spec.registered_name in loaded
        assert spec.registered_name in OPTIONAL_TOOL_REGISTRY
        assert spec.registered_name not in TOOL_REGISTRY

        enable_optional_tools(spec.registered_name)
        assert spec.registered_name in TOOL_REGISTRY
