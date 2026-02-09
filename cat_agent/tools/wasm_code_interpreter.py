"""
WASM-sandboxed Python code interpreter powered by wasmtime.

Runs agent-generated Python code inside a WebAssembly sandbox using the
``wasmtime`` Python package and a WASI build of CPython.  The WASM
sandbox provides memory isolation with **no** host filesystem or network
access by default — without requiring Docker or Node.js.

Requirements:
    pip install wasmtime

The WASI CPython binary and standard library are bundled under
``cat_agent/tools/resource/wasm_runtime/`` — no extra downloads needed.
You can override the runtime directory via the ``runtime_dir`` config key.

Limitations:
    - Only the Python **standard library** is available (no numpy/pandas/
      matplotlib).
    - Each execution starts fresh (no persistent state between calls).
    - Fuel-based execution limits prevent infinite loops.
"""

import glob
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import json5

from cat_agent.log import logger
from cat_agent.tools.base import BaseTool, register_tool
from cat_agent.utils.utils import extract_code, has_chinese_chars

# Default fuel budget — enough for most reasonable computations.
# 400M fuel ≈ a few seconds of CPU work; raise for heavier tasks.
DEFAULT_FUEL = 400_000_000

# Bundled WASI CPython runtime shipped with the package.
BUNDLED_RUNTIME_DIR = str(Path(__file__).absolute().parent / 'resource' / 'wasm_runtime')


# ---------------------------------------------------------------------------
# WASM runtime manager
# ---------------------------------------------------------------------------

class WasmPythonRuntime:
    """Manages a cached wasmtime engine + pre-compiled CPython WASM module."""

    def __init__(self, runtime_dir: str):
        self.runtime_dir = runtime_dir
        self._engine = None
        self._module = None
        self._wasm_path: Optional[str] = None

    def _find_wasm_binary(self) -> str:
        """Locate the python*.wasm file anywhere under runtime_dir."""
        if self._wasm_path and os.path.isfile(self._wasm_path):
            return self._wasm_path

        # Search for python*.wasm recursively
        pattern = os.path.join(self.runtime_dir, '**', 'python*.wasm')
        candidates = glob.glob(pattern, recursive=True)

        if not candidates:
            raise FileNotFoundError(
                f'No python*.wasm file found under {self.runtime_dir}.\n'
                'The bundled runtime may be missing or corrupted. '
                'Re-install the package or set "runtime_dir" in the tool config.'
            )

        self._wasm_path = candidates[0]
        logger.info('Found WASM binary: {}', self._wasm_path)
        return self._wasm_path

    def _get_engine_and_module(self):
        """Return a cached (Engine, Module) pair, creating them on first call."""
        if self._engine is not None and self._module is not None:
            return self._engine, self._module

        from wasmtime import Config, Engine, Module

        wasm_path = self._find_wasm_binary()

        cfg = Config()
        cfg.consume_fuel = True
        cfg.cache = True

        self._engine = Engine(cfg)

        logger.info('Compiling Python WASM module (cached after first load) ...')
        self._module = Module.from_file(self._engine, wasm_path)

        return self._engine, self._module

    # -- execution ------------------------------------------------------------

    def execute(self, code: str, fuel: int = DEFAULT_FUEL) -> dict:
        """Run *code* inside the WASM sandbox and return results.

        Returns a dict with keys: stdout, stderr, error, fuel_consumed.
        """
        from wasmtime import Linker, Store, WasiConfig

        engine, module = self._get_engine_and_module()

        linker = Linker(engine)
        linker.define_wasi()

        wasi_cfg = WasiConfig()
        wasi_cfg.argv = ('python', '-c', code)
        # Map the runtime directory (contains lib/) to / inside the sandbox.
        # This gives CPython access to its stdlib but nothing else on the host.
        wasi_cfg.preopen_dir(self.runtime_dir, '/')

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout_path = os.path.join(tmpdir, 'stdout.log')
            stderr_path = os.path.join(tmpdir, 'stderr.log')
            wasi_cfg.stdout_file = stdout_path
            wasi_cfg.stderr_file = stderr_path

            store = Store(engine)
            store.set_fuel(fuel)
            store.set_wasi(wasi_cfg)

            instance = linker.instantiate(store, module)
            start_fn = instance.exports(store)['_start']

            error = None
            try:
                start_fn(store)
            except Exception as exc:
                error_msg = str(exc)
                # Make the fuel-exhaustion message friendlier
                if 'all fuel consumed' in error_msg:
                    error = 'Timeout: Code execution exceeded the fuel (instruction) limit.'
                else:
                    error = error_msg

            stdout = ''
            stderr = ''
            if os.path.isfile(stdout_path):
                with open(stdout_path) as f:
                    stdout = f.read()
            if os.path.isfile(stderr_path):
                with open(stderr_path) as f:
                    stderr = f.read()

            return {
                'stdout': stdout.rstrip('\n'),
                'stderr': stderr.rstrip('\n'),
                'error': error,
                'fuel_consumed': fuel - store.get_fuel(),
            }


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

@register_tool('wasm_code_interpreter')
class WasmCodeInterpreter(BaseTool):
    """Sandboxed Python code executor using WebAssembly (wasmtime).

    Runs code inside a WASI sandbox with no host filesystem or network
    access.  Only requires ``pip install wasmtime`` — no Docker or
    Node.js needed.

    Available packages: Python standard library (json, math, re,
    sqlite3, itertools, collections, datetime, etc.).
    """

    description = (
        'Sandboxed Python code executor powered by WebAssembly (WASI). '
        'Supports the full Python standard library.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'description': 'The Python code to execute.',
                'type': 'string',
            },
        },
        'required': ['code'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.runtime_dir: str = self.cfg.get('runtime_dir', BUNDLED_RUNTIME_DIR)
        self.fuel: int = self.cfg.get('fuel', DEFAULT_FUEL)
        self._runtime: Optional[WasmPythonRuntime] = None
        _check_wasmtime_available()

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars(
                [self.name_for_human, self.name, self.description, self.parameters]
            ):
                fmt = 'The input for this tool should be a Markdown code block.'
            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def _get_runtime(self) -> WasmPythonRuntime:
        if self._runtime is None:
            self._runtime = WasmPythonRuntime(self.runtime_dir)
        return self._runtime

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = json5.loads(params)
            code = params['code']
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return ''

        runtime = self._get_runtime()
        result = runtime.execute(code, fuel=self.fuel)
        return self._format_result(result)

    @staticmethod
    def _format_result(result: dict) -> str:
        """Turn a WASM execution result into a human-readable string."""
        parts: List[str] = []

        if result.get('error'):
            parts.append(f'error:\n\n```\n{result["error"]}\n```')

        if result.get('stdout'):
            parts.append(f'stdout:\n\n```\n{result["stdout"]}\n```')

        if result.get('stderr'):
            parts.append(f'stderr:\n\n```\n{result["stderr"]}\n```')

        output = '\n\n'.join(parts)
        return output if output.strip() else 'Finished execution.'


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_wasmtime_available():
    """Verify that the wasmtime package is installed."""
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        raise ImportError(
            'The wasmtime package is required for the WASM code interpreter. '
            'Install it with: pip install wasmtime'
        )
