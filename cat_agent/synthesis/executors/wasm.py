"""WASM backend for :class:`SandboxExecutor`."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from cat_agent.log import logger
from cat_agent.synthesis.executors.base import ExecResult
from cat_agent.synthesis.harness import (
    build_harness,
    map_timeout_to_fuel,
    parse_harness_stdout,
)
from cat_agent.tools.wasm_code_interpreter import (
    DEFAULT_FUEL,
    DEFAULT_MAX_OUTPUT_BYTES,
    WasmPythonRuntime,
)
from cat_agent.tools.resource.wasm_runtime_loader import ensure_wasm_runtime

# Harness (import json + loads/dumps) costs ~2x a trivial one-liner under WASI CPython.
_DEFAULT_HARNESS_FUEL = DEFAULT_FUEL * 2


class WasmExecutor:
    """Run synthesised Python inside the WASI CPython sandbox."""

    name = 'wasm'
    supports_dependencies = False

    def __init__(
        self,
        *,
        runtime_dir: Optional[str] = None,
        fuel: Optional[int] = None,
        max_output_bytes: Optional[int] = None,
    ):
        self._runtime_dir = ensure_wasm_runtime(runtime_dir)
        self._fuel = fuel if fuel is not None else _DEFAULT_HARNESS_FUEL
        self._max_output_bytes = (
            max_output_bytes if max_output_bytes is not None else DEFAULT_MAX_OUTPUT_BYTES
        )
        self._runtime: Optional[WasmPythonRuntime] = None
        self._runtime_lock = threading.Lock()

    def _get_runtime(self) -> WasmPythonRuntime:
        if self._runtime is None:
            with self._runtime_lock:
                if self._runtime is None:
                    self._runtime = WasmPythonRuntime(self._runtime_dir)
        return self._runtime

    def run(
        self,
        code: str,
        inputs: Dict[str, Any],
        deps: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
        *,
        function_name: str = 'main',
        fuel: Optional[int] = None,
    ) -> ExecResult:
        if deps:
            return ExecResult(
                ok=False,
                stdout='',
                stderr='',
                error=(
                    'WasmExecutor does not support third-party dependencies '
                    f'(requested: {deps!r}). Use a backend with '
                    'supports_dependencies=True when available.'
                ),
            )

        fuel_budget = fuel if fuel is not None else map_timeout_to_fuel(timeout_s, self._fuel)
        harness = build_harness(code, function_name, inputs)
        started = time.perf_counter()
        try:
            raw = self._get_runtime().execute(
                harness,
                fuel=fuel_budget,
                max_output_bytes=self._max_output_bytes,
            )
        except Exception as exc:  # pragma: no cover - runtime bootstrap failures
            logger.exception('WasmExecutor failed to start runtime')
            return ExecResult(
                ok=False,
                stdout='',
                stderr='',
                error=f'WasmExecutor runtime error: {exc}',
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        duration_ms = (time.perf_counter() - started) * 1000
        stdout = raw.get('stdout') or ''
        stderr = raw.get('stderr') or ''
        error = raw.get('error')
        meta = {
            'fuel_consumed': raw.get('fuel_consumed'),
            'fuel_budget': fuel_budget,
            'stdout_truncated': bool(raw.get('stdout_truncated')),
            'stderr_truncated': bool(raw.get('stderr_truncated')),
            'truncated': bool(raw.get('stdout_truncated') or raw.get('stderr_truncated')),
        }

        if error:
            return ExecResult(
                ok=False,
                stdout=stdout,
                stderr=stderr,
                error=error,
                duration_ms=duration_ms,
                meta=meta,
            )

        try:
            clean_stdout, returned = parse_harness_stdout(stdout)
        except ValueError as exc:
            return ExecResult(
                ok=False,
                stdout=stdout,
                stderr=stderr,
                error=str(exc),
                duration_ms=duration_ms,
                meta=meta,
            )

        return ExecResult(
            ok=True,
            stdout=clean_stdout,
            stderr=stderr,
            error=None,
            returned=returned,
            duration_ms=duration_ms,
            meta=meta,
        )
