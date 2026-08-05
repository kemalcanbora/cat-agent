"""Sandbox executors for synthesised tool code."""

from __future__ import annotations

from cat_agent.synthesis.executors.base import ExecResult, SandboxExecutor
from cat_agent.synthesis.executors.wasm import WasmExecutor

__all__ = [
    'ExecResult',
    'SandboxExecutor',
    'WasmExecutor',
    'get_executor',
]


def get_executor(name: str = 'wasm', **kwargs) -> SandboxExecutor:
    """Return a :class:`SandboxExecutor` by backend name."""
    key = (name or 'wasm').strip().lower()
    if key == 'wasm':
        return WasmExecutor(**kwargs)
    raise ValueError(
        f'Unknown sandbox executor {name!r}. Available: wasm. '
        'Docker/nono backends are deferred.'
    )
