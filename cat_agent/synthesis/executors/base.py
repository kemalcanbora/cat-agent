"""Sandbox executor protocol and shared result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


RESULT_SENTINEL = '__CAT_RESULT__'
ERROR_SENTINEL = '__CAT_ERROR__'


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    error: Optional[str]
    returned: Any = None
    duration_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SandboxExecutor(Protocol):
    name: str
    supports_dependencies: bool

    def run(
        self,
        code: str,
        inputs: Dict[str, Any],
        deps: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
        *,
        function_name: str = 'main',
    ) -> ExecResult:
        ...
