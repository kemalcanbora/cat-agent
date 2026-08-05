"""Explicit loader for synthesised tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from cat_agent.log import logger
from cat_agent.synthesis.artifacts import (
    generated_tools_root,
    read_manifest,
    verify_impl_hash,
)
from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY, BaseTool


def load_generated_tools(
    path: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, BaseTool]:
    """Scan generated tool dirs, verify impl hashes, and register proxies.

    Never called automatically on import — call explicitly, then optionally
    ``enable_optional_tools(...)`` to move them into the active registry.
    """
    root = Path(path) if path else generated_tools_root()
    if not root.is_dir():
        logger.warning('No generated tools directory at {}', root)
        return {}

    wanted = set(names) if names else None
    loaded: Dict[str, BaseTool] = {}

    for tool_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        function_name = tool_dir.name
        if wanted is not None and function_name not in wanted and (
            f'generated_{function_name}' not in wanted
        ):
            continue

        tool_py = tool_dir / 'tool.py'
        manifest_path = tool_dir / 'manifest.json'
        impl_path = tool_dir / 'impl.py'
        if not (tool_py.is_file() and manifest_path.is_file() and impl_path.is_file()):
            logger.warning('Skipping incomplete generated tool dir {}', tool_dir)
            continue

        try:
            manifest = read_manifest(tool_dir)
        except Exception as exc:
            logger.warning('Skipping {}: cannot read manifest ({})', tool_dir, exc)
            continue

        if not verify_impl_hash(tool_dir, manifest):
            logger.warning(
                'Refusing to load generated tool {!r}: impl.py hash mismatch '
                '(file was edited after validation).',
                manifest.get('registered_name') or function_name,
            )
            continue

        registered_name = manifest.get('registered_name') or f'generated_{function_name}'
        module_name = f'cat_agent_generated_{function_name}'
        try:
            tool_obj = _import_proxy(tool_py, module_name)
        except Exception as exc:
            logger.warning('Failed to import {}: {}', tool_py, exc)
            continue

        loaded[registered_name] = tool_obj
        logger.info('Loaded generated tool {} from {}', registered_name, tool_dir)

    return loaded


def _import_proxy(tool_py: Path, module_name: str) -> BaseTool:
    spec = importlib.util.spec_from_file_location(module_name, tool_py)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {tool_py}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # The @tool decorator returns a BaseTool instance bound to the function name.
    for value in vars(module).values():
        if isinstance(value, BaseTool):
            return value
    raise RuntimeError(f'No BaseTool instance found in {tool_py}')


def list_generated_tool_names(path: Optional[str] = None) -> List[str]:
    root = Path(path) if path else generated_tools_root()
    if not root.is_dir():
        return []
    names = []
    for tool_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (tool_dir / 'manifest.json').is_file():
            names.append(tool_dir.name)
    return names


def is_generated_tool_registered(name: str) -> bool:
    return name in TOOL_REGISTRY or name in OPTIONAL_TOOL_REGISTRY
