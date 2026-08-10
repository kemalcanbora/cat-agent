"""Explicit loader for synthesised tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from cat_agent.log import logger
from cat_agent.security.principal import Principal, owner_registered_name
from cat_agent.synthesis.artifacts import (
    active_pointers_path,
    artifact_version_dir,
    artifact_version_dir_for_group,
    generated_tools_root,
    parse_active_pointer_key,
    read_json_pointers,
    read_manifest,
    verify_impl_hash,
    warn_legacy_generated_tools,
)
from cat_agent.tools.base import (
    OPTIONAL_TOOL_REGISTRY,
    TOOL_REGISTRY,
    BaseTool,
    is_generated_tool_name,
    tool_allowed_for_group,
)


class AdoptedToolError(RuntimeError):
    """Raised when an adopted pointer cannot be loaded (revoked / missing)."""


def tools_for_principal(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, BaseTool]:
    """Resolve tool classes visible to *principal*.

    Built-in (non-``generated_*``) tools in ``TOOL_REGISTRY`` are shared.
    Generated tools are included when enabled and either owned by
    ``principal.group_id`` or adopted into that group. Opt-in / demoted
    entries in ``OPTIONAL_TOOL_REGISTRY`` are not resolvable until re-enabled.
    """
    if principal is None:
        raise ValueError('tools_for_principal requires a Principal')
    from cat_agent.synthesis.share import adopted_registered_names

    adopted = adopted_registered_names(principal, workspace=workspace)

    out: Dict[str, BaseTool] = {}
    for name, cls in TOOL_REGISTRY.items():
        if not is_generated_tool_name(name):
            out[name] = cls
            continue
        if tool_allowed_for_group(name, principal.group_id):
            out[name] = cls
            continue
        if name in adopted:
            out[name] = cls
    return out


def load_generated_tools(
    principal: Optional[Principal] = None,
    path: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, BaseTool]:
    """Scan active pointers (or *path*), verify hashes, register proxies.

    Never called automatically on import — call explicitly, then optionally
    ``enable_optional_tools(...)`` to move them into the active registry.

    *path* overrides the scan root (kept for tests / legacy flat layouts).
    When *path* is omitted, *principal* is required and only versions listed
    in that group's ``active.json`` are loaded — staging is never loaded.
    Adopted pointers (``owner/tool``) load from the owning group's artifacts
    without copying; revoked shares raise :class:`AdoptedToolError`.
    """
    if path is not None:
        return _load_from_directory(Path(path), names=names, group_id=None)

    if principal is None:
        raise ValueError(
            'load_generated_tools requires a Principal when path is not set. '
            'There is no default group.'
        )
    warn_legacy_generated_tools(workspace)
    from cat_agent.synthesis.share import auto_adopt_org_shared

    auto_adopt_org_shared(principal, workspace=workspace)
    return _load_from_active_pointers(principal, names=names, workspace=workspace)


def _load_from_active_pointers(
    principal: Principal,
    *,
    names: Optional[Sequence[str]],
    workspace: Optional[str],
) -> Dict[str, BaseTool]:
    from cat_agent.synthesis.share import is_shared_with, revocation_reason

    pointers = read_json_pointers(active_pointers_path(principal, workspace))
    if not pointers:
        logger.warning(
            'No active tools for group {} (missing or empty active.json)',
            principal.group_id,
        )
        return {}
    wanted = set(names) if names else None
    loaded: Dict[str, BaseTool] = {}
    for key, version in sorted(pointers.items()):
        owner, tool_name = parse_active_pointer_key(key)
        if wanted is not None and tool_name not in wanted and key not in wanted and (
            f'generated_{tool_name}' not in wanted
            and not any(w.endswith(f'_{tool_name}') for w in wanted)
        ):
            continue

        if owner is None:
            tool_dir = artifact_version_dir(
                principal, tool_name, version, workspace,
            )
            group_id = principal.group_id
        else:
            if not is_shared_with(
                owner, tool_name, principal.group_id, workspace=workspace,
            ):
                reason = revocation_reason(
                    owner, tool_name, principal.group_id, workspace=workspace,
                )
                reason_text = (
                    f'reason={reason!r}'
                    if reason is not None
                    else 'reason=(none recorded)'
                )
                raise AdoptedToolError(
                    f'Refused to load adopted tool {owner}/{tool_name} '
                    f'(owning group {owner!r}): share revoked or missing '
                    f'({reason_text}). Remove the pointer from active.json '
                    f'or ask the owner to re-share.'
                )
            tool_dir = artifact_version_dir_for_group(
                owner, tool_name, version, workspace,
            )
            if not tool_dir.is_dir():
                raise AdoptedToolError(
                    f'Refused to load adopted tool {owner}/{tool_name}@{version}: '
                    f'artifact missing at {tool_dir} (owning group {owner!r}).'
                )
            group_id = owner

        tool_obj = _load_one_tool_dir(
            tool_dir,
            function_name=tool_name,
            group_id=group_id,
        )
        if tool_obj is None:
            if owner is not None:
                raise AdoptedToolError(
                    f'Refused to load adopted tool {owner}/{tool_name}@{version}: '
                    f'incomplete or hash-mismatched artifact (owning group {owner!r}).'
                )
            continue
        loaded[tool_obj.name] = tool_obj
    return loaded


def _load_from_directory(
    root: Path,
    *,
    names: Optional[Sequence[str]],
    group_id: Optional[str],
) -> Dict[str, BaseTool]:
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
            if not any(w.endswith(f'_{function_name}') for w in wanted):
                continue
        tool_obj = _load_one_tool_dir(
            tool_dir, function_name=function_name, group_id=group_id,
        )
        if tool_obj is None:
            continue
        loaded[tool_obj.name] = tool_obj
    return loaded


def _load_one_tool_dir(
    tool_dir: Path,
    *,
    function_name: str,
    group_id: Optional[str],
) -> Optional[BaseTool]:
    tool_py = tool_dir / 'tool.py'
    manifest_path = tool_dir / 'manifest.json'
    impl_path = tool_dir / 'impl.py'
    if not (tool_py.is_file() and manifest_path.is_file() and impl_path.is_file()):
        logger.warning('Skipping incomplete generated tool dir {}', tool_dir)
        return None
    try:
        manifest = read_manifest(tool_dir)
    except Exception as exc:
        logger.warning('Skipping {}: cannot read manifest ({})', tool_dir, exc)
        return None
    if not verify_impl_hash(tool_dir, manifest):
        logger.warning(
            'Refusing to load generated tool {!r}: impl.py hash mismatch '
            '(file was edited after validation).',
            manifest.get('registered_name') or function_name,
        )
        return None

    registered_name = manifest.get('registered_name') or f'generated_{function_name}'
    if group_id and not registered_name.startswith(f'generated_{group_id}_'):
        registered_name = owner_registered_name(group_id, function_name)

    existing = TOOL_REGISTRY.get(registered_name) or OPTIONAL_TOOL_REGISTRY.get(
        registered_name,
    )
    if existing is not None:
        # Already loaded (owner and consumer in the same process).
        if isinstance(existing, BaseTool):
            return existing
        try:
            return existing()  # type: ignore[misc]
        except Exception:
            logger.info(
                'Reusing registered class for {} without new instance',
                registered_name,
            )
            return existing  # type: ignore[return-value]

    module_key = registered_name.replace('.', '_')
    module_name = f'cat_agent_generated_{module_key}'
    try:
        tool_obj = _import_proxy(tool_py, module_name)
    except Exception as exc:
        logger.warning('Failed to import {}: {}', tool_py, exc)
        return None

    if getattr(tool_obj, 'name', None) != registered_name:
        tool_obj.name = registered_name
    logger.info('Loaded generated tool {} from {}', registered_name, tool_dir)
    return tool_obj


def _import_proxy(tool_py: Path, module_name: str) -> BaseTool:
    spec = importlib.util.spec_from_file_location(module_name, tool_py)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {tool_py}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for value in vars(module).values():
        if isinstance(value, BaseTool):
            return value
    raise RuntimeError(f'No BaseTool instance found in {tool_py}')


def list_generated_tool_names(
    principal: Optional[Principal] = None,
    path: Optional[str] = None,
    *,
    workspace: Optional[str] = None,
) -> List[str]:
    if path is not None:
        root = Path(path)
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / 'manifest.json').is_file()
        )
    if principal is not None:
        return sorted(read_json_pointers(active_pointers_path(principal, workspace)).keys())
    root = generated_tools_root(workspace)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / 'manifest.json').is_file()
    )


def is_generated_tool_registered(name: str) -> bool:
    return name in TOOL_REGISTRY or name in OPTIONAL_TOOL_REGISTRY
