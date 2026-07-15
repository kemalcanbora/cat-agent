"""Tool registration metadata for air-gap policy enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

from cat_agent.security.offline import is_offline_mode

if TYPE_CHECKING:
    from cat_agent.tools.base import BaseTool

TOOL_METADATA: Dict[str, Dict[str, bool]] = {}


def record_tool_metadata(
    name: str,
    *,
    requires_network: bool,
    cloud_service: bool,
    register_by_default: bool,
) -> None:
    TOOL_METADATA[name] = {
        'requires_network': requires_network,
        'cloud_service': cloud_service,
        'register_by_default': register_by_default,
    }


def is_tool_allowed_in_offline_mode(tool_name: str, tool_cls: Type['BaseTool'] | None = None) -> bool:
    if not is_offline_mode():
        return True
    meta = TOOL_METADATA.get(tool_name, {})
    if meta:
        return not meta.get('requires_network', False)
    if tool_cls is not None:
        return not getattr(tool_cls, 'requires_network', False)
    return True


def tool_requires_network(tool_name: str, tool_cls: Type['BaseTool'] | None = None) -> bool:
    meta = TOOL_METADATA.get(tool_name, {})
    if meta:
        return bool(meta.get('requires_network', False))
    if tool_cls is not None:
        return bool(getattr(tool_cls, 'requires_network', False))
    return False


def tool_is_cloud_service(tool_name: str, tool_cls: Type['BaseTool'] | None = None) -> bool:
    meta = TOOL_METADATA.get(tool_name, {})
    if meta:
        return bool(meta.get('cloud_service', False))
    if tool_cls is not None:
        return bool(getattr(tool_cls, 'cloud_service', False))
    return False
