"""Cross-group tool sharing: share → adopt (two-sided consent)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from cat_agent.log import logger
from cat_agent.security.principal import (
    Principal,
    owner_registered_name,
    validate_group_id,
)
from cat_agent.synthesis.artifacts import (
    ORG_SHARE_TARGET,
    active_pointers_path,
    adopted_pointer_key,
    adoptions_path,
    artifact_version_dir,
    artifact_version_dir_for_group,
    auto_adopt_org_tools_enabled,
    groups_root,
    parse_active_pointer_key,
    read_json_object,
    read_json_pointers,
    read_manifest,
    shares_path,
    shares_path_for_group,
    verify_impl_hash,
    warn_legacy_generated_tools,
    write_json_object,
    write_json_pointers,
)
from cat_agent.synthesis.promote import (
    _ensure_writable,
    _make_tree_readonly,
    list_active_tools,
)


def read_shares(group_id: str, *, workspace: Optional[str] = None) -> Dict[str, Any]:
    return read_json_object(shares_path_for_group(group_id, workspace))


def read_adoptions(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    return read_json_object(adoptions_path(principal, workspace))


def is_shared_with(
    owner_group: str,
    tool_name: str,
    consumer_group: str,
    *,
    workspace: Optional[str] = None,
) -> bool:
    """True when *consumer_group* may adopt *tool_name* from *owner_group*."""
    shares = read_shares(owner_group, workspace=workspace)
    entry = shares.get(tool_name) or {}
    targets = list(entry.get('shared_with') or [])
    if consumer_group in targets:
        return True
    if ORG_SHARE_TARGET in targets:
        return True
    return False


def revocation_reason(
    owner_group: str,
    tool_name: str,
    consumer_group: str,
    *,
    workspace: Optional[str] = None,
) -> Optional[str]:
    shares = read_shares(owner_group, workspace=workspace)
    entry = shares.get(tool_name) or {}
    for rev in reversed(list(entry.get('revocations') or [])):
        if not isinstance(rev, dict):
            continue
        if rev.get('group') == consumer_group or rev.get('group') == ORG_SHARE_TARGET:
            reason = rev.get('reason')
            return str(reason) if reason is not None else ''
    return None


def share(
    principal: Principal,
    tool_name: str,
    *,
    with_groups: Sequence[str],
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark an active tool as offerable to *with_groups* (publisher side)."""
    warn_legacy_generated_tools(workspace)
    active = list_active_tools(principal, workspace=workspace)
    if tool_name not in active or '/' in tool_name:
        raise FileNotFoundError(
            f'No owned active tool {tool_name!r} for group {principal.group_id!r}'
        )
    version = active[tool_name]
    targets = _normalize_share_targets(with_groups, publisher=principal.group_id)

    shares = read_shares(principal.group_id, workspace=workspace)
    entry = dict(shares.get(tool_name) or {})
    existing = list(entry.get('shared_with') or [])
    merged = sorted(set(existing) | set(targets))
    now = datetime.now(timezone.utc).isoformat()
    entry.update({
        'shared_with': merged,
        'shared_by': principal.user_id,
        'shared_at': now,
        'active_version_at_share': version,
    })
    entry.setdefault('revocations', [])
    shares[tool_name] = entry
    write_json_object(shares_path(principal, workspace), shares)

    tool_dir = artifact_version_dir(principal, tool_name, version, workspace)
    _append_sharing_on_manifest(
        tool_dir,
        shared_with=merged,
        shared_by=principal.user_id,
        shared_at=now,
    )
    logger.info(
        'Shared {}@{} from {} with {}',
        tool_name, version, principal.group_id, merged,
    )
    return entry


def unshare(
    principal: Principal,
    tool_name: str,
    *,
    with_groups: Sequence[str],
    reason: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Revoke offerability for *with_groups* (publisher side)."""
    warn_legacy_generated_tools(workspace)
    shares = read_shares(principal.group_id, workspace=workspace)
    if tool_name not in shares:
        raise FileNotFoundError(
            f'Tool {tool_name!r} is not shared by group {principal.group_id!r}'
        )
    targets = _normalize_share_targets(with_groups, publisher=principal.group_id)
    entry = dict(shares[tool_name])
    current = list(entry.get('shared_with') or [])
    remaining = [g for g in current if g not in targets]
    now = datetime.now(timezone.utc).isoformat()
    revocations = list(entry.get('revocations') or [])
    for g in targets:
        if g in current or g == ORG_SHARE_TARGET:
            revocations.append({
                'group': g,
                'by': principal.user_id,
                'at': now,
                'reason': reason or '',
            })
    entry['shared_with'] = remaining
    entry['revocations'] = revocations
    entry['shared_by'] = principal.user_id
    entry['shared_at'] = entry.get('shared_at') or now
    shares[tool_name] = entry
    write_json_object(shares_path(principal, workspace), shares)

    active = list_active_tools(principal, workspace=workspace)
    if tool_name in active:
        tool_dir = artifact_version_dir(
            principal, tool_name, active[tool_name], workspace,
        )
        _append_sharing_on_manifest(
            tool_dir,
            shared_with=remaining,
            shared_by=principal.user_id,
            shared_at=now,
            revocations=revocations,
        )
    logger.info(
        'Unshared {} from {} for {} (reason={!r})',
        tool_name, principal.group_id, targets, reason or '',
    )
    return entry


def adopt(
    principal: Principal,
    ref: str,
    *,
    version: str,
    workspace: Optional[str] = None,
    yes: bool = False,
    confirm: Optional[Callable[[str], bool]] = None,
    automatic: bool = False,
) -> Dict[str, Any]:
    """Pin an offered foreign tool into this group's active.json.

    *ref* is ``owner_group/tool_name``. Artifacts are not copied.
    """
    warn_legacy_generated_tools(workspace)
    if not version or not str(version).strip():
        raise ValueError('adopt requires an explicit --version (content hash pin)')
    version = str(version).strip()
    owner_group, tool_name = _parse_adopt_ref(ref)
    if owner_group == principal.group_id:
        raise ValueError('Cannot adopt a tool owned by your own group; use promote')

    if not is_shared_with(
        owner_group, tool_name, principal.group_id, workspace=workspace,
    ):
        raise PermissionError(
            f'Tool {owner_group}/{tool_name} is not shared with group '
            f'{principal.group_id!r} (and not org-shared)'
        )

    tool_dir = artifact_version_dir_for_group(
        owner_group, tool_name, version, workspace,
    )
    if not tool_dir.is_dir():
        raise FileNotFoundError(
            f'No artifact for {owner_group}/{tool_name}@{version} at {tool_dir}'
        )
    manifest = read_manifest(tool_dir)
    if not verify_impl_hash(tool_dir, manifest):
        raise ValueError(
            f'Refusing to adopt {owner_group}/{tool_name}@{version}: '
            'impl.py hash mismatch'
        )

    summary = format_adopt_review(
        owner_group=owner_group,
        tool_name=tool_name,
        version=version,
        tool_dir=tool_dir,
        manifest=manifest,
    )
    confirmation_skipped = bool(yes) or bool(automatic)
    if not confirmation_skipped:
        ask = confirm or _default_adopt_confirm
        if not ask(summary):
            raise RuntimeError('Adoption cancelled by operator')

    key = adopted_pointer_key(owner_group, tool_name)
    active = dict(read_json_pointers(active_pointers_path(principal, workspace)))
    active[key] = version
    write_json_pointers(active_pointers_path(principal, workspace), active)

    record = {
        'owner_group': owner_group,
        'tool_name': tool_name,
        'version': version,
        'adopted_by': principal.user_id,
        'adopted_at': datetime.now(timezone.utc).isoformat(),
        'confirmation_skipped': confirmation_skipped,
        'automatic': bool(automatic),
        'registered_name': owner_registered_name(owner_group, tool_name),
        'impl_sha256': manifest.get('impl_sha256'),
    }
    adoptions = read_adoptions(principal, workspace=workspace)
    adoptions[key] = record
    write_json_object(adoptions_path(principal, workspace), adoptions)
    logger.info(
        'Adopted {}@{} into group {} as {} (by {}, automatic={})',
        key, version, principal.group_id, record['registered_name'],
        principal.user_id, automatic,
    )
    return record


def format_adopt_review(
    *,
    owner_group: str,
    tool_name: str,
    version: str,
    tool_dir: Path,
    manifest: Dict[str, Any],
) -> str:
    """Publisher evidence shown before adopt confirmation."""
    verification = manifest.get('verification') or {}
    promotion = manifest.get('promotion') or {}
    lines = [
        f'Adopt review for {owner_group}/{tool_name}@{version}',
        f'  artifact: {tool_dir}',
        f'  registered_name: {owner_registered_name(owner_group, tool_name)}',
        f'  impl_sha256: {manifest.get("impl_sha256")}',
        '',
        '  verification:',
        f'    {json.dumps(verification, indent=6, default=str)}',
        '',
        '  promotion:',
        f'    {json.dumps(promotion, indent=6, default=str)}',
        '',
        'This is information for the adopting group — not an enforcement gate.',
    ]
    return '\n'.join(lines)


def auto_adopt_org_shared(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Adopt currently org-shared tools when group setting enables it."""
    if not auto_adopt_org_tools_enabled(principal.group_id, workspace):
        return []
    adopted: List[Dict[str, Any]] = []
    root = groups_root(workspace)
    if not root.is_dir():
        return adopted
    existing = read_json_pointers(active_pointers_path(principal, workspace))
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        owner = group_dir.name
        if owner == principal.group_id:
            continue
        try:
            validate_group_id(owner)
        except Exception:  # noqa: BLE001
            continue
        shares = read_shares(owner, workspace=workspace)
        owner_active = read_json_pointers(
            groups_root(workspace) / owner / 'active.json',
        )
        for tool_name, entry in shares.items():
            targets = list((entry or {}).get('shared_with') or [])
            if ORG_SHARE_TARGET not in targets:
                continue
            if tool_name not in owner_active or '/' in tool_name:
                continue
            key = adopted_pointer_key(owner, tool_name)
            if key in existing:
                continue
            version = owner_active[tool_name]
            try:
                record = adopt(
                    principal,
                    f'{owner}/{tool_name}',
                    version=version,
                    workspace=workspace,
                    yes=True,
                    automatic=True,
                )
                adopted.append(record)
                existing[key] = version
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'auto_adopt_org_tools skipped {}/{}@{}: {}',
                    owner, tool_name, version, exc,
                )
    return adopted


def owned_and_adopted_pointers(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Split active.json into owned ``{tool: ver}`` and adopted ``{owner/tool: ver}``."""
    raw = read_json_pointers(active_pointers_path(principal, workspace))
    owned: Dict[str, str] = {}
    adopted: Dict[str, str] = {}
    for key, version in raw.items():
        owner, tool = parse_active_pointer_key(key)
        if owner is None:
            owned[tool] = version
        else:
            adopted[key] = version
    return owned, adopted


def adopted_registered_names(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Set[str]:
    _, adopted = owned_and_adopted_pointers(principal, workspace=workspace)
    names: Set[str] = set()
    for key in adopted:
        owner, tool = parse_active_pointer_key(key)
        if owner:
            names.add(owner_registered_name(owner, tool))
    return names


def collect_external_pins(
    owner_group: str,
    tool_name: str,
    *,
    workspace: Optional[str] = None,
) -> Set[str]:
    """Versions of *owner_group/tool_name* pinned by other groups' active.json."""
    pins: Set[str] = set()
    root = groups_root(workspace)
    if not root.is_dir():
        return pins
    key = adopted_pointer_key(owner_group, tool_name)
    for group_dir in root.iterdir():
        if not group_dir.is_dir() or group_dir.name == owner_group:
            continue
        active = read_json_pointers(group_dir / 'active.json')
        if key in active:
            pins.add(str(active[key]))
    return pins


def upstream_shared_version(
    owner_group: str,
    tool_name: str,
    *,
    workspace: Optional[str] = None,
) -> Optional[str]:
    """Owning group's current active version, if still shared."""
    active = read_json_pointers(groups_root(workspace) / owner_group / 'active.json')
    return active.get(tool_name)


def _normalize_share_targets(
    with_groups: Sequence[str],
    *,
    publisher: str,
) -> List[str]:
    if not with_groups:
        raise ValueError('Provide at least one target group via --with')
    out: List[str] = []
    for raw in with_groups:
        for part in str(raw).split(','):
            g = part.strip()
            if not g:
                continue
            if g == ORG_SHARE_TARGET:
                out.append(ORG_SHARE_TARGET)
                continue
            gid = validate_group_id(g)
            if gid == publisher:
                raise ValueError('Cannot share a tool with the owning group itself')
            out.append(gid)
    if not out:
        raise ValueError('Provide at least one target group via --with')
    return sorted(set(out))


def _parse_adopt_ref(ref: str) -> Tuple[str, str]:
    raw = (ref or '').strip()
    if '/' not in raw:
        raise ValueError(
            f'Adopt ref must be owner_group/tool_name, got {ref!r}'
        )
    owner, _, tool = raw.partition('/')
    if not owner or not tool or '/' in tool:
        raise ValueError(f'Invalid adopt ref {ref!r}')
    return validate_group_id(owner), tool


def _append_sharing_on_manifest(
    tool_dir: Path,
    *,
    shared_with: List[str],
    shared_by: str,
    shared_at: str,
    revocations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not (tool_dir / 'manifest.json').is_file():
        return
    try:
        _ensure_writable(tool_dir / 'manifest.json')
        manifest = read_manifest(tool_dir)
        sharing = dict(manifest.get('sharing') or {})
        sharing['shared_with'] = list(shared_with)
        sharing['shared_by'] = shared_by
        sharing['shared_at'] = shared_at
        if revocations is not None:
            sharing['revocations'] = list(revocations)
        else:
            sharing.setdefault('revocations', list(sharing.get('revocations') or []))
        manifest['sharing'] = sharing
        (tool_dir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + '\n',
            encoding='utf-8',
        )
        _make_tree_readonly(tool_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning('Could not update sharing on {}: {}', tool_dir, exc)


def _default_adopt_confirm(summary: str) -> bool:
    print(summary)
    print('Adopt this tool at the pinned version? [y/N] ', end='', flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {'y', 'yes'}
