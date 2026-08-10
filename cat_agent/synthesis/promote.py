"""Promote / demote synthesised tools via content-addressed pointers."""

from __future__ import annotations

import ast
import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cat_agent.log import logger
from cat_agent.security.principal import Principal, namespaced_registered_name
from cat_agent.synthesis.artifacts import (
    active_pointers_path,
    active_root,
    artifact_version_dir,
    artifacts_root,
    read_json_pointers,
    read_manifest,
    sha256_text,
    staging_pointers_path,
    staging_root,
    verify_impl_hash,
    version_id_from_hash,
    warn_legacy_generated_tools,
    write_json_pointers,
)
from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY, disable_tools

_DANGEROUS_NAMES = frozenset({
    'eval', 'exec', '__import__', 'open', 'socket', 'subprocess',
})


@dataclass(frozen=True)
class ReviewPayload:
    """AST / verification review shown to the approver and stored on promotion."""

    imports: List[str]
    flagged_names: List[str]
    verification_summary: Dict[str, Any]
    display: str
    parse_error: Optional[str] = None

    def as_record(self) -> Dict[str, Any]:
        return {
            'imports': list(self.imports),
            'flagged_names': list(self.flagged_names),
            'verification_summary': dict(self.verification_summary),
        }


@dataclass
class DemoteResult:
    tool_name: str
    registered_name: str
    restart_required: bool
    disabled: List[str]


def list_staging_tools(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``{tool_name: version}`` from staging.json."""
    return dict(read_json_pointers(staging_pointers_path(principal, workspace)))


def list_active_tools(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``{tool_name: version}`` from active.json."""
    return dict(read_json_pointers(active_pointers_path(principal, workspace)))


def list_artifact_versions(
    principal: Principal,
    tool_name: str,
    *,
    workspace: Optional[str] = None,
) -> List[str]:
    root = artifacts_root(principal, workspace) / tool_name
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / 'manifest.json').is_file()
    )


def inspect_impl(impl_path: Path) -> Dict[str, Any]:
    """AST walk of ``impl.py`` — imports and dangerous name mentions."""
    text = impl_path.read_text(encoding='utf-8')
    imports: List[str] = []
    dangerous: List[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            'imports': [],
            'dangerous_names': [],
            'parse_error': str(exc),
        }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                top = alias.name.split('.')[0]
                if top in _DANGEROUS_NAMES and top not in dangerous:
                    dangerous.append(top)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            imports.append(mod if mod else '.' * (node.level or 0))
            top = mod.split('.')[0] if mod else ''
            if top in _DANGEROUS_NAMES and top not in dangerous:
                dangerous.append(top)
        elif isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            if node.id not in dangerous:
                dangerous.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_NAMES:
            if node.attr not in dangerous:
                dangerous.append(node.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_NAMES:
                if func.id not in dangerous:
                    dangerous.append(func.id)
    return {
        'imports': sorted(set(imports)),
        'dangerous_names': sorted(dangerous),
        'parse_error': None,
    }


def _verification_summary(verification: Any) -> Dict[str, Any]:
    if not isinstance(verification, dict):
        return {}
    summary: Dict[str, Any] = {}
    code_mut = verification.get('code_mutation')
    if isinstance(code_mut, dict):
        killed = code_mut.get('killed')
        total = code_mut.get('total')
        if killed is not None and total is not None:
            summary['code_mutation'] = f'{killed}/{total}'
        else:
            summary['code_mutation'] = code_mut
    elif code_mut is not None:
        summary['code_mutation'] = code_mut
    flags = verification.get('input_sensitivity')
    if isinstance(flags, list):
        summary['input_sensitivity_flags'] = len(flags)
    elif flags is not None:
        summary['input_sensitivity_flags'] = flags
    return summary


def build_review_payload(
    tool_name: str,
    tool_dir: Path,
    manifest: Dict[str, Any],
) -> ReviewPayload:
    """Single source for CLI review display and the promotion ``review_shown`` record."""
    inspection = inspect_impl(tool_dir / 'impl.py')
    verification = manifest.get('verification') or {}
    v_summary = _verification_summary(verification)
    imports = list(inspection.get('imports') or [])
    flagged = list(inspection.get('dangerous_names') or [])
    parse_error = inspection.get('parse_error')
    lines = [
        f'Review summary for {tool_name!r}',
        f'  artifact: {tool_dir}',
        f'  registered_name (after promote): '
        f'{manifest.get("registered_name")}',
        f'  version: {manifest.get("artifact_version") or tool_dir.name}',
        '',
        '  verification_summary:',
        f'    {json.dumps(v_summary, indent=6, default=str)}',
        '',
        '  module-level imports (AST):',
        f'    {imports or "(none)"}',
        '  flagged names present:',
        f'    {flagged or "(none)"}',
    ]
    if parse_error:
        lines.append(f'  parse_error: {parse_error}')
    lines.append('')
    lines.append(
        'This is information for the human reviewer — not an enforcement gate.'
    )
    return ReviewPayload(
        imports=imports,
        flagged_names=flagged,
        verification_summary=v_summary,
        display='\n'.join(lines),
        parse_error=parse_error,
    )


def format_review_summary(
    tool_name: str,
    tool_dir: Path,
    manifest: Dict[str, Any],
) -> str:
    """Human-readable review block for the promote confirmation prompt."""
    return build_review_payload(tool_name, tool_dir, manifest).display


def promote(
    principal: Principal,
    tool_name: str,
    *,
    workspace: Optional[str] = None,
    version: Optional[str] = None,
    yes: bool = False,
    confirm: Optional[Callable[[str], bool]] = None,
) -> Path:
    """Point ``active.json`` at a content-addressed artifact after review.

    *version* selects an explicit artifact directory (rollback). When omitted,
    the current ``staging.json`` pointer is used.
    """
    warn_legacy_generated_tools(workspace)
    if version:
        chosen = version
    else:
        staging = list_staging_tools(principal, workspace=workspace)
        if tool_name not in staging:
            raise FileNotFoundError(
                f'No staging pointer for {tool_name!r} '
                f'(expected in {staging_pointers_path(principal, workspace)})'
            )
        chosen = staging[tool_name]

    tool_dir = artifact_version_dir(principal, tool_name, chosen, workspace)
    if not tool_dir.is_dir():
        raise FileNotFoundError(
            f'No artifact for {tool_name!r} version {chosen!r} at {tool_dir}'
        )

    manifest = read_manifest(tool_dir)
    if not verify_impl_hash(tool_dir, manifest):
        raise ValueError(
            f'Refusing to promote {tool_name!r}: impl.py hash mismatch '
            '(file was edited after validation). Re-synthesise or restore impl.py.'
        )

    reg = namespaced_registered_name(principal, tool_name)
    review = build_review_payload(tool_name, tool_dir, {
        **manifest,
        'registered_name': reg,
        'artifact_version': chosen,
    })
    confirmation_skipped = bool(yes)
    if not yes:
        ask = confirm or _default_confirm
        if not ask(review.display):
            raise RuntimeError('Promotion cancelled by operator')

    impl_text = (tool_dir / 'impl.py').read_text(encoding='utf-8')
    impl_hash = sha256_text(impl_text if impl_text.endswith('\n') else impl_text + '\n')

    synthesized_by = (
        (manifest.get('synthesized_by')
         or (manifest.get('provenance') or {}).get('synthesized_by')
         or principal.user_id)
    )
    promotion = {
        'promoted_by': principal.user_id,
        'promoted_at': datetime.now(timezone.utc).isoformat(),
        'impl_sha256': impl_hash,
        'artifact_version': chosen,
        'synthesized_by': synthesized_by,
        'group_id': principal.group_id,
        'confirmation_skipped': confirmation_skipped,
        'review_shown': review.as_record(),
    }

    # Update pointer first (promote is a pointer change).
    active = list_active_tools(principal, workspace=workspace)
    active[tool_name] = chosen
    write_json_pointers(active_pointers_path(principal, workspace), active)

    # Append promotion record on the immutable artifact's manifest.
    _ensure_writable(tool_dir / 'manifest.json')
    active_manifest = read_manifest(tool_dir)
    active_manifest['registered_name'] = reg
    active_manifest['function_name'] = tool_name
    active_manifest['group_id'] = principal.group_id
    active_manifest['artifact_version'] = chosen
    active_manifest['promotion'] = promotion
    history = list(active_manifest.get('promotion_history') or [])
    history.append(promotion)
    active_manifest['promotion_history'] = history
    _rewrite_registered_name_in_proxies(
        tool_dir, manifest.get('registered_name'), reg,
    )
    (tool_dir / 'manifest.json').write_text(
        json.dumps(active_manifest, indent=2, ensure_ascii=False, default=str) + '\n',
        encoding='utf-8',
    )
    _make_tree_readonly(tool_dir)
    logger.info(
        'Promoted {}@{} → active as {} (by {})',
        tool_name, chosen, reg, principal.user_id,
    )
    return tool_dir


def demote(
    principal: Principal,
    tool_name: str,
    *,
    workspace: Optional[str] = None,
) -> DemoteResult:
    """Remove tool from ``active.json`` and disable it in-process when possible."""
    warn_legacy_generated_tools(workspace)
    active = list_active_tools(principal, workspace=workspace)
    if tool_name not in active:
        raise FileNotFoundError(
            f'No active pointer for {tool_name!r} '
            f'(expected in {active_pointers_path(principal, workspace)})'
        )
    version = active.pop(tool_name)
    write_json_pointers(active_pointers_path(principal, workspace), active)

    demotion = {
        'demoted_by': principal.user_id,
        'demoted_at': datetime.now(timezone.utc).isoformat(),
        'group_id': principal.group_id,
        'artifact_version': version,
    }
    tool_dir = artifact_version_dir(principal, tool_name, version, workspace)
    if tool_dir.is_dir() and (tool_dir / 'manifest.json').is_file():
        try:
            _ensure_writable(tool_dir / 'manifest.json')
            sm = read_manifest(tool_dir)
            history = list(sm.get('demotion_history') or [])
            history.append(demotion)
            sm['demotion_history'] = history
            sm['last_demotion'] = demotion
            (tool_dir / 'manifest.json').write_text(
                json.dumps(sm, indent=2, ensure_ascii=False, default=str) + '\n',
                encoding='utf-8',
            )
            _make_tree_readonly(tool_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Could not append demotion record to artifact: {}', exc)

    reg = namespaced_registered_name(principal, tool_name)
    disabled = disable_tools(reg)
    # Fully unregister so demoted tools are not re-enabled without reload.
    OPTIONAL_TOOL_REGISTRY.pop(reg, None)
    still_live = reg in TOOL_REGISTRY or reg in OPTIONAL_TOOL_REGISTRY
    restart_required = still_live
    if restart_required:
        logger.warning(
            'Demoted {} but {} remains in-process; restart required to unload',
            tool_name, reg,
        )
    else:
        logger.info(
            'Demoted {} from active (artifact kept at {}, in-process disabled)',
            tool_name, tool_dir,
        )
    return DemoteResult(
        tool_name=tool_name,
        registered_name=reg,
        restart_required=restart_required,
        disabled=disabled,
    )


def gc_artifacts(
    principal: Principal,
    *,
    keep: int = 3,
    workspace: Optional[str] = None,
) -> List[Path]:
    """Remove artifact versions that are neither active nor among the newest *keep*.

    Never removes a version referenced by this group's ``active.json`` or by
    any other group's adopted pointer to these artifacts.
    """
    if keep < 0:
        raise ValueError('keep must be >= 0')
    from cat_agent.synthesis.share import collect_external_pins

    active = list_active_tools(principal, workspace=workspace)
    root = artifacts_root(principal, workspace)
    removed: List[Path] = []
    if not root.is_dir():
        return removed
    for tool_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tool_name = tool_dir.name
        versions = [
            p for p in tool_dir.iterdir()
            if p.is_dir() and (p / 'manifest.json').is_file()
        ]
        versions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        protected: set = set()
        # Owned active pointer (bare key).
        if tool_name in active and '/' not in tool_name:
            protected.add(active[tool_name])
        # Also protect if somehow referenced as self-adopt (shouldn't happen).
        self_key = f'{principal.group_id}/{tool_name}'
        if self_key in active:
            protected.add(active[self_key])
        protected |= collect_external_pins(
            principal.group_id, tool_name, workspace=workspace,
        )
        keep_set = {p.name for p in versions[:keep]} | protected
        for ver_dir in versions:
            if ver_dir.name in keep_set:
                continue
            shutil.rmtree(ver_dir)
            removed.append(ver_dir)
            logger.info('gc removed {}', ver_dir)
    return removed


def migrate_flat_layout(
    principal: Principal,
    *,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Migrate legacy ``staging/`` + ``active/`` dirs into content-addressed layout.

    Hashes each existing artifact, moves it under ``artifacts/<tool>/<sha12>/``,
    and writes ``staging.json`` / ``active.json`` pointers.
    """
    warn_legacy_generated_tools(workspace)
    report: Dict[str, Any] = {
        'group_id': principal.group_id,
        'staging': {},
        'active': {},
        'moved': [],
    }
    staging_ptr: Dict[str, str] = read_json_pointers(
        staging_pointers_path(principal, workspace),
    )
    active_ptr: Dict[str, str] = read_json_pointers(
        active_pointers_path(principal, workspace),
    )

    for label, legacy_root, pointers in (
        ('staging', staging_root(principal, workspace), staging_ptr),
        ('active', active_root(principal, workspace), active_ptr),
    ):
        if not legacy_root.is_dir():
            continue
        for tool_dir in sorted(p for p in legacy_root.iterdir() if p.is_dir()):
            if not (tool_dir / 'manifest.json').is_file():
                continue
            tool_name = tool_dir.name
            impl = tool_dir / 'impl.py'
            if not impl.is_file():
                logger.warning('Skip incomplete legacy tool {}', tool_dir)
                continue
            impl_text = impl.read_text(encoding='utf-8')
            if not impl_text.endswith('\n'):
                impl_text += '\n'
            version = version_id_from_hash(sha256_text(impl_text))
            dest = artifact_version_dir(principal, tool_name, version, workspace)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tool_dir), str(dest))
                report['moved'].append({'from': str(tool_dir), 'to': str(dest)})
            else:
                # Destination already present — drop the duplicate legacy copy.
                shutil.rmtree(tool_dir)
                report['moved'].append({
                    'from': str(tool_dir),
                    'to': str(dest),
                    'deduped': True,
                })
            pointers[tool_name] = version
            report[label][tool_name] = version
            # Refresh registered_name on migrated active tools.
            if label == 'active':
                try:
                    _ensure_writable(dest / 'manifest.json')
                    m = read_manifest(dest)
                    reg = namespaced_registered_name(principal, tool_name)
                    old = m.get('registered_name')
                    m['registered_name'] = reg
                    m['artifact_version'] = version
                    m['group_id'] = principal.group_id
                    _rewrite_registered_name_in_proxies(dest, old, reg)
                    (dest / 'manifest.json').write_text(
                        json.dumps(m, indent=2, ensure_ascii=False, default=str) + '\n',
                        encoding='utf-8',
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning('Could not rewrite migrated manifest {}: {}', dest, exc)

        # Remove empty legacy directories.
        try:
            if legacy_root.is_dir() and not any(legacy_root.iterdir()):
                legacy_root.rmdir()
        except OSError:
            pass

    write_json_pointers(staging_pointers_path(principal, workspace), staging_ptr)
    write_json_pointers(active_pointers_path(principal, workspace), active_ptr)
    report['staging'] = staging_ptr
    report['active'] = active_ptr
    return report


def format_tool_list(principal: Principal, *, workspace: Optional[str] = None) -> str:
    from cat_agent.synthesis.share import (
        owned_and_adopted_pointers,
        upstream_shared_version,
    )

    staging = list_staging_tools(principal, workspace=workspace)
    owned, adopted = owned_and_adopted_pointers(principal, workspace=workspace)
    lines = [
        f'group={principal.group_id} user={principal.user_id}',
        'staging:',
    ]
    if not staging:
        lines.append('  (empty)')
    else:
        for name, ver in sorted(staging.items()):
            lines.append(f'  {name}@{ver}')
    lines.append('active (owned):')
    if not owned:
        lines.append('  (empty)')
    else:
        for name, ver in sorted(owned.items()):
            available = list_artifact_versions(principal, name, workspace=workspace)
            lines.append(
                f'  {name}@{ver}  (versions: {", ".join(available) or ver})'
            )
    lines.append('active (adopted):')
    if not adopted:
        lines.append('  (empty)')
    else:
        for key, ver in sorted(adopted.items()):
            owner, tool = key.split('/', 1)
            upstream = upstream_shared_version(owner, tool, workspace=workspace)
            note = ''
            if upstream and upstream != ver:
                note = f'  [newer upstream: {upstream} — re-adopt to move]'
            elif upstream is None:
                note = '  [owner has no active version]'
            lines.append(f'  {key}@{ver}{note}')
    return '\n'.join(lines)


def _default_confirm(summary: str) -> bool:
    print(summary)
    print('Promote this tool? [y/N] ', end='', flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {'y', 'yes'}


def _rewrite_registered_name_in_proxies(
    tool_dir: Path,
    old_name: Optional[str],
    new_name: str,
) -> None:
    if not old_name or old_name == new_name:
        old_name = old_name or ''
    for filename in ('tool.py',):
        path = tool_dir / filename
        if not path.is_file():
            continue
        _ensure_writable(path)
        text = path.read_text(encoding='utf-8')
        if old_name and old_name in text:
            text = text.replace(old_name, new_name)
        path.write_text(text, encoding='utf-8')
    for path in tool_dir.glob('*.py'):
        if path.name in {'tool.py', 'impl.py'}:
            continue
        _ensure_writable(path)
        text = path.read_text(encoding='utf-8')
        if old_name and old_name in text:
            path.write_text(text.replace(old_name, new_name), encoding='utf-8')


def _make_tree_readonly(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                mode = path.stat().st_mode
                path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            except OSError as exc:
                logger.warning('Could not make {} read-only: {}', path, exc)


def _ensure_writable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR)
    except OSError:
        pass
