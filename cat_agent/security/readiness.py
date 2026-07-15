"""Offline deployment readiness checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from cat_agent.security.offline import get_offline_allow_hosts, is_offline_mode
from cat_agent.security.tool_policy import TOOL_METADATA, tool_is_cloud_service, tool_requires_network


@dataclass
class OfflineReadinessReport:
    offline_mode: bool
    disabled_tools: List[str] = field(default_factory=list)
    cloud_tools: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    wasm_runtime_ready: bool = False
    wasm_runtime_path: str = ''
    encrypt_at_rest_enabled: bool = False
    encryption_key_ready: bool = False
    plaintext_storage_items: int = 0
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.issues

    def format_report(self) -> str:
        lines = [
            f'Cat-Agent offline readiness (CAT_AGENT_OFFLINE={int(self.offline_mode)})',
            f'  WASM runtime ready: {self.wasm_runtime_ready}',
        ]
        if self.wasm_runtime_path:
            lines.append(f'  WASM runtime path: {self.wasm_runtime_path}')
        lines.append(f'  Encrypt at rest: {self.encrypt_at_rest_enabled}')
        lines.append(f'  Encryption key ready: {self.encryption_key_ready}')
        if self.plaintext_storage_items:
            lines.append(f'  Plaintext storage items: {self.plaintext_storage_items}')
        if self.disabled_tools:
            lines.append(f'  Network tools disabled: {", ".join(sorted(self.disabled_tools))}')
        if self.cloud_tools:
            lines.append(f'  Cloud-backed tools (not for air-gap): {", ".join(sorted(self.cloud_tools))}')
        if self.allowed_hosts:
            lines.append(f'  Offline allowlist hosts: {", ".join(self.allowed_hosts)}')
        for note in self.notes:
            lines.append(f'  note: {note}')
        for issue in self.issues:
            lines.append(f'  ISSUE: {issue}')
        return '\n'.join(lines)


def _check_wasm_runtime() -> tuple[bool, str, List[str]]:
    from pathlib import Path

    from cat_agent.tools.resource.wasm_runtime_loader import (
        BUNDLED_RUNTIME_DIR,
        _RUNTIME_ASSETS,
        _sha256,
    )

    issues: List[str] = []
    candidates = [
        BUNDLED_RUNTIME_DIR,
        Path(__file__).resolve().parents[1] / 'tools' / 'resource' / 'wasm_runtime',
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        ready = True
        for relative_path, expected in _RUNTIME_ASSETS.items():
            asset = base / relative_path
            if not asset.is_file() or _sha256(asset) != expected:
                ready = False
                break
        if ready:
            return True, str(base), issues

    if is_offline_mode():
        issues.append(
            'WASM runtime assets are missing locally. '
            'Bundle them via pip install cat-agent[wasm-bundled] or set runtime_dir '
            'to a pre-provisioned copy.'
        )
    return False, '', issues


def _check_encryption(workspace: str) -> tuple[bool, bool, int, List[str]]:
    from cat_agent.security.at_rest import is_encrypt_at_rest_enabled, is_encrypted_at_rest_required
    from cat_agent.security.crypto import EncryptionKeyError, resolve_encryption_key
    from cat_agent.security.encrypted_cache import count_plaintext_values
    from cat_agent.security.encrypted_migrate import count_plaintext_index_files
    from cat_agent.tools.storage import _sqlite_path

    issues: List[str] = []
    enabled = is_encrypt_at_rest_enabled()
    key_ready = False
    plaintext_items = 0

    if not enabled:
        return enabled, key_ready, plaintext_items, issues

    try:
        resolve_encryption_key(create_if_missing=False)
        key_ready = True
    except EncryptionKeyError:
        if is_encrypted_at_rest_required() or is_offline_mode():
            issues.append(
                'Encryption key is missing. Set CAT_AGENT_ENCRYPTION_KEY or store a key in the OS keyring.'
            )

    for subdir in ('doc_parser', 'simple_doc_parser', 'storage'):
        db_path = _sqlite_path(os.path.join(workspace, 'tools', subdir))
        plaintext_items += count_plaintext_values(db_path)
    plaintext_items += count_plaintext_index_files(workspace)

    if plaintext_items and is_encrypted_at_rest_required():
        issues.append(
            f'Found {plaintext_items} plaintext storage item(s). Run `cat-agent encrypt-storage`.'
        )

    return enabled, key_ready, plaintext_items, issues


def run_offline_readiness_check(*, strict: bool = False) -> OfflineReadinessReport:
    from cat_agent.settings import DEFAULT_WORKSPACE
    from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY

    offline = is_offline_mode()
    report = OfflineReadinessReport(offline_mode=offline)
    report.allowed_hosts = get_offline_allow_hosts()

    for registry in (TOOL_REGISTRY, OPTIONAL_TOOL_REGISTRY):
        for name in sorted(registry.keys()):
            if tool_requires_network(name):
                if offline:
                    report.disabled_tools.append(name)
                if tool_is_cloud_service(name):
                    report.cloud_tools.append(name)

    for name, meta in TOOL_METADATA.items():
        if name in TOOL_REGISTRY and meta.get('cloud_service') and name not in report.cloud_tools:
            report.cloud_tools.append(name)

    wasm_ready, wasm_path, wasm_issues = _check_wasm_runtime()
    report.wasm_runtime_ready = wasm_ready
    report.wasm_runtime_path = wasm_path
    report.issues.extend(wasm_issues)

    encrypt_enabled, key_ready, plaintext_items, encrypt_issues = _check_encryption(DEFAULT_WORKSPACE)
    report.encrypt_at_rest_enabled = encrypt_enabled
    report.encryption_key_ready = key_ready
    report.plaintext_storage_items = plaintext_items
    report.issues.extend(encrypt_issues)

    if offline and report.cloud_tools:
        report.issues.append(
            'Cloud-backed tools are registered; remove them for strict air-gap deployments.'
        )

    if strict and not report.ok():
        raise RuntimeError(report.format_report())

    return report
