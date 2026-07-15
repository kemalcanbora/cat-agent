"""Command-line utilities for on-prem Cat-Agent deployments."""

from __future__ import annotations

import argparse
import os
import sys

from cat_agent.security.audit import export_audit_log, verify_audit_log
from cat_agent.security.encrypted_cache import migrate_plaintext_cache
from cat_agent.security.encrypted_migrate import migrate_workspace_storage
from cat_agent.security import install_offline_guards, run_offline_readiness_check
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.resource.wasm_runtime_loader import fetch_runtime_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='cat-agent')
    subparsers = parser.add_subparsers(dest='command', required=True)

    fetch_parser = subparsers.add_parser(
        'fetch-runtime',
        help='Copy bundled WASM runtime assets to a directory for offline transfer',
    )
    fetch_parser.add_argument('--output', required=True, help='Destination directory')

    check_parser = subparsers.add_parser(
        'offline-check',
        help='Run offline readiness checks (respects CAT_AGENT_OFFLINE)',
    )
    check_parser.add_argument('--strict', action='store_true', help='Exit non-zero on issues')

    encrypt_parser = subparsers.add_parser(
        'encrypt-cache',
        help='Encrypt plaintext doc-parser cache records in place',
    )
    encrypt_parser.add_argument(
        '--path',
        default=os.path.join(DEFAULT_WORKSPACE, 'tools', 'doc_parser'),
        help='Doc parser cache directory (default: workspace/tools/doc_parser)',
    )

    encrypt_storage_parser = subparsers.add_parser(
        'encrypt-storage',
        help='Encrypt plaintext workspace caches, agent memory, and RAG indexes',
    )
    encrypt_storage_parser.add_argument(
        '--workspace',
        default=DEFAULT_WORKSPACE,
        help='Workspace root (default: CAT_AGENT_DEFAULT_WORKSPACE)',
    )

    audit_verify_parser = subparsers.add_parser(
        'audit-verify',
        help='Verify tamper-evident audit log hash chain integrity',
    )
    audit_verify_parser.add_argument('--path', required=True, help='Path to audit.jsonl')

    audit_export_parser = subparsers.add_parser(
        'audit-export',
        help='Export audit log records for auditors',
    )
    audit_export_parser.add_argument('--path', required=True, help='Path to audit.jsonl')
    audit_export_parser.add_argument('--output', required=True, help='Destination JSONL file')

    args = parser.parse_args(argv)
    install_offline_guards()

    if args.command == 'fetch-runtime':
        path = fetch_runtime_assets(args.output)
        print(f'WASM runtime copied to {path}')
        return 0

    if args.command == 'offline-check':
        report = run_offline_readiness_check(strict=args.strict)
        print(report.format_report())
        return 0 if report.ok() else 1

    if args.command == 'encrypt-cache':
        migrated = migrate_plaintext_cache(args.path)
        print(f'Encrypted {migrated} plaintext cache record(s) under {args.path}')
        return 0

    if args.command == 'encrypt-storage':
        report = migrate_workspace_storage(args.workspace)
        print(
            'Encrypted workspace storage under '
            f'{args.workspace}: '
            f"{report['sqlite_records']} sqlite record(s), "
            f"{report['index_files']} index file(s)"
        )
        return 0

    if args.command == 'audit-verify':
        report = verify_audit_log(args.path)
        print(
            f'Audit log {args.path}: {report.record_count} record(s), '
            f'valid={report.valid}'
        )
        if report.first_error:
            print(f'ERROR: {report.first_error}')
        return 0 if report.ok() else 1

    if args.command == 'audit-export':
        count = export_audit_log(args.path, args.output)
        print(f'Exported {count} audit record(s) to {args.output}')
        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
