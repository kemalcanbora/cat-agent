"""Command-line utilities for on-prem Cat-Agent deployments."""

from __future__ import annotations

import argparse
import os
import sys

from cat_agent.env import load_env_file

load_env_file()

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

    synth_parser = subparsers.add_parser(
        'synth',
        help='Tool synthesis intake (draft → interview → sandboxed tool)',
    )
    synth_sub = synth_parser.add_subparsers(dest='synth_command', required=True)

    synth_init = synth_sub.add_parser(
        'init',
        help='Write a blank Markdown draft template',
    )
    synth_init.add_argument('name', help='Tool name used in the output filename')
    synth_init.add_argument(
        '--lang',
        default='en',
        help='Template language: en, de, fr, es, it, nl, tr (default: en)',
    )
    synth_init.add_argument(
        '--output',
        default=None,
        help='Output path (default: ./<name>_draft.md)',
    )

    synth_run = synth_sub.add_parser(
        'run',
        help='Interview + synthesise a tool from a Markdown draft',
    )
    synth_run.add_argument('draft', help='Path to draft.md')
    synth_run.add_argument('--locale', default=None, help='Override locale (e.g. de-DE)')
    synth_run.add_argument(
        '--lang',
        default=None,
        help='Override working language for questions (e.g. de)',
    )
    synth_run.add_argument(
        '--output-dir',
        default=None,
        help='Workspace root for generated_tools/',
    )

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

    if args.command == 'synth':
        return _cmd_synth(args)

    return 1


def _cmd_synth(args: argparse.Namespace) -> int:
    from pathlib import Path

    if args.synth_command == 'init':
        from cat_agent.synthesis.intake.template import write_template

        out = Path(args.output) if args.output else Path(f'{args.name}_draft.md')
        write_template(out, lang=args.lang)
        print(f'Wrote draft template to {out} (lang={args.lang})')
        return 0

    if args.synth_command == 'run':
        import os

        if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
            print(
                'Missing OLLAMA_API_KEY or OPENAI_API_KEY — see .env.example.'
            )
            return 1

        from cat_agent.synthesis.intake.pipeline import synthesize_from_draft

        llm_cfg = _build_llm_cfg()
        intake_cfg = dict(llm_cfg)
        # Intake benefits from a stronger model when INTAKE_LLM_MODEL is set.
        if os.getenv('INTAKE_LLM_MODEL'):
            intake_cfg = dict(llm_cfg)
            intake_cfg['model'] = os.getenv('INTAKE_LLM_MODEL')

        result = synthesize_from_draft(
            args.draft,
            llm=llm_cfg,
            intake_llm=intake_cfg,
            locale=args.locale,
            lang=args.lang,
            output_dir=args.output_dir,
        )
        if result.synthesis and result.synthesis.artifact_dir:
            print(f'artifacts: {result.synthesis.artifact_dir}')
        if result.ok:
            print(f'ok: {result.spec.registered_name if result.spec else "?"}')
            return 0
        print(f'failed: {result.error}')
        return 1

    return 1


def _build_llm_cfg() -> dict:
    import os

    api_key = (
        os.getenv('OLLAMA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or 'EMPTY'
    )
    model = os.getenv('LLM_MODEL', 'minimax-m2.7:cloud')
    base_url = (os.getenv('OLLAMA_BASE_URL') or 'https://ollama.com/v1').rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = base_url + '/v1'
    return {
        'model': model,
        'model_type': 'oai',
        'model_server': base_url,
        'api_key': api_key,
        'generate_cfg': {
            'temperature': 0.2,
            'top_p': 0.8,
            # Reasoning models need headroom; 1024 often yields empty content.
            'max_tokens': 8192,
        },
    }


if __name__ == '__main__':
    sys.exit(main())
