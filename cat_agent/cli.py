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
        help='Workspace root for groups/<id>/artifacts/ (or legacy generated_tools/)',
    )
    synth_run.add_argument(
        '--workspace',
        default=None,
        help='Workspace root (default: CAT_AGENT_DEFAULT_WORKSPACE)',
    )
    synth_run.add_argument(
        '--group',
        default=None,
        help='Group id (required when the OS user belongs to multiple groups)',
    )
    synth_run.add_argument(
        '--membership',
        default=None,
        help=(
            'Path to groups.json (default: /etc/cat-agent/groups.json '
            'or CAT_AGENT_MEMBERSHIP_PATH)'
        ),
    )

    synth_promote = synth_sub.add_parser(
        'promote',
        help='Promote a staging tool into active.json after human review',
    )
    synth_promote.add_argument('tool_name', help='Tool name (function_name)')
    synth_promote.add_argument('--group', default=None, help='Group id (disambiguation)')
    synth_promote.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_promote.add_argument('--membership', default=None)
    synth_promote.add_argument(
        '--version',
        default=None,
        help='Artifact version (impl_sha256[:12]); default: current staging.json pointer',
    )
    synth_promote.add_argument(
        '--yes', action='store_true',
        help='Skip interactive confirmation (recorded on the promotion record)',
    )

    synth_demote = synth_sub.add_parser(
        'demote',
        help='Remove a tool from active.json (artifact versions are kept)',
    )
    synth_demote.add_argument('tool_name', help='Tool name under active.json')
    synth_demote.add_argument('--group', default=None)
    synth_demote.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_demote.add_argument('--membership', default=None)

    synth_list = synth_sub.add_parser(
        'list',
        help='List staging/active pointers and available versions for a group',
    )
    synth_list.add_argument('--group', default=None)
    synth_list.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_list.add_argument('--membership', default=None)
    synth_list.add_argument(
        '--all-groups',
        action='store_true',
        help='List every group under <workspace>/groups/ (operator view)',
    )

    synth_gc = synth_sub.add_parser(
        'gc',
        help='Garbage-collect old artifact versions (never removes active pointers)',
    )
    synth_gc.add_argument('--group', default=None)
    synth_gc.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_gc.add_argument('--membership', default=None)
    synth_gc.add_argument(
        '--keep',
        type=int,
        default=3,
        help='Keep the newest N versions per tool in addition to active (default: 3)',
    )

    synth_migrate = synth_sub.add_parser(
        'migrate',
        help='Migrate legacy staging/ + active/ dirs into content-addressed artifacts/',
    )
    synth_migrate.add_argument('--group', default=None)
    synth_migrate.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_migrate.add_argument('--membership', default=None)

    synth_share = synth_sub.add_parser(
        'share',
        help='Offer an active tool to other groups (does not auto-install)',
    )
    synth_share.add_argument('tool_name', help='Owned tool name under active.json')
    synth_share.add_argument(
        '--with',
        dest='with_groups',
        required=True,
        help='Comma-separated group ids, or "org" for every group',
    )
    synth_share.add_argument('--group', default=None)
    synth_share.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_share.add_argument('--membership', default=None)

    synth_unshare = synth_sub.add_parser(
        'unshare',
        help='Revoke a prior share offer (consumers fail loudly on next load)',
    )
    synth_unshare.add_argument('tool_name', help='Owned tool name')
    synth_unshare.add_argument(
        '--with',
        dest='with_groups',
        required=True,
        help='Comma-separated group ids, or "org"',
    )
    synth_unshare.add_argument(
        '--reason',
        default=None,
        help='Revocation reason recorded for consuming operators',
    )
    synth_unshare.add_argument('--group', default=None)
    synth_unshare.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_unshare.add_argument('--membership', default=None)

    synth_adopt = synth_sub.add_parser(
        'adopt',
        help='Pin another group\'s shared tool into this group\'s active.json',
    )
    synth_adopt.add_argument(
        'ref',
        help='owner_group/tool_name (keeps owner-qualified registry name)',
    )
    synth_adopt.add_argument(
        '--version',
        required=True,
        help='Content hash pin (impl_sha256[:12]); never "latest"',
    )
    synth_adopt.add_argument('--group', default=None)
    synth_adopt.add_argument('--workspace', default=DEFAULT_WORKSPACE)
    synth_adopt.add_argument('--membership', default=None)
    synth_adopt.add_argument(
        '--yes',
        action='store_true',
        help='Skip interactive confirmation (recorded on the adoption record)',
    )

    schedule_parser = subparsers.add_parser(
        'schedule',
        help='Scheduled source collection and report delivery',
    )
    schedule_sub = schedule_parser.add_subparsers(dest='schedule_command', required=True)

    schedule_list = schedule_sub.add_parser('list', help='Show jobs, cadence, next run')
    schedule_list.add_argument('--user', default=None, help='Filter by user id')

    schedule_add = schedule_sub.add_parser('add', help='Create a collect-and-report job')
    schedule_add.add_argument('--user', required=True)
    schedule_add.add_argument('--topic', required=True)
    schedule_add.add_argument('--every', type=float, required=True, help='Hours between runs')
    schedule_add.add_argument('--channel', required=True, choices=['smtp', 'resend', 'webhook'])
    schedule_add.add_argument('--target', required=True, help='Email or webhook URL')

    schedule_rm = schedule_sub.add_parser('rm', help='Delete a job')
    schedule_rm.add_argument('job_id')

    schedule_run = schedule_sub.add_parser('run', help='Run one job now')
    schedule_run.add_argument('job_id')
    schedule_run.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip delivery and print the report body',
    )

    schedule_run_due = schedule_sub.add_parser(
        'run-due',
        help='Claim and run due jobs (same path as the k8s CronJob)',
    )
    schedule_run_due.add_argument('--limit', type=int, default=None)

    schedule_sub.add_parser(
        'doctor',
        help='Validate DSN, channel credentials, LLM reachability, clock skew',
    )

    serve_parser = subparsers.add_parser(
        'serve',
        help='Serve named agents over HTTP (requires cat-agent[serve])',
    )
    serve_parser.add_argument(
        '--factory',
        required=True,
        help='Import path "module:callable" that returns AgentRegistry, Agent, or dict[str, Agent]',
    )
    serve_parser.add_argument(
        '--host',
        default=None,
        help='Bind host (default: CAT_AGENT_SERVE_HOST or 127.0.0.1)',
    )
    serve_parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Bind port (default: CAT_AGENT_SERVE_PORT or 8080)',
    )
    serve_parser.add_argument(
        '--token',
        default=None,
        help='Optional Bearer token (default: CAT_AGENT_SERVE_TOKEN)',
    )

    # --- platform (Nomad) commands; require cat-agent[platform] ---
    platform_parent = argparse.ArgumentParser(add_help=False)
    platform_parent.add_argument(
        '--config',
        default=None,
        help='Operator config.toml (default: ~/.cat-agent/config.toml)',
    )
    platform_parent.add_argument('--nomad-addr', default=None)
    platform_parent.add_argument('--registry', default=None)

    deploy_parser = subparsers.add_parser(
        'deploy',
        parents=[platform_parent],
        help='Build and deploy an agent.yaml to Nomad (requires cat-agent[platform])',
    )
    deploy_parser.add_argument('--dir', default='.', help='Directory containing agent.yaml')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Print HCL and exit')
    deploy_parser.add_argument('--image-tag', default=None, help='Skip build; use this tag')
    deploy_parser.add_argument(
        '--no-push',
        action='store_true',
        help='Local-image mode: tag locally, never push (default when registry=local)',
    )
    deploy_parser.add_argument(
        '--skip-alias-check',
        action='store_true',
        help=(
            'Skip verifying model.alias against the live LLM backend '
            '(default: fail if the model id is unknown to Ollama/gateway)'
        ),
    )

    ls_parser = subparsers.add_parser(
        'ls',
        parents=[platform_parent],
        help='List cat-agent managed Nomad jobs',
    )
    ls_parser.add_argument('--team', default=None)
    ls_parser.add_argument('--json', action='store_true')

    status_parser = subparsers.add_parser(
        'status',
        parents=[platform_parent],
        help='Show allocations for a deployed agent',
    )
    status_parser.add_argument('name')
    status_parser.add_argument('--team', default=None)
    status_parser.add_argument('--dir', default=None, help='Local dir to compare manifest_sha')

    logs_parser = subparsers.add_parser(
        'logs',
        parents=[platform_parent],
        help='Show allocation logs for a deployed agent',
    )
    logs_parser.add_argument('name')
    logs_parser.add_argument('--team', default=None)
    logs_parser.add_argument('--stderr', action='store_true')

    rm_parser = subparsers.add_parser(
        'rm',
        parents=[platform_parent],
        help='Stop and purge a deployed agent',
    )
    rm_parser.add_argument('name')
    rm_parser.add_argument('--team', default=None)
    rm_parser.add_argument('--yes', action='store_true')
    rm_parser.add_argument('--force', action='store_true')

    rollback_parser = subparsers.add_parser(
        'rollback',
        parents=[platform_parent],
        help='Re-submit a previous Nomad job version',
    )
    rollback_parser.add_argument('name')
    rollback_parser.add_argument('--team', default=None)
    rollback_parser.add_argument('--to', default=None, help='Version number')

    build_base_parser = subparsers.add_parser(
        'build-base',
        parents=[platform_parent],
        help='Build the runtime base image',
    )
    build_base_parser.add_argument('--no-push', action='store_true')

    doctor_parser = subparsers.add_parser(
        'doctor',
        parents=[platform_parent],
        help='Check Nomad, docker driver, LLM gateway, Consul DNS, and Vault',
    )
    doctor_parser.add_argument(
        '--team',
        default='demo',
        help='Team whose Vault LLM virtual-key path to check (default: demo)',
    )

    stack_parser = subparsers.add_parser(
        'stack',
        help='Local Nomad compose stack (up/down/seed from .env; requires cat-agent[platform])',
    )
    stack_sub = stack_parser.add_subparsers(dest='stack_command', required=True)
    stack_dir_parent = argparse.ArgumentParser(add_help=False)
    stack_dir_parent.add_argument(
        '--dir',
        default=None,
        help='Stack directory with docker-compose.yml (default: CAT_AGENT_STACK_DIR, cwd, or sibling cat-agent-stack)',
    )
    stack_dir_parent.add_argument(
        '--config',
        default=None,
        help='Operator config.toml (default: CAT_AGENT_CONFIG or ~/.cat-agent/config.toml)',
    )
    stack_dir_parent.add_argument(
        '--profile',
        action='append',
        default=[],
        help='Compose profile (repeatable), e.g. --profile registry',
    )
    stack_dir_parent.add_argument('--nomad-addr', default=None)

    stack_up = stack_sub.add_parser(
        'up',
        parents=[stack_dir_parent],
        help='docker compose up (sets HOST_NOMAD_DATA / HOST_ZOT_DATA)',
    )
    stack_up.add_argument('--build', action='store_true')
    stack_up.add_argument('-d', '--detach', action='store_true')
    stack_up.add_argument(
        '--seed',
        action='store_true',
        help='After up, seed Vault + team key from stack .env',
    )
    stack_up.add_argument(
        '--registry',
        action='store_true',
        help='With --seed, also seed registry Vault secrets',
    )
    stack_up.add_argument('--team', default=None, help='Team for --seed (default: demo)')
    stack_up.add_argument('compose_args', nargs='*', help='Extra docker compose up args')

    stack_down = stack_sub.add_parser(
        'down',
        parents=[stack_dir_parent],
        help='docker compose down',
    )
    stack_down.add_argument('compose_args', nargs='*', help='Extra docker compose down args')

    stack_compose = stack_sub.add_parser(
        'compose',
        parents=[stack_dir_parent],
        help='Pass-through: cat-agent stack compose -- <args>',
    )
    stack_compose.add_argument(
        'compose_args',
        nargs=argparse.REMAINDER,
        help='Args after -- go to docker compose',
    )

    stack_seed = stack_sub.add_parser(
        'seed',
        parents=[stack_dir_parent],
        help='Seed Vault LLM (+ team key) from stack .env',
    )
    stack_seed.add_argument('--team', default=None, help='Team virtual key (default: demo)')
    stack_seed.add_argument(
        '--registry',
        action='store_true',
        help='Also seed Zot push/pull Vault secrets',
    )
    stack_seed.add_argument('--max-tokens-per-day', type=int, default=None)
    stack_seed.add_argument('--tpm-limit', type=int, default=None)
    stack_seed.add_argument('--rpm-limit', type=int, default=None)

    stack_boot = stack_sub.add_parser(
        'bootstrap',
        parents=[stack_dir_parent],
        help='up --build -d + seed from .env (first-time happy path)',
    )
    stack_boot.add_argument('--team', default=None)
    stack_boot.add_argument(
        '--registry',
        action='store_true',
        help='Also seed registry secrets (use with --profile registry)',
    )
    stack_boot.add_argument('compose_args', nargs='*', help='Extra docker compose up args')

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

    if args.command == 'schedule':
        return _cmd_schedule(args)

    if args.command == 'serve':
        return _cmd_serve(args)

    if args.command in {
        'deploy',
        'ls',
        'status',
        'logs',
        'rm',
        'rollback',
        'build-base',
        'doctor',
    }:
        return _cmd_platform(args.command, args)

    if args.command == 'stack':
        return _cmd_stack(args)

    return 1


def _cmd_stack(args: argparse.Namespace) -> int:
    try:
        from cat_agent.platform.commands import run_command
    except ImportError:
        print(
            "platform commands require: pip install 'cat-agent[platform]'",
            file=sys.stderr,
        )
        return 1
    rem = getattr(args, 'compose_args', None)
    if isinstance(rem, list) and rem and rem[0] == '--':
        args.compose_args = rem[1:]
    return run_command(f'stack-{args.stack_command}', args)


def _cmd_platform(command: str, args: argparse.Namespace) -> int:
    try:
        from cat_agent.platform.commands import run_command
    except ImportError:
        print(
            "platform commands require: pip install 'cat-agent[platform]'",
            file=sys.stderr,
        )
        return 1
    return run_command(command, args)


def _cmd_serve(args: argparse.Namespace) -> int:
    from cat_agent.settings import SERVE_TOKEN
    from cat_agent.serve import create_app, load_registry, run_app

    registry = load_registry(args.factory)
    token = args.token if args.token is not None else SERVE_TOKEN
    app = create_app(registry, bearer_token=token or None)
    run_kwargs = {}
    if args.host is not None:
        run_kwargs['host'] = args.host
    if args.port is not None:
        run_kwargs['port'] = args.port
    # Resolve for display (same rules as run_app when kwargs omitted)
    from cat_agent.serve.server import _resolve_host, _resolve_port, _UNSET
    display_host = _resolve_host(run_kwargs.get('host', _UNSET))
    display_port = _resolve_port(run_kwargs.get('port', _UNSET))
    print(f'Serving {len(registry)} agent(s) on http://{display_host}:{display_port}')
    for info in registry.list_info():
        print(f'  - {info.name} ({info.agent_class})')
    run_app(app, **run_kwargs)
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    import asyncio
    import json
    import socket
    import time
    from datetime import datetime, timezone

    from cat_agent.scheduling.store import JobStore, default_scheduler_dsn
    from cat_agent.scheduling.tools import create_schedule, scheduling_context
    from cat_agent.settings import SCHEDULER_JOB_LIMIT, SCHEDULER_LEASE_SECONDS

    store = JobStore(dsn=default_scheduler_dsn())

    if args.schedule_command == 'list':
        jobs = store.list_jobs(user_id=args.user)
        if not jobs:
            print('(no jobs)')
            return 0
        for j in jobs:
            nxt = datetime.fromtimestamp(j.next_run_at, tz=timezone.utc).isoformat()
            cadence = (
                f'every {j.interval_seconds}s'
                if j.interval_seconds
                else f'cron {j.cron_expr}'
            )
            print(
                f'{j.id}\tuser={j.user_id}\t{cadence}\t'
                f'next={nxt}\tenabled={j.enabled}\t'
                f'channel={j.channel}\tfail={j.consecutive_failures}'
            )
        return 0

    if args.schedule_command == 'add':
        with scheduling_context(store):
            out = create_schedule(
                user_id=args.user,
                topic=args.topic,
                every_hours=args.every,
                channel=args.channel,
                target=args.target,
            )
        print(out)
        return 0

    if args.schedule_command == 'rm':
        ok = store.delete_job(args.job_id)
        print(json.dumps({'job_id': args.job_id, 'deleted': ok}))
        return 0 if ok else 1

    if args.schedule_command == 'run':
        from cat_agent.scheduling.runner import execute_job

        async def _run():
            return await execute_job(
                args.job_id,
                store=store,
                owner=f'cli-{socket.gethostname()}',
                dry_run=bool(args.dry_run),
            )

        try:
            run = asyncio.run(_run())
        except Exception as exc:
            print(f'FAILED: {type(exc).__name__}: {exc}')
            return 1
        print(json.dumps({
            'job_id': run.job_id,
            'run_id': run.id,
            'status': run.status,
            'sources_count': run.sources_count,
            'error': run.error,
        }, ensure_ascii=False))
        if args.dry_run and run.error:
            print('--- report (dry-run) ---')
            print(run.error)
        return 0 if run.status != 'failed' else 1

    if args.schedule_command == 'run-due':
        from cat_agent.scheduling.runner import run_due_once

        limit = args.limit if args.limit is not None else SCHEDULER_JOB_LIMIT
        owner = f'cli-{socket.gethostname()}'

        async def _due():
            return await run_due_once(
                store, owner=owner, limit=limit, lease_seconds=SCHEDULER_LEASE_SECONDS,
            )

        results = asyncio.run(_due())
        failed = 0
        for run in results:
            print(json.dumps({
                'job_id': run.job_id,
                'run_id': run.id,
                'status': run.status,
                'sources_count': run.sources_count,
                'error': run.error,
            }, ensure_ascii=False))
            if run.status == 'failed':
                failed += 1
        return 1 if failed else 0

    if args.schedule_command == 'doctor':
        issues = []
        dsn = default_scheduler_dsn()
        print(f'DSN: {dsn}')
        try:
            store.list_jobs()
            print('store: ok')
        except Exception as exc:
            issues.append(f'store: {exc}')
            print(f'store: FAIL ({exc})')

        import os

        channel = os.getenv('CAT_AGENT_DOCTOR_CHANNEL', 'smtp')
        try:
            from cat_agent.scheduling.channels.base import get_channel

            if channel == 'smtp' and not os.getenv('SMTP_HOST'):
                print('smtp: SKIP (SMTP_HOST unset)')
            else:
                get_channel(channel)
                print(f'channel[{channel}]: config ok')
        except Exception as exc:
            issues.append(f'channel: {exc}')
            print(f'channel: FAIL ({exc})')

        # LLM reachability (optional)
        if os.getenv('OPENAI_API_KEY') or os.getenv('OLLAMA_API_KEY'):
            print('llm credentials: present')
        else:
            print('llm credentials: WARN (OPENAI_API_KEY / OLLAMA_API_KEY unset)')

        skew = abs(time.time() - time.time())
        print(f'clock: ok (skew probe {skew:.6f}s)')
        return 1 if issues else 0

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

        from cat_agent.security.principal import (
            ROLE_MEMBER,
            PrincipalError,
            membership_index_from_cli,
            require_role,
            resolve_principal,
        )
        from cat_agent.synthesis.intake.pipeline import synthesize_from_draft

        try:
            if getattr(args, 'workspace', None) is None and args.output_dir:
                args.workspace = args.output_dir
            index = membership_index_from_cli(args)
            import getpass
            user_id = (
                getattr(args, 'user', None)
                or os.environ.get('CAT_AGENT_USER')
                or getpass.getuser()
            )
            principal = resolve_principal(
                user_id=str(user_id),
                group_id=getattr(args, 'group', None),
                membership=index.as_group_map(),
                source='config',
            )
            require_role(index, principal, ROLE_MEMBER)
        except PrincipalError as exc:
            print(f'identity error: {exc}')
            return 1

        llm_cfg = _build_llm_cfg()
        intake_cfg = dict(llm_cfg)
        if os.getenv('INTAKE_LLM_MODEL'):
            intake_cfg = dict(llm_cfg)
            intake_cfg['model'] = os.getenv('INTAKE_LLM_MODEL')

        result = synthesize_from_draft(
            args.draft,
            llm=llm_cfg,
            intake_llm=intake_cfg,
            locale=args.locale,
            lang=args.lang,
            output_dir=args.output_dir or args.workspace,
            principal=principal,
        )
        if result.synthesis and result.synthesis.artifact_dir:
            print(f'artifacts: {result.synthesis.artifact_dir}')
        if result.ok:
            print(f'ok: {result.spec.registered_name if result.spec else "?"}')
            return 0
        print(f'failed: {result.error}')
        return 1

    if args.synth_command in {
        'promote', 'demote', 'list', 'gc', 'migrate',
        'share', 'unshare', 'adopt',
    }:
        return _cmd_synth_lifecycle(args)

    return 1


def _cmd_synth_lifecycle(args: argparse.Namespace) -> int:
    import getpass
    import os

    from cat_agent.security.principal import (
        ROLE_MEMBER,
        ROLE_PROMOTER,
        ROLE_SHARER,
        PrincipalError,
        membership_index_from_cli,
        require_role,
        resolve_principal,
    )
    from cat_agent.synthesis import promote as promote_mod
    from cat_agent.synthesis import share as share_mod
    from cat_agent.synthesis.artifacts import groups_root

    role_by_cmd = {
        'list': ROLE_MEMBER,
        'promote': ROLE_PROMOTER,
        'demote': ROLE_PROMOTER,
        'gc': ROLE_PROMOTER,
        'migrate': ROLE_PROMOTER,
        'share': ROLE_SHARER,
        'unshare': ROLE_SHARER,
        'adopt': ROLE_SHARER,
    }

    try:
        if args.synth_command == 'list' and args.all_groups:
            root = groups_root(args.workspace)
            if not root.is_dir():
                print(f'No groups under {root}')
                return 0
            for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                print(f'[{group_dir.name}]')
                for label, filename in (
                    ('staging', 'staging.json'),
                    ('active', 'active.json'),
                    ('shares', 'shares.json'),
                    ('adoptions', 'adoptions.json'),
                ):
                    ptr_path = group_dir / filename
                    if not ptr_path.is_file():
                        print(f'  {label}: (empty)')
                        continue
                    try:
                        data = __import__('json').loads(
                            ptr_path.read_text(encoding='utf-8')
                        )
                    except Exception:  # noqa: BLE001
                        print(f'  {label}: (unreadable)')
                        continue
                    if not isinstance(data, dict) or not data:
                        print(f'  {label}: (empty)')
                    else:
                        items = ', '.join(
                            f'{k}@{v}' if not isinstance(v, dict)
                            else k
                            for k, v in sorted(data.items())
                        )
                        print(f'  {label}: {items}')
            return 0

        # Role check before any filesystem work on the workspace.
        index = membership_index_from_cli(args)
        user_id = (
            getattr(args, 'user', None)
            or os.environ.get('CAT_AGENT_USER')
            or getpass.getuser()
        )
        principal = resolve_principal(
            user_id=str(user_id),
            group_id=getattr(args, 'group', None),
            membership=index.as_group_map(),
            source='config',
        )
        require_role(index, principal, role_by_cmd[args.synth_command])
    except PrincipalError as exc:
        print(f'identity error: {exc}')
        return 1

    try:
        if args.synth_command == 'promote':
            path = promote_mod.promote(
                principal,
                args.tool_name,
                workspace=args.workspace,
                version=getattr(args, 'version', None),
                yes=bool(args.yes),
            )
            print(f'promoted: {path}')
            return 0
        if args.synth_command == 'demote':
            result = promote_mod.demote(
                principal, args.tool_name, workspace=args.workspace,
            )
            print(
                f'demoted {result.tool_name}; artifact versions kept; '
                f'disabled={result.disabled or "(already inactive)"}'
            )
            if result.restart_required:
                print(
                    f'WARNING: {result.registered_name} could not be fully '
                    'unloaded in-process — restart is required before the '
                    'tool is guaranteed unreachable.'
                )
            return 0
        if args.synth_command == 'list':
            print(promote_mod.format_tool_list(principal, workspace=args.workspace))
            return 0
        if args.synth_command == 'gc':
            removed = promote_mod.gc_artifacts(
                principal, keep=int(args.keep), workspace=args.workspace,
            )
            print(f'gc removed {len(removed)} version(s)')
            for path in removed:
                print(f'  {path}')
            return 0
        if args.synth_command == 'migrate':
            report = promote_mod.migrate_flat_layout(
                principal, workspace=args.workspace,
            )
            print(
                f'migrated group={report["group_id"]} '
                f'moved={len(report["moved"])} '
                f'staging={report["staging"]} active={report["active"]}'
            )
            return 0
        if args.synth_command == 'share':
            entry = share_mod.share(
                principal,
                args.tool_name,
                with_groups=[args.with_groups],
                workspace=args.workspace,
            )
            print(
                f'shared {args.tool_name} with {entry.get("shared_with")} '
                f'(offer only — consumers must adopt)'
            )
            return 0
        if args.synth_command == 'unshare':
            entry = share_mod.unshare(
                principal,
                args.tool_name,
                with_groups=[args.with_groups],
                reason=getattr(args, 'reason', None),
                workspace=args.workspace,
            )
            print(
                f'unshared {args.tool_name}; remaining={entry.get("shared_with")}'
            )
            return 0
        if args.synth_command == 'adopt':
            record = share_mod.adopt(
                principal,
                args.ref,
                version=args.version,
                workspace=args.workspace,
                yes=bool(args.yes),
            )
            print(
                f'adopted {args.ref}@{record["version"]} as '
                f'{record["registered_name"]} '
                f'(confirmation_skipped={record["confirmation_skipped"]})'
            )
            return 0
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        PermissionError,
    ) as exc:
        print(f'error: {exc}')
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
    base_url = (
        os.getenv('OLLAMA_API_BASE')
        or os.getenv('OLLAMA_BASE_URL')
        or 'https://ollama.com/v1'
    ).rstrip('/')
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
