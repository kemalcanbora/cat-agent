# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic coverage tests for ``cat_agent.cli.main`` (mocked side effects)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cat_agent.cli import main
from cat_agent.security.audit import AuditVerificationReport
from cat_agent.security.principal import PrincipalError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _noop_offline_guards(monkeypatch):
    monkeypatch.setattr('cat_agent.cli.install_offline_guards', lambda: None)


def _install_platform_commands(monkeypatch, *, run_command=None, raise_import=False):
    """Inject or block ``cat_agent.platform.commands`` for ImportError / success paths."""
    if raise_import:
        # ``sys.modules[name] = None`` makes ``import name`` raise ImportError;
        # monkeypatch restores the prior entry on teardown.
        monkeypatch.setitem(sys.modules, 'cat_agent.platform.commands', None)
        return None

    mod = types.ModuleType('cat_agent.platform.commands')
    stub = run_command if run_command is not None else MagicMock(return_value=0)
    mod.run_command = stub
    monkeypatch.setitem(sys.modules, 'cat_agent.platform.commands', mod)
    if 'cat_agent.platform' not in sys.modules:
        pkg = types.ModuleType('cat_agent.platform')
        pkg.__path__ = []  # mark as package
        pkg.commands = mod
        monkeypatch.setitem(sys.modules, 'cat_agent.platform', pkg)
    else:
        monkeypatch.setattr(
            sys.modules['cat_agent.platform'], 'commands', mod, raising=False,
        )
    return stub


@pytest.fixture()
def scheduler_dsn(tmp_path, monkeypatch):
    dsn = f'sqlite:///{tmp_path / "cli_sched.sqlite"}'
    monkeypatch.setattr(
        'cat_agent.scheduling.store.default_scheduler_dsn',
        lambda: dsn,
    )
    return dsn


# ---------------------------------------------------------------------------
# argparse / unknown command
# ---------------------------------------------------------------------------


def test_main_missing_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_main_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(['not-a-real-command'])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# simple top-level commands
# ---------------------------------------------------------------------------


def test_fetch_runtime(monkeypatch, tmp_path, capsys):
    out = tmp_path / 'runtime'
    monkeypatch.setattr(
        'cat_agent.cli.fetch_runtime_assets',
        lambda dest: str(Path(dest) / 'copied'),
    )
    assert main(['fetch-runtime', '--output', str(out)]) == 0
    assert 'WASM runtime copied' in capsys.readouterr().out


def test_offline_check_ok_and_fail(monkeypatch, capsys):
    ok_report = SimpleNamespace(
        ok=lambda: True,
        format_report=lambda: 'ready-ok',
    )
    monkeypatch.setattr(
        'cat_agent.cli.run_offline_readiness_check',
        lambda strict=False: ok_report,
    )
    assert main(['offline-check']) == 0
    assert 'ready-ok' in capsys.readouterr().out

    bad_report = SimpleNamespace(
        ok=lambda: False,
        format_report=lambda: 'ready-bad',
    )
    monkeypatch.setattr(
        'cat_agent.cli.run_offline_readiness_check',
        lambda strict=False: bad_report,
    )
    assert main(['offline-check', '--strict']) == 1
    assert 'ready-bad' in capsys.readouterr().out


def test_encrypt_cache(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr('cat_agent.cli.migrate_plaintext_cache', lambda path: 3)
    assert main(['encrypt-cache', '--path', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'Encrypted 3' in out
    assert str(tmp_path) in out


def test_encrypt_storage(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        'cat_agent.cli.migrate_workspace_storage',
        lambda workspace: {'sqlite_records': 2, 'index_files': 5},
    )
    assert main(['encrypt-storage', '--workspace', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '2 sqlite record(s)' in out
    assert '5 index file(s)' in out


def test_audit_verify_valid_and_invalid(monkeypatch, tmp_path, capsys):
    path = str(tmp_path / 'audit.jsonl')

    monkeypatch.setattr(
        'cat_agent.cli.verify_audit_log',
        lambda p: AuditVerificationReport(
            path=p, record_count=4, valid=True, first_error=None,
        ),
    )
    assert main(['audit-verify', '--path', path]) == 0
    assert 'valid=True' in capsys.readouterr().out

    monkeypatch.setattr(
        'cat_agent.cli.verify_audit_log',
        lambda p: AuditVerificationReport(
            path=p, record_count=2, valid=False, first_error='chain break',
        ),
    )
    assert main(['audit-verify', '--path', path]) == 1
    out = capsys.readouterr().out
    assert 'valid=False' in out
    assert 'ERROR: chain break' in out


def test_audit_export(monkeypatch, tmp_path, capsys):
    src = tmp_path / 'audit.jsonl'
    dest = tmp_path / 'export.jsonl'
    monkeypatch.setattr('cat_agent.cli.export_audit_log', lambda a, b: 7)
    assert main(['audit-export', '--path', str(src), '--output', str(dest)]) == 0
    assert 'Exported 7' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def test_serve_with_host_port_token(monkeypatch, capsys):
    info = SimpleNamespace(name='demo', agent_class='Agent')

    class _Reg:
        def __len__(self):
            return 1

        def list_info(self):
            return [info]

    monkeypatch.setattr('cat_agent.serve.load_registry', lambda factory: _Reg())
    monkeypatch.setattr('cat_agent.serve.create_app', lambda reg, bearer_token=None: 'app')
    run_app = MagicMock()
    monkeypatch.setattr('cat_agent.serve.run_app', run_app)
    monkeypatch.setattr('cat_agent.serve.server._resolve_host', lambda h: '0.0.0.0')
    monkeypatch.setattr('cat_agent.serve.server._resolve_port', lambda p: 9090)

    code = main([
        'serve',
        '--factory', 'mod:factory',
        '--host', '0.0.0.0',
        '--port', '9090',
        '--token', 'secret',
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert 'Serving 1 agent(s)' in out
    assert 'demo (Agent)' in out
    run_app.assert_called_once()
    assert run_app.call_args.kwargs.get('host') == '0.0.0.0'
    assert run_app.call_args.kwargs.get('port') == 9090


def test_serve_defaults_no_host_port(monkeypatch, capsys):
    class _Reg:
        def __len__(self):
            return 0

        def list_info(self):
            return []

    monkeypatch.setattr('cat_agent.serve.load_registry', lambda factory: _Reg())
    monkeypatch.setattr('cat_agent.serve.create_app', lambda reg, bearer_token=None: 'app')
    run_app = MagicMock()
    monkeypatch.setattr('cat_agent.serve.run_app', run_app)
    monkeypatch.setattr('cat_agent.serve.server._resolve_host', lambda h: '127.0.0.1')
    monkeypatch.setattr('cat_agent.serve.server._resolve_port', lambda p: 8080)

    assert main(['serve', '--factory', 'mod:f']) == 0
    run_app.assert_called_once_with('app')


# ---------------------------------------------------------------------------
# platform + stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'argv',
    [
        ['deploy', '--dir', '.', '--dry-run'],
        ['ls', '--json'],
        ['status', 'my-agent'],
        ['logs', 'my-agent', '--stderr'],
        ['rm', 'my-agent', '--yes'],
        ['rollback', 'my-agent', '--to', '3'],
        ['build-base', '--no-push'],
        ['doctor', '--team', 'demo'],
    ],
)
def test_platform_commands_success(monkeypatch, argv):
    stub = _install_platform_commands(monkeypatch)
    assert main(argv) == 0
    stub.assert_called_once()
    assert stub.call_args[0][0] == argv[0]


def test_platform_import_error(monkeypatch, capsys):
    _install_platform_commands(monkeypatch, raise_import=True)
    assert main(['doctor']) == 1
    assert "cat-agent[platform]" in capsys.readouterr().err


@pytest.mark.parametrize(
    'argv,expected_cmd',
    [
        (['stack', 'up', '-d', '--seed'], 'stack-up'),
        (['stack', 'down'], 'stack-down'),
        (['stack', 'compose', '--', 'ps'], 'stack-compose'),
        (['stack', 'seed', '--team', 'demo'], 'stack-seed'),
        (['stack', 'bootstrap', '--registry'], 'stack-bootstrap'),
    ],
)
def test_stack_commands_success(monkeypatch, argv, expected_cmd):
    stub = _install_platform_commands(monkeypatch)
    assert main(argv) == 0
    stub.assert_called_once()
    assert stub.call_args[0][0] == expected_cmd
    if expected_cmd == 'stack-compose':
        # leading ``--`` stripped for compose passthrough
        assert stub.call_args[0][1].compose_args == ['ps']


def test_stack_import_error(monkeypatch, capsys):
    _install_platform_commands(monkeypatch, raise_import=True)
    assert main(['stack', 'up']) == 1
    assert "cat-agent[platform]" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def test_schedule_list_empty(scheduler_dsn, capsys):
    assert main(['schedule', 'list']) == 0
    assert '(no jobs)' in capsys.readouterr().out


def test_schedule_list_cron_job(scheduler_dsn, capsys):
    import time

    from cat_agent.scheduling.models import Job
    from cat_agent.scheduling.store import JobStore

    store = JobStore(dsn=scheduler_dsn)
    now = time.time()
    store.upsert_job(Job(
        id='cron:bob:digest',
        user_id='bob',
        kind='collect_and_report',
        topic='digest',
        interval_seconds=None,
        cron_expr='0 * * * *',
        channel='smtp',
        target='bob@example.com',
        enabled=True,
        next_run_at=now,
        created_at=now,
        updated_at=now,
    ))
    assert main(['schedule', 'list', '--user', 'bob']) == 0
    out = capsys.readouterr().out
    assert 'cron:bob:digest' in out
    assert 'cron 0 * * * *' in out


def test_schedule_add_list_rm(scheduler_dsn, capsys):
    assert main([
        'schedule', 'add',
        '--user', 'alice',
        '--topic', 'ai-news',
        '--every', '1',
        '--channel', 'webhook',
        '--target', 'https://example.com/hook',
    ]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    job_id = payload['job_id']

    assert main(['schedule', 'list', '--user', 'alice']) == 0
    listed = capsys.readouterr().out
    assert job_id in listed
    assert 'alice' in listed

    assert main(['schedule', 'rm', job_id]) == 0
    deleted = json.loads(capsys.readouterr().out.strip())
    assert deleted == {'job_id': job_id, 'deleted': True}

    assert main(['schedule', 'rm', job_id]) == 1


def test_schedule_rm_missing(scheduler_dsn, capsys):
    assert main(['schedule', 'rm', 'no-such-job']) == 1
    assert 'deleted' in capsys.readouterr().out


def test_schedule_run_success_and_fail(monkeypatch, scheduler_dsn, capsys):
    ok_run = SimpleNamespace(
        id='r1', job_id='j1', status='ok', sources_count=2, error=None,
    )

    async def _ok(*a, **k):
        return ok_run

    monkeypatch.setattr('cat_agent.scheduling.runner.execute_job', _ok)
    assert main(['schedule', 'run', 'j1']) == 0
    assert '"status": "ok"' in capsys.readouterr().out

    async def _boom(*a, **k):
        raise RuntimeError('boom')

    monkeypatch.setattr('cat_agent.scheduling.runner.execute_job', _boom)
    assert main(['schedule', 'run', 'j1']) == 1
    assert 'FAILED' in capsys.readouterr().out


def test_schedule_run_dry_run_failed_prints_report(monkeypatch, scheduler_dsn, capsys):
    failed = SimpleNamespace(
        id='r2', job_id='j2', status='failed',
        sources_count=0, error='report body here',
    )

    async def _fail(*a, **k):
        return failed

    monkeypatch.setattr('cat_agent.scheduling.runner.execute_job', _fail)
    assert main(['schedule', 'run', 'j2', '--dry-run']) == 1
    out = capsys.readouterr().out
    assert 'report (dry-run)' in out
    assert 'report body here' in out


def test_schedule_run_due(monkeypatch, scheduler_dsn, capsys):
    runs = [
        SimpleNamespace(
            id='a', job_id='j1', status='ok', sources_count=1, error=None,
        ),
        SimpleNamespace(
            id='b', job_id='j2', status='failed', sources_count=0, error='x',
        ),
    ]

    async def _due(*a, **k):
        return runs

    monkeypatch.setattr('cat_agent.scheduling.runner.run_due_once', _due)
    assert main(['schedule', 'run-due', '--limit', '5']) == 1
    out = capsys.readouterr().out
    assert 'j1' in out and 'j2' in out

    async def _ok_only(*a, **k):
        return runs[:1]

    monkeypatch.setattr('cat_agent.scheduling.runner.run_due_once', _ok_only)
    assert main(['schedule', 'run-due']) == 0


def test_schedule_doctor_ok(monkeypatch, scheduler_dsn, capsys):
    monkeypatch.delenv('SMTP_HOST', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
    monkeypatch.setenv('CAT_AGENT_DOCTOR_CHANNEL', 'smtp')
    assert main(['schedule', 'doctor']) == 0
    out = capsys.readouterr().out
    assert 'store: ok' in out
    assert 'smtp: SKIP' in out
    assert 'llm credentials: WARN' in out


def test_schedule_doctor_channel_config_ok(monkeypatch, scheduler_dsn, capsys):
    monkeypatch.setenv('CAT_AGENT_DOCTOR_CHANNEL', 'webhook')
    monkeypatch.setenv('OLLAMA_API_KEY', 'ollama-key')
    monkeypatch.setattr(
        'cat_agent.scheduling.channels.base.get_channel',
        lambda name: object(),
    )
    assert main(['schedule', 'doctor']) == 0
    out = capsys.readouterr().out
    assert 'channel[webhook]: config ok' in out
    assert 'llm credentials: present' in out


def test_schedule_doctor_store_and_channel_fail(monkeypatch, scheduler_dsn, capsys):
    class BoomStore:
        def __init__(self, *a, **k):
            pass

        def list_jobs(self, user_id=None):
            raise RuntimeError('db down')

    monkeypatch.setattr('cat_agent.scheduling.store.JobStore', BoomStore)
    monkeypatch.setenv('CAT_AGENT_DOCTOR_CHANNEL', 'webhook')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')

    def _bad_channel(name):
        raise ValueError('bad channel cfg')

    monkeypatch.setattr(
        'cat_agent.scheduling.channels.base.get_channel',
        _bad_channel,
    )
    assert main(['schedule', 'doctor']) == 1
    out = capsys.readouterr().out
    assert 'store: FAIL' in out
    assert 'channel: FAIL' in out
    assert 'llm credentials: present' in out


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def test_synth_init_writes_draft(tmp_path, capsys):
    out = tmp_path / 'mytool_draft.md'
    assert main(['synth', 'init', 'mytool', '--lang', 'en', '--output', str(out)]) == 0
    assert out.is_file()
    assert out.stat().st_size > 0
    assert 'Wrote draft template' in capsys.readouterr().out


def test_synth_run_missing_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    draft = tmp_path / 'draft.md'
    draft.write_text('# draft\n', encoding='utf-8')
    assert main(['synth', 'run', str(draft)]) == 1
    assert 'Missing OLLAMA_API_KEY' in capsys.readouterr().out


def test_synth_run_identity_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    draft = tmp_path / 'draft.md'
    draft.write_text('# draft\n', encoding='utf-8')

    def _boom(args):
        raise PrincipalError('no membership')

    monkeypatch.setattr(
        'cat_agent.security.principal.membership_index_from_cli',
        _boom,
    )
    assert main(['synth', 'run', str(draft), '--workspace', str(tmp_path)]) == 1
    assert 'identity error' in capsys.readouterr().out


def test_synth_run_ok_and_fail(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('INTAKE_LLM_MODEL', 'tiny')
    draft = tmp_path / 'draft.md'
    draft.write_text('# draft\n', encoding='utf-8')

    principal = SimpleNamespace(user_id='u', group_id='g')
    index = MagicMock()
    index.as_group_map.return_value = {'g': ['u']}
    monkeypatch.setattr(
        'cat_agent.security.principal.membership_index_from_cli',
        lambda args: index,
    )
    monkeypatch.setattr(
        'cat_agent.security.principal.resolve_principal',
        lambda **k: principal,
    )
    monkeypatch.setattr(
        'cat_agent.security.principal.require_role',
        lambda *a, **k: None,
    )

    ok = SimpleNamespace(
        ok=True,
        error=None,
        spec=SimpleNamespace(registered_name='g.tool'),
        synthesis=SimpleNamespace(artifact_dir=str(tmp_path / 'art')),
    )
    monkeypatch.setattr(
        'cat_agent.synthesis.intake.pipeline.synthesize_from_draft',
        lambda *a, **k: ok,
    )
    assert main([
        'synth', 'run', str(draft),
        '--output-dir', str(tmp_path),
        '--locale', 'en-US',
        '--lang', 'en',
    ]) == 0
    out = capsys.readouterr().out
    assert 'ok: g.tool' in out
    assert 'artifacts:' in out

    bad = SimpleNamespace(
        ok=False,
        error='synth blew up',
        spec=None,
        synthesis=None,
    )
    monkeypatch.setattr(
        'cat_agent.synthesis.intake.pipeline.synthesize_from_draft',
        lambda *a, **k: bad,
    )
    assert main(['synth', 'run', str(draft), '--workspace', str(tmp_path)]) == 1
    assert 'failed: synth blew up' in capsys.readouterr().out


def test_synth_list_all_groups(tmp_path, capsys):
    root = tmp_path / 'groups'
    g1 = root / 'finance'
    g1.mkdir(parents=True)
    (g1 / 'staging.json').write_text('{"t1": "abc123"}', encoding='utf-8')
    (g1 / 'active.json').write_text('{}', encoding='utf-8')
    (g1 / 'shares.json').write_text('not-json', encoding='utf-8')
    (g1 / 'adoptions.json').write_text(
        json.dumps({'ext.tool': {'version': 'deadbeef'}}),
        encoding='utf-8',
    )
    g2 = root / 'empty'
    g2.mkdir()

    assert main([
        'synth', 'list',
        '--all-groups',
        '--workspace', str(tmp_path),
    ]) == 0
    out = capsys.readouterr().out
    assert '[finance]' in out
    assert 'staging: t1@abc123' in out
    assert 'active: (empty)' in out
    assert 'shares: (unreadable)' in out
    assert 'adoptions: ext.tool' in out
    assert '[empty]' in out


def test_synth_list_all_groups_no_dir(tmp_path, capsys):
    assert main([
        'synth', 'list',
        '--all-groups',
        '--workspace', str(tmp_path / 'missing'),
    ]) == 0
    assert 'No groups under' in capsys.readouterr().out


def _mock_synth_identity(monkeypatch):
    principal = SimpleNamespace(user_id='alice', group_id='finance')
    index = MagicMock()
    index.as_group_map.return_value = {'finance': ['alice']}
    monkeypatch.setattr(
        'cat_agent.security.principal.membership_index_from_cli',
        lambda args: index,
    )
    monkeypatch.setattr(
        'cat_agent.security.principal.resolve_principal',
        lambda **k: principal,
    )
    monkeypatch.setattr(
        'cat_agent.security.principal.require_role',
        lambda *a, **k: None,
    )
    return principal


def test_synth_lifecycle_promote_demote_list_gc_migrate(monkeypatch, tmp_path, capsys):
    _mock_synth_identity(monkeypatch)

    monkeypatch.setattr(
        'cat_agent.synthesis.promote.promote',
        lambda *a, **k: tmp_path / 'active.json',
    )
    assert main([
        'synth', 'promote', 'tool_a',
        '--workspace', str(tmp_path),
        '--yes',
        '--version', 'deadbeefcafe',
    ]) == 0
    assert 'promoted:' in capsys.readouterr().out

    demote_result = SimpleNamespace(
        tool_name='tool_a',
        disabled='tool_a',
        restart_required=True,
        registered_name='finance.tool_a',
    )
    monkeypatch.setattr(
        'cat_agent.synthesis.promote.demote',
        lambda *a, **k: demote_result,
    )
    assert main(['synth', 'demote', 'tool_a', '--workspace', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'demoted tool_a' in out
    assert 'WARNING' in out

    already = SimpleNamespace(
        tool_name='tool_b',
        disabled=None,
        restart_required=False,
        registered_name='finance.tool_b',
    )
    monkeypatch.setattr(
        'cat_agent.synthesis.promote.demote',
        lambda *a, **k: already,
    )
    assert main(['synth', 'demote', 'tool_b', '--workspace', str(tmp_path)]) == 0
    assert '(already inactive)' in capsys.readouterr().out

    monkeypatch.setattr(
        'cat_agent.synthesis.promote.format_tool_list',
        lambda *a, **k: 'tool list body',
    )
    assert main(['synth', 'list', '--workspace', str(tmp_path)]) == 0
    assert 'tool list body' in capsys.readouterr().out

    monkeypatch.setattr(
        'cat_agent.synthesis.promote.gc_artifacts',
        lambda *a, **k: [tmp_path / 'v1', tmp_path / 'v2'],
    )
    assert main(['synth', 'gc', '--keep', '1', '--workspace', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'gc removed 2' in out

    monkeypatch.setattr(
        'cat_agent.synthesis.promote.migrate_flat_layout',
        lambda *a, **k: {
            'group_id': 'finance',
            'moved': ['a', 'b'],
            'staging': 1,
            'active': 2,
        },
    )
    assert main(['synth', 'migrate', '--workspace', str(tmp_path)]) == 0
    assert 'migrated group=finance' in capsys.readouterr().out


def test_synth_lifecycle_share_unshare_adopt(monkeypatch, tmp_path, capsys):
    _mock_synth_identity(monkeypatch)

    monkeypatch.setattr(
        'cat_agent.synthesis.share.share',
        lambda *a, **k: {'shared_with': ['ops']},
    )
    assert main([
        'synth', 'share', 'tool_a',
        '--with', 'ops',
        '--workspace', str(tmp_path),
    ]) == 0
    assert 'shared tool_a' in capsys.readouterr().out

    monkeypatch.setattr(
        'cat_agent.synthesis.share.unshare',
        lambda *a, **k: {'shared_with': []},
    )
    assert main([
        'synth', 'unshare', 'tool_a',
        '--with', 'ops',
        '--reason', 'revoked',
        '--workspace', str(tmp_path),
    ]) == 0
    assert 'unshared tool_a' in capsys.readouterr().out

    monkeypatch.setattr(
        'cat_agent.synthesis.share.adopt',
        lambda *a, **k: {
            'version': 'abc123def456',
            'registered_name': 'ops.tool_a',
            'confirmation_skipped': True,
        },
    )
    assert main([
        'synth', 'adopt', 'ops/tool_a',
        '--version', 'abc123def456',
        '--yes',
        '--workspace', str(tmp_path),
    ]) == 0
    assert 'adopted ops/tool_a' in capsys.readouterr().out


def test_synth_lifecycle_identity_error(monkeypatch, tmp_path, capsys):
    def _boom(args):
        raise PrincipalError('denied')

    monkeypatch.setattr(
        'cat_agent.security.principal.membership_index_from_cli',
        _boom,
    )
    assert main(['synth', 'promote', 't', '--workspace', str(tmp_path)]) == 1
    assert 'identity error: denied' in capsys.readouterr().out


def test_synth_lifecycle_operation_error(monkeypatch, tmp_path, capsys):
    _mock_synth_identity(monkeypatch)

    def _raise(*a, **k):
        raise FileNotFoundError('missing staging')

    monkeypatch.setattr('cat_agent.synthesis.promote.promote', _raise)
    assert main([
        'synth', 'promote', 't',
        '--workspace', str(tmp_path),
        '--yes',
    ]) == 1
    assert 'error: missing staging' in capsys.readouterr().out


def test_build_llm_cfg_appends_v1(monkeypatch):
    from cat_agent.cli import _build_llm_cfg

    monkeypatch.setenv('OLLAMA_API_KEY', 'k')
    monkeypatch.setenv('OLLAMA_API_BASE', 'https://ollama.example.com')
    monkeypatch.delenv('OLLAMA_BASE_URL', raising=False)
    monkeypatch.setenv('LLM_MODEL', 'm1')
    cfg = _build_llm_cfg()
    assert cfg['model'] == 'm1'
    assert cfg['model_server'].endswith('/v1')
    assert cfg['api_key'] == 'k'
