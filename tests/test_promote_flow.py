"""Deploy flow: principal identity, content-addressed promote / demote."""

from __future__ import annotations

import json
import shutil
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cat_agent.agent import Agent
from cat_agent.security.principal import (
    Principal,
    PrincipalError,
    ROLE_PROMOTER,
    ROLE_SHARER,
    load_membership,
    load_membership_index,
    namespaced_registered_name,
    require_role,
    resolve_principal,
    resolve_principal_from_cli,
    validate_group_id,
)
from cat_agent.synthesis.artifacts import (
    active_pointers_path,
    active_root,
    artifact_version_dir,
    artifact_version_dir_for_group,
    read_json_pointers,
    read_manifest,
    sha256_text,
    staging_pointers_path,
    staging_root,
    write_artifacts,
)
from cat_agent.synthesis.promote import (
    demote,
    gc_artifacts,
    migrate_flat_layout,
    promote,
)
from cat_agent.synthesis.registry import (
    AdoptedToolError,
    load_generated_tools,
    tools_for_principal,
)
from cat_agent.synthesis.share import adopt, share, unshare
from cat_agent.synthesis.spec import Example, ToolSpec
from cat_agent.tools.base import (
    OPTIONAL_TOOL_REGISTRY,
    TOOL_REGISTRY,
    disable_tools,
    enable_optional_tools,
)

_IMPL = '''\
def validate_iban(iban: str) -> bool:
    """Validate IBAN shape.

    Args:
        iban: IBAN string.
    """
    return len(iban) >= 15 and iban[:2].isalpha()
'''


def _spec(name: str = 'validate_iban') -> ToolSpec:
    return ToolSpec(
        name=name,
        description='Validate an IBAN.',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=[
            Example(inputs={'iban': 'TR330006100519786457841326'}, expected=True),
            Example(inputs={'iban': 'XX00'}, expected=False),
            Example(inputs={'iban': ''}, expected=False),
            Example(inputs={'iban': 'DE89370400440532013000'}, expected=True),
        ],
        holdout_ratio=0.25,
    )


def _write_membership(path: Path, data: dict, *, mode: int = 0o644) -> Path:
    path.write_text(json.dumps(data), encoding='utf-8')
    path.chmod(mode)
    return path


def _clear_reg(principal: Principal, name: str = 'validate_iban') -> str:
    reg = namespaced_registered_name(principal, name)
    OPTIONAL_TOOL_REGISTRY.pop(reg, None)
    TOOL_REGISTRY.pop(reg, None)
    return reg


def _stage_tool(tmp_path: Path, principal: Principal, name: str = 'validate_iban') -> Path:
    spec = _spec(name)
    _clear_reg(principal, spec.function_name)
    OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
    TOOL_REGISTRY.pop(spec.registered_name, None)
    work, holdout = spec.split_examples()
    return write_artifacts(
        spec=spec,
        code=_IMPL.replace('validate_iban', spec.function_name),
        executor_name='wasm',
        model_name='test',
        attempt_count=1,
        example_results=[],
        work=work,
        holdout=holdout,
        base=str(tmp_path),
        principal=principal,
        provenance={'synthesized_by': 'builder', 'verification': {
            'code_mutation': {'killed': 6, 'total': 6, 'threshold': 0.8},
            'input_sensitivity': [{'flag': 'x'}],
            'spec_warnings': [],
            'warnings_overridden': False,
            'holdout_size': 1,
        }},
    )


def test_user_not_in_membership_is_error_not_default(tmp_path: Path):
    membership = _write_membership(tmp_path / 'groups.json', {'alice': ['finance']})
    with pytest.raises(PrincipalError, match='not listed'):
        resolve_principal(
            user_id='eve',
            membership=load_membership(membership),
        )


def test_user_cannot_pass_foreign_group(tmp_path: Path):
    membership = _write_membership(
        tmp_path / 'groups.json',
        {'alice': ['finance']},
    )
    with pytest.raises(PrincipalError, match='not a member'):
        resolve_principal(
            user_id='alice',
            group_id='ops',
            membership=load_membership(membership),
        )


def test_group_id_path_traversal_rejected_before_fs():
    with pytest.raises(PrincipalError, match='path separator|\\.\\.'):
        validate_group_id('../etc')
    with pytest.raises(PrincipalError, match='path separator|\\.\\.'):
        validate_group_id('finance/../ops')
    with pytest.raises(PrincipalError, match='path separator|\\.\\.'):
        validate_group_id('a/b')


def test_cli_resolve_rejects_unlisted_user(tmp_path: Path):
    _write_membership(tmp_path / 'groups.json', {'alice': ['finance']})
    args = SimpleNamespace(
        user='eve',
        group=None,
        workspace=str(tmp_path),
        membership=str(tmp_path / 'groups.json'),
        output_dir=None,
    )
    with pytest.raises(PrincipalError, match='not listed'):
        resolve_principal_from_cli(args)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX mode bits')
def test_membership_mode_0666_rejected(tmp_path: Path):
    path = _write_membership(
        tmp_path / 'groups.json', {'alice': ['finance']}, mode=0o666,
    )
    with pytest.raises(PrincipalError, match='writable by group or other'):
        load_membership(path)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX mode bits')
def test_membership_mode_0644_loads(tmp_path: Path):
    path = _write_membership(
        tmp_path / 'groups.json', {'alice': ['finance']}, mode=0o644,
    )
    assert load_membership(path) == {'alice': ['finance']}


def test_two_groups_same_tool_name_distinct_registry(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    ops = Principal(user_id='bob', group_id='ops', source='explicit')
    _stage_tool(tmp_path, finance)
    _stage_tool(tmp_path, ops)

    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    promote(ops, 'validate_iban', workspace=str(tmp_path), yes=True)

    for p in (finance, ops):
        _clear_reg(p)

    loaded_a = load_generated_tools(finance, workspace=str(tmp_path))
    loaded_b = load_generated_tools(ops, workspace=str(tmp_path))

    name_a = namespaced_registered_name(finance, 'validate_iban')
    name_b = namespaced_registered_name(ops, 'validate_iban')
    assert name_a == 'generated_finance_validate_iban'
    assert name_b == 'generated_ops_validate_iban'
    assert name_a in loaded_a
    assert name_b in loaded_b
    assert name_a in OPTIONAL_TOOL_REGISTRY
    assert name_b in OPTIONAL_TOOL_REGISTRY
    assert name_a not in loaded_b
    assert name_b not in loaded_a


def test_load_group_a_excludes_group_b(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    ops = Principal(user_id='bob', group_id='ops', source='explicit')
    _stage_tool(tmp_path, finance)
    _stage_tool(tmp_path, ops)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    promote(ops, 'validate_iban', workspace=str(tmp_path), yes=True)

    for p in (finance, ops):
        _clear_reg(p)

    loaded = load_generated_tools(finance, workspace=str(tmp_path))
    assert list(loaded) == ['generated_finance_validate_iban']


def test_staging_tool_is_not_loaded(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    staging = _stage_tool(tmp_path, finance)
    pointers = read_json_pointers(staging_pointers_path(finance, str(tmp_path)))
    assert 'validate_iban' in pointers
    assert staging == artifact_version_dir(
        finance, 'validate_iban', pointers['validate_iban'], str(tmp_path),
    )
    assert staging.is_dir()
    _clear_reg(finance)
    loaded = load_generated_tools(finance, workspace=str(tmp_path))
    assert loaded == {}


def test_edited_staging_impl_fails_promote(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    staging = _stage_tool(tmp_path, finance)
    impl = staging / 'impl.py'
    impl.chmod(impl.stat().st_mode | stat.S_IWUSR)
    impl.write_text(
        impl.read_text(encoding='utf-8') + '\n# tampered\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='hash mismatch'):
        promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)


def test_promote_demote_load_empty_artifact_kept(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    staging = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    demote(finance, 'validate_iban', workspace=str(tmp_path))

    _clear_reg(finance)
    loaded = load_generated_tools(finance, workspace=str(tmp_path))
    assert loaded == {}
    assert staging.is_dir()
    assert (staging / 'manifest.json').is_file()
    assert 'validate_iban' not in read_json_pointers(
        active_pointers_path(finance, str(tmp_path)),
    )


def test_promotion_record_round_trips_distinct_actors(tmp_path: Path):
    builder = Principal(user_id='builder', group_id='finance', source='explicit')
    _stage_tool(tmp_path, builder)
    approver = Principal(user_id='approver', group_id='finance', source='explicit')
    active = promote(approver, 'validate_iban', workspace=str(tmp_path), yes=True)
    manifest = read_manifest(active)
    promo = manifest['promotion']
    assert promo['synthesized_by'] == 'builder'
    assert promo['promoted_by'] == 'approver'
    assert promo['group_id'] == 'finance'
    assert promo['confirmation_skipped'] is True
    assert promo['impl_sha256']
    assert promo['promoted_at']
    review = promo['review_shown']
    assert 'imports' in review
    assert 'flagged_names' in review
    assert review['verification_summary']['code_mutation'] == '6/6'
    assert review['verification_summary']['input_sensitivity_flags'] == 1
    again = read_manifest(active)
    assert again['promotion'] == promo


def test_directory_layout_two_group_fixture(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    ops = Principal(user_id='bob', group_id='ops', source='explicit')
    fin_dir = _stage_tool(tmp_path, finance)
    ops_dir = _stage_tool(tmp_path, ops)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    promote(ops, 'validate_iban', workspace=str(tmp_path), yes=True)

    fin_ver = fin_dir.name
    ops_ver = ops_dir.name
    assert (
        tmp_path / 'groups' / 'finance' / 'artifacts' / 'validate_iban' / fin_ver
    ).is_dir()
    assert (
        tmp_path / 'groups' / 'ops' / 'artifacts' / 'validate_iban' / ops_ver
    ).is_dir()
    assert read_json_pointers(active_pointers_path(finance, str(tmp_path))) == {
        'validate_iban': fin_ver,
    }
    assert read_json_pointers(active_pointers_path(ops, str(tmp_path))) == {
        'validate_iban': ops_ver,
    }
    assert read_json_pointers(staging_pointers_path(finance, str(tmp_path))) == {
        'validate_iban': fin_ver,
    }


def test_ops_agent_excludes_finance_tools_even_when_enabled(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    ops = Principal(user_id='bob', group_id='ops', source='explicit')
    _stage_tool(tmp_path, finance)
    _stage_tool(tmp_path, ops)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    promote(ops, 'validate_iban', workspace=str(tmp_path), yes=True)

    for p in (finance, ops):
        _clear_reg(p)
    load_generated_tools(finance, workspace=str(tmp_path))
    load_generated_tools(ops, workspace=str(tmp_path))
    enable_optional_tools(
        namespaced_registered_name(finance, 'validate_iban'),
        namespaced_registered_name(ops, 'validate_iban'),
    )

    resolved = tools_for_principal(ops)
    assert 'generated_ops_validate_iban' in resolved
    assert 'generated_finance_validate_iban' not in resolved
    for name in TOOL_REGISTRY:
        if not name.startswith('generated_'):
            assert name in resolved

    class _Dummy(Agent):
        def _run(self, messages, **kwargs):
            yield []

    # Only attach generated tools — avoid constructing Docker/network builtins.
    agent = _Dummy(
        function_list=['generated_ops_validate_iban'],
        llm=None,
        principal=ops,
    )
    assert list(agent.function_map) == ['generated_ops_validate_iban']

    with pytest.raises(ValueError, match='another group'):
        _Dummy(
            function_list=['generated_finance_validate_iban'],
            llm=None,
            principal=ops,
        )


def test_disable_tools_symmetric_with_enable():
    name = 'generated_testgroup_tmp_disable'
    OPTIONAL_TOOL_REGISTRY.pop(name, None)
    TOOL_REGISTRY.pop(name, None)

    class _T:
        name = 'generated_testgroup_tmp_disable'

    OPTIONAL_TOOL_REGISTRY[name] = _T
    enable_optional_tools(name)
    assert name in TOOL_REGISTRY
    assert name not in OPTIONAL_TOOL_REGISTRY
    disabled = disable_tools(name)
    assert disabled == [name]
    assert name not in TOOL_REGISTRY
    assert name in OPTIONAL_TOOL_REGISTRY
    OPTIONAL_TOOL_REGISTRY.pop(name, None)


def test_demote_unresolvable_without_restart(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    _clear_reg(finance)
    load_generated_tools(finance, workspace=str(tmp_path))
    reg = namespaced_registered_name(finance, 'validate_iban')
    enable_optional_tools(reg)
    assert reg in TOOL_REGISTRY

    result = demote(finance, 'validate_iban', workspace=str(tmp_path))
    assert result.restart_required is False
    assert reg not in TOOL_REGISTRY
    assert reg not in OPTIONAL_TOOL_REGISTRY
    assert reg not in tools_for_principal(finance)


def test_migrate_flat_layout(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    staging = staging_root(finance, str(tmp_path)) / 'validate_iban'
    staging.mkdir(parents=True)
    code = _IMPL if _IMPL.endswith('\n') else _IMPL + '\n'
    (staging / 'impl.py').write_text(code, encoding='utf-8')
    (staging / 'tool.py').write_text('# proxy\n', encoding='utf-8')
    (staging / 'manifest.json').write_text(
        json.dumps({
            'impl_sha256': sha256_text(code),
            'registered_name': 'generated_validate_iban',
            'function_name': 'validate_iban',
        }),
        encoding='utf-8',
    )
    active = active_root(finance, str(tmp_path)) / 'validate_iban'
    shutil.copytree(staging, active)

    report = migrate_flat_layout(finance, workspace=str(tmp_path))
    assert 'validate_iban' in report['staging']
    assert 'validate_iban' in report['active']
    version = report['active']['validate_iban']
    assert artifact_version_dir(
        finance, 'validate_iban', version, str(tmp_path),
    ).is_dir()
    assert not staging.exists()
    assert not active.exists()


def test_gc_never_removes_active(tmp_path: Path):
    finance = Principal(user_id='alice', group_id='finance', source='explicit')
    v1 = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    work, holdout = _spec().split_examples()
    v2 = write_artifacts(
        spec=_spec(),
        code=_IMPL.replace('return len', 'return  len'),
        executor_name='wasm',
        model_name='test',
        attempt_count=1,
        example_results=[],
        work=work,
        holdout=holdout,
        base=str(tmp_path),
        principal=finance,
    )
    assert v1.name != v2.name
    removed = gc_artifacts(finance, keep=0, workspace=str(tmp_path))
    assert v1.is_dir()
    assert not v2.is_dir()
    assert any(p == v2 for p in removed)


def test_load_without_principal_or_path_errors():
    with pytest.raises(ValueError, match='requires a Principal'):
        load_generated_tools()


# --- Cross-group share → adopt -------------------------------------------------


def test_legacy_flat_membership_resolves_to_member(tmp_path: Path):
    path = _write_membership(tmp_path / 'groups.json', {'alice': ['finance']})
    index = load_membership_index(path)
    assert index.roles_for('alice', 'finance') == frozenset({'member'})
    assert load_membership(path) == {'alice': ['finance']}


def test_member_and_promoter_cannot_share_before_filesystem(tmp_path: Path):
    path = _write_membership(tmp_path / 'groups.json', {
        'builder': {'finance': ['member']},
        'approver': {'finance': ['member', 'promoter']},
        'lead': {'finance': ['member', 'promoter', 'sharer']},
    })
    index = load_membership_index(path)
    builder = resolve_principal(
        user_id='builder', membership=index.as_group_map(),
    )
    approver = resolve_principal(
        user_id='approver', membership=index.as_group_map(),
    )
    shares_file = tmp_path / 'groups' / 'finance' / 'shares.json'
    assert not shares_file.exists()
    with pytest.raises(PrincipalError, match="lacks required role 'sharer'"):
        require_role(index, builder, ROLE_SHARER)
    with pytest.raises(PrincipalError, match="lacks required role 'sharer'"):
        require_role(index, approver, ROLE_SHARER)
    # Promoter is fine for promote, still not sharer.
    require_role(index, approver, ROLE_PROMOTER)
    assert not shares_file.exists()


def test_share_without_adopt_invisible_then_adopt_visible(tmp_path: Path):
    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    staged = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    version = staged.name

    share(finance, 'validate_iban', with_groups=['ops'], workspace=str(tmp_path))

    for p in (finance, ops):
        _clear_reg(p)
    load_generated_tools(finance, workspace=str(tmp_path))
    enable_optional_tools(namespaced_registered_name(finance, 'validate_iban'))

    resolved_before = tools_for_principal(ops, workspace=str(tmp_path))
    assert 'generated_finance_validate_iban' not in resolved_before

    adopt(
        ops, 'finance/validate_iban', version=version,
        workspace=str(tmp_path), yes=True,
    )
    _clear_reg(finance)
    loaded = load_generated_tools(ops, workspace=str(tmp_path))
    assert 'generated_finance_validate_iban' in loaded
    enable_optional_tools('generated_finance_validate_iban')
    resolved = tools_for_principal(ops, workspace=str(tmp_path))
    assert 'generated_finance_validate_iban' in resolved
    assert list(resolved).count('generated_finance_validate_iban') == 1


def test_consumer_pin_survives_publisher_repromote(tmp_path: Path):
    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    v1 = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    share(finance, 'validate_iban', with_groups=['ops'], workspace=str(tmp_path))
    adopt(
        ops, 'finance/validate_iban', version=v1.name,
        workspace=str(tmp_path), yes=True,
    )

    work, holdout = _spec().split_examples()
    v2 = write_artifacts(
        spec=_spec(),
        code=_IMPL.replace('return len', 'return  len'),
        executor_name='wasm',
        model_name='test',
        attempt_count=1,
        example_results=[],
        work=work,
        holdout=holdout,
        base=str(tmp_path),
        principal=finance,
    )
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    assert v2.name != v1.name
    assert read_json_pointers(active_pointers_path(finance, str(tmp_path)))[
        'validate_iban'
    ] == v2.name
    # Consumer still pinned to v1.
    assert read_json_pointers(active_pointers_path(ops, str(tmp_path)))[
        'finance/validate_iban'
    ] == v1.name

    _clear_reg(finance)
    loaded = load_generated_tools(ops, workspace=str(tmp_path))
    assert 'generated_finance_validate_iban' in loaded
    assert artifact_version_dir_for_group(
        'finance', 'validate_iban', v1.name, str(tmp_path),
    ).is_dir()


def test_unshare_then_load_raises_with_reason(tmp_path: Path):
    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    staged = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    share(finance, 'validate_iban', with_groups=['ops'], workspace=str(tmp_path))
    adopt(
        ops, 'finance/validate_iban', version=staged.name,
        workspace=str(tmp_path), yes=True,
    )
    unshare(
        finance, 'validate_iban', with_groups=['ops'],
        reason='iban checksum bug', workspace=str(tmp_path),
    )
    _clear_reg(finance)
    with pytest.raises(AdoptedToolError, match='finance/validate_iban') as exc:
        load_generated_tools(ops, workspace=str(tmp_path))
    msg = str(exc.value)
    assert 'finance' in msg
    assert 'iban checksum bug' in msg


def test_gc_refuses_version_pinned_by_other_group(tmp_path: Path):
    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    v1 = _stage_tool(tmp_path, finance)
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    share(finance, 'validate_iban', with_groups=['ops'], workspace=str(tmp_path))
    adopt(
        ops, 'finance/validate_iban', version=v1.name,
        workspace=str(tmp_path), yes=True,
    )
    # Publisher moves active to a new version so v1 is only held by ops.
    work, holdout = _spec().split_examples()
    v2 = write_artifacts(
        spec=_spec(),
        code=_IMPL.replace('return len', 'return  len'),
        executor_name='wasm',
        model_name='test',
        attempt_count=1,
        example_results=[],
        work=work,
        holdout=holdout,
        base=str(tmp_path),
        principal=finance,
    )
    promote(finance, 'validate_iban', workspace=str(tmp_path), yes=True)
    removed = gc_artifacts(finance, keep=0, workspace=str(tmp_path))
    assert v1.is_dir()
    assert v2.is_dir()
    assert v1 not in removed
    assert v2 not in removed
