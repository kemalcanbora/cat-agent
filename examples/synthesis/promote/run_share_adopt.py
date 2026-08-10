#!/usr/bin/env python3.10
"""Cross-group share → adopt: two-sided consent and pinned versions.

No LLM. Finance promotes and shares ``validate_iban``; ops adopts a specific
content hash. A later finance re-promote does not move ops. Unshare then makes
the next ops load fail with tool, owner, and reason.

    python3.10 examples/synthesis/promote/run_share_adopt.py
    python3.10 examples/synthesis/promote/run_share_adopt.py --workspace /tmp/cat-share
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_DIR = Path(__file__).resolve().parent
MEMBERSHIP = EXAMPLE_DIR / 'groups.json'

_IMPL = '''\
def validate_iban(iban: str) -> bool:
    """Validate IBAN shape.

    Args:
        iban: IBAN string.
    """
    return len(iban) >= 15 and iban[:2].isalpha()
'''


def _clear_iban_registry() -> None:
    from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY

    for name in list(TOOL_REGISTRY) + list(OPTIONAL_TOOL_REGISTRY):
        if 'validate_iban' in name:
            TOOL_REGISTRY.pop(name, None)
            OPTIONAL_TOOL_REGISTRY.pop(name, None)


def _stage(workspace: Path, principal, *, code: str = _IMPL):
    from cat_agent.synthesis.artifacts import write_artifacts
    from cat_agent.synthesis.spec import Example, ToolSpec

    spec = ToolSpec(
        name='validate_iban',
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
    _clear_iban_registry()
    work, holdout = spec.split_examples()
    return write_artifacts(
        spec=spec,
        code=code,
        executor_name='wasm',
        model_name='example-fixed',
        attempt_count=1,
        example_results=[],
        work=work,
        holdout=holdout,
        base=str(workspace),
        principal=principal,
        provenance={
            'synthesized_by': principal.user_id,
            'verification': {
                'code_mutation': {'killed': 6, 'total': 6, 'threshold': 0.8},
                'input_sensitivity': [{'flag': 'empty_string'}],
                'spec_warnings': [],
                'warnings_overridden': False,
                'holdout_size': 1,
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default=None)
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep the workspace directory after the run',
    )
    args = parser.parse_args()

    MEMBERSHIP.chmod(0o644)
    os.environ['CAT_AGENT_MEMBERSHIP_PATH'] = str(MEMBERSHIP)

    from cat_agent.security.principal import (
        ROLE_SHARER,
        Principal,
        load_membership_index,
        require_role,
    )
    from cat_agent.synthesis.artifacts import (
        active_pointers_path,
        read_json_pointers,
        read_manifest,
    )
    from cat_agent.synthesis.promote import format_tool_list, promote
    from cat_agent.synthesis.registry import (
        AdoptedToolError,
        load_generated_tools,
        tools_for_principal,
    )
    from cat_agent.synthesis.share import adopt, share, unshare
    from cat_agent.tools.base import enable_optional_tools

    index = load_membership_index(MEMBERSHIP)
    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    ops_member = Principal(user_id='ops_member', group_id='ops', source='explicit')

    require_role(index, finance, ROLE_SHARER)
    require_role(index, ops, ROLE_SHARER)
    try:
        require_role(index, ops_member, ROLE_SHARER)
    except Exception as exc:
        print(f'ok: ops_member cannot adopt → {exc}')

    if args.workspace:
        workspace = Path(args.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix='cat-share-', dir=str(EXAMPLE_DIR)))
        cleanup = not args.keep

    print(f'workspace: {workspace}\n')

    try:
        v1 = _stage(workspace, finance)
        promote(finance, 'validate_iban', workspace=str(workspace), yes=True)
        print(f'finance promoted version: {v1.name}')

        share(
            finance, 'validate_iban',
            with_groups=['ops'],
            workspace=str(workspace),
        )
        print('finance shared validate_iban with ops (offer only)')

        _clear_iban_registry()
        load_generated_tools(finance, workspace=str(workspace))
        enable_optional_tools('generated_finance_validate_iban')
        before = [
            n for n in tools_for_principal(ops, workspace=str(workspace))
            if n.startswith('generated_')
        ]
        print(f'ops resolved generated tools before adopt: {before}')
        assert before == []

        record = adopt(
            ops,
            'finance/validate_iban',
            version=v1.name,
            workspace=str(workspace),
            yes=True,
        )
        print(
            f'ops adopted → {record["registered_name"]}@{record["version"]} '
            f'(confirmation_skipped={record["confirmation_skipped"]})'
        )
        print()
        print(format_tool_list(ops, workspace=str(workspace)))

        _clear_iban_registry()
        loaded = load_generated_tools(ops, workspace=str(workspace))
        enable_optional_tools(*loaded.keys())
        after = [
            n for n in tools_for_principal(ops, workspace=str(workspace))
            if n.startswith('generated_')
        ]
        print(f'\nops resolved generated tools after adopt: {after}')
        assert after == ['generated_finance_validate_iban']

        # Publisher re-promotes a new content hash — consumer pin must not move.
        v2 = _stage(
            workspace, finance,
            code=_IMPL.replace('return len', 'return  len'),
        )
        promote(finance, 'validate_iban', workspace=str(workspace), yes=True)
        fin_active = read_json_pointers(active_pointers_path(finance, str(workspace)))
        ops_active = read_json_pointers(active_pointers_path(ops, str(workspace)))
        print(f'\nfinance active after re-promote: {fin_active}')
        print(f'ops pin unchanged:              {ops_active}')
        assert fin_active['validate_iban'] == v2.name
        assert ops_active['finance/validate_iban'] == v1.name
        assert v1.name != v2.name

        manifest = read_manifest(v1)
        promo = (manifest.get('promotion') or {})
        print(
            f'pinned artifact review_shown keys: '
            f'{sorted((promo.get("review_shown") or {}).keys())}'
        )

        unshare(
            finance, 'validate_iban',
            with_groups=['ops'],
            reason='iban checksum bug in pinned build',
            workspace=str(workspace),
        )
        print('\nfinance unshared with ops (reason recorded)')
        _clear_iban_registry()
        try:
            load_generated_tools(ops, workspace=str(workspace))
            print('ERROR: expected AdoptedToolError after unshare')
            return 1
        except AdoptedToolError as exc:
            print(f'ok: load refused → {exc}')

        print('\nOK — share → adopt → pin survives re-promote → unshare fails loudly')
        return 0
    finally:
        if cleanup and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
