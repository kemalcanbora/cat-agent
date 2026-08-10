#!/usr/bin/env python3.10
"""In-group promote: stage → promote → principal-scoped tools → demote.

No LLM. Writes a fixed IBAN validator artifact, promotes it for finance,
shows that ops cannot see finance's enabled tool, then demotes in-process.

    python3.10 examples/synthesis/promote/run_promote_demote.py
    python3.10 examples/synthesis/promote/run_promote_demote.py --workspace /tmp/cat-promote
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


def _stage(workspace: Path, principal):
    from cat_agent.synthesis.artifacts import write_artifacts
    from cat_agent.synthesis.spec import Example, ToolSpec
    from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY

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
    for name in list(TOOL_REGISTRY) + list(OPTIONAL_TOOL_REGISTRY):
        if 'validate_iban' in name:
            TOOL_REGISTRY.pop(name, None)
            OPTIONAL_TOOL_REGISTRY.pop(name, None)

    work, holdout = spec.split_examples()
    return write_artifacts(
        spec=spec,
        code=_IMPL,
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
                'input_sensitivity': [],
                'spec_warnings': [],
                'warnings_overridden': False,
                'holdout_size': 1,
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workspace',
        default=None,
        help='Workspace root (default: a fresh temp dir under this example)',
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep the workspace directory after the run',
    )
    args = parser.parse_args()

    # Membership must not be group/other-writable (library enforces this).
    MEMBERSHIP.chmod(0o644)
    os.environ['CAT_AGENT_MEMBERSHIP_PATH'] = str(MEMBERSHIP)

    from cat_agent.security.principal import (
        ROLE_PROMOTER,
        ROLE_SHARER,
        Principal,
        load_membership_index,
        require_role,
    )
    from cat_agent.synthesis.artifacts import (
        active_pointers_path,
        read_json_pointers,
        staging_pointers_path,
    )
    from cat_agent.synthesis.promote import demote, format_tool_list, promote
    from cat_agent.synthesis.registry import load_generated_tools, tools_for_principal
    from cat_agent.tools.base import (
        OPTIONAL_TOOL_REGISTRY,
        TOOL_REGISTRY,
        enable_optional_tools,
    )

    index = load_membership_index(MEMBERSHIP)
    print(f'membership: {MEMBERSHIP}')
    print(f'  lead roles (finance): {sorted(index.roles_for("lead", "finance"))}')
    print(f'  builder roles (finance): {sorted(index.roles_for("builder", "finance"))}')

    # Role gate (same check the CLI runs before filesystem work).
    builder = Principal(user_id='builder', group_id='finance', source='explicit')
    try:
        require_role(index, builder, ROLE_PROMOTER)
    except Exception as exc:
        print(f'ok: builder cannot promote → {exc}')

    finance = Principal(user_id='lead', group_id='finance', source='explicit')
    ops = Principal(user_id='ops_lead', group_id='ops', source='explicit')
    require_role(index, finance, ROLE_PROMOTER)
    require_role(index, finance, ROLE_SHARER)

    if args.workspace:
        workspace = Path(args.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workspace = Path(tempfile.mkdtemp(prefix='cat-promote-', dir=str(EXAMPLE_DIR)))
        cleanup = not args.keep

    print(f'\nworkspace: {workspace}')

    try:
        staged = _stage(workspace, finance)
        print(f'staged artifact: {staged}')
        print(
            'staging.json:',
            read_json_pointers(staging_pointers_path(finance, str(workspace))),
        )

        promote(finance, 'validate_iban', workspace=str(workspace), yes=True)
        print(
            'active.json:',
            read_json_pointers(active_pointers_path(finance, str(workspace))),
        )
        print()
        print(format_tool_list(finance, workspace=str(workspace)))

        for name in list(TOOL_REGISTRY) + list(OPTIONAL_TOOL_REGISTRY):
            if 'validate_iban' in name:
                TOOL_REGISTRY.pop(name, None)
                OPTIONAL_TOOL_REGISTRY.pop(name, None)

        loaded = load_generated_tools(finance, workspace=str(workspace))
        print(f'\nloaded for finance: {sorted(loaded)}')
        enable_optional_tools(*loaded.keys())

        # Ops must not see finance's tool even though it is enabled in-process.
        fin_tools = [
            n for n in tools_for_principal(finance, workspace=str(workspace))
            if n.startswith('generated_')
        ]
        ops_tools = [
            n for n in tools_for_principal(ops, workspace=str(workspace))
            if n.startswith('generated_')
        ]
        print(f'tools_for_principal(finance) generated: {fin_tools}')
        print(f'tools_for_principal(ops) generated:     {ops_tools}')
        assert fin_tools == ['generated_finance_validate_iban']
        assert ops_tools == []

        result = demote(finance, 'validate_iban', workspace=str(workspace))
        print(
            f'\ndemoted: restart_required={result.restart_required} '
            f'disabled={result.disabled}'
        )
        print(
            'active.json after demote:',
            read_json_pointers(active_pointers_path(finance, str(workspace))),
        )
        after = [
            n for n in tools_for_principal(finance, workspace=str(workspace))
            if n.startswith('generated_')
        ]
        print(f'tools_for_principal(finance) after demote: {after}')
        assert after == []
        print('\nOK — promote / principal scoping / demote without restart')
        return 0
    finally:
        if cleanup and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
