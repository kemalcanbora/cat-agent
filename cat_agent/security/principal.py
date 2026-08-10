"""Authenticated group identity for tool synthesis and deploy.

The library does not authenticate. Callers supply an already-authenticated
:class:`Principal`. There is no module-level current user, no thread-local,
and no silent default group — a missing principal is always an error.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from cat_agent.log import logger

_GROUP_ID_RE = re.compile(r'^[a-z][a-z0-9_]*$')

# Operator-owned membership file. Not under the writable workspace tree.
DEFAULT_MEMBERSHIP_PATH = '/etc/cat-agent/groups.json'
MEMBERSHIP_ENV = 'CAT_AGENT_MEMBERSHIP_PATH'

ROLE_MEMBER = 'member'
ROLE_PROMOTER = 'promoter'
ROLE_SHARER = 'sharer'
VALID_ROLES = frozenset({ROLE_MEMBER, ROLE_PROMOTER, ROLE_SHARER})

_legacy_membership_logged = False


class PrincipalError(ValueError):
    """Raised when group identity cannot be resolved safely."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    group_id: str
    source: str  # "config" | "session" | "explicit"

    def __post_init__(self) -> None:
        validate_group_id(self.group_id)
        if not (self.user_id or '').strip():
            raise PrincipalError('user_id must be a non-empty string')
        if self.source not in {'config', 'session', 'explicit'}:
            raise PrincipalError(
                f'Principal.source must be config|session|explicit, got {self.source!r}'
            )


@dataclass(frozen=True)
class MembershipIndex:
    """user_id → group_id → roles.

    Built from ``groups.json``. Legacy flat entries (a list of group ids) are
    normalised to ``{group: ["member"]}``.
    """

    roles: Dict[str, Dict[str, frozenset]]

    def groups_for(self, user_id: str) -> List[str]:
        return sorted((self.roles.get(user_id) or {}).keys())

    def roles_for(self, user_id: str, group_id: str) -> frozenset:
        return frozenset((self.roles.get(user_id) or {}).get(group_id) or ())

    def has_role(self, user_id: str, group_id: str, role: str) -> bool:
        return role in self.roles_for(user_id, group_id)

    def as_group_map(self) -> Dict[str, List[str]]:
        """Flat user → groups map for :func:`resolve_principal`."""
        return {user: self.groups_for(user) for user in self.roles}


def validate_group_id(group_id: str) -> str:
    """Return *group_id* if safe to use as a directory name.

    Rejects path separators and ``..`` before any filesystem access. Must already
    match the sanitised-identifier rules (lowercase letters, digits, underscore).
    """
    raw = group_id if isinstance(group_id, str) else ''
    if not raw:
        raise PrincipalError('group_id must be a non-empty string')
    if '..' in raw or '/' in raw or '\\' in raw or os.sep in raw:
        raise PrincipalError(
            f'group_id {raw!r} contains a path separator or ".."; rejected'
        )
    if not _GROUP_ID_RE.match(raw):
        raise PrincipalError(
            f'group_id {raw!r} is not a safe identifier '
            '(expected lowercase [a-z][a-z0-9_]*)'
        )
    return raw


def default_membership_path() -> Path:
    """Return the membership file path (env override or ``/etc/cat-agent/groups.json``)."""
    override = os.environ.get(MEMBERSHIP_ENV)
    if override:
        return Path(override)
    return Path(DEFAULT_MEMBERSHIP_PATH)


def _assert_membership_permissions(membership_path: Path) -> None:
    """Refuse group/other-writable membership files (Unix). Expected: root-owned 0644."""
    if sys.platform == 'win32':
        return
    try:
        st = membership_path.stat()
    except OSError as exc:
        raise PrincipalError(
            f'Cannot stat membership file {membership_path}: {exc}'
        ) from exc
    mode = stat.S_IMODE(st.st_mode)
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PrincipalError(
            f'Membership file {membership_path} is writable by group or other '
            f'(mode {oct(mode)}). Expected root-owned, mode 0644. '
            'Fix ownership/permissions before continuing — refusing to load.'
        )


def _validate_roles(user: str, group: str, roles: Iterable[str]) -> frozenset:
    out: Set[str] = set()
    for role in roles:
        if not isinstance(role, str):
            raise PrincipalError(
                f'Role for {user!r}/{group!r} must be a string, got {role!r}'
            )
        if role not in VALID_ROLES:
            raise PrincipalError(
                f'Unknown role {role!r} for {user!r}/{group!r} '
                f'(expected one of {sorted(VALID_ROLES)})'
            )
        out.add(role)
    if ROLE_MEMBER not in out:
        # Promoter/sharer imply membership of the group.
        out.add(ROLE_MEMBER)
    return frozenset(out)


def load_membership_index(path: str | Path) -> MembershipIndex:
    """Load user → group → roles from JSON.

    Roleful format::

        {"lead": {"finance": ["member", "promoter", "sharer"]}}

    Legacy flat format (list of group ids) is treated as ``["member"]`` per
    group and logged once per process.
    """
    global _legacy_membership_logged

    membership_path = Path(path)
    if not membership_path.is_file():
        raise PrincipalError(
            f'Membership file not found: {membership_path}. '
            'Ask the operator to create groups.json and add this user.'
        )
    _assert_membership_permissions(membership_path)
    try:
        data = json.loads(membership_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise PrincipalError(f'Invalid membership JSON at {membership_path}: {exc}') from exc
    if not isinstance(data, dict):
        raise PrincipalError('Membership file must be a JSON object mapping user → groups')

    roles: Dict[str, Dict[str, frozenset]] = {}
    saw_legacy = False
    for user, entry in data.items():
        if not isinstance(user, str) or not user.strip():
            raise PrincipalError(f'Invalid user key in membership file: {user!r}')
        uid = user.strip()
        user_roles: Dict[str, frozenset] = {}

        if isinstance(entry, str):
            saw_legacy = True
            gid = validate_group_id(entry)
            user_roles[gid] = frozenset({ROLE_MEMBER})
        elif isinstance(entry, list):
            saw_legacy = True
            for g in entry:
                if not isinstance(g, str):
                    raise PrincipalError(
                        f'Membership for {uid!r} must be group id strings, got {g!r}'
                    )
                gid = validate_group_id(g)
                user_roles[gid] = frozenset({ROLE_MEMBER})
        elif isinstance(entry, dict):
            for g, role_list in entry.items():
                if not isinstance(g, str):
                    raise PrincipalError(f'Group id for {uid!r} must be a string, got {g!r}')
                gid = validate_group_id(g)
                if isinstance(role_list, str):
                    role_iter = [role_list]
                elif isinstance(role_list, list):
                    role_iter = role_list
                else:
                    raise PrincipalError(
                        f'Roles for {uid!r}/{gid!r} must be a string or list, '
                        f'got {type(role_list).__name__}'
                    )
                user_roles[gid] = _validate_roles(uid, gid, role_iter)
        else:
            raise PrincipalError(
                f'Membership for {uid!r} must be a string, list of groups, '
                f'or object of group → roles'
            )
        roles[uid] = user_roles

    if saw_legacy and not _legacy_membership_logged:
        logger.warning(
            'Membership file {} uses legacy flat group lists; '
            'treating each entry as role ["member"]. '
            'Migrate to {{"user": {{"group": ["member", ...]}}}} for promoter/sharer.',
            membership_path,
        )
        _legacy_membership_logged = True

    return MembershipIndex(roles=roles)


def load_membership(path: str | Path) -> Dict[str, List[str]]:
    """Load operator-owned user → groups map from JSON.

    Prefer :func:`load_membership_index` when roles are required. This helper
    keeps the historical return type (user → list of group ids).
    """
    return load_membership_index(path).as_group_map()


def require_role(
    index: MembershipIndex,
    principal: Principal,
    role: str,
) -> None:
    """Raise if *principal* lacks *role* in their current group."""
    if role not in VALID_ROLES:
        raise PrincipalError(f'Unknown required role {role!r}')
    if index.has_role(principal.user_id, principal.group_id, role):
        return
    have = sorted(index.roles_for(principal.user_id, principal.group_id))
    raise PrincipalError(
        f'User {principal.user_id!r} lacks required role {role!r} '
        f'in group {principal.group_id!r} (have: {have or ["(none)"]}). '
        'Ask an operator to grant the role in groups.json.'
    )


def resolve_principal(
    *,
    user_id: str,
    group_id: Optional[str] = None,
    membership: Mapping[str, Sequence[str]],
    source: str = 'config',
) -> Principal:
    """Resolve a principal from an authenticated *user_id* and membership map.

    *group_id* is only for disambiguation when the user belongs to multiple
    groups; it must appear in that user's memberships.
    """
    uid = (user_id or '').strip()
    if not uid:
        raise PrincipalError('user_id must be a non-empty string')
    groups = list(membership.get(uid) or [])
    if not groups:
        raise PrincipalError(
            f'User {uid!r} is not listed in the membership file. '
            'Ask the operator to add them — there is no default group.'
        )
    if group_id is None or group_id == '':
        if len(groups) == 1:
            return Principal(user_id=uid, group_id=groups[0], source=source)
        raise PrincipalError(
            f'User {uid!r} belongs to multiple groups {groups!r}; '
            'pass --group to disambiguate.'
        )
    chosen = validate_group_id(group_id)
    if chosen not in groups:
        raise PrincipalError(
            f'User {uid!r} is not a member of group {chosen!r} '
            f'(memberships: {groups}). Rejected.'
        )
    return Principal(user_id=uid, group_id=chosen, source=source)


def resolve_principal_from_cli(args: Any) -> Principal:
    """Resolve identity for CLI entry points (OS user + membership file).

    ``--group`` is accepted only to disambiguate multi-group membership and is
    validated against the membership file — never trusted as self-declaration.
    """
    import getpass

    user_id = (
        getattr(args, 'user', None)
        or os.environ.get('CAT_AGENT_USER')
        or getpass.getuser()
    )
    membership_path = getattr(args, 'membership', None)
    if not membership_path:
        membership_path = default_membership_path()
    membership = load_membership(membership_path)
    group = getattr(args, 'group', None)
    return resolve_principal(
        user_id=str(user_id),
        group_id=group,
        membership=membership,
        source='config',
    )


def membership_index_from_cli(args: Any) -> MembershipIndex:
    """Load the roleful membership index for CLI role checks."""
    membership_path = getattr(args, 'membership', None)
    if not membership_path:
        membership_path = default_membership_path()
    return load_membership_index(membership_path)


def namespaced_registered_name(principal: Principal, function_name: str) -> str:
    """Return ``generated_<group_id>_<tool_name>`` for registry isolation."""
    fn = (function_name or '').strip()
    if not fn:
        raise PrincipalError('function_name must be non-empty')
    return f'generated_{principal.group_id}_{fn}'


def owner_registered_name(owner_group: str, function_name: str) -> str:
    """Return ``generated_<owner_group>_<tool_name>`` for an adopted tool."""
    validate_group_id(owner_group)
    fn = (function_name or '').strip()
    if not fn:
        raise PrincipalError('function_name must be non-empty')
    return f'generated_{owner_group}_{fn}'
