"""On-premise security controls for regulated deployments."""

from cat_agent.security.offline import (
    OfflineViolationError,
    get_offline_allow_hosts,
    guard_outbound_request,
    install_offline_guards,
    is_host_allowed,
    is_offline_mode,
)
from cat_agent.security.principal import (
    Principal,
    PrincipalError,
    load_membership,
    resolve_principal,
    resolve_principal_from_cli,
)

__all__ = [
    'OfflineViolationError',
    'Principal',
    'PrincipalError',
    'get_offline_allow_hosts',
    'guard_outbound_request',
    'install_offline_guards',
    'is_host_allowed',
    'is_offline_mode',
    'load_membership',
    'resolve_principal',
    'resolve_principal_from_cli',
    'run_offline_readiness_check',
]


def run_offline_readiness_check(*args, **kwargs):
    from cat_agent.security.readiness import run_offline_readiness_check as _run

    return _run(*args, **kwargs)
