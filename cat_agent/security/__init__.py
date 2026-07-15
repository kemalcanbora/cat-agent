"""On-premise security controls for regulated deployments."""

from cat_agent.security.offline import (
    OfflineViolationError,
    guard_outbound_request,
    install_offline_guards,
    is_offline_mode,
)

__all__ = [
    'OfflineViolationError',
    'guard_outbound_request',
    'install_offline_guards',
    'is_offline_mode',
    'run_offline_readiness_check',
]


def run_offline_readiness_check(*args, **kwargs):
    from cat_agent.security.readiness import run_offline_readiness_check as _run

    return _run(*args, **kwargs)
