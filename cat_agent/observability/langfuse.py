"""Langfuse OTLP setup helpers (env-file + decorator).

Reads ``LANGFUSE_*`` / ``OTEL_*`` from the process environment (and optionally
a ``.env`` file), configures a global OpenTelemetry TracerProvider that
exports to Langfuse's OTLP/HTTP endpoint, and flushes on exit.

Typical usage::

    from cat_agent.observability import OpenTelemetryHandler, with_langfuse

    @with_langfuse          # or @with_langfuse(env_file=".env")
    def main() -> None:
        agent = Assistant(..., handlers=[OpenTelemetryHandler()])
        list(agent.run(messages))

Requires ``pip install 'cat-agent[otel]'``.
"""

from __future__ import annotations

import base64
import functools
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, overload

F = TypeVar('F', bound=Callable[..., Any])

_DEFAULT_HOST = 'http://localhost:3000'
_DEFAULT_SERVICE = 'cat-agent'


def configure_langfuse_otel(
    *,
    env_file: str | Path | None = None,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    service_name: str | None = None,
    endpoint: str | None = None,
) -> Any:
    """Configure the global OTel provider to export traces to Langfuse.

    Parameters are resolved in order: explicit kwargs → process env → defaults.
    When ``env_file`` is set, that dotenv file is loaded first (does not
    overwrite existing process env vars).

    Returns the installed ``TracerProvider``.
    """
    if env_file is not None:
        _load_dotenv(env_file)
    else:
        from cat_agent.env import load_env_file

        load_env_file()

    # Prefer dict headers over OTEL_EXPORTER_OTLP_HEADERS — Python's OTEL SDK
    # requires URL-encoded values in that env var and drops invalid ones (401).
    os.environ.pop('OTEL_EXPORTER_OTLP_HEADERS', None)
    os.environ.pop('OTEL_EXPORTER_OTLP_TRACES_HEADERS', None)

    resolved_host = (host or os.environ.get('LANGFUSE_HOST') or _DEFAULT_HOST).rstrip('/')
    resolved_public = public_key or os.environ.get('LANGFUSE_PUBLIC_KEY') or ''
    resolved_secret = secret_key or os.environ.get('LANGFUSE_SECRET_KEY') or ''
    resolved_service = (
        service_name
        or os.environ.get('OTEL_SERVICE_NAME')
        or _DEFAULT_SERVICE
    )
    resolved_endpoint = (
        endpoint
        or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        or f'{resolved_host}/api/public/otel/v1/traces'
    )
    # Env often stores the base `/api/public/otel`; append the traces path.
    if resolved_endpoint.rstrip('/').endswith('/otel'):
        resolved_endpoint = resolved_endpoint.rstrip('/') + '/v1/traces'

    if not resolved_public or not resolved_secret:
        raise ValueError(
            'LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required '
            '(set them in .env or pass public_key=/secret_key=).'
        )

    auth = base64.b64encode(f'{resolved_public}:{resolved_secret}'.encode()).decode()
    headers = {
        'Authorization': f'Basic {auth}',
        'x-langfuse-ingestion-version': '4',
    }

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        raise ImportError(
            'Langfuse OTel export requires the OpenTelemetry SDK. '
            "Install it with `pip install 'cat-agent[otel]'`."
        ) from e

    resource = Resource.create({'service.name': resolved_service})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=resolved_endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    return provider


def flush_langfuse_otel(timeout_millis: int = 10_000) -> None:
    """Force-flush the global TracerProvider (no-op if unset / unsupported)."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'force_flush'):
        provider.force_flush(timeout_millis=timeout_millis)


@overload
def with_langfuse(func: F) -> F: ...


@overload
def with_langfuse(
    func: None = None,
    *,
    env_file: str | Path | None = None,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    service_name: str | None = None,
    endpoint: str | None = None,
    flush_ms: int = 10_000,
) -> Callable[[F], F]: ...


def with_langfuse(
    func: Optional[F] = None,
    *,
    env_file: str | Path | None = None,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    service_name: str | None = None,
    endpoint: str | None = None,
    flush_ms: int = 10_000,
) -> Union[F, Callable[[F], F]]:
    """Decorator: configure Langfuse OTel, run the function, then flush.

    Works with or without parentheses::

        @with_langfuse
        def main(): ...

        @with_langfuse(env_file="examples/langfuse/.env")
        def main(): ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            configure_langfuse_otel(
                env_file=env_file,
                host=host,
                public_key=public_key,
                secret_key=secret_key,
                service_name=service_name,
                endpoint=endpoint,
            )
            try:
                return fn(*args, **kwargs)
            finally:
                flush_langfuse_otel(timeout_millis=flush_ms)

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


def _load_dotenv(path: str | Path) -> None:
    from dotenv import load_dotenv

    candidate = Path(path)
    if candidate.is_file():
        load_dotenv(candidate, override=False)
