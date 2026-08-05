"""Tests for Langfuse OTel helpers (env + decorator).

These tests must not require ``pip install 'cat-agent[otel]'`` — CI base jobs
do not install the optional OpenTelemetry stack. Runtime imports are stubbed
via ``sys.modules`` where needed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from cat_agent.observability.langfuse import (
    configure_langfuse_otel,
    flush_langfuse_otel,
    with_langfuse,
)


@pytest.fixture
def langfuse_env(monkeypatch, tmp_path):
    env_path = tmp_path / '.env'
    env_path.write_text(
        '\n'.join([
            'LANGFUSE_HOST=http://langfuse.test:3000',
            'LANGFUSE_PUBLIC_KEY=pk-lf-test',
            'LANGFUSE_SECRET_KEY=sk-lf-test',
            'OTEL_SERVICE_NAME=test-service',
        ]) + '\n',
        encoding='utf-8',
    )
    for key in (
        'LANGFUSE_HOST',
        'LANGFUSE_PUBLIC_KEY',
        'LANGFUSE_SECRET_KEY',
        'OTEL_SERVICE_NAME',
        'OTEL_EXPORTER_OTLP_ENDPOINT',
        'OTEL_EXPORTER_OTLP_HEADERS',
        'OTEL_EXPORTER_OTLP_TRACES_HEADERS',
    ):
        monkeypatch.delenv(key, raising=False)
    return env_path


@pytest.fixture
def fake_otel(monkeypatch):
    """Install stub OpenTelemetry modules so configure_langfuse_otel can import."""
    provider = MagicMock()
    exporter = MagicMock()
    exporter_cls = MagicMock(return_value=exporter)
    set_provider = MagicMock()
    batch_cls = MagicMock()
    resource_create = MagicMock(return_value=MagicMock())

    fake_trace = MagicMock()
    fake_trace.set_tracer_provider = set_provider

    fake_sdk_trace = MagicMock()
    fake_sdk_trace.TracerProvider = MagicMock(return_value=provider)

    fake_export = MagicMock()
    fake_export.BatchSpanProcessor = batch_cls

    fake_resources = MagicMock()
    fake_resources.Resource = MagicMock()
    fake_resources.Resource.create = resource_create

    fake_exporter_mod = MagicMock()
    fake_exporter_mod.OTLPSpanExporter = exporter_cls

    stubs = {
        'opentelemetry': MagicMock(trace=fake_trace),
        'opentelemetry.trace': fake_trace,
        'opentelemetry.sdk': MagicMock(),
        'opentelemetry.sdk.trace': fake_sdk_trace,
        'opentelemetry.sdk.trace.export': fake_export,
        'opentelemetry.sdk.resources': fake_resources,
        'opentelemetry.exporter': MagicMock(),
        'opentelemetry.exporter.otlp': MagicMock(),
        'opentelemetry.exporter.otlp.proto': MagicMock(),
        'opentelemetry.exporter.otlp.proto.http': MagicMock(),
        'opentelemetry.exporter.otlp.proto.http.trace_exporter': fake_exporter_mod,
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return {
        'provider': provider,
        'exporter_cls': exporter_cls,
        'set_provider': set_provider,
        'batch_cls': batch_cls,
        'resource_create': resource_create,
    }


def test_configure_reads_dotenv(langfuse_env, fake_otel):
    configure_langfuse_otel(env_file=langfuse_env)

    fake_otel['set_provider'].assert_called_once_with(fake_otel['provider'])
    kwargs = fake_otel['exporter_cls'].call_args.kwargs
    assert kwargs['endpoint'] == 'http://langfuse.test:3000/api/public/otel/v1/traces'
    assert kwargs['headers']['Authorization'].startswith('Basic ')
    assert kwargs['headers']['x-langfuse-ingestion-version'] == '4'
    fake_otel['provider'].add_span_processor.assert_called_once()


def test_configure_requires_keys(monkeypatch, tmp_path):
    monkeypatch.delenv('LANGFUSE_PUBLIC_KEY', raising=False)
    monkeypatch.delenv('LANGFUSE_SECRET_KEY', raising=False)
    empty = tmp_path / 'empty.env'
    empty.write_text('LANGFUSE_HOST=http://localhost:3000\n', encoding='utf-8')
    with pytest.raises(ValueError, match='LANGFUSE_PUBLIC_KEY'):
        configure_langfuse_otel(env_file=empty)


def test_configure_requires_otel(langfuse_env, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'opentelemetry' or name.startswith('opentelemetry.'):
            raise ImportError('no otel')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match='cat-agent\\[otel\\]'):
        configure_langfuse_otel(env_file=langfuse_env)


def test_with_langfuse_decorator_flushes(langfuse_env, monkeypatch):
    calls = []

    @with_langfuse(env_file=langfuse_env, flush_ms=1234)
    def run():
        calls.append('run')
        return 42

    configure = MagicMock(return_value=MagicMock())
    flush = MagicMock()
    monkeypatch.setattr(
        'cat_agent.observability.langfuse.configure_langfuse_otel', configure
    )
    monkeypatch.setattr(
        'cat_agent.observability.langfuse.flush_langfuse_otel', flush
    )
    assert run() == 42

    assert calls == ['run']
    configure.assert_called_once()
    assert configure.call_args.kwargs['env_file'] == langfuse_env
    flush.assert_called_once_with(timeout_millis=1234)


def test_with_langfuse_bare_decorator(monkeypatch):
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-x')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-x')

    @with_langfuse
    def run():
        return 'ok'

    monkeypatch.setattr(
        'cat_agent.observability.langfuse.configure_langfuse_otel',
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        'cat_agent.observability.langfuse.flush_langfuse_otel', MagicMock()
    )
    assert run() == 'ok'


def test_flush_noop_without_otel(monkeypatch):
    # Should not raise when OTel is unavailable / provider has no force_flush.
    flush_langfuse_otel(timeout_millis=1)
