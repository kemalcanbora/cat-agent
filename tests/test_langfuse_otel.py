"""Tests for Langfuse OTel helpers (env + decorator)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_configure_reads_dotenv(langfuse_env):
    provider = MagicMock()
    exporter = MagicMock()

    with patch('opentelemetry.trace.set_tracer_provider') as set_provider, \
         patch('opentelemetry.sdk.trace.TracerProvider', return_value=provider), \
         patch(
             'opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter',
             return_value=exporter,
         ) as exporter_cls, \
         patch('opentelemetry.sdk.trace.export.BatchSpanProcessor'), \
         patch('opentelemetry.sdk.resources.Resource.create', return_value=MagicMock()):
        configure_langfuse_otel(env_file=langfuse_env)

    set_provider.assert_called_once_with(provider)
    kwargs = exporter_cls.call_args.kwargs
    assert kwargs['endpoint'] == 'http://langfuse.test:3000/api/public/otel/v1/traces'
    assert kwargs['headers']['Authorization'].startswith('Basic ')
    assert kwargs['headers']['x-langfuse-ingestion-version'] == '4'
    provider.add_span_processor.assert_called_once()


def test_configure_requires_keys(monkeypatch, tmp_path):
    monkeypatch.delenv('LANGFUSE_PUBLIC_KEY', raising=False)
    monkeypatch.delenv('LANGFUSE_SECRET_KEY', raising=False)
    empty = tmp_path / 'empty.env'
    empty.write_text('LANGFUSE_HOST=http://localhost:3000\n', encoding='utf-8')
    with pytest.raises(ValueError, match='LANGFUSE_PUBLIC_KEY'):
        configure_langfuse_otel(env_file=empty)


def test_with_langfuse_decorator_flushes(langfuse_env):
    calls = []

    @with_langfuse(env_file=langfuse_env, flush_ms=1234)
    def run():
        calls.append('run')
        return 42

    with patch(
        'cat_agent.observability.langfuse.configure_langfuse_otel',
        return_value=MagicMock(),
    ) as configure, \
         patch('cat_agent.observability.langfuse.flush_langfuse_otel') as flush:
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

    with patch(
        'cat_agent.observability.langfuse.configure_langfuse_otel',
        return_value=MagicMock(),
    ), patch('cat_agent.observability.langfuse.flush_langfuse_otel'):
        assert run() == 'ok'


def test_flush_noop_without_otel(monkeypatch):
    # Should not raise when OTel is unavailable / provider has no force_flush.
    flush_langfuse_otel(timeout_millis=1)
