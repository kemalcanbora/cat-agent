"""Tests for intake provenance artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cat_agent.synthesis.artifacts import sha256_text, write_artifacts
from cat_agent.synthesis.spec import Example, ParameterSpec, ToolSpec


def _spec(name: str = 'prov_vat_tool') -> ToolSpec:
    return ToolSpec(
        name=name,
        description='Split VAT',
        parameters={
            'gross': ParameterSpec(type='number', description='Gross'),
            'rate': ParameterSpec(type='number', description='Rate'),
        },
        returns='object',
        examples=[
            Example(inputs={'gross': 120.0, 'rate': 0.2}, expected={'net': 100.0, 'tax': 20.0}),
            Example(inputs={'gross': 100.0, 'rate': 0.0}, expected={'net': 100.0, 'tax': 0.0}),
            Example(inputs={'gross': 1.0, 'rate': 0.2}, expected={'net': 0.83, 'tax': 0.17}),
        ],
    )


class TestProvenance:

    def test_draft_and_interview_utf8_verbatim(self, tmp_path: Path):
        draft_md = (
            '# MwSt aufteilen\n\n'
            'Brutto 1.500,50 € — Übersetzung: ürün tutarı\n'
        )
        interview = {
            'turns': [{'question': 'Wie runden?', 'answer': 'kaufmännisch'}],
            'confirmation': 'Sie teilen MwSt. Richtig?',
            'confirmed': True,
            'questions_asked': 1,
            'corrections': [],
            'added_examples': [],
            'example_traces': [],
        }
        spec = _spec('prov_utf8_tool')
        work, holdout = spec.split_examples()
        out = write_artifacts(
            spec=spec,
            code='def prov_utf8_tool(gross, rate):\n    return {"net": 0, "tax": 0}\n',
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
            provenance={
                'draft_markdown': draft_md,
                'draft_lang': 'de',
                'locale': 'de-DE',
                'interview': interview,
            },
        )
        recovered = (out / 'draft.md').read_text(encoding='utf-8')
        assert recovered == draft_md
        assert 'ürün' in recovered
        assert '1.500,50' in recovered
        interview_text = (out / 'interview.json').read_text(encoding='utf-8')
        assert json.loads(interview_text)['confirmation'] == interview['confirmation']

        manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['draft_sha256'] == sha256_text(draft_md)
        assert manifest['interview_sha256'] == sha256_text(interview_text)
        assert manifest['draft_lang'] == 'de'
        assert manifest['locale'] == 'de-DE'

    def test_audit_record_when_enabled(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_AUDIT', '1')
        audit_path = tmp_path / 'audit.jsonl'
        monkeypatch.setenv('CAT_AGENT_AUDIT_PATH', str(audit_path))

        from cat_agent.security import audit as audit_mod

        # Force the module to re-read path helpers if cached — call append directly path
        original = audit_mod.append_audit_record
        seen = []

        def _wrap(event_type, payload, **kwargs):
            seen.append(event_type)
            return original(event_type, payload, **kwargs)

        monkeypatch.setattr(audit_mod, 'append_audit_record', _wrap)
        # write_artifacts imports append inside the function — patch the source module
        monkeypatch.setattr(
            'cat_agent.security.audit.append_audit_record', _wrap
        )

        spec = _spec('prov_audit_tool')
        work, holdout = spec.split_examples()
        write_artifacts(
            spec=spec,
            code='def prov_audit_tool(gross, rate):\n    return {}\n',
            executor_name='wasm',
            model_name='m',
            attempt_count=1,
            example_results=[],
            work=work,
            holdout=holdout,
            base=str(tmp_path),
            provenance={
                'draft_markdown': '# x\n',
                'draft_lang': 'en',
                'locale': None,
                'interview': {'turns': [], 'confirmed': True},
            },
        )
        assert 'synthesis.intake_completed' in seen
        assert (tmp_path / 'generated_tools' / 'prov_audit_tool' / 'draft.md').is_file()
