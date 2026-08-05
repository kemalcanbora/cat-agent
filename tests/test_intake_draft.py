"""Tests for intake draft loading and locale-aware number parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from cat_agent.synthesis.intake.draft import Draft, extract_markdown_tables
from cat_agent.synthesis.intake.numbers import is_ambiguous_number, parse_cell
from cat_agent.synthesis.intake.template import write_template


class TestTableExtraction:

    def test_finds_table_under_renamed_heading(self):
        md = """\
# Tool

## Mes exemples personnalisés
| a | result |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
"""
        draft = Draft.from_markdown(md, locale='en-IE')
        assert len(draft.examples) == 3
        assert draft.examples[0].inputs == {'a': 1}
        assert draft.examples[0].expected == 2

    def test_non_english_headings(self):
        md = """\
# Werkzeug

## Was auch immer
Text

## Tabelle
| x | ergebnis |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
"""
        draft = Draft.from_markdown(md, locale='de-DE')
        assert len(draft.examples) == 3

    def test_largest_table_wins_with_note(self):
        md = """\
| a | r |
|---|---|
| 1 | 1 |

| b | c | r |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |
"""
        draft = Draft.from_markdown(md, locale='en-IE')
        assert draft.example_columns == ['b', 'c', 'r']
        assert draft.table_ambiguity
        assert any(q.kind == 'table' for q in draft.open_questions)

    def test_missing_table_actionable(self):
        with pytest.raises(ValueError, match='No examples table'):
            Draft.from_markdown('# No table here\n')

    def test_extract_multiple(self):
        md = """\
| a | r |
|---|---|
| 1 | 2 |

| b | r |
|---|---|
| 3 | 4 |
| 5 | 6 |
"""
        tables = extract_markdown_tables(md)
        assert len(tables) == 2


class TestLocaleMatrix:

    @pytest.mark.parametrize(
        'raw,locale,expected',
        [
            ('1,500', 'en-IE', 1500),
            ('1,500', 'en-MT', 1500),
            ('1,500', 'de-DE', 1.5),
            ('1,500', 'es-ES', 1.5),
            ('1,500', 'it-IT', 1.5),
            ('1.500', 'de-DE', 1500),
            ('1.500', 'en-IE', 1.5),
            ('1.500,50', 'de-DE', 1500.5),
            ('1 500,50', 'fr-FR', 1500.5),
            ('1\u00a0500,50', 'fr-FR', 1500.5),  # NBSP
            ('1\u202f500,50', 'fr-FR', 1500.5),  # narrow NBSP
        ],
    )
    def test_locale_number_matrix(self, raw, locale, expected):
        cell = parse_cell(raw, locale=locale)
        assert cell.ok and not cell.ambiguous
        assert cell.value == expected

    def test_ambiguous_without_locale(self):
        cell = parse_cell('1,500')
        assert cell.ambiguous
        assert cell.question
        assert '1,500' in cell.question
        assert is_ambiguous_number('1,500')
        assert is_ambiguous_number('1.500')

    def test_percent_and_currency(self):
        p = parse_cell('20%', locale='en-IE')
        assert p.value == 20
        assert p.unit == '%'
        e = parse_cell('12,50€', locale='de-DE')
        assert e.value == 12.5
        assert e.unit == '€'
        p2 = parse_cell('£1,500', locale='en-IE')
        assert p2.value == 1500
        assert p2.unit == '£'

    def test_draft_propagates_ambiguous(self):
        md = """\
# t
| amount | result |
|---|---|
| 1,500 | 2 |
| 2 | 3 |
| 3 | 4 |
"""
        draft = Draft.from_markdown(md)  # no locale
        assert any(q.kind == 'number' for q in draft.open_questions)

    def test_german_vat_draft(self):
        path = Path('examples/synthesis/from_draft/vat_draft_de.md')
        draft = Draft.from_path(path)
        assert draft.locale and draft.locale.lower().startswith('de')
        assert draft.detected_lang == 'de'
        values = [ex.inputs['brutto'] for ex in draft.examples]
        assert 1500.5 in values


class TestTemplate:

    def test_write_template_fallback(self, tmp_path):
        path = write_template(tmp_path / 'x.md', lang='xx')
        text = path.read_text(encoding='utf-8')
        assert 'What it should do' in text

    def test_translated_templates_exist(self, tmp_path):
        for lang in ('en', 'de', 'fr', 'es', 'it', 'nl', 'tr'):
            p = write_template(tmp_path / f'{lang}.md', lang=lang)
            assert p.read_text(encoding='utf-8').strip()


class TestExcelPath:

    def test_from_excel_matches_markdown(self, tmp_path):
        pytest.importorskip('openpyxl')
        pytest.importorskip('polars')
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(['x', 'result'])
        ws.append([1, 2])
        ws.append([2, 3])
        ws.append([3, 4])
        path = tmp_path / 'ex.xlsx'
        wb.save(path)

        draft = Draft.from_excel(path, description_md='# excel tool\n\nAdds one.')
        assert len(draft.examples) == 3
        assert draft.examples[0].inputs == {'x': 1}
        assert draft.examples[0].expected == 2
        # Native ints — no locale ambiguity
        assert draft.open_questions == [] or all(
            q.kind != 'number' for q in draft.open_questions
        )
