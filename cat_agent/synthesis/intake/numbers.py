"""Locale-aware cell parsing for intake example tables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Narrow no-break space (U+202F) and no-break space (U+00A0) — Excel emits these.
_THOUSANDS_SPACES = ('\u00a0', '\u202f', ' ')

# Locale families for decimal / thousands separators.
_LOCALE_DOT_DECIMAL = {
    'en', 'en-ie', 'en-mt', 'en-gb', 'en-us', 'en-au', 'en-ca',
}
_LOCALE_COMMA_DECIMAL_DOT_THOUSANDS = {
    'de', 'de-de', 'de-at', 'de-ch',
    'es', 'es-es', 'es-mx',
    'it', 'it-it',
    'nl', 'nl-nl', 'nl-be',
    'tr', 'tr-tr',
    'pt', 'pt-pt', 'pt-br',
}
_LOCALE_COMMA_DECIMAL_SPACE_THOUSANDS = {
    'fr', 'fr-fr', 'fr-be', 'fr-ch',
    'pl', 'pl-pl',
    'sv', 'sv-se',
    'fi', 'fi-fi',
    'cs', 'cs-cz',
    'sk', 'sk-sk',
    'no', 'nb-no', 'nn-no',
}

_CURRENCY = set('€£$¥₺złkr')
_BOOL_TRUE = {'true', 'yes', 'y', 'evet', 'oui', 'sí', 'si', 'ja', 'wahr', 'vrai'}
_BOOL_FALSE = {'false', 'no', 'n', 'hayır', 'hayir', 'non', 'nein', 'falsch', 'faux'}

_DATE_LIKE = re.compile(
    r'^\d{1,4}([./\-])\d{1,2}\1\d{1,4}$'
)


@dataclass
class CellParse:
    """Result of parsing one table cell."""

    value: Any
    ok: bool = True
    ambiguous: bool = False
    question: Optional[str] = None
    unit: Optional[str] = None
    raw: str = ''


@dataclass
class ColumnParseReport:
    ambiguous: List[Tuple[int, str, str]] = field(default_factory=list)
    # (row_index, raw_cell, question)


def normalise_locale(locale: Optional[str]) -> Optional[str]:
    if not locale:
        return None
    return locale.strip().lower().replace('_', '-')


def locale_family(locale: Optional[str]) -> Optional[str]:
    """Return ``dot``, ``comma_dot``, ``comma_space``, or ``None`` if unknown."""
    key = normalise_locale(locale)
    if not key:
        return None
    if key in _LOCALE_DOT_DECIMAL or key.startswith('en-'):
        return 'dot'
    if key in _LOCALE_COMMA_DECIMAL_DOT_THOUSANDS or key.split('-')[0] in {
        'de', 'es', 'it', 'nl', 'tr', 'pt',
    }:
        return 'comma_dot'
    if key in _LOCALE_COMMA_DECIMAL_SPACE_THOUSANDS or key.split('-')[0] in {
        'fr', 'pl', 'sv', 'fi', 'cs', 'sk', 'no', 'nb', 'nn',
    }:
        return 'comma_space'
    return None


def _strip_unit(text: str) -> Tuple[str, Optional[str]]:
    s = text.strip()
    unit = None
    if s.endswith('%'):
        unit = '%'
        s = s[:-1].strip()
    elif s and s[-1] in _CURRENCY:
        unit = s[-1]
        s = s[:-1].strip()
    elif s and s[0] in _CURRENCY:
        unit = s[0]
        s = s[1:].strip()
    return s, unit


def _has_space_thousands(text: str) -> bool:
    for sep in _THOUSANDS_SPACES:
        if sep in text:
            return True
    return False


def _collapse_thousands_spaces(text: str) -> str:
    out = text
    for sep in _THOUSANDS_SPACES:
        out = out.replace(sep, '')
    return out


def is_ambiguous_number(text: str) -> bool:
    """True when the token needs a locale to decide between thousand vs decimal."""
    s, _ = _strip_unit(text)
    s = s.strip()
    if not s:
        return False
    # Compound forms with BOTH separators are unambiguous.
    if re.fullmatch(r'\d{1,3}(\.\d{3})+,\d+', s):
        return False  # 1.500,50 → de-style
    if re.fullmatch(r'\d{1,3}(,\d{3})+\.\d+', s):
        return False  # 1,500.50 → en-style
    if _has_space_thousands(s) and ',' in s:
        return False  # 1 500,50 → fr-style
    if _has_space_thousands(s) and '.' in s and re.search(r'\d\.\d{3}\b', s):
        return True
    # Classic collisions without locale
    if re.fullmatch(r'\d{1,3},\d{3}', s):  # 1,500
        return True
    if re.fullmatch(r'\d{1,3}\.\d{3}', s):  # 1.500
        return True
    if re.fullmatch(r'\d{1,3},\d{1,2}', s):  # 1,5
        return True
    return False


def is_ambiguous_date(text: str) -> bool:
    s = text.strip()
    if not _DATE_LIKE.match(s):
        return False
    # ISO YYYY-MM-DD is fine
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        return False
    parts = re.split(r'[./\-]', s)
    if len(parts) != 3:
        return False
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        return False
    # Both first components could be day or month
    if 1 <= a <= 12 and 1 <= b <= 12 and a != b:
        return True
    return False


def parse_number_with_locale(text: str, family: str) -> Optional[float]:
    s, _ = _strip_unit(text)
    s = s.strip()
    if not s:
        return None
    try:
        if family == 'dot':
            cleaned = _collapse_thousands_spaces(s).replace(',', '')
            return float(cleaned) if '.' in cleaned or 'e' in cleaned.lower() else float(int(cleaned))
        if family == 'comma_dot':
            cleaned = _collapse_thousands_spaces(s).replace('.', '').replace(',', '.')
            return float(cleaned)
        if family == 'comma_space':
            cleaned = _collapse_thousands_spaces(s).replace(',', '.')
            return float(cleaned)
    except ValueError:
        return None
    return None


def _looks_like_int(value: float) -> bool:
    return float(value).is_integer()


def parse_cell(
    raw: Any,
    *,
    locale: Optional[str] = None,
    prefer_native: bool = True,
) -> CellParse:
    """Parse a single cell. Never guess when locale-ambiguous."""
    if prefer_native and not isinstance(raw, str):
        if raw is None:
            return CellParse(value=None, raw='')
        if isinstance(raw, bool):
            return CellParse(value=raw, raw=str(raw))
        if isinstance(raw, int):
            return CellParse(value=raw, raw=str(raw))
        if isinstance(raw, float):
            if _looks_like_int(raw):
                return CellParse(value=int(raw), raw=str(raw))
            return CellParse(value=raw, raw=str(raw))
        raw = str(raw)

    text = '' if raw is None else str(raw)
    stripped = text.strip()
    if stripped == '':
        return CellParse(value=None, raw=text)

    low = stripped.lower()
    if low in _BOOL_TRUE:
        return CellParse(value=True, raw=text)
    if low in _BOOL_FALSE:
        return CellParse(value=False, raw=text)

    body, unit = _strip_unit(stripped)

    if is_ambiguous_date(body):
        return CellParse(
            value=stripped,
            ok=False,
            ambiguous=True,
            question=(
                f'In your examples, is `{stripped}` day-first (DD/MM) or month-first (MM/DD)?'
            ),
            raw=text,
        )

    family = locale_family(locale)

    if is_ambiguous_number(body) and family is None:
        return CellParse(
            value=stripped,
            ok=False,
            ambiguous=True,
            question=(
                f'In your examples, does `{stripped}` mean one thousand five hundred, '
                f'or one point five? Please tell us which, or set a Locale (e.g. de-DE, en-IE).'
            ),
            unit=unit,
            raw=text,
        )

    # Unambiguous compound forms even without locale
    if family is None:
        if re.fullmatch(r'\d{1,3}(\.\d{3})+,\d+', body):
            family = 'comma_dot'
        elif re.fullmatch(r'\d{1,3}(,\d{3})+\.\d+', body):
            family = 'dot'
        elif _has_space_thousands(body) and ',' in body:
            family = 'comma_space'
        elif re.fullmatch(r'-?\d+$', body):
            value: Any = int(body)
            if unit:
                return CellParse(value=value, unit=unit, raw=text)
            return CellParse(value=value, raw=text)
        elif re.fullmatch(r'-?\d+\.\d+$', body):
            # Bare decimal with dot and no thousands grouping — safe as float
            value = float(body)
            if _looks_like_int(value):
                value = int(value)
            return CellParse(value=value, unit=unit, raw=text)
        elif re.fullmatch(r'-?\d+,\d+$', body):
            # 1,5 without locale — ambiguous between EU decimal and en thousands-ish
            return CellParse(
                value=stripped,
                ok=False,
                ambiguous=True,
                question=(
                    f'In your examples, does `{stripped}` mean a decimal number '
                    f'(one point five), or something else? Please set a Locale.'
                ),
                unit=unit,
                raw=text,
            )

    if family is not None:
        if re.fullmatch(r'-?\d+$', body):
            value = int(body)
            return CellParse(value=value, unit=unit, raw=text)
        num = parse_number_with_locale(body, family)
        if num is not None:
            value = int(num) if _looks_like_int(num) else num
            return CellParse(value=value, unit=unit, raw=text)

    # ISO date
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', body):
        return CellParse(value=body, raw=text)

    return CellParse(value=stripped, raw=text)


def parse_column_values(
    cells: Sequence[Any],
    *,
    locale: Optional[str] = None,
) -> Tuple[List[Any], ColumnParseReport]:
    """Parse a column; collect ambiguity questions without guessing."""
    report = ColumnParseReport()
    values: List[Any] = []
    # First pass: if any cell is ambiguous and no locale, flag all ambiguous ones
    parsed = [parse_cell(c, locale=locale) for c in cells]
    for index, cell in enumerate(parsed):
        if cell.ambiguous:
            report.ambiguous.append((index, cell.raw, cell.question or ''))
            values.append(cell.raw)  # keep raw until resolved
        else:
            values.append(cell.value)
    return values, report
