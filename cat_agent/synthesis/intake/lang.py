"""Lightweight working-language detection for intake (D5)."""

from __future__ import annotations

import re
from typing import Optional

# Stopword / function-word hints — enough to prefer one EU language over English.
_HINTS = {
    'de': re.compile(
        r'\b(und|oder|nicht|für|über|soll|werden|eingabe|ausgabe|regel|wenn|dann)\b',
        re.I,
    ),
    'fr': re.compile(
        r'\b(et|ou|pas|pour|dans|doit|règle|regle|lorsque|montant|taux)\b',
        re.I,
    ),
    'es': re.compile(
        r'\b(y|o|no|para|debe|regla|cuando|importe|entrada|salida)\b',
        re.I,
    ),
    'it': re.compile(
        r'\b(e|o|non|per|deve|regola|quando|importo|ingresso|uscita)\b',
        re.I,
    ),
    'nl': re.compile(
        r'\b(en|of|niet|voor|moet|regel|wanneer|bedrag|invoer|uitvoer)\b',
        re.I,
    ),
    'tr': re.compile(
        r'\b(ve|veya|için|icin|olmalı|olmali|kural|eğer|eger|tutar|oran)\b',
        re.I,
    ),
}

# Uniquely Turkish letters (ü/ö also appear in German — do not use them here).
_TURKISH_CHARS = re.compile(r'[ıİşŞğĞ]')
_CJK = re.compile(r'[\u4e00-\u9fff]')


def detect_lang(text: str, override: Optional[str] = None) -> str:
    """Detect the draft's working language. Default English when inconclusive."""
    if override:
        return override.strip().lower().split('-')[0]
    sample = (text or '')[:8000]
    if not sample.strip():
        return 'en'
    if _CJK.search(sample):
        return 'zh'

    scores = {code: len(pat.findall(sample)) for code, pat in _HINTS.items()}
    # Boost Turkish when its distinctive letters appear.
    if _TURKISH_CHARS.search(sample):
        scores['tr'] = scores.get('tr', 0) + 3

    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    return 'en'
