"""Offline PII detection and redaction for regulated deployments."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Iterable, List, Optional, Union

from cat_agent.llm.schema import CONTENT, Message

PII_PLACEHOLDER = '[PII]'

_EMAIL = re.compile(
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
    re.IGNORECASE,
)
_PHONE = re.compile(
    r'(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3}[\s\-.]?\d{2}[\s\-.]?\d{2,4}',
)
_IBAN = re.compile(
    r'\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b',
    re.IGNORECASE,
)
_CREDIT_CARD = re.compile(
    r'\b(?:\d[ -]*?){13,19}\b',
)
_TURKISH_TC_ID = re.compile(r'\b[1-9]\d{10}\b')

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL, PII_PLACEHOLDER),
    (_IBAN, PII_PLACEHOLDER),
    (_CREDIT_CARD, PII_PLACEHOLDER),
    (_PHONE, PII_PLACEHOLDER),
)


def _env_flag(name: str, *, default: bool = True) -> bool:
    master = os.getenv('CAT_AGENT_PII_REDACT', '').strip().lower()
    if master in {'0', 'false', 'no', 'off'}:
        return False
    value = os.getenv(name, '').strip().lower()
    if not value:
        return default
    return value in {'1', 'true', 'yes', 'on'}


def is_pii_redact_rag_enabled() -> bool:
    return _env_flag('CAT_AGENT_PII_REDACT_RAG', default=True)


def is_pii_redact_prompts_enabled() -> bool:
    return _env_flag('CAT_AGENT_PII_REDACT_PROMPTS', default=True)


def is_pii_redact_audit_enabled() -> bool:
    return _env_flag('CAT_AGENT_PII_REDACT_AUDIT', default=True)


def _valid_turkish_tc_id(number: str) -> bool:
    if len(number) != 11 or number[0] == '0' or not number.isdigit():
        return False
    digits = [int(char) for char in number]
    tenth = (
        (digits[0] + digits[2] + digits[4] + digits[6] + digits[8]) * 7
        - (digits[1] + digits[3] + digits[5] + digits[7])
    ) % 10
    eleventh = sum(digits[:10]) % 10
    return digits[9] == tenth and digits[10] == eleventh


def _redact_turkish_tc_ids(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        candidate = match.group(0)
        return PII_PLACEHOLDER if _valid_turkish_tc_id(candidate) else candidate

    return _TURKISH_TC_ID.sub(replacer, text)


def _apply_regex_redaction(text: str) -> str:
    redacted = text
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return _redact_turkish_tc_ids(redacted)


def _try_presidio_redact(text: str) -> str:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return text

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language='en')
    if not results:
        return text
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def redact_text(text: str) -> str:
    if not text:
        return text
    redacted = _apply_regex_redaction(text)
    return _try_presidio_redact(redacted)


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


def maybe_redact_for_rag(text: str) -> str:
    if not is_pii_redact_rag_enabled():
        return text
    return redact_text(text)


def maybe_redact_for_prompt(text: str) -> str:
    if not is_pii_redact_prompts_enabled():
        return text
    return redact_text(text)


def maybe_redact_for_audit(payload: Any) -> Any:
    if not is_pii_redact_audit_enabled():
        return payload
    return redact_value(payload)


def maybe_redact_messages_for_prompt(messages: List[Message]) -> List[Message]:
    if not is_pii_redact_prompts_enabled():
        return messages

    redacted_messages: List[Message] = []
    for message in messages:
        cloned = copy.deepcopy(message)
        content = cloned.get(CONTENT)
        if isinstance(content, str):
            cloned[CONTENT] = maybe_redact_for_prompt(content)
        elif isinstance(content, list):
            for item in content:
                if hasattr(item, 'text') and item.text:
                    item.text = maybe_redact_for_prompt(item.text)
        redacted_messages.append(cloned)
    return redacted_messages


def redact_structured_doc(doc: List[dict]) -> List[dict]:
    if not is_pii_redact_rag_enabled():
        return doc

    redacted_doc = copy.deepcopy(doc)
    for page in redacted_doc:
        if 'title' in page and isinstance(page['title'], str):
            page['title'] = maybe_redact_for_rag(page['title'])
        for paragraph in page.get('content', []):
            if isinstance(paragraph.get('text'), str):
                paragraph['text'] = maybe_redact_for_rag(paragraph['text'])
    return redacted_doc
