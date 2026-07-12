"""PDF document parser using the mandatory Rust text extractor."""

from importlib import import_module
from typing import List

from cat_agent.tools.parsers.base import clean_paragraph


def parse_pdf(pdf_path: str, extract_image: bool = False) -> List[dict]:
    if extract_image:
        raise ValueError('The native PDF parser does not support extracting images.')
    native = import_module('cat_agent._native')
    return [
        {
            'page_num': page_number,
            'content': [{'text': clean_paragraph(text.strip())}] if text.strip() else [],
        }
        for page_number, text in native.parse_pdf_text(pdf_path)
    ]
