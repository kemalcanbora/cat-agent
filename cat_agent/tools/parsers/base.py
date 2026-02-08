"""Shared constants, errors and helpers used by all parsers."""

from typing import Optional

from cat_agent.utils.str_processing import rm_cid, rm_continuous_placeholders, rm_hexadecimal

PARAGRAPH_SPLIT_SYMBOL = '\n'


class DocParserError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[dict] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


def clean_paragraph(text: str) -> str:
    text = rm_cid(text)
    text = rm_hexadecimal(text)
    text = rm_continuous_placeholders(text)
    return text


def get_plain_doc(doc: list) -> str:
    paras = []
    for page in doc:
        for para in page['content']:
            for k, v in para.items():
                if k in ('text', 'table', 'image'):
                    paras.append(v)
    return PARAGRAPH_SPLIT_SYMBOL.join(paras)
