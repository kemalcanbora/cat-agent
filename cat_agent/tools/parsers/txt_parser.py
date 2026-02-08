"""Plain-text (.txt) document parser."""

from typing import List

from cat_agent.tools.parsers.base import PARAGRAPH_SPLIT_SYMBOL
from cat_agent.utils.file_utils import read_text_from_file


def parse_txt(path: str) -> List[dict]:
    text = read_text_from_file(path)
    paras = text.split(PARAGRAPH_SPLIT_SYMBOL)
    content = [{'text': p} for p in paras]
    return [{'page_num': 1, 'content': content}]
