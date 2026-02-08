"""HTML document parser using BeautifulSoup."""

import re
from typing import List

from cat_agent.tools.parsers.base import PARAGRAPH_SPLIT_SYMBOL, clean_paragraph


def parse_html_bs(path: str, extract_image: bool = False) -> List[dict]:
    if extract_image:
        raise ValueError('Currently, extracting images is not supported!')

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ValueError('Please install bs4 by `pip install beautifulsoup4`')

    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, features='lxml')

    text = soup.get_text()
    title = str(soup.title.string) if soup.title else ''

    # Collapse multiple newlines
    text = re.sub(r'\n+', '\n', text)
    text = text.replace("Add to Qwen's Reading List", '')

    paras = text.split(PARAGRAPH_SPLIT_SYMBOL)
    content = []
    for p in paras:
        p = clean_paragraph(p)
        if p.strip():
            content.append({'text': p})

    return [{'page_num': 1, 'content': content, 'title': title}]
