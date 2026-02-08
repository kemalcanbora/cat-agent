"""PDF document parser using pdfminer + pdfplumber."""

from collections import Counter
from typing import List

from cat_agent.tools.parsers.base import clean_paragraph


def parse_pdf(pdf_path: str, extract_image: bool = False) -> List[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTImage, LTRect, LTTextContainer
    import pdfplumber

    doc = []
    pdf = pdfplumber.open(pdf_path)
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        page = {'page_num': page_layout.pageid, 'content': []}
        elements = list(page_layout)

        table_num = 0
        tables = []

        for element in elements:
            if isinstance(element, LTRect):
                if not tables:
                    tables = _extract_tables(pdf, i)
                if table_num < len(tables):
                    table_string = _table_to_string(tables[table_num])
                    table_num += 1
                    if table_string:
                        page['content'].append({'table': table_string, 'obj': element})
            elif isinstance(element, LTTextContainer):
                text = element.get_text()
                font = _get_font(element)
                if text.strip():
                    new_content_item = {'text': text, 'obj': element}
                    if font:
                        new_content_item['font-size'] = round(font[1])
                    page['content'].append(new_content_item)
            elif extract_image and isinstance(element, LTImage):
                raise ValueError('Currently, extracting images is not supported!')

        page['content'] = _postprocess_page_content(page['content'])
        doc.append(page)

    return doc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _postprocess_page_content(page_content: list) -> list:
    """Remove duplicates between table/text regions, then merge split paragraphs."""
    # Remove text elements that overlap with table bounding boxes
    table_obj = [p['obj'] for p in page_content if 'table' in p]
    filtered = []
    for p in page_content:
        if 'text' in p:
            overlaps_table = any(
                t.bbox[0] <= p['obj'].bbox[0] and p['obj'].bbox[1] <= t.bbox[1]
                and t.bbox[2] <= p['obj'].bbox[2] and p['obj'].bbox[3] <= t.bbox[3]
                for t in table_obj
            )
            if overlaps_table:
                continue
        filtered.append(p)

    # Merge paragraph lines that were split by mistake
    merged = []
    for p in filtered:
        if (merged
                and 'text' in merged[-1]
                and 'text' in p
                and abs(p.get('font-size', 12) - merged[-1].get('font-size', 12)) < 2
                and p['obj'].height < p.get('font-size', 12) + 1):
            merged[-1]['text'] += f' {p["text"]}'
            merged[-1]['font-size'] = p.get('font-size', 12)
        else:
            p.pop('obj', None)
            merged.append(p)

    for item in merged:
        if 'text' in item:
            item['text'] = clean_paragraph(item['text'])
        item.pop('obj', None)

    return merged


def _get_font(element):
    from pdfminer.layout import LTChar, LTTextContainer

    fonts_list = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    fonts_list.append((character.fontname, character.size))

    if fonts_list:
        counter = Counter(fonts_list)
        return counter.most_common(1)[0][0]
    return []


def _extract_tables(pdf, page_num):
    return pdf.pages[page_num].extract_tables()


def _table_to_string(table) -> str:
    rows = []
    for row in table:
        cleaned = [
            item.replace('\n', ' ') if item is not None and '\n' in item
            else 'None' if item is None
            else item
            for item in row
        ]
        rows.append('|' + '|'.join(cleaned) + '|')
    return '\n'.join(rows)
