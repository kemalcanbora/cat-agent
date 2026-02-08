"""Document parsers -- one module per file format.

Public API:
    parse_document(path, extract_image=False) -> list[dict]
    PARSER_SUPPORTED_FILE_TYPES
    PARAGRAPH_SPLIT_SYMBOL
    get_plain_doc(doc)
    DocParserError
"""

from cat_agent.tools.parsers.base import (  # noqa: F401
    DocParserError,
    PARAGRAPH_SPLIT_SYMBOL,
    clean_paragraph,
    get_plain_doc,
)

PARSER_SUPPORTED_FILE_TYPES = ['pdf', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls']


def parse_document(path: str, extract_image: bool = False, file_type: str = None) -> list:
    """Dispatch to the appropriate parser based on file extension.

    Args:
        path: Path to the document file.
        extract_image: Whether to extract images (limited support).
        file_type: Optional pre-computed file type. If ``None``, detected automatically.

    Returns the structured page list::

        [{'page_num': 1, 'content': [{'text': '...'}, {'table': '...'}]}, ...]
    """
    if file_type is None:
        from cat_agent.utils.file_utils import get_file_type
        file_type = get_file_type(path)

    f_type = file_type

    if f_type == 'pdf':
        from cat_agent.tools.parsers.pdf_parser import parse_pdf
        return parse_pdf(path, extract_image)
    elif f_type == 'docx':
        from cat_agent.tools.parsers.word_parser import parse_word
        return parse_word(path, extract_image)
    elif f_type == 'pptx':
        from cat_agent.tools.parsers.ppt_parser import parse_ppt
        return parse_ppt(path, extract_image)
    elif f_type == 'txt':
        from cat_agent.tools.parsers.txt_parser import parse_txt
        return parse_txt(path)
    elif f_type == 'html':
        from cat_agent.tools.parsers.html_parser import parse_html_bs
        return parse_html_bs(path, extract_image)
    elif f_type == 'csv':
        from cat_agent.tools.parsers.excel_parser import parse_csv
        return parse_csv(path, extract_image)
    elif f_type == 'tsv':
        from cat_agent.tools.parsers.excel_parser import parse_tsv
        return parse_tsv(path, extract_image)
    elif f_type in ('xlsx', 'xls'):
        from cat_agent.tools.parsers.excel_parser import parse_excel
        return parse_excel(path, extract_image)
    else:
        _t = '/'.join(PARSER_SUPPORTED_FILE_TYPES)
        raise ValueError(f'Failed: The current parser does not support this file type! Supported types: {_t}')
