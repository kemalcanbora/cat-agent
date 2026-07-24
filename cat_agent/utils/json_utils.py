"""JSON parsing, serialisation, and code-extraction helpers."""

import json
import re

import json5
from pydantic import BaseModel

from cat_agent.utils.misc import print_traceback


def json_loads(text: str) -> dict:
    text = text.strip().strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    text = _light_json_repair(text)
    try:
        result = json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            result = json5.loads(text)
        except ValueError:
            raise json_err
    # Some models double-encode tool args as a JSON string.
    for _ in range(2):
        if isinstance(result, str):
            result = json_loads(result)
        else:
            break
    return result


def _light_json_repair(text: str) -> str:
    """Fix common LLM tool-arg mistakes without changing valid JSON."""
    s = text.strip()
    # Trailing ')' instead of '}'
    if s.startswith('{') and s.endswith(')'):
        s = s[:-1] + '}'
    # Extra closing braces: {"key": "x"}}
    while s.startswith('{') and s.endswith('}') and s.count('{') < s.count('}'):
        s = s[:-1]
    # Trailing ']' instead of '}' for object payloads
    if s.startswith('{') and s.endswith(']') and s.count('{') >= s.count('}'):
        s = s[:-1] + '}'
    # Raw newlines inside the payload → escaped (helps multiline "content")
    if '\n' in s and s.startswith('{'):
        s = _escape_newlines_in_json_strings(s)
    return s


def _escape_newlines_in_json_strings(text: str) -> str:
    """Escape literal newlines that appear inside JSON string values."""
    out = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == '\\' and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == '\n':
            out.append('\\n')
            continue
        if in_string and ch == '\r':
            continue
        out.append(ch)
    return ''.join(out)

class PydanticJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def json_dumps_pretty(obj: dict, ensure_ascii=False, indent=2, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)


def json_dumps_compact(obj: dict, ensure_ascii=False, indent=None, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, cls=PydanticJSONEncoder, **kwargs)


def extract_code(text: str) -> str:
    """Extract code from a markdown-fenced block or a JSON ``{"code": ...}`` wrapper."""
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        text = triple_match.group(1)
    else:
        try:
            text = json5.loads(text)['code']
        except Exception:
            print_traceback(is_error=False)
    return text
