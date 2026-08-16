# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pluggable factual-residue extractors for observation masking.

When bulk tool output is elided, a compact residue keeps identifiers, counts,
repeated status tokens, and low-frequency salient middle lines so downstream
reasoning does not lose structured facts (e.g. a one-off exit code buried in
a long log dump).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

ResidueExtractor = Callable[[str, str], str]


@runtime_checkable
class ResidueExtractorProtocol(Protocol):
    def __call__(self, tool_name: str, text: str) -> str: ...


# Status-like tokens that often carry the factual signal in logs / API dumps.
_STATUS_TOKEN = re.compile(
    r'\b(?:'
    r'ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL|'
    r'OOMKilled|CrashLoopBackOff|ImagePullBackOff|Pending|Running|Succeeded|Failed|'
    r'OK|FAIL|TIMEOUT|DENIED|UNAUTHORIZED|'
    r'(?:HTTP[/\s-]?)?(?:[1-5]\d{2})|'  # status codes 100–599
    r'pod-\d+|node-\d+|ns/[A-Za-z0-9._-]+|'
    r'[A-Za-z][\w.-]*Error'
    r')\b',
    re.IGNORECASE,
)

_LINE_TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_.-]*|\d+')

# Soft preference for lines that look like facts even among rare lines.
_FACTISH = re.compile(
    r'exit[_ ]?codes?|traceback|exception|\b[A-Z][A-Za-z0-9]*(?:Error|Exception)\b|'
    r'\b(?:FATAL|CRITICAL|DENIED|UNAUTHORIZED)\b',
    re.IGNORECASE,
)


def generic_residue_extractor(
    tool_name: str,
    text: str,
    *,
    head_lines: int = 2,
    tail_lines: int = 2,
    max_line_chars: int = 120,
    min_status_repeats: int = 2,
    max_status_kinds: int = 12,
    salient_top_k: int = 3,
) -> str:
    """Default residue: head/tail lines + repeated status/ids + rare mid lines.

    Middle lines are scored by mean inverse document frequency of their tokens
    within *this* tool output (cheap, no LLM). Top-``salient_top_k`` outliers
    are kept so a single-occurrence fact (e.g. ``exit_code=1`` mid-dump) is not
    dropped solely because it never repeats.
    """
    if not text:
        return ''
    lines = text.splitlines() or [text]

    def _clip(ln: str) -> str:
        ln = ln.strip()
        if len(ln) <= max_line_chars:
            return ln
        half = max_line_chars // 2
        return f'{ln[:half]}…{ln[-half:]}'

    if len(lines) <= head_lines + tail_lines:
        kept_lines = [_clip(ln) for ln in lines]
        salient: List[str] = []
    else:
        kept_lines = [_clip(ln) for ln in lines[:head_lines]]
        kept_lines.append('…')
        kept_lines.extend(_clip(ln) for ln in lines[-tail_lines:])
        salient = [
            _clip(ln)
            for ln in _salient_middle_lines(
                lines,
                head_lines=head_lines,
                tail_lines=tail_lines,
                top_k=salient_top_k,
            )
            if _clip(ln)
        ]

    tokens = _STATUS_TOKEN.findall(text)
    counts = Counter(t if not t.isdigit() else t for t in tokens)
    status_bits = []
    for tok, n in counts.most_common():
        if n >= min_status_repeats or re.match(r'(?i)pod-\d+|node-\d+|ns/', tok) or tok.isdigit():
            status_bits.append(f'{tok}×{n}' if n > 1 else tok)
        if len(status_bits) >= max_status_kinds:
            break

    parts = [f'[residue:{tool_name}]']
    if kept_lines:
        parts.append('lines: ' + ' | '.join(ln for ln in kept_lines if ln))
    if salient:
        parts.append('salient: ' + ' | '.join(salient))
    if status_bits:
        parts.append('status: ' + ', '.join(status_bits))
    return '\n'.join(parts)


def _line_tokens(line: str) -> List[str]:
    return [t.lower() for t in _LINE_TOKEN.findall(line)]


def _salient_middle_lines(
    lines: Sequence[str],
    *,
    head_lines: int,
    tail_lines: int,
    top_k: int,
) -> List[str]:
    """Return up to *top_k* mid-body lines with highest mean token IDF."""
    n = len(lines)
    if top_k <= 0 or n <= head_lines + tail_lines:
        return []
    protected = set(range(head_lines)) | set(range(max(head_lines, n - tail_lines), n))

    line_tok_sets: List[set] = []
    df: Counter = Counter()
    for ln in lines:
        toks = set(_line_tokens(ln))
        line_tok_sets.append(toks)
        for t in toks:
            df[t] += 1

    n_lines = float(max(n, 1))
    scored: List[Tuple[float, int, str]] = []
    for i, ln in enumerate(lines):
        if i in protected:
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        toks = line_tok_sets[i]
        # Ignore pure numerics so unique seq=/index values do not look "rare".
        content_toks = {t for t in toks if not t.isdigit()}
        if not content_toks:
            continue
        # Mean smoothed IDF — tokens that appear in few lines score higher.
        idf = sum(
            math.log((n_lines + 1.0) / (df[t] + 1.0)) + 1.0 for t in content_toks
        ) / len(content_toks)
        if _FACTISH.search(stripped):
            idf += 2.0
        scored.append((idf, i, stripped))

    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored:
        return []

    # Prefer fact-like lines. Pad with other rare lines only when they score
    # *strictly above* the non-factish median — otherwise a sea of near-identical
    # filler lines (unique seq=N but shared vocabulary) all tie at the decile cut.
    factish = [x for x in scored if _FACTISH.search(x[2])]
    non_fact = [x for x in scored if not _FACTISH.search(x[2])]
    if non_fact:
        nf_scores = sorted(s for s, _, _ in non_fact)
        nf_med = nf_scores[len(nf_scores) // 2]
        other = [x for x in non_fact if x[0] > nf_med]
    else:
        other = []

    chosen: List[Tuple[int, str]] = []
    seen: set = set()
    for _score, i, ln in factish + other:
        key = ln.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((i, ln))
        if len(chosen) >= top_k:
            break
    chosen.sort(key=lambda x: x[0])
    return [ln for _, ln in chosen]


class ResidueRegistry:
    """Per-tool extractors with a generic default."""

    def __init__(
        self,
        default: Optional[ResidueExtractor] = None,
        extractors: Optional[Dict[str, ResidueExtractor]] = None,
    ) -> None:
        self.default: ResidueExtractor = default or generic_residue_extractor
        self._by_tool: Dict[str, ResidueExtractor] = dict(extractors or {})

    def register(self, tool_name: str, extractor: ResidueExtractor) -> None:
        self._by_tool[tool_name] = extractor

    def extract(self, tool_name: str, text: str) -> str:
        fn = self._by_tool.get(tool_name, self.default)
        return fn(tool_name, text)


DEFAULT_RESIDUE_REGISTRY = ResidueRegistry()
