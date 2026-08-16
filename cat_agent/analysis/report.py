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

"""Analysis report rendering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from cat_agent.analysis.detectors import Finding
from cat_agent.analysis.taxonomy import CATEGORIES, MODES_BY_ID, PAPER_CITATION


@dataclass
class AnalysisResult:
    run_id: str
    findings: List[Finding] = field(default_factory=list)
    category_rollups: Dict[str, int] = field(default_factory=dict)
    summary: str = ''
    limitations: str = (
        'MAST was developed on multi-agent traces (Cemri et al., 2025). '
        'Deterministic detectors are our operationalisation of published definitions, '
        'not the paper\'s annotation method. Judge scores are noisy estimates.'
    )

    def present_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.present]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'findings': [asdict(f) for f in self.findings],
            'category_rollups': self.category_rollups,
            'summary': self.summary,
            'limitations': self.limitations,
            'citation': PAPER_CITATION,
        }


def build_rollups(findings: List[Finding]) -> Dict[str, int]:
    counts = {cid: 0 for cid in CATEGORIES}
    for f in findings:
        if not f.present:
            continue
        mode = MODES_BY_ID.get(f.mode_id)
        if mode:
            counts[mode.category] += 1
    return counts


def render_text_report(result: AnalysisResult) -> str:
    lines = [
        f'MAST analysis for run {result.run_id}',
        f'Citation: {PAPER_CITATION}',
        '',
        result.summary or '(no summary)',
        '',
        'Category rollups (present modes):',
    ]
    for cid, name in CATEGORIES.items():
        lines.append(f'  {cid} {name}: {result.category_rollups.get(cid, 0)}')
    lines.append('')
    lines.append('Findings:')
    for f in result.findings:
        mode = MODES_BY_ID.get(f.mode_id)
        label = mode.name if mode else f.mode_id
        flag = 'YES' if f.present else 'no'
        src = 'det' if f.deterministic else 'judge'
        lines.append(
            f'  [{flag}] {f.mode_id} {label} ({src}, conf={f.confidence:.2f}) '
            f'steps={f.evidence_steps} — {f.explanation}'
        )
    lines.append('')
    lines.append(f'Limitations: {result.limitations}')
    return '\n'.join(lines)


def render_json_report(result: AnalysisResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class BatchStats:
    n_runs: int = 0
    mode_prevalence: Dict[str, float] = field(default_factory=dict)
    most_common_category: Optional[str] = None
    per_agent_class: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
