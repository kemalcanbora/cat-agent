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

"""Public API: analyze_trace / analyze_traces.

Read-only over traces. Does not import cat_agent.agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from cat_agent.analysis.detectors import DetectorConfig, Finding, run_deterministic_detectors
from cat_agent.analysis.judge import judge_trace
from cat_agent.analysis.report import AnalysisResult, BatchStats, build_rollups, render_json_report, render_text_report
from cat_agent.analysis.taxonomy import CATEGORIES, MAST_MODES, MODES_BY_ID, PAPER_CITATION
from cat_agent.trace.schema import Run
from cat_agent.trace.store import load_runs_from_jsonl

__all__ = [
    'AnalysisResult',
    'BatchStats',
    'PAPER_CITATION',
    'analyze_trace',
    'analyze_traces',
    'render_json_report',
    'render_text_report',
]


def analyze_trace(
    run: Run,
    judge_llm: Any = None,
    tiers: Sequence[str] = ('deterministic', 'judge'),
    *,
    detector_config: Optional[DetectorConfig] = None,
    judge_opt_in: Optional[bool] = None,
) -> AnalysisResult:
    """Classify MAST failure modes present in *run*.

    When ``judge_llm`` is None, only Tier-1 deterministic detectors run.
    Deterministic findings are never overridden by the judge.
    """
    findings: List[Finding] = []
    locked_ids = set()

    if 'deterministic' in tiers:
        for f in run_deterministic_detectors(run, detector_config):
            findings.append(f)
            locked_ids.add(f.mode_id)

    if 'judge' in tiers and judge_llm is not None:
        opt_in = True if judge_opt_in is None else bool(judge_opt_in)
        for f in judge_trace(run, judge_llm, opt_in=opt_in):
            if f.mode_id in locked_ids:
                continue
            findings.append(f)

    findings.sort(key=lambda f: f.mode_id)
    rollups = build_rollups(findings)
    present = [f for f in findings if f.present]
    if present:
        names = [
            f'{f.mode_id} ({MODES_BY_ID[f.mode_id].name})'
            for f in present if f.mode_id in MODES_BY_ID
        ]
        summary = f'{len(present)} mode(s) present: ' + ', '.join(names)
    else:
        summary = 'No MAST failure modes detected.'

    return AnalysisResult(
        run_id=run.run_id,
        findings=findings,
        category_rollups=rollups,
        summary=summary,
    )


def analyze_traces(
    paths: Iterable[Union[str, Path]],
    judge_llm: Any = None,
    tiers: Sequence[str] = ('deterministic',),
) -> Tuple[List[AnalysisResult], BatchStats]:
    results: List[AnalysisResult] = []
    prevalence_counts = {m.id: 0 for m in MAST_MODES}
    cat_counts = {cid: 0 for cid in CATEGORIES}
    per_class: dict = {}

    for path in paths:
        runs = load_runs_from_jsonl(path)
        for run in runs.values():
            result = analyze_trace(run, judge_llm=judge_llm, tiers=tiers)
            results.append(result)
            agent_class = run.agent_class or 'unknown'
            per_class.setdefault(agent_class, {m.id: 0 for m in MAST_MODES})
            for f in result.findings:
                if f.present:
                    prevalence_counts[f.mode_id] = prevalence_counts.get(f.mode_id, 0) + 1
                    mode = MODES_BY_ID.get(f.mode_id)
                    if mode:
                        cat_counts[mode.category] += 1
                    per_class[agent_class][f.mode_id] = (
                        per_class[agent_class].get(f.mode_id, 0) + 1
                    )

    n = max(len(results), 1)
    prevalence = {k: v / n for k, v in prevalence_counts.items()}
    most_common = (
        max(cat_counts, key=lambda c: cat_counts[c]) if any(cat_counts.values()) else None
    )
    stats = BatchStats(
        n_runs=len(results),
        mode_prevalence=prevalence,
        most_common_category=(
            f'{most_common} {CATEGORIES[most_common]}' if most_common else None
        ),
        per_agent_class=per_class,
    )
    return results, stats
