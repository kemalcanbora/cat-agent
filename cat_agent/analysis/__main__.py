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

"""CLI: ``python -m cat_agent.analysis <trace.jsonl>``."""

from __future__ import annotations

import argparse
import sys

from cat_agent.analysis import analyze_trace, analyze_traces
from cat_agent.analysis.report import render_json_report, render_text_report
from cat_agent.trace.store import load_runs_from_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m cat_agent.analysis',
        description='MAST failure analysis over cat-agent JSONL traces '
                    '(Cemri et al., arXiv:2503.13657).',
    )
    parser.add_argument('trace', nargs='+', help='Path(s) to trace JSONL file(s)')
    parser.add_argument('--json', action='store_true', help='Machine-readable JSON output')
    parser.add_argument(
        '--no-judge',
        action='store_true',
        help='Deterministic Tier-1 only (default when no judge model configured)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Aggregate statistics across all runs in the given files',
    )
    args = parser.parse_args(argv)

    tiers = ('deterministic',) if args.no_judge else ('deterministic',)
    # Judge requires explicit model wiring; CLI defaults to Tier-1 only.
    # Use --no-judge for clarity; without a configured judge we never send traces.

    if args.batch or len(args.trace) > 1:
        results, stats = analyze_traces(args.trace, tiers=tiers)
        if args.json:
            import json
            print(json.dumps({
                'results': [r.to_dict() for r in results],
                'batch': stats.to_dict(),
            }, indent=2, ensure_ascii=False))
        else:
            print(f'Batch: {stats.n_runs} runs')
            print(f'Most common category: {stats.most_common_category}')
            print('Mode prevalence:')
            for mode_id, rate in sorted(stats.mode_prevalence.items()):
                if rate > 0:
                    print(f'  {mode_id}: {rate:.1%}')
            for r in results:
                print()
                print(render_text_report(r))
        return 0

    runs = load_runs_from_jsonl(args.trace[0])
    if not runs:
        print(f'No runs found in {args.trace[0]}', file=sys.stderr)
        return 1
    for run in runs.values():
        result = analyze_trace(run, tiers=tiers)
        if args.json:
            print(render_json_report(result))
        else:
            print(render_text_report(result))
            print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
