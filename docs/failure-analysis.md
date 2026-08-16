# Trajectory failure analysis (MAST)

`analyze_trace(run)` classifies failure modes in a recorded execution trace using
the **MAST** taxonomy from:

> Cemri, M. et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657, NeurIPS 2025.

## Honest limitations

- MAST was developed and validated on **multi-agent** traces (κ = 0.88 on expert labels).
- Our **deterministic detectors** are an operationalisation of the published definitions
  for cat-agent's trace schema — **not** the paper's annotation method.
- **LLM-as-judge** scores are noisy estimates, not measurements. Do not treat output
  percentages as the paper's reported prevalence figures.
- Trace content is **never** sent to a judge model without explicit opt-in.

## Taxonomy (14 modes / 3 categories)

| ID | Category | Name | Tier |
| --- | --- | --- | --- |
| 1.1 | System Design Issues | Disobey Task Specification | Judge |
| 1.2 | System Design Issues | Disobey Role Specification | Judge |
| 1.3 | System Design Issues | Step Repetition | **Deterministic** — contiguous successful stuck-loop only; excludes retries, pagination, and spaced re-reads |
| 1.4 | System Design Issues | Loss of Conversation History | **Deterministic** |
| 1.5 | System Design Issues | Unaware of Termination Conditions | **Deterministic** |
| 2.1 | Inter-Agent Misalignment | Conversation Reset | Judge |
| 2.2 | Inter-Agent Misalignment | Fail to Ask for Clarification | Judge |
| 2.3 | Inter-Agent Misalignment | Task Derailment | Judge |
| 2.4 | Inter-Agent Misalignment | Information Withholding | Judge |
| 2.5 | Inter-Agent Misalignment | Ignored Other Agent's Input | Judge |
| 2.6 | Inter-Agent Misalignment | Reasoning-Action Mismatch | Judge |
| 3.1 | Task Verification | Premature Termination | Judge |
| 3.2 | Task Verification | No or Incomplete Verification | Judge |
| 3.3 | Task Verification | Incorrect Verification | Judge |

Definitions follow Appendix A of the paper (see `cat_agent/analysis/taxonomy.py`).

## API

```python
from cat_agent.analysis import analyze_trace
from cat_agent.trace import load_runs_from_jsonl

run = next(iter(load_runs_from_jsonl('traces.jsonl').values()))
result = analyze_trace(run)                    # Tier-1 only (no API key)
result = analyze_trace(run, judge_llm=llm)      # Tier-1 + Tier-2 (opt-in)
print(result.summary)
```

CLI:

```bash
python -m cat_agent.analysis traces.jsonl
python -m cat_agent.analysis traces.jsonl --json --no-judge
python -m cat_agent.analysis *.jsonl --batch
```

Deterministic findings are never overridden by the judge.

See `examples/failure_analysis/` for an end-to-end failing agent + analysis.
