# Context management

`cat_agent.memory.Memory` does **RAG over user files**. It does **not** manage the
growing conversation window during a long agent run. That is `cat_agent.context`.

These concerns stay separate on purpose: retrieval fills the window; context
management decides what stays.

References: Lindenbauer et al. (2025) [arXiv:2508.21433](https://arxiv.org/abs/2508.21433);
Sun et al. (2025) [arXiv:2510.11967](https://arxiv.org/abs/2510.11967);
Mei et al. (2025) [arXiv:2507.13334](https://arxiv.org/abs/2507.13334);
Hu et al. (2026) [arXiv:2603.07670](https://arxiv.org/abs/2603.07670);
[Anthropic — Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

## Decision table

| Situation | Strategy |
| --- | --- |
| Default / cheap long tool loops | **Observation masking** (default) |
| Need a narrative of decisions after masking is not enough | **Summary compaction** (optional LLM) |
| Deep sub-task that would blow the window | **Context folding** (`with mgr.fold(...)`) |
| Still over budget after all strategies | **`ContextOverflowError`** (never silent truncate) |

## Usage

```python
from cat_agent.agents import Assistant
from cat_agent.context import ContextManager, ObservationMaskingStrategy, SummaryCompactionStrategy

mgr = ContextManager(strategies=[
    ObservationMaskingStrategy(keep_recent=3),
    SummaryCompactionStrategy(llm=cheap_llm, persist_dir='workspace/context_archive'),
])
bot = Assistant(llm={...}, context_manager=mgr)

# Disable: context_manager=False  or  CAT_AGENT_CONTEXT=0
```

Folding is explicit:

```python
with bot.context_manager.fold(task='enumerate failing pods') as sub:
    # run scratch work against sub.scratch
    sub.set_result('3 CrashLoopBackOff pods in ns=payments')
messages = bot.context_manager.fold_into(messages, sub)
```

Every applied strategy emits a `context_op` step into the execution trace when
tracing is enabled.

## Measured reduction (single example — not a general claim)

These numbers are **one scripted demo under stated conditions**. They are not a
framework-wide efficiency claim.

### Primary: quality A/B with traced prompt tokens

One agent loop, one stub model, seed `42`. Answers checked equivalent; prompt
tokens come from `Run.totals` (the same `cat_agent.trace` accounting users get
with `CAT_AGENT_TRACE` / `trace=True`).

| Condition | Value |
| --- | --- |
| Script | `examples/long_horizon_agent/run.py` → `run_quality_ab()` |
| Model | `stub-kube-llm` (`DeterministicKubeLLM`), seed **42** |
| Task | 8 kubectl log fetches (pods 0–7) then a final answer; **pod-3** has a single-occurrence mid-log outlier (`exit_code=1` + one-line `ConfigError` stack); others are OOM / `exit_code=137` |
| OFF | `context_manager=False` |
| ON | `ObservationMaskingStrategy(keep_recent=2)`, `max_context_tokens=3200`, `trigger_ratio=0.4`, structured residue (head/tail + repeats + IDF salient mid-lines) |
| Metric | Sum of **prompt tokens** across all LLM-call steps (`RunTotals.prompt_tokens`); stub omits usage metadata so tokens are **estimated** via the trace tokenizer fallback (`tokens_estimated=True`) |
| Result | **31094 → 15269** prompt tokens across **9** LLM turns (**50.9%** fewer prompts under masking) |
| Quality | Final answers **equivalent** (entities: pods, exit codes, `ConfigError` / `OOMKilled`); tool-call sequences **identical**; outlier recovered in the masked answer |

Residue (pluggable `ResidueRegistry` / `generic_residue_extractor`) is what keeps
the one-off outlier visible after elision. That costs some compression versus
head/tail + repeats alone; the A/B figure above already includes that tradeoff.

### Separate: static `prepare()` fixture (synthetic upper bound)

A different measurement: one shot of `ContextManager.prepare()` on a fixed
history, **no LLM turns**, heuristic `HeuristicTokenCounter` only. Useful as a
quick compression ceiling for the fixture text — **not** comparable to the A/B
prompt-token total and **not** “same answer, fewer tokens.”

| Condition | Value |
| --- | --- |
| Script | `examples/long_horizon_agent/run.py` → `run_token_demo()` |
| Input | Static history: system + user + 8× (assistant call + bulky log) + final line |
| Counter | Built-in ``o200k_base`` tiktoken heuristic — **not** a live model tokenizer |
| Result | 6948 → 2460 heuristic tokens (**64.6%** on that snapshot) |

## Invariants

- System prompt and original user task survive every strategy
- Message ordering preserved; role sequencing stays legal (no tool result without a preceding tool call)
- Token count never increases
- Under-threshold histories are returned unchanged
- Multimodal (image) parts are preserved when masking or compacting text
