#!/usr/bin/env python3
"""Long-horizon context demo: quality A/B + optional static prepare() ceiling.

Primary measurement (``run_quality_ab``): same stubbed agent twice (seed 42)
with ``context_manager=False`` vs observation masking. Compares final answers,
tool-call sequences, and **prompt tokens from ``Run.totals``** across all LLM
turns (trace accounting path).

Separate (``run_token_demo``): one ``prepare()`` on a fixed history with a
heuristic counter — a synthetic upper bound, not the A/B figure.

The fixture embeds a single-occurrence mid-log outlier (exit_code=1 + stack
trace on one pod) so residue extractors that only keep head/tail + *repeated*
tokens lose a fact the correct answer must mention.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cat_agent.agent import Agent
from cat_agent.context import ContextManager, ObservationMaskingStrategy
from cat_agent.context.budget import HeuristicTokenCounter
from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, FunctionCall, Message
from cat_agent.tools import tool
from cat_agent.trace import InMemoryTraceStore
from cat_agent.trace.schema import Run

SEED = 42
N_PODS = 8
KEEP_RECENT = 2
OUTLIER_POD = 3  # middle of the fetch order; not in the keep_recent tail
EXIT_OOM = 137
EXIT_OUTLIER = 1
OUTLIER_ERROR = 'ConfigError'
# Single-occurrence stack line — ConfigError must not be kept merely by repeats.
OUTLIER_STACK = (
    f'Traceback (most recent call last): File "payments.py", line 42, '
    f'in charge: raise {OUTLIER_ERROR}("missing MERCHANT_KEY")'
)


def pod_log(pod_index: int, *, filler_lines: int = 80) -> str:
    """Bulky synthetic log. Outlier pod has a unique mid-body fact once."""
    header = f'=== logs for pod-{pod_index} ==='
    # Homogeneous filler so head/tail clips and repeated-token filters see noise.
    filler = [f'INFO heartbeat seq={j} ok' for j in range(filler_lines)]
    mid = filler_lines // 2
    if pod_index == OUTLIER_POD:
        # Exactly one exit_code=1 line and one ConfigError mention, both mid-body.
        body = (
            filler[:mid]
            + [
                f'exit_code={EXIT_OUTLIER}',
                OUTLIER_STACK,
            ]
            + filler[mid:]
        )
    else:
        body = (
            filler[:mid]
            + [f'exit_code={EXIT_OOM}', 'ERROR OOMKilled']
            + filler[mid:]
        )
        # Repeat the dominant failure so the old residue path keeps it.
        body.extend(['ERROR OOMKilled'] * 40)
    return header + '\n' + '\n'.join(body)


@tool(allow_overwrite=True)
def kubectl(cmd: str) -> str:
    """Fetch kubectl / pod logs (stub)."""
    m = re.search(r'logs\s+pod-(\d+)', cmd)
    if not m:
        return f'unknown cmd: {cmd}'
    return pod_log(int(m.group(1)))


@dataclass
class RecordedTurn:
    tool_calls: List[Tuple[str, dict]] = field(default_factory=list)
    final: Optional[str] = None
    messages_in_snapshot: List[dict] = field(default_factory=list)


class DeterministicKubeLLM:
    """Stub backend: fetch pod-0..N-1 in order, then conclude from visible logs.

    Decision rule (seed-independent except for the fixed N_PODS):
    - Count kubectl tool *intents* already issued (assistant function_call).
    - If fewer than N_PODS, request the next pod.
    - Else conclude from non-elided / residue-visible facts; if OOM evidence is
      missing because it was elided, re-request the lowest missing pod index.
    - The final answer must mention the mid-log outlier when its facts are still
      visible (exit_code=1 / ConfigError on OUTLIER_POD).
    """

    model = 'stub-kube-llm'
    model_type = 'oai'
    model_cfg = {'model': 'stub-kube-llm', 'model_type': 'oai', 'seed': SEED}

    def __init__(self, n_pods: int = N_PODS):
        self.n_pods = n_pods
        self.turns: List[RecordedTurn] = []

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        turn = RecordedTurn(
            messages_in_snapshot=[_msg_brief(m) for m in messages],
        )
        issued = _issued_pod_indices(messages)
        visible_oom_pods = _visible_oom_pods(messages)
        elided_pods = _elided_pod_indices(messages)
        outlier_visible = _outlier_visible(messages)

        if len(issued) < self.n_pods:
            nxt = len(issued)
            args = {'cmd': f'logs pod-{nxt}'}
            turn.tool_calls.append(('kubectl', args))
            out = [_fc('kubectl', args)]
        elif visible_oom_pods:
            text = (
                f'Root cause appears to be OOMKilled '
                f'(evidence in pods {sorted(visible_oom_pods)}; '
                f'exit_code={EXIT_OOM}).'
            )
            if outlier_visible:
                text += (
                    f' Outlier: pod-{OUTLIER_POD} exit_code={EXIT_OUTLIER} '
                    f'{OUTLIER_ERROR} (not OOM).'
                )
            turn.final = text
            out = [_answer(text)]
        elif elided_pods:
            nxt = min(elided_pods)
            args = {'cmd': f'logs pod-{nxt}'}
            turn.tool_calls.append(('kubectl', args))
            out = [_fc('kubectl', args)]
        else:
            text = 'Unable to determine root cause from available logs.'
            turn.final = text
            out = [_answer(text)]

        self.turns.append(turn)
        if stream:
            yield out
        else:
            return out


def _fc(name: str, args: dict) -> Message:
    # No fabricated usage — TraceRecorder resolves prompt tokens from
    # messages_in (same path as backends that omit usage metadata).
    return Message(
        ASSISTANT, '',
        function_call=FunctionCall(name=name, arguments=json.dumps(args)),
    )


def _answer(text: str) -> Message:
    return Message(ASSISTANT, text)



def _msg_brief(m: Message) -> dict:
    content = m.content
    if isinstance(content, str) and len(content) > 120:
        content = content[:120] + '…'
    return {'role': m.role, 'name': m.name, 'content': content,
            'tool_calls': bool(m.tool_calls or m.function_call)}


def _issued_pod_indices(messages: List[Message]) -> List[int]:
    found = []
    for m in messages:
        if m.role != ASSISTANT:
            continue
        fc = m.function_call
        if not fc or fc.name != 'kubectl':
            continue
        try:
            args = json.loads(fc.arguments or '{}')
        except json.JSONDecodeError:
            continue
        m2 = re.search(r'pod-(\d+)', str(args.get('cmd', '')))
        if m2:
            found.append(int(m2.group(1)))
    return found


def _visible_oom_pods(messages: List[Message]) -> set[int]:
    """Pods with OOM evidence in full bodies *or* structured elision residue."""
    pods = set()
    for m in messages:
        if m.role not in (FUNCTION, 'tool'):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        if 'OOMKilled' not in text and 'oomkilled' not in text.lower():
            continue
        for m2 in re.finditer(r'pod-(\d+)', text, flags=re.IGNORECASE):
            pods.add(int(m2.group(1)))
    return pods


def _elided_pod_indices(messages: List[Message]) -> set[int]:
    pods = set()
    for m in messages:
        if m.role not in (FUNCTION, 'tool'):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        if 'elided' not in text.lower():
            continue
        # Only count elided pods that lost OOM evidence entirely (no residue).
        if 'OOMKilled' in text or 'oomkilled' in text.lower():
            continue
        for m3 in re.finditer(r'pod-(\d+)', text, flags=re.IGNORECASE):
            pods.add(int(m3.group(1)))
    return pods


def _outlier_visible(messages: List[Message]) -> bool:
    """True when the unique mid-log outlier facts are still in context."""
    for m in messages:
        if m.role not in (FUNCTION, 'tool'):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        has_exit = (
            f'exit_code={EXIT_OUTLIER}' in text
            or re.search(rf'exit[_ ]?codes?\s*[=:]?\s*{EXIT_OUTLIER}\b', text, re.I)
        )
        has_err = OUTLIER_ERROR in text
        if has_exit and has_err:
            return True
    return False


class KubeAgent(Agent):
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        response: List[Message] = []
        for _ in range(40):
            output: List[Message] = []
            for output in self._call_llm(
                messages,
                functions=[f.function for f in self.function_map.values()],
            ):
                if output:
                    yield response + output
            if not output:
                break
            response.extend(output)
            messages.extend(output)
            used = False
            for _src, tc_id, tool_name, tool_args in self._iter_tool_call_jobs(output):
                result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                fn = Message(
                    role=FUNCTION, name=tool_name, content=result,
                    tool_call_id=tc_id, extra={'function_id': tc_id},
                )
                messages.append(fn)
                response.append(fn)
                yield response
                used = True
            if not used:
                break
        yield response


def _tool_sequence(llm: DeterministicKubeLLM) -> List[Tuple[str, dict]]:
    seq = []
    for turn in llm.turns:
        seq.extend(turn.tool_calls)
    return seq


def _first_divergence(
    a: List[Tuple[str, dict]],
    b: List[Tuple[str, dict]],
) -> Optional[Tuple[int, Tuple[str, dict], Tuple[str, dict]]]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, a[i] if i < len(a) else None, b[i] if i < len(b) else None  # type: ignore[return-value]
    return None


def _run_traced(agent: Agent, user: list) -> Tuple[Any, Run]:
    """Execute ``agent.run`` with an in-memory store; return (last outputs, Run)."""
    store = InMemoryTraceStore()
    out = list(agent.run(user, trace=True, trace_store=store))
    runs = list(store.iter_runs())
    if len(runs) != 1:
        raise RuntimeError(f'expected 1 traced run, got {len(runs)}')
    return out, runs[0]


def _prompt_token_summary(run: Run) -> dict:
    llm_steps = [s for s in run.steps if s.kind == 'llm_call']
    return {
        'prompt_tokens': run.totals.prompt_tokens,
        'completion_tokens': run.totals.completion_tokens,
        'total_tokens': run.totals.total_tokens,
        'tokens_estimated': run.totals.tokens_estimated,
        'llm_calls': len(llm_steps),
        'steps': run.totals.steps,
    }


def run_quality_ab() -> dict:
    user = [{'role': 'user', 'content': 'Why are payments pods crash-looping?'}]

    llm_off = DeterministicKubeLLM(N_PODS)
    agent_off = KubeAgent(
        llm=llm_off,
        name='kube-off',
        system_message='You are a Kubernetes Q&A agent.',
        function_list=[kubectl],
        context_manager=False,
    )
    out_off, run_off = _run_traced(agent_off, user)
    final_off = out_off[-1][-1]['content'] if isinstance(out_off[-1][-1], dict) else out_off[-1][-1].content

    counter = HeuristicTokenCounter()
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy(keep_recent=KEEP_RECENT, counter=counter)],
        # Richer salient residue costs tokens vs head/tail-only; keep the budget
        # above the post-mask size so overflow is not confused with quality loss.
        max_context_tokens=3200,
        reserved_output_tokens=128,
        trigger_ratio=0.4,
    )
    llm_on = DeterministicKubeLLM(N_PODS)
    agent_on = KubeAgent(
        llm=llm_on,
        name='kube-on',
        system_message='You are a Kubernetes Q&A agent.',
        function_list=[kubectl],
        context_manager=mgr,
    )
    out_on, run_on = _run_traced(agent_on, user)
    final_on = out_on[-1][-1]['content'] if isinstance(out_on[-1][-1], dict) else out_on[-1][-1].content

    seq_off = _tool_sequence(llm_off)
    seq_on = _tool_sequence(llm_on)
    div = _first_divergence(seq_off, seq_on)

    elided_seen = any(
        'elided' in str(snap.get('content', '')).lower()
        for turn in llm_on.turns
        for snap in turn.messages_in_snapshot
    )

    tok_off = _prompt_token_summary(run_off)
    tok_on = _prompt_token_summary(run_on)
    prompt_off = tok_off['prompt_tokens']
    prompt_on = tok_on['prompt_tokens']
    prompt_reduction_pct = (
        100.0 * (1.0 - prompt_on / prompt_off) if prompt_off else 0.0
    )

    return {
        'seed': SEED,
        'model': 'stub-kube-llm (DeterministicKubeLLM)',
        'n_pods': N_PODS,
        'outlier_pod': OUTLIER_POD,
        'keep_recent': KEEP_RECENT,
        'final_off': final_off,
        'final_on': final_on,
        'answers_equivalent': _answers_equivalent(final_off, final_on),
        'content_diff': _content_diff(final_off, final_on),
        'tool_seq_off': seq_off,
        'tool_seq_on': seq_on,
        'tool_seqs_equal': seq_off == seq_on,
        'steps_off': len(llm_off.turns),
        'steps_on': len(llm_on.turns),
        'elided_seen_in_masked_prompts': elided_seen,
        'first_divergence': div,
        'redundant_or_rerequest': bool(div) or len(seq_on) > len(seq_off),
        'outlier_in_unmasked': _answer_mentions_outlier(final_off),
        'outlier_in_masked': _answer_mentions_outlier(final_on),
        # Trace RunTotals — same accounting path as CAT_AGENT_TRACE users see.
        'trace_totals_off': tok_off,
        'trace_totals_on': tok_on,
        'prompt_tokens_off': prompt_off,
        'prompt_tokens_on': prompt_on,
        'prompt_reduction_pct': prompt_reduction_pct,
        'llm_calls_off': tok_off['llm_calls'],
        'llm_calls_on': tok_on['llm_calls'],
        'tokens_estimated': tok_off['tokens_estimated'] or tok_on['tokens_estimated'],
    }



def _extract_pod_entities(text: Any) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r'pod[s\s\[,-]*(\d+)', str(text), flags=re.I)}


def _extract_entities(text: Any) -> set[str]:
    """Content-level entities drawn from the answer text itself.

    Not a fixed allow-list of known kubectl tokens: whatever exit codes, error
    types, and status labels the *unmasked* answer actually uttered must still
    appear after masking.
    """
    s = str(text)
    ents: set[str] = set()
    for p in _extract_pod_entities(s):
        ents.add(f'pod-{p}')
    bracket = re.search(r'pods?\s*\[([^\]]+)\]', s, flags=re.I)
    if bracket:
        for part in re.findall(r'\d+', bracket.group(1)):
            ents.add(f'pod-{int(part)}')
    for m in re.finditer(r'exit[_ ]?codes?\s*[=:]?\s*(\d+)', s, flags=re.I):
        ents.add(f'exit_code={m.group(1)}')
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9]*(?:Error|Exception))\b', s):
        ents.add(m.group(1))
    for tok in re.findall(
        r'\b(OOMKilled|CrashLoopBackOff|ImagePullBackOff|Pending|Running)\b',
        s,
        flags=re.I,
    ):
        ents.add(tok)
    return ents


def _content_diff(unmasked: Any, masked: Any) -> dict:
    """Entities present in unmasked but missing from masked (and vice versa)."""
    a, b = _extract_entities(unmasked), _extract_entities(masked)
    return {
        'missing_in_masked': sorted(a - b),
        'extra_in_masked': sorted(b - a),
    }


def _answer_mentions_outlier(text: Any) -> bool:
    s = str(text)
    return (
        f'exit_code={EXIT_OUTLIER}' in s
        and OUTLIER_ERROR in s
        and f'pod-{OUTLIER_POD}' in s
    )


def _answers_equivalent(a: Any, b: Any) -> bool:
    """Fail when masked omits entities that unmasked reported."""
    diff = _content_diff(a, b)
    if diff['missing_in_masked']:
        return False
    if 'OOMKilled' in str(a) and 'OOMKilled' not in str(b):
        return False
    # Outlier fact in the unmasked answer is mandatory for equivalence.
    if _answer_mentions_outlier(a) and not _answer_mentions_outlier(b):
        return False
    return True


def build_history(n: int = N_PODS) -> List[Message]:
    msgs = [
        Message(SYSTEM, 'You are a Kubernetes Q&A agent.'),
        Message(USER, 'Why are payments pods crash-looping?'),
    ]
    for i in range(n):
        msgs.append(Message(
            ASSISTANT, '',
            function_call=FunctionCall('kubectl', f'{{"cmd":"logs pod-{i}"}}'),
        ))
        msgs.append(Message(
            FUNCTION, name='kubectl',
            content=pod_log(i),
        ))
    msgs.append(Message(
        ASSISTANT,
        f'Root cause appears to be OOMKilled (exit_code={EXIT_OOM}). '
        f'Outlier: pod-{OUTLIER_POD} exit_code={EXIT_OUTLIER} {OUTLIER_ERROR}.',
    ))
    return msgs


def run_token_demo() -> dict:
    counter = HeuristicTokenCounter()
    history = build_history(N_PODS)
    before = counter.count_messages(history)
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy(keep_recent=KEEP_RECENT, counter=counter)],
        max_context_tokens=max(before // 2, 500),
        reserved_output_tokens=128,
        trigger_ratio=0.4,
    )
    result = mgr.prepare(history)
    after = result.stats.tokens_after
    reduction = 100.0 * (1.0 - after / before) if before else 0.0
    # Inspect whether the outlier survived in any masked body.
    outlier_in_residue = False
    for m in result.messages:
        if m.role not in (FUNCTION, 'tool'):
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        if f'exit_code={EXIT_OUTLIER}' in text and OUTLIER_ERROR in text:
            outlier_in_residue = True
            break
    return {
        'tokens_before': before,
        'tokens_after': after,
        'reduction_pct': reduction,
        'messages_before': result.stats.messages_before,
        'messages_after': result.stats.messages_after,
        'operations': len(result.operations),
        'outlier_in_residue': outlier_in_residue,
    }


def main():
    print('=== Quality A/B (stub LLM, seed=%s) — primary measurement ===' % SEED)
    q = run_quality_ab()
    print(f"model={q['model']}")
    print(f"final OFF: {q['final_off']}")
    print(f"final ON : {q['final_on']}")
    print(f"answers_equivalent={q['answers_equivalent']} content_diff={q['content_diff']}")
    print(f"outlier_in_unmasked={q['outlier_in_unmasked']} outlier_in_masked={q['outlier_in_masked']}")
    print(
        f"prompt_tokens (RunTotals across {q['llm_calls_off']} LLM calls): "
        f"OFF={q['prompt_tokens_off']} ON={q['prompt_tokens_on']} "
        f"reduction={q['prompt_reduction_pct']:.1f}% "
        f"(tokens_estimated={q['tokens_estimated']})"
    )
    print(f"tool_seqs_equal={q['tool_seqs_equal']} steps_off={q['steps_off']} steps_on={q['steps_on']}")
    print(f"elided_seen_in_masked_prompts={q['elided_seen_in_masked_prompts']}")
    print(f"redundant_or_rerequest={q['redundant_or_rerequest']}")
    if q['first_divergence']:
        i, a, b = q['first_divergence']
        print(f'FIRST DIVERGENCE at tool-call index {i}: off={a} on={b}')
        print('(inspect DeterministicKubeLLM.turns for masked message snapshots)')
    else:
        print('No tool-sequence divergence.')

    assert q['answers_equivalent'], (
        f"Masked answer omitted entities: {q['content_diff']}"
    )
    assert q['tool_seqs_equal'], f"Tool sequences diverged: {q['first_divergence']}"
    assert not q['redundant_or_rerequest']
    assert q['llm_calls_off'] == q['llm_calls_on'] == N_PODS + 1
    assert q['prompt_tokens_on'] < q['prompt_tokens_off']

    print('\n=== Static prepare() demo (synthetic upper bound; not the A/B) ===')
    tokens = run_token_demo()
    print(
        f"heuristic tokens before={tokens['tokens_before']} after={tokens['tokens_after']} "
        f"reduction={tokens['reduction_pct']:.1f}% "
        f"(single prepare() on a fixed history; no LLM turns)"
    )
    print(f"outlier_in_residue={tokens['outlier_in_residue']}")


if __name__ == '__main__':
    main()
