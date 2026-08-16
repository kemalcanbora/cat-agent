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

"""Tests for cat_agent.context."""

from __future__ import annotations

from typing import List, Sequence, Union

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from cat_agent.context import (
    ContextManager,
    ContextOverflowError,
    ObservationMaskingStrategy,
    SummaryCompactionStrategy,
)
from cat_agent.context.budget import ContextBudget, HeuristicTokenCounter
from cat_agent.context.strategies.base import roles_are_legal
from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, ContentItem, FunctionCall, Message


# ---------------------------------------------------------------------------
# History builders
# ---------------------------------------------------------------------------

def _history(n_tools: int = 6) -> List[Message]:
    msgs = [
        Message(role=SYSTEM, content='You are helpful.'),
        Message(role=USER, content='Solve the ticket.'),
    ]
    for i in range(n_tools):
        msgs.append(Message(
            role=ASSISTANT, content='',
            function_call=FunctionCall(name='web_search', arguments=f'{{"q": "{i}"}}'),
        ))
        msgs.append(Message(
            role=FUNCTION, name='web_search',
            content=('RESULT ' + str(i) + ' ') * 200,
        ))
    msgs.append(Message(role=ASSISTANT, content='Almost done.'))
    return msgs


def _roles(messages: Sequence[Message]) -> List[str]:
    return [m.role for m in messages]


def _image_uris(messages: Sequence[Message]) -> List[str]:
    uris = []
    for m in messages:
        if isinstance(m.content, list):
            for part in m.content:
                img = getattr(part, 'image', None)
                if img:
                    uris.append(img)
    return uris


# ---------------------------------------------------------------------------
# Hypothesis strategies — legal chat histories
# ---------------------------------------------------------------------------

@st.composite
def legal_histories(draw) -> List[Message]:
    """Varying-length histories with legal tool call/result pairing."""
    kind = draw(st.sampled_from(['empty', 'single', 'tools', 'multimodal', 'all_tools']))
    system = Message(SYSTEM, draw(st.sampled_from(['You are helpful.', 'Sys.', ''])))
    task = Message(USER, draw(st.text(min_size=1, max_size=80)))
    msgs: List[Message] = [system, task]

    if kind == 'empty':
        return msgs

    if kind == 'single':
        msgs.append(Message(ASSISTANT, draw(st.text(min_size=0, max_size=40))))
        return msgs

    n = draw(st.integers(min_value=1, max_value=6))
    for i in range(n):
        name = draw(st.sampled_from(['web_search', 'kubectl', 'lookup']))
        msgs.append(Message(
            ASSISTANT, '',
            function_call=FunctionCall(name=name, arguments=f'{{"q": "{i}"}}'),
        ))
        if kind == 'multimodal':
            # Every tool result carries an image so multimodal props don't filter.
            body: Union[str, List[ContentItem]] = [
                ContentItem(text=('blob ' * draw(st.integers(1, 40)))),
                ContentItem(image=f'file://img-{i}.png'),
            ]
        else:
            empty = draw(st.booleans())
            body = '' if empty else (('RESULT ' + str(i) + ' ') * draw(st.integers(1, 30)))
        msgs.append(Message(FUNCTION, name=name, content=body))

    if kind != 'all_tools':
        msgs.append(Message(ASSISTANT, draw(st.text(min_size=0, max_size=60))))
    return msgs


@st.composite
def multimodal_histories(draw) -> List[Message]:
    """Histories that always include at least one image part."""
    base = draw(legal_histories())
    # Force a multimodal tool pair if somehow missing.
    if not _image_uris(base):
        base = [
            Message(SYSTEM, 'You are helpful.'),
            Message(USER, 'describe'),
            Message(ASSISTANT, '', function_call=FunctionCall('vision', '{}')),
            Message(
                FUNCTION, name='vision',
                content=[
                    ContentItem(text='caption ' * 20),
                    ContentItem(image='file://forced.png'),
                ],
            ),
            Message(ASSISTANT, 'ok'),
        ]
    return base


def _apply_masking(msgs: List[Message], *, keep_recent: int = 1) -> tuple:
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=keep_recent, counter=counter)
    before = counter.count_messages(msgs)
    # Force apply for property tests that check masking invariants.
    budget = ContextBudget(
        max_context_tokens=max(before, 1),
        current_token_count=before,
        trigger_ratio=0.0,
        reserved_output_tokens=0,
    )
    if not any(m.role == FUNCTION for m in msgs):
        # Nothing to mask — strategy may no-op via should_apply in manager;
        # call apply directly only when there are observations.
        return msgs, msgs, before, before
    result = strat.apply(msgs, budget)
    after = counter.count_messages(result.messages)
    return msgs, result.messages, before, after


def _apply_compaction(msgs: List[Message]) -> tuple:
    counter = HeuristicTokenCounter()
    strat = SummaryCompactionStrategy(llm=None, counter=counter, min_block=2)
    before = counter.count_messages(msgs)
    budget = ContextBudget(
        max_context_tokens=max(before, 1),
        current_token_count=before,
        trigger_ratio=0.0,
        reserved_output_tokens=0,
    )
    if len(msgs) < 4:
        return msgs, msgs, before, before
    result = strat.apply(msgs, budget)
    after = counter.count_messages(result.messages)
    return msgs, result.messages, before, after


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

def test_under_budget_unchanged():
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy()],
        max_context_tokens=100_000,
        trigger_ratio=0.99,
    )
    msgs = _history(2)
    result = mgr.prepare(msgs)
    assert [m.id for m in result.messages] == [m.id for m in msgs]


def test_masking_residue_keeps_status_and_ids():
    from cat_agent.context.residue import generic_residue_extractor
    text = '=== logs for pod-3 ===\n' + ('ERROR OOMKilled\n' * 50)
    residue = generic_residue_extractor('kubectl', text)
    assert 'pod-3' in residue
    assert 'OOMKilled' in residue

    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=0, counter=counter)
    msgs = [
        Message(SYSTEM, 'sys'),
        Message(USER, 'task'),
        Message(ASSISTANT, '', function_call=FunctionCall('kubectl', '{"cmd":"logs pod-3"}')),
        Message(FUNCTION, name='kubectl', content=text),
        Message(ASSISTANT, 'done'),
    ]
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    masked = next(m for m in result.messages if m.role == FUNCTION)
    body = str(masked.content)
    assert 'elided' in body.lower()
    assert 'pod-3' in body
    assert 'OOMKilled' in body


def test_generic_residue_keeps_single_occurrence_mid_log_outlier():
    """A fact that appears once in the middle must survive residue extraction.

    Head/tail clips and repeated-token filters both miss this class of signal.
    """
    from cat_agent.context.residue import generic_residue_extractor

    filler = [f'INFO heartbeat seq={j} ok' for j in range(80)]
    mid = 40
    lines = (
        ['=== logs for pod-3 ===']
        + filler[:mid]
        + [
            'exit_code=1',
            'Traceback (most recent call last): File "payments.py", line 42, '
            'in charge: raise ConfigError("missing MERCHANT_KEY")',
        ]
        + filler[mid:]
    )
    text = '\n'.join(lines)
    residue = generic_residue_extractor('kubectl', text)
    assert 'exit_code=1' in residue, residue
    assert 'ConfigError' in residue, residue
    # Still a single occurrence in the source — not kept via repeat counting alone.
    assert text.count('exit_code=1') == 1
    assert text.count('ConfigError') == 1


def _load_long_horizon_mod():
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples' / 'long_horizon_agent' / 'run.py'
    name = 'long_horizon_agent_run'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_content_equivalence_tracks_answer_entities_not_just_pods():
    """Equivalence must notice exit codes / Error types uttered in the answer."""
    mod = _load_long_horizon_mod()
    unmasked = (
        'Root cause appears to be OOMKilled (evidence in pods [0, 1, 2, 4, 5, 6, 7]; '
        'exit_code=137). Outlier: pod-3 exit_code=1 ConfigError (not OOM).'
    )
    masked_weak = (
        'Root cause appears to be OOMKilled (evidence in pods [0, 1, 2, 3, 4, 5, 6, 7]; '
        'exit_code=137).'
    )
    # Old check that only compared pod-N + OOMKilled would incorrectly pass.
    assert 'OOMKilled' in unmasked and 'OOMKilled' in masked_weak
    assert mod._extract_pod_entities(unmasked) <= mod._extract_pod_entities(masked_weak) | {3}
    assert mod._answers_equivalent(unmasked, masked_weak) is False
    diff = mod._content_diff(unmasked, masked_weak)
    assert 'exit_code=1' in diff['missing_in_masked'] or 'ConfigError' in diff['missing_in_masked']
    assert mod._answers_equivalent(unmasked, unmasked) is True


def test_long_horizon_outlier_ab_requires_residue():
    """Masked run must retain the mid-log outlier fact in the final answer."""
    mod = _load_long_horizon_mod()
    q = mod.run_quality_ab()
    assert q['outlier_in_unmasked'] is True, q['final_off']
    assert q['answers_equivalent'] is True, q
    assert q['outlier_in_masked'] is True, q['final_on']
    assert q['tool_seqs_equal']
    assert not q['redundant_or_rerequest']
    assert not q['content_diff']['missing_in_masked'], q['content_diff']
    # Trace RunTotals: same accounting path; masking must cut prompt tokens.
    assert q['llm_calls_off'] == q['llm_calls_on'] == mod.N_PODS + 1
    assert q['prompt_tokens_on'] < q['prompt_tokens_off']
    assert q['tokens_estimated'] is True


def test_masking_reduces_tokens_and_keeps_system_task():
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=1, counter=counter)
    msgs = _history(5)
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.1)
    assert strat.should_apply(msgs, budget)
    result = strat.apply(msgs, budget)
    assert result.stats.tokens_after <= result.stats.tokens_before
    assert result.messages[0].role == SYSTEM
    assert result.messages[0].content == msgs[0].content
    assert result.messages[1].role == USER
    assert result.messages[1].content == msgs[1].content
    assert any('elided' in str(m.content) for m in result.messages if m.role == FUNCTION)


def test_overflow_raises_rather_than_truncate():
    class Noop:
        name = 'noop'

        def should_apply(self, messages, budget):
            return True

        def apply(self, messages, budget):
            from cat_agent.context.budget import ContextResult, ContextStats
            return ContextResult(
                messages=list(messages),
                stats=ContextStats(budget.current_token_count, budget.current_token_count,
                                   len(messages), len(messages)),
                strategy_name='noop',
            )

    mgr = ContextManager(
        strategies=[Noop()],
        max_context_tokens=10,
        reserved_output_tokens=0,
        trigger_ratio=0.01,
    )
    with pytest.raises(ContextOverflowError):
        mgr.prepare(_history(3))


def test_multimodal_mask_preserves_image_parts():
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=0, counter=counter)
    msgs = [
        Message(SYSTEM, 'sys'),
        Message(USER, 'task'),
        Message(ASSISTANT, '', function_call=FunctionCall('vision', '{}')),
        Message(
            FUNCTION, name='vision',
            content=[ContentItem(text='huge text ' * 100), ContentItem(image='file://x.png')],
        ),
        Message(ASSISTANT, 'ok'),
        Message(FUNCTION, name='other', content='keep recent'),
    ]
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    masked = next(m for m in result.messages if m.role == FUNCTION and m.name == 'vision')
    assert isinstance(masked.content, list)
    assert any(getattr(p, 'image', None) for p in masked.content)


def test_fold_api():
    mgr = ContextManager()
    with mgr.fold(task='list pods') as sub:
        sub.add(Message(USER, 'work'))
        sub.set_result('3 pods')
    msgs = mgr.fold_into([Message(USER, 'main')], sub)
    assert any('Folded result' in str(m.content) for m in msgs)


def test_end_to_end_long_run_stub():
    counter = HeuristicTokenCounter()
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy(keep_recent=2, counter=counter)],
        max_context_tokens=counter.count_messages(_history(8)) // 2 + 500,
        reserved_output_tokens=50,
        trigger_ratio=0.3,
    )
    result = mgr.prepare(_history(8))
    assert result.stats.tokens_after <= result.stats.tokens_before
    assert result.messages[0].role == SYSTEM


def test_summary_compaction_persists(tmp_path):
    counter = HeuristicTokenCounter()
    msgs = _history(5)
    before = counter.count_messages(msgs)
    strat = SummaryCompactionStrategy(
        llm=None, counter=counter, persist_dir=str(tmp_path), min_block=2,
    )
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    assert result.stats.tokens_after <= result.stats.tokens_before
    assert list(tmp_path.glob('pre_compaction_*.jsonl'))


def test_long_horizon_quality_ab_equivalent():
    """Import the example A/B harness — answers + tool sequences must match."""
    mod = _load_long_horizon_mod()
    q = mod.run_quality_ab()
    assert q['answers_equivalent']
    assert q['tool_seqs_equal']
    assert not q['redundant_or_rerequest']
    assert not q['content_diff']['missing_in_masked'], q['content_diff']
    assert q['outlier_in_unmasked'] and q['outlier_in_masked']
    assert q['prompt_tokens_on'] < q['prompt_tokens_off']
    assert q['llm_calls_off'] == mod.N_PODS + 1


# ---------------------------------------------------------------------------
# Property-based invariants (hypothesis)
# ---------------------------------------------------------------------------

@given(n=st.integers(min_value=2, max_value=8))
@settings(max_examples=20, deadline=None)
def test_invariant_system_and_task_survive(n):
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=1, counter=counter)
    msgs = _history(n)
    sys_c, task_c = msgs[0].content, msgs[1].content
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    assert result.messages[0].content == sys_c
    assert result.messages[1].content == task_c


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=15, deadline=None)
def test_invariant_never_increases_tokens(n):
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=1, counter=counter)
    msgs = _history(n)
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    after = counter.count_messages(result.messages)
    assert after <= before


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=15, deadline=None)
def test_invariant_roles_legal(n):
    counter = HeuristicTokenCounter()
    strat = ObservationMaskingStrategy(keep_recent=2, counter=counter)
    msgs = _history(n)
    before = counter.count_messages(msgs)
    budget = ContextBudget(max_context_tokens=before, current_token_count=before, trigger_ratio=0.0)
    result = strat.apply(msgs, budget)
    assert roles_are_legal(result.messages)


@given(legal_histories())
@settings(max_examples=40, deadline=None)
def test_prop_system_and_task_survive_generated(msgs: List[Message]):
    assume(len(msgs) >= 2 and msgs[0].role == SYSTEM and msgs[1].role == USER)
    _, out, _, _ = _apply_masking(msgs)
    assert out[0].role == SYSTEM
    assert out[0].content == msgs[0].content
    assert out[1].role == USER
    assert out[1].content == msgs[1].content
    # Compaction path when long enough
    if len(msgs) >= 6:
        _, out2, _, _ = _apply_compaction(msgs)
        # System or summary may be first after compaction; original task text must remain.
        blob = ' '.join(
            (m.content if isinstance(m.content, str) else str(m.content)) for m in out2
        )
        assert msgs[0].content in blob or out2[0].role == SYSTEM
        assert msgs[1].content in blob


@given(legal_histories())
@settings(max_examples=40, deadline=None)
def test_prop_ordering_and_role_sequencing(msgs: List[Message]):
    assume(roles_are_legal(msgs))
    _, out, _, _ = _apply_masking(msgs)
    assert _roles(out) == _roles(msgs), 'masking must preserve message order/roles'
    assert roles_are_legal(out)
    if len(msgs) >= 6:
        _, out2, _, _ = _apply_compaction(msgs)
        assert roles_are_legal(out2)


@given(legal_histories())
@settings(max_examples=40, deadline=None)
def test_prop_never_increases_tokens_generated(msgs: List[Message]):
    _, out, before, after = _apply_masking(msgs)
    assert after <= before
    if len(msgs) >= 6:
        _, _, b2, a2 = _apply_compaction(msgs)
        assert a2 <= b2


@given(legal_histories())
@settings(max_examples=30, deadline=None)
def test_prop_under_budget_unchanged(msgs: List[Message]):
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy(keep_recent=2)],
        max_context_tokens=10_000_000,
        trigger_ratio=0.99,
        reserved_output_tokens=0,
    )
    result = mgr.prepare(msgs)
    assert [m.id for m in result.messages] == [m.id for m in msgs]
    assert result.stats.tokens_after == result.stats.tokens_before


@given(multimodal_histories())
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_prop_multimodal_not_corrupted(msgs: List[Message]):
    before_imgs = _image_uris(msgs)
    assert before_imgs
    _, out, _, _ = _apply_masking(msgs, keep_recent=0)
    after_imgs = _image_uris(out)
    assert after_imgs == before_imgs
    if len(msgs) >= 6:
        _, out2, _, _ = _apply_compaction(msgs)
        for m in out2:
            if isinstance(m.content, list):
                for part in m.content:
                    if getattr(part, 'image', None):
                        assert part.image in before_imgs
