#!/usr/bin/env python3
"""Context management (cat_agent.context).

Shows observation masking with residue, then a short agent turn that uses
the same ContextManager. No tool spam — keep the history synthetic.

    export OLLAMA_API_KEY=...
    export OLLAMA_BASE_URL=https://ollama.com/v1
    python examples/context/run.py
"""

from __future__ import annotations

import os

from cat_agent.agents import Assistant
from cat_agent.context import ContextManager, ObservationMaskingStrategy
from cat_agent.context.budget import HeuristicTokenCounter
from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, FunctionCall, Message


def llm_cfg() -> dict:
    base = (
        os.getenv('OLLAMA_BASE_URL')
        or os.getenv('OLLAMA_API_BASE')
        or 'https://ollama.com/v1'
    ).rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'
    return {
        'model': os.getenv('LLM_MODEL', 'minimax-m2.7:cloud'),
        'model_type': 'oai',
        'model_server': base,
        'api_key': os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY') or 'EMPTY',
        'generate_cfg': {'temperature': 0.2, 'max_tokens': 256},
    }


def bulky_history() -> list:
    """Synthetic long tool loop — enough text to trip masking."""
    msgs = [
        Message(SYSTEM, 'You are a concise helper.'),
        Message(USER, 'Summarise the cluster status.'),
    ]
    for i in range(6):
        msgs.append(Message(
            ASSISTANT, '',
            function_call=FunctionCall('logs', f'{{"pod":"pod-{i}"}}'),
        ))
        body = f'=== logs for pod-{i} ===\n' + ('INFO heartbeat ok\n' * 40)
        if i == 2:
            body += 'exit_code=1\nConfigError: missing MERCHANT_KEY\n'
        else:
            body += 'exit_code=137\nERROR OOMKilled\n' * 20
        msgs.append(Message(FUNCTION, body, name='logs'))
    msgs.append(Message(ASSISTANT, 'Most pods OOM; pod-2 is ConfigError.'))
    return msgs


def main():
    counter = HeuristicTokenCounter()
    mgr = ContextManager(
        strategies=[ObservationMaskingStrategy(keep_recent=2, counter=counter)],
        max_context_tokens=2000,
        reserved_output_tokens=128,
        trigger_ratio=0.4,
    )

    history = bulky_history()
    before = counter.count_messages(history)
    result = mgr.prepare(history)
    after = result.stats.tokens_after
    print('=== observation masking (static prepare) ===')
    print(f'tokens {before} → {after}  ops={len(result.operations)}')
    elided = sum(1 for m in result.messages if 'elided' in str(m.content).lower())
    print(f'elided observations: {elided}')
    residue_hit = any(
        'ConfigError' in str(m.content) or 'exit_code=1' in str(m.content)
        for m in result.messages
    )
    print(f'outlier residue kept: {residue_hit}')

    # Explicit fold — scratch work collapses to one result message.
    with mgr.fold(task='check payments namespace') as sub:
        sub.add(Message(USER, 'list failing pods'))
        sub.set_result('3 CrashLoopBackOff in ns=payments')
    folded = mgr.fold_into(
        [Message(SYSTEM, 'sys'), Message(USER, 'why are payments down?')],
        sub,
    )
    print('\n=== fold() ===')
    print(f'messages after fold: {len(folded)}')
    print(folded[-1].content)

    print('\n=== short agent turn with context_manager ===')
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print('skip — set OLLAMA_API_KEY to run the live turn')
        return
    bot = Assistant(
        llm=llm_cfg(),
        name='context-demo',
        system_message='Answer in one short sentence.',
        context_manager=mgr,
    )
    rsp = []
    for rsp in bot.run([{'role': 'user', 'content': 'Say hello in five words or fewer.'}]):
        pass
    if rsp:
        last = rsp[-1]
        print('reply:', last.get('content') if isinstance(last, dict) else last.content)


if __name__ == '__main__':
    main()
