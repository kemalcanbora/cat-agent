"""Long-term agent memory: remember, recall, compaction, and cross-session persistence.

Runs fully offline — recall uses hash embeddings + the native HNSW index.
Compaction normally calls your LLM; here a tiny stub stands in so the demo
needs no model. See `agent_with_memory()` below for real agent wiring.
"""

import tempfile

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.memory import MemoryManager


class StubSummarizerLLM:
    """Stands in for a real LLM; only `quick_chat` is used by compaction."""

    def quick_chat(self, prompt: str) -> str:
        return 'User is planning an on-prem deployment for a bank with a 50k EUR budget.'


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Session 1: remember facts and record an exchange -------------
        memory = MemoryManager(cfg={'scope': 'user:alice', 'path': tmpdir})

        memory.remember('Alice works at a bank with strict compliance rules', kind='fact')
        memory.remember('Alice prefers replies in Turkish', kind='fact')
        memory.remember('The staging cluster runs air-gapped Kubernetes', kind='fact')

        memory.record_exchange(
            messages=[Message(role=USER, content='What encryption does cat-agent use at rest?')],
            response=[Message(role=ASSISTANT, content='AES-GCM with keys from the OS keyring or env.')],
        )

        # --- Recall: search over stored memories ---------------------------
        # Default hash embeddings match on shared keywords; use the onnx
        # embedding_backend for true semantic matching.
        print('Recall for "does alice prefer turkish replies?":')
        for record in memory.recall('does alice prefer turkish replies?', top_k=2):
            print(f'  [{record.kind}] score={record.score:.3f}  {record.text}')

        # --- Injection: memories land in the system prompt -----------------
        messages = [
            Message(role=SYSTEM, content='You are a helpful assistant.'),
            Message(role=USER, content='Where does Alice work?'),
        ]
        injected = memory.inject_memories(messages)
        print('\nSystem prompt after injection:')
        print(injected[0].content)

        # --- Compaction: long history -> LLM summary + stored memory -------
        compacting = MemoryManager(
            cfg={'scope': 'user:alice', 'path': tmpdir,
                 'session_window_tokens': 100, 'keep_recent_turns': 1},
            llm=StubSummarizerLLM(),
        )
        long_history = [
            Message(role=SYSTEM, content='You are a helpful assistant.'),
            Message(role=USER, content='word ' * 300),
            Message(role=ASSISTANT, content='word ' * 300),
            Message(role=USER, content='And what is the final budget?'),
        ]
        compacted = compacting.compact_messages(long_history)
        print(f'\nCompacted {len(long_history)} messages down to {len(compacted)}:')
        for message in compacted:
            preview = message.content if isinstance(message.content, str) else str(message.content)
            print(f'  {message.role}: {preview[:90]}...')

        # --- Session 2: a fresh manager sees everything --------------------
        next_session = MemoryManager(cfg={'scope': 'user:alice', 'path': tmpdir})
        print('\nNext session recalls the budget from the compaction summary:')
        for record in next_session.recall('deployment budget', top_k=1):
            print(f'  [{record.kind}] {record.text}')


def agent_with_memory():
    """Wire memory into a real agent — every run recalls, records, and compacts."""
    from cat_agent.agents import Assistant

    return Assistant(
        llm={'model': 'qwen2.5', 'model_server': 'http://llm.internal:8080/v1', 'api_key': 'EMPTY'},
        memory_cfg={
            'scope': 'user:alice',            # cross-session namespace
            'top_k': 5,                       # memories recalled per query
            'auto_record': True,              # store each completed exchange
            'auto_summarize': True,           # compact long histories via the LLM
            'session_window_tokens': 8000,    # budget before compaction kicks in
        },
    )


if __name__ == '__main__':
    main()
