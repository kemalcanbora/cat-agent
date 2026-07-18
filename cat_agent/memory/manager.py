"""Session memory management: recall injection, auto-record, and compaction."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, ContentItem, Message
from cat_agent.log import logger
from cat_agent.memory.store import MemoryRecord, MemoryStore
from cat_agent.utils.utils import extract_text_from_message

MEMORY_PROMPT_TEMPLATE = """# Long-term memory

Relevant information remembered from previous conversations:

{memories}

Use this context when helpful. Do not mention the memory system unless asked."""

SUMMARIZE_PROMPT = """Summarize the following conversation excerpt in a compact form.
Preserve concrete facts, names, numbers, decisions, and user preferences.
Reply with the summary only.

{transcript}"""


def _message_text(message: Message) -> str:
    return extract_text_from_message(message, add_upload_info=False)


def _count_tokens(text: str) -> int:
    try:
        from cat_agent.utils.tokenization_qwen import count_tokens

        return count_tokens(text)
    except Exception:  # noqa: BLE001 - native tokenizer optional
        return max(1, len(text) // 4)


class MemoryManager:
    """Coordinates long-term memory for an agent.

    Config keys (all optional):
        scope: Namespace for cross-session memory (e.g. "user:alice"). Default "default".
        path: Directory for the memory database. Default workspace/memory.
        top_k: Number of memories recalled per query. Default 5.
        auto_record: Store each completed user/assistant exchange. Default True.
        auto_summarize: Compact old turns via the LLM when over budget. Default True.
        session_window_tokens: Token budget for dialogue history. Default 8000.
        keep_recent_turns: User turns always kept verbatim during compaction. Default 3.
        embedding_backend / embedding_dimensions / embedding_model_path: recall embeddings.
    """

    def __init__(self, cfg: Optional[Dict] = None, llm=None):
        cfg = cfg or {}
        self.scope: str = cfg.get('scope', 'default')
        self.top_k: int = int(cfg.get('top_k', 5))
        self.auto_record: bool = bool(cfg.get('auto_record', True))
        self.auto_summarize: bool = bool(cfg.get('auto_summarize', True))
        self.session_window_tokens: int = int(cfg.get('session_window_tokens', 8000))
        self.keep_recent_turns: int = max(1, int(cfg.get('keep_recent_turns', 3)))
        self.llm = llm
        self.store = MemoryStore(
            path=cfg.get('path'),
            embedding_cfg={
                key: cfg[key]
                for key in ('embedding_backend', 'embedding_dimensions', 'embedding_model_path')
                if key in cfg
            },
        )

    # ------------------------------------------------------------------ API

    def remember(self, text: str, *, kind: str = 'fact', metadata: Optional[Dict] = None) -> MemoryRecord:
        return self.store.add(text, scope=self.scope, kind=kind, metadata=metadata)

    def recall(self, query: str, top_k: Optional[int] = None) -> List[MemoryRecord]:
        return self.store.search(query, scope=self.scope, top_k=top_k or self.top_k)

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    # ------------------------------------------------- prompt integration

    def memory_prompt(self, query: str) -> str:
        records = self.recall(query)
        if not records:
            return ''
        lines = [f'- ({record.kind}, {record.created_at[:10]}) {record.text}' for record in records]
        return MEMORY_PROMPT_TEMPLATE.format(memories='\n'.join(lines))

    def inject_memories(self, messages: List[Message]) -> List[Message]:
        """Prepend recalled memories to the system message based on the last user query."""
        query = ''
        for message in reversed(messages):
            if message.role == USER:
                query = _message_text(message)
                break
        prompt = self.memory_prompt(query) if query else ''
        if not prompt:
            return messages

        messages = copy.deepcopy(messages)
        if messages and messages[0].role == SYSTEM:
            if isinstance(messages[0].content, str):
                messages[0].content += '\n\n' + prompt
            else:
                messages[0].content += [ContentItem(text='\n\n' + prompt)]
        else:
            messages = [Message(role=SYSTEM, content=prompt)] + messages
        return messages

    def record_exchange(self, messages: List[Message], response: List[Message]) -> Optional[MemoryRecord]:
        """Store the last user query and final assistant reply as an episodic memory."""
        if not self.auto_record:
            return None
        user_text = ''
        for message in reversed(messages):
            if message.role == USER:
                user_text = _message_text(message)
                break
        assistant_text = ''
        for message in reversed(response):
            if message.role == ASSISTANT and not message.function_call:
                assistant_text = _message_text(message)
                break
        if not user_text or not assistant_text:
            return None
        episode = f'User: {user_text}\nAssistant: {assistant_text}'
        return self.store.add(episode, scope=self.scope, kind='episode')

    # ---------------------------------------------------------- compaction

    def compact_messages(self, messages: List[Message]) -> List[Message]:
        """Summarize old dialogue turns into long-term memory when over the token budget.

        The system message and the most recent turns are kept verbatim; older
        turns are replaced by an LLM-generated summary that is also stored as
        a `summary` memory for future sessions.
        """
        if not self.auto_summarize or self.llm is None:
            return messages

        total_tokens = sum(_count_tokens(_message_text(message)) for message in messages)
        if total_tokens <= self.session_window_tokens:
            return messages

        system_messages = [message for message in messages if message.role == SYSTEM]
        dialogue = [message for message in messages if message.role != SYSTEM]

        # Find the cut point that keeps the last `keep_recent_turns` user turns.
        user_seen = 0
        cut_index = 0
        for index in range(len(dialogue) - 1, -1, -1):
            if dialogue[index].role == USER:
                user_seen += 1
                if user_seen >= self.keep_recent_turns:
                    cut_index = index
                    break
        old_turns, recent_turns = dialogue[:cut_index], dialogue[cut_index:]
        if not old_turns:
            return messages

        transcript = '\n'.join(
            f'{message.role}: {_message_text(message)}' for message in old_turns
            if _message_text(message).strip()
        )
        try:
            summary = self.llm.quick_chat(SUMMARIZE_PROMPT.format(transcript=transcript)).strip()
        except Exception as error:  # noqa: BLE001 - degrade to unsummarized history
            logger.warning('[MemoryManager] Summarization failed; keeping full history: {}', error)
            return messages
        if not summary:
            return messages

        self.store.add(summary, scope=self.scope, kind='summary')
        summary_message = Message(
            role=USER,
            content=f'[Summary of earlier conversation]\n{summary}',
        )
        logger.info(
            '[MemoryManager] Compacted {} old message(s) into a summary ({} -> {} tokens budgeted).',
            len(old_turns), total_tokens, self.session_window_tokens,
        )
        return system_messages + [summary_message] + recent_turns
