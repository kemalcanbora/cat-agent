"""End-to-end _run coverage for BasicDocQA / DocQAAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cat_agent.agents import DocQAAgent
from cat_agent.agents.doc_qa.basic_doc_qa import BasicDocQA
from cat_agent.llm.schema import ASSISTANT, USER, Message


class _CaptureLLM:
    def __init__(self, reply: str = 'DOCQA_ANSWER'):
        self.model = 'fake'
        self.model_type = 'fake'
        self.reply = reply
        self.last_messages = None
        self.calls = 0

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        self.calls += 1
        self.last_messages = messages
        out = [Message(role=ASSISTANT, content=self.reply)]
        if stream:
            return iter([out])
        return out


def _flatten_text(messages) -> str:
    parts = []
    for m in messages or []:
        c = getattr(m, 'content', None)
        if isinstance(c, str):
            parts.append(c)
    return '\n'.join(parts)


class TestBasicDocQARunE2E:

    def test_docqa_alias_is_basic_doc_qa(self):
        assert DocQAAgent is BasicDocQA

    def test_run_uses_mem_knowledge_in_prompt_and_yields_answer(self):
        fake = _CaptureLLM(reply='TABLE_1_SUMMARY')
        mock_mem = MagicMock()
        mock_mem.run.return_value = iter([[
            Message(role=ASSISTANT, content='TABLE_1_MARKER: revenue grew 12% YoY'),
        ]])

        with patch('cat_agent.agents.fncall_agent.Memory', return_value=mock_mem):
            agent = BasicDocQA(llm=fake, files=[], system_message='')

        out = list(agent.run([Message(role=USER, content='Introduce Table 1')]))
        assert out, 'BasicDocQA._run must produce a non-empty iterable of responses'
        assert out[-1][-1].content == 'TABLE_1_SUMMARY'
        mock_mem.run.assert_called()
        prompt = _flatten_text(fake.last_messages)
        assert 'TABLE_1_MARKER' in prompt
        assert 'Reference' in prompt or 'reference' in prompt.lower()
        assert fake.calls >= 1
