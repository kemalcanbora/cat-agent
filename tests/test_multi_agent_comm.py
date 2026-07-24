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

"""Tests for the multi-agent communication layer."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.agents.assistant import Assistant
from cat_agent.agents.group_chat import GroupChat
from cat_agent.agents.router import Router
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.multi_agent import (
    AgentMessage,
    Blackboard,
    Handoff,
    HubEvent,
    filter_visible,
    parse_mentions,
    render_for_agent,
)
from cat_agent.multi_agent_hub import MultiAgentHub


def _named_agent(name: str, description: str = 'd') -> BasicAgent:
    mock_llm = MagicMock()
    mock_llm.model = 'gpt-4'
    mock_llm.model_type = 'openai'
    agent = BasicAgent(llm=mock_llm)
    agent.name = name
    agent.description = description
    return agent


# ---------------------------------------------------------------------------
# AgentMessage / mentions / visibility
# ---------------------------------------------------------------------------


class TestAgentMessage:

    def test_broadcast_visible_to_all(self):
        msg = AgentMessage(sender='Alice', content='hello')
        assert msg.visible_to('Bob')
        assert msg.visible_to('Carol')
        assert msg.visible_to('Alice')

    def test_directed_message_visibility(self):
        msg = AgentMessage(sender='Alice', content='secret', recipients=['Bob'])
        assert msg.visible_to('Bob')
        assert msg.visible_to('Alice')  # sender always sees own
        assert not msg.visible_to('Carol')

    def test_render_for_agent_filters_carol(self):
        msgs = [
            AgentMessage(sender='user', content='hi all'),
            AgentMessage(sender='Alice', content='for Bob only', recipients=['Bob']),
            AgentMessage(sender='Bob', content='got it', recipients=['Alice']),
        ]
        carol = render_for_agent(msgs, 'Carol')
        texts = [m.content for m in carol]
        assert any('hi all' in t for t in texts)
        assert not any('for Bob only' in t for t in texts)
        assert not any('got it' in t for t in texts)

        bob = render_for_agent(msgs, 'Bob')
        bob_texts = [m.content for m in bob]
        assert any('for Bob only' in t for t in bob_texts)

    def test_mention_word_boundary_no_false_positive(self):
        names = ['Alice', 'Bob', 'Carol']
        assert parse_mentions('email me at user@example.com please', names) == []
        assert parse_mentions('code: x = "@Alice" in str', names) == []
        assert parse_mentions('mail user@Bob.com later', names) == []
        # @Alice as a real mention (whitespace or start)
        assert parse_mentions('Hey @Alice can you help?', names) == ['Alice']
        assert parse_mentions('@Bob and @Carol', names) == ['Bob', 'Carol']

    def test_from_message_resolves_mentions(self):
        wire = Message(USER, 'Please review @Bob', name='Alice')
        am = AgentMessage.from_message(wire, known_names=['Alice', 'Bob', 'Carol'])
        assert am.recipients == ['Bob']
        assert am.sender == 'Alice'


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


class TestBlackboard:

    def test_write_read_describe(self):
        bb = Blackboard()
        ref = bb.write('parser_v1', 'def parse():\n  pass', author='Alice', summary='parser stub')
        assert ref == 'artifact:parser_v1'
        assert bb.read('parser_v1') == 'def parse():\n  pass'
        assert bb.read('artifact:parser_v1') == 'def parse():\n  pass'
        desc = bb.describe()
        assert 'parser_v1' in desc
        assert 'Alice' in desc

    def test_only_reader_pays(self):
        bb = Blackboard()
        bb.write('big', 'x' * 5000, author='Alice')
        # Five agents; only Bob reads
        readers = []
        for name in ['Alice', 'Bob', 'Carol', 'Dan', 'Eve']:
            if name == 'Bob':
                readers.append(bb.read('big'))
        assert len(readers) == 1
        assert len(readers[0]) == 5000


# ---------------------------------------------------------------------------
# Hub ask_agent / handoff / events
# ---------------------------------------------------------------------------


class _TestHub(MultiAgentHub):
    """Minimal concrete hub for unit tests."""

    def __init__(self, agents, **kwargs):
        self._agents = agents
        events = []
        self.events = events

        def on_event(e: HubEvent):
            events.append(e)

        self._init_hub(on_event=on_event, **kwargs)
        self._inject_hub_tools(agents)


class TestAskAgent:

    def test_cycle_rejected(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        hub = _TestHub([a, b])
        hub._call_stack = ['Alice', 'Bob']
        out = hub.handle_ask(caller='Bob', target_name='Alice', question='loop?')
        assert 'cycle rejected' in out.lower() or 'already handling' in out.lower()

    def test_depth_limit(self):
        agents = [_named_agent(n) for n in ['A', 'B', 'C', 'D']]
        hub = _TestHub(agents, max_ask_depth=3)
        hub._call_stack = ['A', 'B', 'C']  # depth already 3
        out = hub.handle_ask(caller='C', target_name='D', question='go deeper?')
        assert 'maximum delegation depth' in out.lower() or 'depth' in out.lower()

    def test_ask_runs_target_isolated(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        b.run = MagicMock(return_value=iter([[Message(ASSISTANT, '42', name='Bob')]]))
        hub = _TestHub([a, b])
        out = hub.handle_ask(caller='Alice', target_name='Bob', question='What is 6*7?')
        assert out == '42'
        # Isolated: only the question, not caller history
        args, kwargs = b.run.call_args
        msgs = kwargs.get('messages') or args[0]
        assert len(msgs) == 1
        assert msgs[0].content == 'What is 6*7?'
        assert any(e.type == 'ask' for e in hub.events)
        assert any(e.type == 'agent_start' and e.agent == 'Bob' for e in hub.events)
        assert any(e.type == 'agent_end' and e.agent == 'Bob' for e in hub.events)

    def test_ask_strips_caller_messages_kwarg(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        b.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'ok', name='Bob')]]))
        hub = _TestHub([a, b])
        # Simulate FnCallAgent._call_tool forwarding the caller's messages=
        out = hub.handle_ask(
            caller='Alice',
            target_name='Bob',
            question='Need R and T',
            messages=[Message(ASSISTANT, 'caller context', name='Alice')],
            lang='en',
        )
        assert out == 'ok'
        kwargs = b.run.call_args.kwargs
        assert 'messages' in kwargs
        assert kwargs['messages'][0].content == 'Need R and T'
        assert all(m.content != 'caller context' for m in kwargs['messages'])

    def test_allow_list(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        hub = _TestHub([a, b], allow_list={'Alice': []})
        out = hub.handle_ask(caller='Alice', target_name='Bob', question='hi')
        assert 'not allowed' in out.lower()


class TestHandoff:

    def test_set_and_consume(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        hub = _TestHub([a, b])
        hub.set_pending_handoff(Handoff(to='Bob', context='take over'), caller='Alice')
        assert any(e.type == 'handoff' for e in hub.events)
        h = hub.consume_pending_handoff()
        assert h.to == 'Bob'
        assert hub.consume_pending_handoff() is None


class TestHubEvents:

    def test_agent_start_end_pairs_on_ask(self):
        a = _named_agent('Alice')
        b = _named_agent('Bob')
        b.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'ok', name='Bob')]]))
        hub = _TestHub([a, b])
        hub.handle_ask(caller='Alice', target_name='Bob', question='q')
        starts = [e for e in hub.events if e.type == 'agent_start']
        ends = [e for e in hub.events if e.type == 'agent_end']
        assert len(starts) == len(ends) == 1


# ---------------------------------------------------------------------------
# Router multi-hop
# ---------------------------------------------------------------------------


class TestRouterMultiHop:

    def _make_router(self, agents, events=None):
        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        on_event = (lambda e: events.append(e)) if events is not None else None
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            return Router(
                llm=mock_llm,
                agents=agents,
                max_turns=5,
                on_event=on_event,
                inject_hub_tools=False,
            )

    def test_multi_hop_then_synthesize(self):
        events = []
        a = _named_agent('AgentA')
        b = _named_agent('AgentB')
        a.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'fact from A', name='AgentA')]]))
        b.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'fact from B', name='AgentB')]]))
        router = self._make_router([a, b], events=events)

        calls = iter([
            [Message(ASSISTANT, 'Call: AgentA')],
            [Message(ASSISTANT, 'Call: AgentB')],
            [Message(ASSISTANT, 'Synthesis: A then B')],
        ])

        def fake_super_run(self, messages, lang=None, **kwargs):
            yield next(calls)

        with patch.object(Assistant, '_run', fake_super_run):
            out = list(router._run([Message(USER, 'Need both')], lang='en'))

        assert a.run.call_count == 1
        assert b.run.call_count == 1
        assert out[-1][-1].content == 'Synthesis: A then B'
        assert any(e.type == 'agent_start' for e in events)
        assert any(e.type == 'agent_end' for e in events)

    def test_max_turns_forced_summary(self):
        a = _named_agent('AgentA')
        a.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'partial', name='AgentA')]]))
        router = self._make_router([a])
        router.max_turns = 2

        # Every router turn tries to Call again; after budget, forced summary
        def fake_super_run(self, messages, lang=None, **kwargs):
            # Detect forced-summary marker
            texts = []
            for m in messages:
                c = m.content if isinstance(m.content, str) else ''
                texts.append(c)
            if any('maximum number of delegation turns' in t for t in texts):
                yield [Message(ASSISTANT, 'Forced final answer')]
            else:
                yield [Message(ASSISTANT, 'Call: AgentA')]

        with patch.object(Assistant, '_run', fake_super_run):
            out = list(router._run([Message(USER, 'Hi')], lang='en', max_turns=2))

        assert out[-1][-1].content == 'Forced final answer'
        # Agent called at most once per unique (agent, user) then blocked by seen_calls;
        # with max_turns=2: turn0 Call+run, turn1 seen_calls block (no run), then summary
        assert a.run.call_count >= 1

    def test_unknown_agent_fed_back_not_fallback(self):
        a = _named_agent('OnlyAgent')
        a.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'Ok', name='OnlyAgent')]]))
        router = self._make_router([a])

        calls = iter([
            [Message(ASSISTANT, 'Call: NonExistent')],
            [Message(ASSISTANT, 'I will answer myself')],
        ])

        def fake_super_run(self, messages, lang=None, **kwargs):
            yield next(calls)

        with patch.object(Assistant, '_run', fake_super_run):
            out = list(router._run([Message(SYSTEM, 'Sys'), Message(USER, 'Hi')], lang='en'))

        a.run.assert_not_called()
        assert out[-1][-1].content == 'I will answer myself'


# ---------------------------------------------------------------------------
# GroupChat: mentions + handoff chain + events
# ---------------------------------------------------------------------------


class TestGroupChatComm:

    def test_mention_parsing_in_batch(self):
        alice = _named_agent('Alice')
        bob = _named_agent('Bob')
        bob.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'Here', name='Bob')]]))
        chat = GroupChat(
            agents=[alice, bob],
            agent_selection_method='round_robin',
            inject_hub_tools=False,
        )
        messages = [Message(USER, 'Hey @Bob please reply', name='user')]
        out = list(chat._gen_batch_response(messages, max_round=1))
        bob.run.assert_called_once()
        assert out[-1][-1].name == 'Bob'

    def test_email_at_does_not_mention(self):
        alice = _named_agent('Alice')
        bob = _named_agent('Bob')
        # round_robin from last speaker Alice → Bob; we need to ensure @example
        # does not add a false mention that would reorder selection
        bob.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'ok', name='Bob')]]))
        alice.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'ok', name='Alice')]]))
        chat = GroupChat(
            agents=[alice, bob],
            agent_selection_method='round_robin',
            inject_hub_tools=False,
        )
        messages = [
            Message(ASSISTANT, 'Hi', name='Alice'),
            Message(USER, 'mail user@Bob.com later', name='user'),
        ]
        # "@Bob" must NOT match inside email; round_robin → next after Alice is Bob
        # (Bob is still selected by round_robin, but mentioned_agents_name stays empty)
        mentions = parse_mentions('mail user@Bob.com later', chat.agent_names)
        assert mentions == []

    def test_handoff_chain_a_to_b_to_c(self):
        events = []
        a = _named_agent('A')
        b = _named_agent('B')
        c = _named_agent('C')

        def a_run(messages=None, **kwargs):
            # Simulate handoff tool by setting pending on the chat mid-run
            chat.set_pending_handoff(Handoff(to='B', context='brief B'), caller='A')
            yield [Message(ASSISTANT, 'triage done', name='A')]

        def b_run(messages=None, **kwargs):
            chat.set_pending_handoff(Handoff(to='C', context='brief C'), caller='B')
            yield [Message(ASSISTANT, 'routed', name='B')]

        def c_run(messages=None, **kwargs):
            yield [Message(ASSISTANT, 'final from C', name='C')]

        a.run = a_run
        b.run = b_run
        c.run = c_run

        chat = GroupChat(
            agents=[a, b, c],
            agent_selection_method='round_robin',
            on_event=lambda e: events.append(e),
            inject_hub_tools=False,
        )
        messages = [Message(USER, 'start', name='user')]
        # Round-robin: first speaker after user name 'user' → index -1 → A
        out = list(chat._gen_batch_response(messages, max_round=5))
        final = out[-1]
        assert final[-1].content == 'final from C'
        assert final[-1].name == 'C'
        # agent_start / agent_end for A, B, C
        start_agents = [e.agent for e in events if e.type == 'agent_start']
        end_agents = [e.agent for e in events if e.type == 'agent_end']
        assert start_agents == end_agents
        assert set(start_agents) >= {'A', 'B', 'C'}

    def test_directed_message_not_in_carol_context(self):
        alice = _named_agent('Alice')
        bob = _named_agent('Bob')
        carol = _named_agent('Carol')
        captured = {}

        def carol_run(messages=None, **kwargs):
            captured['messages'] = list(messages or [])
            yield [Message(ASSISTANT, 'carol speaks', name='Carol')]

        carol.run = carol_run
        bob.run = MagicMock(return_value=iter([[Message(ASSISTANT, 'bob private', name='Bob')]]))

        chat = GroupChat(
            agents=[alice, bob, carol],
            agent_selection_method='round_robin',
            inject_hub_tools=False,
        )
        # Seed hub with a directed Alice→Bob message, then let Carol speak
        hub_messages = [
            AgentMessage(sender='user', content='hello everyone'),
            AgentMessage(sender='Alice', content='secret for Bob', recipients=['Bob']),
        ]
        # Force Carol via mentioned_agents_name
        list(chat._gen_one_response(
            messages=[Message(USER, 'hello everyone', name='user')],
            hub_messages=hub_messages,
            mentioned_agents_name=['Carol'],
        ))
        texts = ' '.join(str(m.content) for m in captured['messages'])
        assert 'secret for Bob' not in texts
        assert 'hello everyone' in texts
