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

"""Coverage tests for cat_agent.trace.instrument helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.trace import instrument as inst


def test_llm_cfg_snapshot_none():
    assert inst.llm_cfg_snapshot(None) == {}


def test_llm_cfg_snapshot_model_cfg_dict():
    llm = SimpleNamespace(model_cfg={'model': 'm', 'api_key': 'x'})
    assert inst.llm_cfg_snapshot(llm) == {'model': 'm', 'api_key': 'x'}


def test_llm_cfg_snapshot_cfg_dict():
    llm = SimpleNamespace(cfg={'model': 'via-cfg'})
    assert inst.llm_cfg_snapshot(llm) == {'model': 'via-cfg'}


def test_llm_cfg_snapshot_attribute_fallback():
    llm = SimpleNamespace(
        model='m1',
        model_type='oai',
        model_server='http://x',
        generate_cfg={'temperature': 0.2},
    )
    out = inst.llm_cfg_snapshot(llm)
    assert out['model'] == 'm1'
    assert out['model_type'] == 'oai'
    assert out['model_server'] == 'http://x'
    assert out['generate_cfg'] == {'temperature': 0.2}


def test_llm_cfg_snapshot_skips_missing_attrs():
    llm = SimpleNamespace(model='only')
    assert inst.llm_cfg_snapshot(llm) == {'model': 'only'}


def test_gen_ai_system_empty():
    assert inst.gen_ai_system(None) is None
    assert inst.gen_ai_system('') is None


def test_gen_ai_system_known_mappings():
    assert inst.gen_ai_system('oai') == 'openai'
    assert inst.gen_ai_system('OpenAI') == 'openai'
    assert inst.gen_ai_system('transformers') == 'huggingface'
    assert inst.gen_ai_system('llama_cpp') == 'llama.cpp'
    assert inst.gen_ai_system('llama_cpp_vision') == 'llama.cpp'
    assert inst.gen_ai_system('mlx_lm') == 'mlx'
    assert inst.gen_ai_system('openvino') == 'openvino'


def test_gen_ai_system_unknown_passthrough():
    assert inst.gen_ai_system('CustomBackend') == 'custombackend'


def test_final_output_text_empty():
    assert inst.final_output_text([]) is None


def test_final_output_text_dict_str():
    assert inst.final_output_text([{'role': 'assistant', 'content': 'hi'}]) == 'hi'


def test_final_output_text_dict_none_content():
    assert inst.final_output_text([{'role': 'assistant', 'content': None}]) is None


def test_final_output_text_message_str():
    assert inst.final_output_text([Message(ASSISTANT, 'done')]) == 'done'


def test_final_output_text_non_str_content():
    assert inst.final_output_text([{'content': 42}]) == '42'


def test_should_trace_run_explicit_false():
    agent = SimpleNamespace(run_limits=None)
    assert inst.should_trace_run(agent, {'trace': False}) is False


def test_should_trace_run_explicit_true():
    agent = SimpleNamespace(run_limits=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        with patch.object(inst, 'is_trace_enabled', return_value=False):
            assert inst.should_trace_run(agent, {'trace': True}) is True


def test_should_trace_run_trace_store():
    agent = SimpleNamespace(run_limits=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        with patch.object(inst, 'is_trace_enabled', return_value=False):
            assert inst.should_trace_run(agent, {'trace_store': object()}) is True


def test_should_trace_run_agent_run_limits():
    agent = SimpleNamespace(run_limits=object())
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        with patch.object(inst, 'is_trace_enabled', return_value=False):
            assert inst.should_trace_run(agent, {}) is True


def test_should_trace_run_active_recorder():
    agent = SimpleNamespace(run_limits=None)
    with patch.object(inst, 'get_trace_recorder', return_value=MagicMock()):
        with patch.object(inst, 'is_trace_enabled', return_value=False):
            assert inst.should_trace_run(agent, {}) is True


def test_should_trace_run_env_enabled():
    agent = SimpleNamespace(run_limits=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        with patch.object(inst, 'is_trace_enabled', return_value=True):
            assert inst.should_trace_run(agent, {}) is True


def test_should_trace_run_all_off():
    agent = SimpleNamespace(run_limits=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        with patch.object(inst, 'is_trace_enabled', return_value=False):
            assert inst.should_trace_run(agent, {}) is False


def test_apply_context_manager_false():
    agent = SimpleNamespace(context_manager=False, llm=None)
    msgs = [Message(USER, 'a')]
    assert inst.apply_context_manager(agent, msgs) is msgs


def test_apply_context_manager_none_default_raises():
    agent = SimpleNamespace(context_manager=None, llm=None)
    msgs = [Message(USER, 'a')]
    with patch(
        'cat_agent.context.get_default_context_manager',
        side_effect=RuntimeError('no default'),
    ):
        assert inst.apply_context_manager(agent, msgs) is msgs


def test_apply_context_manager_none_default_returns_none():
    agent = SimpleNamespace(context_manager=None, llm=None)
    msgs = [Message(USER, 'a')]
    with patch(
        'cat_agent.context.get_default_context_manager',
        return_value=None,
    ):
        assert inst.apply_context_manager(agent, msgs) is msgs


def test_apply_context_manager_with_mgr_records_ops():
    prepared = SimpleNamespace(
        messages=[Message(USER, 'trimmed')],
        operations=[{'op': 'fold'}],
    )
    mgr = MagicMock()
    mgr.prepare.return_value = prepared
    agent = SimpleNamespace(context_manager=mgr, llm='llm')
    recorder = MagicMock()
    with patch.object(inst, 'get_trace_recorder', return_value=recorder):
        out = inst.apply_context_manager(agent, [Message(USER, 'raw')])
    assert out == prepared.messages
    mgr.prepare.assert_called_once()
    recorder.record_context_op.assert_called_once_with({'op': 'fold'})


def test_apply_context_manager_with_mgr_no_recorder():
    prepared = SimpleNamespace(messages=[Message(USER, 'ok')], operations=None)
    mgr = MagicMock()
    mgr.prepare.return_value = prepared
    agent = SimpleNamespace(context_manager=mgr, llm=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        out = inst.apply_context_manager(agent, [Message(USER, 'raw')])
    assert out == prepared.messages


def test_record_llm_call_noop_without_recorder():
    agent = SimpleNamespace(llm=None)
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        inst.record_llm_call(
            agent,
            messages_for_llm=[],
            final_output=[],
            started_at=0.0,
        )


def test_record_llm_call_delegates_to_recorder():
    recorder = MagicMock()
    llm = SimpleNamespace(model_type='oai', model_cfg={'model_type': 'oai'})
    agent = SimpleNamespace(llm=llm)
    with patch.object(inst, 'get_trace_recorder', return_value=recorder):
        with patch(
            'cat_agent.observability.helpers.agent_model_name',
            return_value='stub-model',
        ):
            inst.record_llm_call(
                agent,
                messages_for_llm=[Message(USER, 'q')],
                final_output=[Message(ASSISTANT, 'a')],
                started_at=0.0,
                extra_cfg={'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 16},
            )
    recorder.record_llm_call.assert_called_once()
    kwargs = recorder.record_llm_call.call_args.kwargs
    assert kwargs['model'] == 'stub-model'
    assert kwargs['model_type'] == 'oai'
    assert kwargs['gen_ai_system'] == 'openai'
    assert kwargs['temperature'] == 0.1
    assert kwargs['max_tokens'] == 16


def test_record_tool_call_noop_without_recorder():
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        inst.record_tool_call(
            tool_name='t',
            tool_args={},
            started_at=0.0,
        )


def test_record_tool_call_delegates_to_recorder():
    recorder = MagicMock()
    with patch.object(inst, 'get_trace_recorder', return_value=recorder):
        inst.record_tool_call(
            tool_name='search',
            tool_args='{"q":1}',
            result='ok',
            succeeded=True,
            error=None,
            started_at=0.0,
        )
    recorder.record_tool_call.assert_called_once()
    kwargs = recorder.record_tool_call.call_args.kwargs
    assert kwargs['tool_name'] == 'search'
    assert kwargs['arguments'] == '{"q":1}'
    assert kwargs['result'] == 'ok'
    assert kwargs['succeeded'] is True


def test_check_run_limit_stop_no_recorder():
    with patch.object(inst, 'get_trace_recorder', return_value=None):
        assert inst.check_run_limit_stop() is None


def test_check_run_limit_stop_with_recorder():
    recorder = MagicMock()
    recorder.check_limits.return_value = 'max_steps'
    with patch.object(inst, 'get_trace_recorder', return_value=recorder):
        assert inst.check_run_limit_stop() == 'max_steps'
