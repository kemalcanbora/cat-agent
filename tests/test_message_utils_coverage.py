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

"""Coverage tests for cat_agent.utils.message_utils."""

from unittest.mock import MagicMock

import pytest

from cat_agent.llm.schema import (
    ASSISTANT,
    DEFAULT_SYSTEM_MESSAGE,
    SYSTEM,
    USER,
    ContentItem,
    FunctionCall,
    Message,
)
from cat_agent.utils.message_utils import (
    build_text_completion_prompt,
    extract_files_from_messages,
    extract_images_from_messages,
    extract_markdown_urls,
    extract_text_from_message,
    extract_urls,
    format_as_multimodal_message,
    format_as_text_message,
    get_last_usr_msg_idx,
    has_chinese_messages,
    rm_default_system,
)


def test_format_as_multimodal_string_content():
    msg = Message(USER, 'hello')
    out = format_as_multimodal_message(
        msg,
        add_upload_info=False,
        add_multimodel_upload_info=False,
        add_audio_upload_info=False,
    )
    assert isinstance(out.content, list)
    assert out.content[0].text == 'hello'


def test_format_as_multimodal_file_upload_en_and_zh():
    msg = Message(USER, [
        ContentItem(text='see file'),
        ContentItem(file='https://example.com/path/report.pdf'),
    ])
    en = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=False,
        add_audio_upload_info=False,
        lang='en',
    )
    assert en.content[0].text.startswith('(Uploaded [file](report.pdf))')
    assert en.content[0].text.endswith(' ')
    assert en.content[1].text == 'see file'

    zh = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=False,
        add_audio_upload_info=False,
        lang='zh',
    )
    assert zh.content[0].text == '(Uploaded [file](report.pdf))'
    assert not zh.content[0].text.endswith(' ')


def test_format_as_multimodal_image_and_video_upload_info():
    msg = Message(USER, [
        ContentItem(text='media'),
        ContentItem(image='https://cdn.example.com/a.png'),
        ContentItem(video=['https://cdn.example.com/v1.mp4', 'https://cdn.example.com/v2.mp4']),
    ])
    out = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=True,
        add_audio_upload_info=False,
        lang='en',
    )
    prefix = out.content[0].text
    assert '![image](a.png)' in prefix
    assert '![video](v1.mp4)' in prefix
    assert '![video](v2.mp4)' in prefix


def test_format_as_multimodal_audio_str_and_dict():
    msg = Message(USER, [
        ContentItem(text='listen'),
        ContentItem(audio='https://cdn.example.com/a.wav'),
        ContentItem(audio={'data': 'https://cdn.example.com/b.mp3'}),
    ])
    out = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=False,
        add_audio_upload_info=True,
        lang='zh',
    )
    assert '![audio](a.wav)' in out.content[0].text
    assert '![audio](b.mp3)' in out.content[0].text


def test_format_as_multimodal_auto_lang_chinese():
    msg = Message(USER, [
        ContentItem(text='你好世界'),
        ContentItem(file='/tmp/doc.txt'),
    ])
    out = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=False,
        add_audio_upload_info=False,
        lang='auto',
    )
    assert out.content[0].text == '(Uploaded [file](doc.txt))'


def test_format_as_multimodal_skips_duplicate_upload_info():
    upload = '(Uploaded [file](report.pdf)) '
    msg = Message(USER, [
        ContentItem(text=f'{upload}already'),
        ContentItem(file='https://example.com/report.pdf'),
    ])
    out = format_as_multimodal_message(
        msg,
        add_upload_info=True,
        add_multimodel_upload_info=False,
        add_audio_upload_info=False,
        lang='en',
    )
    # File parts are not kept as content items; duplicate upload tag is not re-prepended.
    assert out.content[0].text == f'{upload}already'
    assert len(out.content) == 1


def test_format_as_multimodal_typeerror_bad_content():
    msg = Message.model_construct(role=USER, content=123)
    with pytest.raises(TypeError):
        format_as_multimodal_message(
            msg,
            add_upload_info=False,
            add_multimodel_upload_info=False,
            add_audio_upload_info=False,
        )


def test_format_as_multimodal_typeerror_bad_image_value():
    bad = MagicMock()
    bad.get_type_and_value.return_value = ('image', 99)
    bad.text = None
    msg = Message.model_construct(role=USER, content=[bad])
    with pytest.raises(TypeError):
        format_as_multimodal_message(
            msg,
            add_upload_info=True,
            add_multimodel_upload_info=True,
            add_audio_upload_info=False,
        )


def test_format_as_multimodal_typeerror_bad_audio_value():
    bad = MagicMock()
    bad.get_type_and_value.return_value = ('audio', ['x'])
    bad.text = None
    msg = Message.model_construct(role=USER, content=[bad])
    with pytest.raises(TypeError):
        format_as_multimodal_message(
            msg,
            add_upload_info=True,
            add_multimodel_upload_info=False,
            add_audio_upload_info=True,
        )


def test_format_as_text_message_flattens():
    msg = Message(USER, [
        ContentItem(text='a'),
        ContentItem(image='https://example.com/x.png'),
        ContentItem(text='b'),
    ])
    out = format_as_text_message(msg, add_upload_info=True, lang='en')
    assert isinstance(out.content, str)
    assert 'a' in out.content
    assert 'b' in out.content
    assert '![image](x.png)' in out.content


def test_extract_text_from_message_list_and_str():
    assert extract_text_from_message(Message(USER, '  hi  '), add_upload_info=False) == 'hi'
    multi = Message(USER, [ContentItem(text='x'), ContentItem(file='/tmp/f.txt')])
    text = extract_text_from_message(multi, add_upload_info=True, lang='en')
    assert 'x' in text
    assert 'Uploaded' in text


def test_extract_text_from_message_typeerror():
    msg = Message.model_construct(role=USER, content=object())
    with pytest.raises(TypeError, match='List of str or str expected'):
        extract_text_from_message(msg, add_upload_info=False)


def test_extract_files_and_images():
    messages = [
        Message(USER, [
            ContentItem(file='/a.pdf'),
            ContentItem(image='https://ex.com/i.png'),
            ContentItem(file='/a.pdf'),
        ]),
        Message(ASSISTANT, 'ok'),
    ]
    assert extract_files_from_messages(messages, include_images=False) == ['/a.pdf']
    assert extract_files_from_messages(messages, include_images=True) == [
        '/a.pdf',
        'https://ex.com/i.png',
    ]
    assert extract_images_from_messages(messages) == ['https://ex.com/i.png']


def test_extract_urls_helpers():
    assert extract_urls('see https://a.com/x and http://b.com/y') == [
        'https://a.com/x',
        'http://b.com/y',
    ]
    assert extract_markdown_urls('![img](https://i.com/a.png) [t](https://t.com)') == [
        'https://i.com/a.png',
        'https://t.com',
    ]


def test_has_chinese_messages_and_last_user_idx():
    msgs = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'ok'},
        {'role': 'user', 'content': '中文'},
    ]
    assert has_chinese_messages(msgs) is True
    assert has_chinese_messages(
        [{'role': 'user', 'content': 'ascii only'}],
        check_roles=('user',),
    ) is False
    assert get_last_usr_msg_idx(msgs) == 3


def test_rm_default_system_variants():
    user = Message(USER, 'hi')
    default_sys = Message(SYSTEM, DEFAULT_SYSTEM_MESSAGE)
    assert rm_default_system([default_sys, user]) == [user]

    custom = Message(SYSTEM, 'custom')
    assert rm_default_system([custom, user]) == [custom, user]

    default_list = Message(SYSTEM, [ContentItem(text=DEFAULT_SYSTEM_MESSAGE)])
    assert rm_default_system([default_list, user]) == [user]

    multi_sys = Message(SYSTEM, [ContentItem(text='a'), ContentItem(text='b')])
    assert rm_default_system([multi_sys, user]) == [multi_sys, user]

    assert rm_default_system([user]) == [user]
    assert rm_default_system([default_sys]) == [default_sys]

    bad = Message.model_construct(role=SYSTEM, content=object())
    with pytest.raises(TypeError):
        rm_default_system([bad, user])


def test_build_text_completion_prompt_basic_and_special():
    msgs = [Message(SYSTEM, 'sys'), Message(USER, 'q')]
    prompt = build_text_completion_prompt(msgs, allow_special=False)
    assert '<|im_start|>system' in prompt
    assert 'sys' in prompt
    assert '<|im_start|>user' in prompt
    assert prompt.endswith('<|im_start|>assistant\n')

    with_fc = [
        Message(USER, 'call it'),
        Message(
            ASSISTANT,
            'thinking',
            function_call=FunctionCall(name='add', arguments='{"x": 1}'),
        ),
    ]
    special = build_text_completion_prompt(with_fc, allow_special=True, default_system='')
    assert '<tool_call>' in special
    assert 'add' in special

    bad_json = [
        Message(USER, 'x'),
        Message(
            ASSISTANT,
            '',
            function_call=FunctionCall(name='f', arguments='{not-json'),
        ),
    ]
    prompt2 = build_text_completion_prompt(bad_json, allow_special=True, default_system='d')
    assert '"name": "f"' in prompt2
