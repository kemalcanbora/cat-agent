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

"""Coverage tests for cat_agent.tools.image_search (mocked HTTP)."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import USER, ContentItem, Message
from cat_agent.tools import image_search as mod
from cat_agent.tools.image_search import ImageResult, ImageSearch, check_image_url_accessibility, serper_search


@pytest.fixture(autouse=True)
def _logger():
    """image_search references logger without importing it; inject a mock."""
    previous = getattr(mod, 'logger', None)
    mod.logger = MagicMock()
    yield
    if previous is None:
        delattr(mod, 'logger')
    else:
        mod.logger = previous


def test_image_result_str_and_getitem():
    img = ImageResult(
        id='1',
        title='t',
        imgurl='https://ex.com/i.jpg',
        url='https://ex.com/page',
        width='10',
        height='20',
        content='desc',
    )
    s = str(img)
    assert 'i.jpg' in s
    assert img['title'] == 't'
    img['title'] = 'new'
    assert img.title == 'new'


def test_check_image_url_accessibility():
    ok = MagicMock()
    ok.status_code = 200
    with patch.object(mod.requests, 'head', return_value=ok):
        url, accessible = check_image_url_accessibility('https://ex.com/a.png')
    assert url.endswith('a.png')
    assert accessible is True

    with patch.object(mod.requests, 'head', side_effect=RuntimeError('down')):
        url, accessible = check_image_url_accessibility('https://ex.com/b.png')
    assert accessible is False


def test_serper_search_requires_api_key():
    with patch.object(mod, 'SERPAPI_IMAGE_SEARCH_KEY', ''):
        with pytest.raises(ValueError, match='SERPAPI_IMAGE_SEARCH_KEY'):
            serper_search('https://ex.com/q.png')


def test_serper_search_success_without_accessibility_check():
    payload = {
        'image_results': [
            {
                'position': 1,
                'title': 'Cat',
                'original': 'https://cdn.ex.com/cat.jpg',
                'link': 'https://ex.com/1',
                'width': '1',
                'height': '2',
                'snippet': 'cute',
            },
            {
                'position': 2,
                'title': '',
                'thumbnail': 'https://cdn.ex.com/dog.jpg',
                'link': 'https://ex.com/2',
            },
            {
                'position': 3,
                'title': 'skip',
                # no original/thumbnail
                'link': 'https://ex.com/3',
            },
        ],
        'inline_images': [],
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload

    with patch.object(mod, 'SERPAPI_IMAGE_SEARCH_KEY', 'k'), \
            patch.object(mod, 'guard_outbound_request'), \
            patch.object(mod.requests, 'get', return_value=resp):
        results = serper_search('https://ex.com/q.png', check_accessibility=False, max_retry=1)

    assert len(results) == 2
    assert results[0].imgurl.endswith('cat.jpg')
    assert results[1].imgurl.endswith('dog.jpg')


def test_serper_search_with_accessibility_filter():
    payload = {
        'image_results': [{
            'position': 1,
            'title': 'A',
            'original': 'https://cdn.ex.com/a.jpg',
            'link': 'https://ex.com/a',
            'snippet': '',
        }],
        'inline_images': [{
            'position': 2,
            'title': 'B',
            'original': 'https://cdn.ex.com/b.jpg',
            'link': 'https://ex.com/b',
        }],
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload

    def access(url, timeout=10):
        return url, url.endswith('a.jpg')

    with patch.object(mod, 'SERPAPI_IMAGE_SEARCH_KEY', 'k'), \
            patch.object(mod, 'guard_outbound_request'), \
            patch.object(mod.requests, 'get', return_value=resp), \
            patch.object(mod, 'check_image_url_accessibility', side_effect=access):
        results = serper_search('https://ex.com/q.png', check_accessibility=True, max_retry=1)

    assert len(results) == 1
    assert results[0].imgurl.endswith('a.jpg')


def test_serper_search_retries_then_empty():
    with patch.object(mod, 'SERPAPI_IMAGE_SEARCH_KEY', 'k'), \
            patch.object(mod, 'guard_outbound_request'), \
            patch.object(mod.requests, 'get', side_effect=RuntimeError('net')), \
            patch.object(mod.time, 'sleep'), \
            patch.object(mod.random, 'uniform', return_value=0.0):
        assert serper_search('https://ex.com/q.png', max_retry=2) == []


def test_serper_search_skips_bad_items():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        'image_results': [
            {'title': 'bad', 'original': 'https://cdn.ex.com/bad.jpg', 'link': ''},
            {'title': 'ok', 'original': 'https://cdn.ex.com/ok.jpg', 'link': '', 'position': 2},
        ],
        'inline_images': [],
    }
    real = mod.ImageResult

    def ctor(**kwargs):
        if kwargs.get('title') == 'bad':
            raise ValueError('parse fail')
        return real(**kwargs)

    with patch.object(mod, 'SERPAPI_IMAGE_SEARCH_KEY', 'k'), \
            patch.object(mod, 'guard_outbound_request'), \
            patch.object(mod.requests, 'get', return_value=resp), \
            patch.object(mod, 'ImageResult', side_effect=ctor):
        results = serper_search('https://ex.com/q.png', check_accessibility=False, max_retry=1)

    assert len(results) == 1
    assert results[0].title == 'ok'


def test_image_search_call_no_images():
    tool = ImageSearch()
    assert tool.call({'img_idx': 0}, messages=[]) == 'Error: no images found in the messages.'


def test_image_search_call_clamps_index_and_formats():
    tool = ImageSearch()
    messages = [Message(USER, [ContentItem(image='https://ex.com/only.png')])]
    fake = ImageResult(
        id='1',
        title='T',
        imgurl='https://cdn.ex.com/r.jpg',
        url='https://ex.com/p',
        width='1',
        height='1',
        content='body',
    )
    with patch.object(mod, 'serper_search', return_value=[fake]):
        out = tool.call({'img_idx': 99}, messages=messages)

    assert isinstance(out, list)
    texts = [c.text for c in out if c.text]
    images = [c.image for c in out if c.image]
    assert any('r.jpg' in (t or '') for t in texts)
    assert 'https://cdn.ex.com/r.jpg' in images


def test_image_search_call_exception_returns_empty():
    tool = ImageSearch()
    messages = [Message(USER, [ContentItem(image='https://ex.com/x.png')])]
    with patch.object(mod, 'serper_search', side_effect=RuntimeError('boom')):
        out = tool.call({'img_idx': 0}, messages=messages)
    assert out == []
