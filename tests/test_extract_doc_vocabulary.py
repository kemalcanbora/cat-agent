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

"""Coverage tests for cat_agent.tools.extract_doc_vocabulary."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.storage import KeyNotExistsError


def _make_tool():
    with patch('cat_agent.tools.extract_doc_vocabulary.SimpleDocParser') as MockParser, \
            patch('cat_agent.tools.extract_doc_vocabulary.Storage') as MockStorage:
        from cat_agent.tools.extract_doc_vocabulary import ExtractDocVocabulary

        tool = ExtractDocVocabulary({'path': '/tmp/extract_doc_vocab_test'})
        return tool, MockParser.return_value, MockStorage.return_value


def test_extract_doc_vocabulary_cache_hit():
    tool, parser, db = _make_tool()
    parser.call.return_value = 'ignored because cached'
    db.call.return_value = 'alpha, beta'

    out = tool.call({'files': ['/docs/a.txt', '/docs/b.txt']})
    assert out == 'alpha, beta'
    db.call.assert_called_with({'operate': 'get', 'key': str(['/docs/a.txt', '/docs/b.txt'])})


def test_extract_doc_vocabulary_files_json_string():
    """Schema requires array; string form is accepted only after verify is bypassed."""
    tool, parser, db = _make_tool()
    parser.call.return_value = 'doc text'
    db.call.return_value = 'cached'

    tool._verify_json_format_args = lambda params: (
        {'files': params['files']} if isinstance(params, dict) else params
    )
    out = tool.call({'files': '["/a.txt"]'})
    assert out == 'cached'
    parser.call.assert_called()


def test_extract_doc_vocabulary_sklearn_path():
    tool, parser, db = _make_tool()
    parser.call.side_effect = lambda params, **kwargs: f"text for {params['url']}"

    def db_call(params):
        if params.get('operate') == 'get':
            raise KeyNotExistsError(params['key'])
        return None

    db.call.side_effect = db_call

    vectorizer = MagicMock()
    matrix = MagicMock()
    matrix.toarray.return_value.flatten.return_value = [0.2, 0.9]
    vectorizer.fit_transform.return_value = matrix
    vectorizer.get_feature_names_out.return_value = ['low', 'high']

    fake_module = MagicMock()
    fake_module.TfidfVectorizer.return_value = vectorizer

    with patch.dict(sys.modules, {
        'sklearn': MagicMock(),
        'sklearn.feature_extraction': MagicMock(),
        'sklearn.feature_extraction.text': fake_module,
    }):
        out = tool.call({'files': ['/docs/one.txt']})

    assert out == 'high, low'
    put_calls = [c for c in db.call.call_args_list if c.args[0].get('operate') == 'put']
    assert put_calls
    stored = json.loads(put_calls[0].args[0]['value'])
    assert stored == 'high, low'


def test_extract_doc_vocabulary_sklearn_missing():
    tool, parser, db = _make_tool()
    parser.call.return_value = 'some document text'

    def db_call(params):
        if params.get('operate') == 'get':
            raise KeyNotExistsError('missing')
        return None

    db.call.side_effect = db_call

    with patch.dict(sys.modules, {
        'sklearn': None,
        'sklearn.feature_extraction': None,
        'sklearn.feature_extraction.text': None,
    }):
        with pytest.raises(ModuleNotFoundError, match='scikit-learn'):
            tool.call({'files': ['/docs/a.txt']})
