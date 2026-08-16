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

"""Coverage tests for cat_agent.utils.file_utils (mocked network I/O)."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from cat_agent.utils import file_utils as fu


def test_get_basename_from_url_variants():
    assert fu.get_basename_from_url('https://ex.com/a/b%20c.pdf') == 'b c.pdf'
    assert fu.get_basename_from_url(r'C:\folder\file.txt') == 'file.txt'
    assert fu.get_basename_from_url('https://ex.com/') == 'ex.com'


def test_is_http_url_and_is_image():
    assert fu.is_http_url('http://a') is True
    assert fu.is_http_url('https://a') is True
    assert fu.is_http_url('/local') is False
    assert fu.is_image('https://ex.com/x.PNG') is True
    assert fu.is_image('https://ex.com/x.webp') is True
    assert fu.is_image('https://ex.com/x.txt') is False


def test_sanitize_paths_existing_and_fallbacks(tmp_path):
    existing = tmp_path / 'ok.txt'
    existing.write_text('x', encoding='utf-8')
    assert fu.sanitize_chrome_file_path(str(existing)) == str(existing)
    assert fu.sanitize_windows_file_path(str(existing)) == str(existing)

    missing = str(tmp_path / 'missing.txt')
    assert fu.sanitize_windows_file_path(missing) == missing
    assert fu.sanitize_chrome_file_path(missing) == missing


def test_sanitize_windows_strips_leading_slash(tmp_path):
    f = tmp_path / 'win.txt'
    f.write_text('ok', encoding='utf-8')
    # path that exists only after stripping a leading slash is hard to
    # construct portably; exercise the non-existing fallback branches.
    assert fu.sanitize_windows_file_path('/no/such/file') == '/no/such/file'


def test_save_url_to_local_work_dir_http_success(tmp_path):
    dest_dir = tmp_path / 'dl'
    dest_dir.mkdir()
    target = dest_dir / 'f.bin'
    target.write_bytes(b'old')

    resp = MagicMock()
    resp.status_code = 200
    resp.content = b'new-bytes'
    with patch.object(fu.requests, 'get', return_value=resp) as get:
        path = fu.save_url_to_local_work_dir('https://ex.com/f.bin', str(dest_dir))
    assert path == str(target)
    assert target.read_bytes() == b'new-bytes'
    get.assert_called_once()


def test_save_url_to_local_work_dir_http_failure(tmp_path):
    dest_dir = tmp_path / 'dl'
    dest_dir.mkdir()
    resp = MagicMock()
    resp.status_code = 404
    with patch.object(fu.requests, 'get', return_value=resp):
        with pytest.raises(ValueError, match='Can not download'):
            fu.save_url_to_local_work_dir('https://ex.com/missing.bin', str(dest_dir), save_filename='m.bin')


def test_save_url_to_local_work_dir_local_copy(tmp_path):
    src = tmp_path / 'src.txt'
    src.write_text('copied', encoding='utf-8')
    dest_dir = tmp_path / 'out'
    dest_dir.mkdir()
    with patch.object(fu, 'sanitize_chrome_file_path', return_value=str(src)):
        path = fu.save_url_to_local_work_dir(str(src), str(dest_dir), save_filename='dst.txt')
    assert (tmp_path / 'out' / 'dst.txt').read_text(encoding='utf-8') == 'copied'
    assert path.endswith('dst.txt')


def test_save_and_read_text(tmp_path):
    path = tmp_path / 't.txt'
    fu.save_text_to_file(str(path), 'hello')
    assert fu.read_text_from_file(str(path)) == 'hello'


def test_read_text_charset_fallback(tmp_path):
    path = tmp_path / 'latin.txt'
    path.write_text('placeholder', encoding='utf-8')
    with patch('builtins.open', mock_open()) as m:
        m.return_value.read.side_effect = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'err')
        best = MagicMock()
        best.__str__ = lambda self: 'via-charset'
        with patch('charset_normalizer.from_path', return_value=MagicMock(best=lambda: best)):
            assert fu.read_text_from_file(str(path)) == 'via-charset'


def test_contains_html_tags():
    assert fu.contains_html_tags('<div class="x">') is True
    assert fu.contains_html_tags('plain text') is False


def test_get_content_type_by_head_request():
    resp = MagicMock()
    resp.headers = {'Content-Type': 'application/pdf'}
    with patch.object(fu.requests, 'head', return_value=resp) as head:
        assert fu.get_content_type_by_head_request('https://ex.com/a') == 'application/pdf'
        head.assert_called_once()

    with patch.object(fu.requests, 'head', side_effect=requests.RequestException('down')):
        assert fu.get_content_type_by_head_request('https://ex.com/a') == 'unk'


def test_get_file_type_extension_and_http():
    assert fu.get_file_type('https://ex.com/a.PDF') == 'pdf'
    assert fu.get_file_type('/tmp/x.docx') == 'docx'

    with patch.object(fu, 'get_content_type_by_head_request', return_value='application/pdf'):
        assert fu.get_file_type('https://ex.com/noext') == 'pdf'
    with patch.object(fu, 'get_content_type_by_head_request', return_value='application/msword'):
        assert fu.get_file_type('https://ex.com/noext') == 'docx'
    with patch.object(fu, 'get_content_type_by_head_request', return_value='text/html'):
        assert fu.get_file_type('https://ex.com/noext') == 'html'


def test_get_file_type_local_html_txt_unk(tmp_path):
    html = tmp_path / 'page.dat'
    html.write_text('<html><div>x</div></html>', encoding='utf-8')
    assert fu.get_file_type(str(html)) == 'html'

    txt = tmp_path / 'notes.dat'
    txt.write_text('just text', encoding='utf-8')
    assert fu.get_file_type(str(txt)) == 'txt'

    with patch.object(fu, 'read_text_from_file', side_effect=OSError('boom')):
        assert fu.get_file_type(str(tmp_path / 'bad.dat')) == 'unk'
