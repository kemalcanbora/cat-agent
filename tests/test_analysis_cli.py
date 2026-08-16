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

"""Tests for python -m cat_agent.analysis CLI."""

from pathlib import Path

from cat_agent.analysis.__main__ import main


FIXTURE = Path(__file__).parent / 'fixtures' / 'clean_traces' / 'clean_chat_01.jsonl'


def test_analysis_cli_text_report(capsys):
    assert FIXTURE.exists()
    code = main([str(FIXTURE), '--no-judge'])
    assert code == 0
    out = capsys.readouterr().out
    assert 'MAST' in out or 'mode' in out.lower() or 'No MAST' in out


def test_analysis_cli_json(capsys):
    code = main([str(FIXTURE), '--json', '--no-judge'])
    assert code == 0
    out = capsys.readouterr().out
    assert '"run_id"' in out or '"findings"' in out


def test_analysis_cli_missing_file(capsys, tmp_path):
    missing = tmp_path / 'empty.jsonl'
    missing.write_text('', encoding='utf-8')
    code = main([str(missing), '--no-judge'])
    assert code == 1
    err = capsys.readouterr().err
    assert 'No runs found' in err


def test_analysis_cli_batch(capsys):
    code = main([str(FIXTURE), str(FIXTURE), '--batch', '--no-judge'])
    assert code == 0
    out = capsys.readouterr().out
    assert 'Batch:' in out
