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

"""Tests for cat_agent.utils.misc helpers not covered via utils re-exports."""

import signal

from cat_agent.utils.misc import append_signal_handler, get_local_ip, print_traceback


def test_get_local_ip_returns_string():
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert ip  # non-empty


def test_print_traceback_warning(caplog):
    try:
        raise ValueError('boom')
    except ValueError:
        print_traceback(is_error=False)


def test_append_signal_handler_chains(monkeypatch):
    calls = []

    def first(*_a, **_k):
        calls.append('first')

    def second(*_a, **_k):
        calls.append('second')

    # Use a rarely used signal and restore afterward.
    sig = signal.SIGUSR1
    prev = signal.getsignal(sig)
    try:
        signal.signal(sig, first)
        append_signal_handler(sig, second)
        signal.raise_signal(sig)
        assert calls == ['second', 'first']
    finally:
        signal.signal(sig, prev)
