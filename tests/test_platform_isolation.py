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

"""Ensure importing core packages does not load cat_agent.platform."""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent


def test_core_import_does_not_load_platform():
    # Run in a fresh interpreter so clearing sys.modules cannot leave stale
    # module objects bound by other test modules in this process.
    script = dedent(
        """
        import sys
        import cat_agent.agent  # noqa: F401
        import cat_agent.serve  # noqa: F401
        import cat_agent.tools.base  # noqa: F401
        assert 'cat_agent.platform' not in sys.modules
        assert not any(k.startswith('cat_agent.platform.') for k in sys.modules)
        """
    )
    proc = subprocess.run(
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
