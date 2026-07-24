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

"""One-way control transfer between agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Handoff:
    """Sentinel signalling that control should transfer to another agent.

    Unlike ``ask_agent`` (which returns to the caller), a handoff means the
    receiving agent owns the rest of the conversation.
    """

    to: str
    context: Optional[str] = None
