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

"""Controllable fake agents for serve tests (no network, no models)."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator, Iterator, List, Optional

from cat_agent.agent import Agent
from cat_agent.llm.schema import ASSISTANT, Message


class FakeAgent(Agent):
    """Agent with controllable construction and run duration."""

    def __init__(
        self,
        name: str,
        reply: str = 'ok',
        *,
        description: str = '',
        run_delay: float = 0.0,
        raise_on_run: Optional[BaseException] = None,
        raise_after_first_yield: Optional[BaseException] = None,
        started: Optional[threading.Event] = None,
        release: Optional[threading.Event] = None,
    ):
        super().__init__(name=name, description=description, system_message='')
        self._reply = reply
        self._run_delay = run_delay
        self._raise_on_run = raise_on_run
        self._raise_after_first_yield = raise_after_first_yield
        self._started = started
        self._release = release

    def _prepare(self) -> None:
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            self._release.wait(timeout=30)
        if self._run_delay > 0:
            time.sleep(self._run_delay)

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        self._prepare()
        if self._raise_on_run is not None:
            raise self._raise_on_run
        yield [Message(role=ASSISTANT, content=self._reply, name=self.name)]
        if self._raise_after_first_yield is not None:
            raise self._raise_after_first_yield

    async def _arun(
        self, messages: List[Message], lang: str = 'en', **kwargs,
    ) -> AsyncIterator[List[Message]]:
        # True async generator so mid-stream failures are not collapsed by
        # Agent._arun's default list(self._run(...)) collection.
        await asyncio.to_thread(self._prepare)
        if self._raise_on_run is not None:
            raise self._raise_on_run
        yield [Message(role=ASSISTANT, content=self._reply, name=self.name)]
        if self._raise_after_first_yield is not None:
            raise self._raise_after_first_yield


class ConstructionTracker:
    """Records overlapping factory builds to assert sequential construction."""

    def __init__(self):
        self.events: List[tuple] = []
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def begin(self, name: str) -> None:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.events.append(('start', name, time.monotonic()))

    def end(self, name: str) -> None:
        with self._lock:
            self._active -= 1
            self.events.append(('end', name, time.monotonic()))


def make_factory(
    name: str,
    *,
    reply: str = 'ok',
    delay: float = 0.0,
    error: Optional[BaseException] = None,
    tracker: Optional[ConstructionTracker] = None,
):
    """Return a zero-arg factory for :meth:`AgentRegistry.register_factory`."""

    def factory() -> FakeAgent:
        if tracker is not None:
            tracker.begin(name)
        try:
            if delay > 0:
                time.sleep(delay)
            if error is not None:
                raise error
            return FakeAgent(name=name, reply=reply)
        finally:
            if tracker is not None:
                tracker.end(name)

    return factory
