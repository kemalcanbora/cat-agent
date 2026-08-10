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

"""Named registry of long-lived agents for HTTP serving."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
)

from cat_agent.agent import Agent
from cat_agent.log import logger

_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')

AgentState = Literal['pending', 'ready', 'failed']
AgentFactory = Callable[[], Agent]


class CapacityFull(Exception):
    """Raised when the per-agent waiter queue is full (HTTP 429)."""

    def __init__(self, agent: str):
        self.agent = agent
        super().__init__(f'agent {agent!r} busy; queue full')


def normalize_agent_name(name: str) -> str:
    """Validate a URL-safe agent key (used in ``/agents/{name}``)."""
    key = (name or '').strip()
    if not _NAME_RE.match(key):
        raise ValueError(
            f'Invalid agent name {name!r}: use letters, digits, '
            "'.', '_', or '-' (1–128 chars, start alphanumeric)."
        )
    return key


@dataclass(frozen=True)
class AgentInfo:
    """Public metadata for a registered agent."""

    name: str
    description: str
    agent_class: str
    max_concurrency: int
    max_queue: int = 8
    state: AgentState = 'ready'


@dataclass
class _Slot:
    name: str
    max_concurrency: int
    max_queue: int
    state: AgentState = 'pending'
    agent: Optional[Agent] = None
    factory: Optional[AgentFactory] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    semaphore: Optional[asyncio.Semaphore] = None
    inflight: int = 0
    waiters: int = 0
    from_factory: bool = False


def _default_concurrency() -> int:
    from cat_agent.settings import SERVE_MAX_CONCURRENCY
    return SERVE_MAX_CONCURRENCY


def _default_max_queue() -> int:
    from cat_agent.settings import SERVE_MAX_QUEUE
    return SERVE_MAX_QUEUE


class AgentRegistry:
    """Map URL names to Agent instances with per-agent concurrency limits.

    Agents may be registered eagerly via :meth:`register`, or deferred via
    :meth:`register_factory` and built sequentially during app lifespan startup.

    Capacity: each agent has an ``asyncio.Semaphore(max_concurrency)`` and a
    bounded waiter counter (``max_queue``). When the agent is busy and
    ``waiters >= max_queue``, new requests are rejected with
    :class:`CapacityFull` (HTTP 429) instead of queueing unboundedly.
    ``max_queue=0`` means reject immediately when busy, but still serve when idle.
    """

    def __init__(
        self,
        *,
        default_max_concurrency: Optional[int] = None,
        default_max_queue: Optional[int] = None,
    ):
        if default_max_concurrency is None:
            default_max_concurrency = _default_concurrency()
        if default_max_queue is None:
            default_max_queue = _default_max_queue()
        if default_max_concurrency < 1:
            raise ValueError('default_max_concurrency must be >= 1')
        if default_max_queue < 0:
            raise ValueError('default_max_queue must be >= 0')
        self._default_max_concurrency = default_max_concurrency
        self._default_max_queue = default_max_queue
        self._slots: Dict[str, _Slot] = {}

    def _limit(self, max_concurrency: Optional[int]) -> int:
        limit = self._default_max_concurrency if max_concurrency is None else int(max_concurrency)
        if limit < 1:
            raise ValueError('max_concurrency must be >= 1')
        return limit

    def _queue_limit(self, max_queue: Optional[int]) -> int:
        limit = self._default_max_queue if max_queue is None else int(max_queue)
        if limit < 0:
            raise ValueError('max_queue must be >= 0')
        return limit

    def _ensure_unique(self, key: str) -> None:
        if key in self._slots:
            raise ValueError(f'Agent already registered: {key!r}')

    def register(
        self,
        agent: Agent,
        *,
        name: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        max_queue: Optional[int] = None,
    ) -> str:
        """Register an already-built *agent* under *name* (defaults to ``agent.name``).

        Returns the registry key. State is ``ready`` immediately.
        """
        if not isinstance(agent, Agent):
            raise TypeError(f'Expected Agent, got {type(agent)!r}')
        key = normalize_agent_name(name or (agent.name or ''))
        self._ensure_unique(key)
        limit = self._limit(max_concurrency)
        qlimit = self._queue_limit(max_queue)
        self._slots[key] = _Slot(
            name=key,
            max_concurrency=limit,
            max_queue=qlimit,
            state='ready',
            agent=agent,
            semaphore=asyncio.Semaphore(limit),
            from_factory=False,
        )
        return key

    def register_factory(
        self,
        factory: AgentFactory,
        *,
        name: str,
        max_concurrency: Optional[int] = None,
        max_queue: Optional[int] = None,
    ) -> str:
        """Register a zero-arg factory built during lifespan startup.

        State starts as ``pending``. Construction failures become ``failed``
        without crashing the process — readiness reports the reason.
        """
        if not callable(factory):
            raise TypeError(f'Expected callable factory, got {type(factory)!r}')
        key = normalize_agent_name(name)
        self._ensure_unique(key)
        limit = self._limit(max_concurrency)
        qlimit = self._queue_limit(max_queue)
        self._slots[key] = _Slot(
            name=key,
            max_concurrency=limit,
            max_queue=qlimit,
            state='pending',
            factory=factory,
            from_factory=True,
        )
        return key

    @property
    def has_deferred_factories(self) -> bool:
        """True if any agent was registered via :meth:`register_factory`.

        Used by ``run_app`` to reject ``workers > 1`` (fork-before-lifespan).
        """
        return any(slot.from_factory for slot in self._slots.values())

    def get(self, name: str) -> Agent:
        key = normalize_agent_name(name)
        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(f'Unknown agent: {key!r}')
        if slot.state != 'ready' or slot.agent is None:
            raise KeyError(f'Agent not ready: {key!r} (state={slot.state})')
        return slot.agent

    def get_slot_state(self, name: str) -> AgentState:
        key = normalize_agent_name(name)
        try:
            return self._slots[key].state
        except KeyError as exc:
            raise KeyError(f'Unknown agent: {key!r}') from exc

    def semaphore(self, name: str) -> asyncio.Semaphore:
        key = normalize_agent_name(name)
        slot = self._slots.get(key)
        if slot is None:
            raise KeyError(f'Unknown agent: {key!r}')
        if slot.semaphore is None:
            raise KeyError(f'Agent not ready: {key!r} (state={slot.state})')
        return slot.semaphore

    def capacity_stats(self, name: str) -> Dict[str, int]:
        """Return ``inflight``, ``waiters``, ``max_concurrency``, ``max_queue``."""
        key = normalize_agent_name(name)
        slot = self._slots[key]
        return {
            'inflight': slot.inflight,
            'waiters': slot.waiters,
            'max_concurrency': slot.max_concurrency,
            'max_queue': slot.max_queue,
        }

    async def acquire_run_slot(self, name: str) -> None:
        """Acquire a concurrency slot or raise :class:`CapacityFull`.

        When the semaphore is exhausted, the request counts as a waiter. If
        ``waiters >= max_queue``, raise immediately (no unbounded queue).
        Pair every successful call with :meth:`release_run_slot`.
        """
        key = normalize_agent_name(name)
        slot = self._slots[key]
        sem = slot.semaphore
        if sem is None:
            raise KeyError(f'Agent not ready: {key!r} (state={slot.state})')

        waiting = False
        if sem.locked():
            if slot.waiters >= slot.max_queue:
                raise CapacityFull(key)
            slot.waiters += 1
            waiting = True
        try:
            await sem.acquire()
        except BaseException:
            if waiting:
                slot.waiters = max(0, slot.waiters - 1)
            raise
        else:
            if waiting:
                slot.waiters = max(0, slot.waiters - 1)
            slot.inflight += 1

    def release_run_slot(self, name: str) -> None:
        """Release a slot acquired via :meth:`acquire_run_slot`."""
        key = normalize_agent_name(name)
        slot = self._slots[key]
        if slot.inflight > 0:
            slot.inflight -= 1
        if slot.semaphore is not None:
            slot.semaphore.release()

    def names(self) -> List[str]:
        return sorted(self._slots)

    def list_info(self) -> List[AgentInfo]:
        out: List[AgentInfo] = []
        for key in self.names():
            slot = self._slots[key]
            if slot.agent is not None:
                description = (slot.agent.description or '') or ''
                agent_class = type(slot.agent).__name__
            else:
                description = ''
                agent_class = 'deferred'
            out.append(
                AgentInfo(
                    name=key,
                    description=description,
                    agent_class=agent_class,
                    max_concurrency=slot.max_concurrency,
                    max_queue=slot.max_queue,
                    state=slot.state,
                )
            )
        return out

    def readiness_payload(self) -> Tuple[bool, Dict[str, dict]]:
        """Return ``(all_ready, per_agent_breakdown)`` for ``/readyz``."""
        agents: Dict[str, dict] = {}
        all_ready = True
        for key in self.names():
            slot = self._slots[key]
            if slot.state == 'ready':
                agents[key] = {
                    'state': 'ready',
                    'inflight': slot.inflight,
                    'capacity': slot.max_concurrency,
                    'queue_waiters': slot.waiters,
                    'max_queue': slot.max_queue,
                }
            elif slot.state == 'failed':
                all_ready = False
                agents[key] = {
                    'state': 'failed',
                    'error_type': slot.error_type or 'Error',
                    'error': slot.error or '',
                }
            else:
                all_ready = False
                agents[key] = {'state': 'pending'}
        return all_ready, agents

    async def build_deferred(self) -> None:
        """Build pending factories sequentially off the event loop.

        Failures are recorded on the slot; they are never re-raised.
        """
        pending = [
            slot for slot in (self._slots[k] for k in self.names())
            if slot.state == 'pending' and slot.factory is not None
        ]
        for slot in pending:
            factory = slot.factory
            assert factory is not None
            logger.info('serve: building agent {!r}…', slot.name)
            started = time.monotonic()
            try:
                agent = await asyncio.to_thread(factory)
                if not isinstance(agent, Agent):
                    raise TypeError(
                        f'Factory for {slot.name!r} returned {type(agent)!r}, expected Agent'
                    )
                slot.agent = agent
                slot.semaphore = asyncio.Semaphore(slot.max_concurrency)
                slot.state = 'ready'
                slot.error_type = None
                slot.error = None
                elapsed = time.monotonic() - started
                logger.info(
                    'serve: agent {!r} ready in {:.2f}s ({})',
                    slot.name,
                    elapsed,
                    type(agent).__name__,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                slot.state = 'failed'
                slot.agent = None
                slot.semaphore = None
                slot.error_type = type(exc).__name__
                slot.error = str(exc)
                logger.error(
                    'serve: agent {!r} failed after {:.2f}s: {}: {}',
                    slot.name,
                    elapsed,
                    slot.error_type,
                    slot.error,
                )

    def ready_agents(self) -> Iterator[Agent]:
        for key in self.names():
            slot = self._slots[key]
            if slot.state == 'ready' and slot.agent is not None:
                yield slot.agent

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            key = normalize_agent_name(name)
        except ValueError:
            return False
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Agent]:
        return self.ready_agents()

    def agents(self) -> Iterable[Agent]:
        return iter(self)
