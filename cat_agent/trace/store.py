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

"""Append-only trace persistence."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, runtime_checkable

from cat_agent.trace.redact import redact_obj
from cat_agent.trace.schema import Run, Step


@runtime_checkable
class TraceStore(Protocol):
    def write_run_header(self, run: Run) -> None: ...

    def append_step(self, run_id: str, step: Step) -> None: ...

    def finalize_run(self, run: Run) -> None: ...

    def load_run(self, run_id: str) -> Optional[Run]: ...

    def iter_runs(self) -> Iterator[Run]: ...


def _run_to_json(run: Run) -> dict:
    return redact_obj(run.model_dump(mode='json'))


def _step_to_json(run_id: str, step: Step) -> dict:
    return redact_obj({
        'record_type': 'step',
        'run_id': run_id,
        **step.model_dump(mode='json'),
    })


class InMemoryTraceStore:
    """Thread-safe in-memory store for tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: Dict[str, Run] = {}
        self._partial: Dict[str, Run] = {}

    def write_run_header(self, run: Run) -> None:
        with self._lock:
            self._partial[run.run_id] = run.model_copy(deep=True)

    def append_step(self, run_id: str, step: Step) -> None:
        with self._lock:
            run = self._partial.get(run_id) or self._runs.get(run_id)
            if run is None:
                return
            run.steps.append(step.model_copy(deep=True))
            run.recompute_totals()
            self._partial[run_id] = run

    def finalize_run(self, run: Run) -> None:
        with self._lock:
            stored = run.model_copy(deep=True)
            self._runs[run.run_id] = stored
            self._partial[run.run_id] = stored

    def load_run(self, run_id: str) -> Optional[Run]:
        with self._lock:
            run = self._runs.get(run_id) or self._partial.get(run_id)
            return run.model_copy(deep=True) if run else None

    def iter_runs(self) -> Iterator[Run]:
        with self._lock:
            ids = list(dict.fromkeys([*self._runs.keys(), *self._partial.keys()]))
            for rid in ids:
                run = self.load_run(rid)
                if run is not None:
                    yield run


class JSONLTraceStore:
    """Append-only JSONL file store.

    Record types:
    - ``run_header`` — written at start (status=running)
    - ``step`` — flushed as each step completes
    - ``run_final`` — written on completion / failure / termination

    A crashed process still leaves a parseable partial trace (header + steps).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, Run] = {}

    def _append_line(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open('a', encoding='utf-8') as fh:
                fh.write(line + '\n')
                fh.flush()

    def write_run_header(self, run: Run) -> None:
        payload = _run_to_json(run)
        payload['record_type'] = 'run_header'
        # Steps are written separately; keep header lean.
        payload['steps'] = []
        self._append_line(payload)
        with self._lock:
            self._cache[run.run_id] = run.model_copy(deep=True)

    def append_step(self, run_id: str, step: Step) -> None:
        self._append_line(_step_to_json(run_id, step))
        with self._lock:
            run = self._cache.get(run_id)
            if run is not None:
                run.steps.append(step.model_copy(deep=True))
                run.recompute_totals()

    def finalize_run(self, run: Run) -> None:
        payload = _run_to_json(run)
        payload['record_type'] = 'run_final'
        self._append_line(payload)
        with self._lock:
            self._cache[run.run_id] = run.model_copy(deep=True)

    def load_run(self, run_id: str) -> Optional[Run]:
        runs = {r.run_id: r for r in self.iter_runs()}
        return runs.get(run_id)

    def iter_runs(self) -> Iterator[Run]:
        yield from load_runs_from_jsonl(self.path).values()


def load_runs_from_jsonl(path: str | Path) -> Dict[str, Run]:
    """Reconstruct runs from a (possibly partial) JSONL file."""
    path = Path(path)
    if not path.exists():
        return {}
    runs: Dict[str, Run] = {}
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = obj.pop('record_type', None)
            if rtype == 'step':
                run_id = obj.pop('run_id', None)
                if not run_id or run_id not in runs:
                    continue
                step = Step.model_validate(obj)
                runs[run_id].steps.append(step)
                runs[run_id].recompute_totals()
            elif rtype in (None, 'run_header', 'run_final'):
                # Full run snapshot (header has empty steps; final may include all).
                run = Run.model_validate(obj)
                existing = runs.get(run.run_id)
                if existing is None:
                    runs[run.run_id] = run
                elif rtype == 'run_final':
                    # Prefer final snapshot but keep any steps already flushed.
                    if not run.steps and existing.steps:
                        run.steps = existing.steps
                        run.recompute_totals()
                    runs[run.run_id] = run
                else:
                    # Header after we already have steps — keep steps.
                    if existing.steps and not run.steps:
                        run.steps = existing.steps
                    runs[run.run_id] = run
    return runs


def parse_partial_jsonl(path: str | Path) -> List[Run]:
    """Return all reconstructable runs from *path*, including incomplete ones."""
    return list(load_runs_from_jsonl(path).values())
