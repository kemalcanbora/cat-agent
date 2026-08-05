"""Benchmark WASM sandbox round-trip latency for tool synthesis.

Run (requires ``wasmtime`` and the bundled WASM runtime)::

    python3.10 benchmarks/benchmark_sandbox_roundtrip.py --iterations 100

Decision thresholds (per-call warm p50):
    < 50ms   — fresh sandbox per call is fine
    50–200ms — cache WasmPythonRuntime on the executor
    > 200ms  — escalate; need a long-lived sandbox pool
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cat_agent.synthesis.executors.wasm import WasmExecutor


_IMPL = '''\
def add_one(x: int) -> int:
    """Return x + 1.

    Args:
        x: Integer input.
    """
    return x + 1
'''


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--fuel', type=int, default=None)
    args = parser.parse_args()

    executor = WasmExecutor(fuel=args.fuel)

    # Cold start: first call after constructing a fresh executor/runtime.
    cold_executor = WasmExecutor(fuel=args.fuel)
    t0 = time.perf_counter()
    cold = cold_executor.run(_IMPL, {'x': 1}, function_name='add_one')
    cold_ms = (time.perf_counter() - t0) * 1000
    if not cold.ok:
        raise SystemExit(f'Cold run failed: {cold.error}')

    # Warm the shared runtime.
    warm_probe = executor.run(_IMPL, {'x': 0}, function_name='add_one')
    if not warm_probe.ok:
        raise SystemExit(f'Warm probe failed: {warm_probe.error}')

    samples: list[float] = []
    for i in range(args.iterations):
        started = time.perf_counter()
        result = executor.run(_IMPL, {'x': i}, function_name='add_one')
        elapsed = (time.perf_counter() - started) * 1000
        if not result.ok:
            raise SystemExit(f'Iteration {i} failed: {result.error}')
        if result.returned != i + 1:
            raise SystemExit(f'Iteration {i} wrong result: {result.returned!r}')
        samples.append(elapsed)

    p50 = statistics.median(samples)
    p95 = _percentile(samples, 95)
    mean = statistics.mean(samples)

    print('WASM sandbox round-trip (add_one)')
    print(f'  iterations : {args.iterations}')
    print(f'  cold_ms    : {cold_ms:.2f}')
    print(f'  warm_mean  : {mean:.2f}')
    print(f'  warm_p50   : {p50:.2f}')
    print(f'  warm_p95   : {p95:.2f}')
    if p50 < 50:
        verdict = 'OK: fresh sandbox per call is fine (<50ms)'
    elif p50 <= 200:
        verdict = 'CACHE: keep WasmPythonRuntime on the executor (50–200ms)'
    else:
        verdict = 'ESCALATE: per-call sandbox too slow (>200ms); need a process pool'
    print(f'  verdict    : {verdict}')


if __name__ == '__main__':
    main()
