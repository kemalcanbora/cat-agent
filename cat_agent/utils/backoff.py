"""Shared exponential-backoff delay math (no sleep — callers decide sync vs async)."""

from __future__ import annotations

import random


def compute_backoff_delay(
    current_delay: float,
    *,
    exponential_base: float = 2.0,
    max_delay: float = 300.0,
    jitter: bool = True,
) -> float:
    """Return the next backoff delay given the previous delay.

    Applies exponential growth capped at *max_delay*, optionally multiplied by
    a uniform jitter factor in ``[1.0, 2.0)``.
    """
    delay = min(current_delay * exponential_base, max_delay)
    if jitter:
        delay *= 1.0 + random.random()
    return delay
