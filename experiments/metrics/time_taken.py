"""
Timing metrics: training wall-clock and per-example inference latency.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - start
    finally:
        pass


def mean_inference_ms(per_example_seconds: list[float]) -> float:
    """Mean inference latency per example, in milliseconds."""
    if not per_example_seconds:
        return 0.0
    return float(sum(per_example_seconds) / len(per_example_seconds) * 1000.0)
