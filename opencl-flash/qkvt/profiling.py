import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable

import numpy as np
import pyopencl as cl


@dataclass(frozen=True)
class ProfilingResult:
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    flops: int
    bandwidth_bytes: int

    @property
    def tflops(self) -> float:
        return self.flops / (self.median_ms / 1000) / 1e12

    @property
    def bandwidth_gbs(self) -> float:
        return self.bandwidth_bytes / (self.median_ms / 1000) / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bandwidth_bytes

    def __repr__(self) -> str:
        return (
            f"median: {self.median_ms:.3f} ms  "
            f"mean: {self.mean_ms:.3f} +- {self.std_ms:.3f} ms  "
            f"[min={self.min_ms:.3f}, max={self.max_ms:.3f}]  "
            f"TFLOPS: {self.tflops:.4f}  "
            f"bandwidth: {self.bandwidth_gbs:.1f} GB/s  "
            f"intensity: {self.arithmetic_intensity:.1f} FLOP/byte"
        )


def sol_tracked(flop_fn: Callable, bytes_fn: Callable):

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, benchmark=False, warmup=3, iters=10, **kwargs):
            if not benchmark:
                return fn(*args, **kwargs)

            orig_queue: cl.CommandQueue = args[0]
            assert (
                orig_queue.properties & cl.command_queue_properties.PROFILING_ENABLE
            ) != 0, "Queue must be created with PROFILING_ENABLE"

            for _ in range(warmup):
                fn(*args, **kwargs)

            elapsed_ns_list = []
            result = None
            for _ in range(iters):
                events = []
                t0 = time.perf_counter()
                result = fn(*args, _prof_events=events, **kwargs)
                t1 = time.perf_counter()
                if events:
                    elapsed_ns_list.append(
                        sum(e.profile.end - e.profile.start for e in events)
                    )
                else:
                    elapsed_ns_list.append((t1 - t0) * 1e9)

            times_ms = np.array(elapsed_ns_list) / 1e6
            return result, ProfilingResult(
                median_ms=float(np.median(times_ms)),
                mean_ms=float(np.mean(times_ms)),
                std_ms=float(np.std(times_ms)),
                min_ms=float(np.min(times_ms)),
                max_ms=float(np.max(times_ms)),
                flops=flop_fn(*args, **kwargs),
                bandwidth_bytes=bytes_fn(*args, **kwargs),
            )

        return wrapper

    return decorator
