import os
from typing import Optional

import numpy as np
import pyopencl as cl
import pytest

from .ops import flash_v1_sdpa, flash_v2_sdpa, native_sdpa
from .profiling import ProfilingResult

RANDOM_SEED = 42

os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "1")

# Measured with clpeak (https://github.com/krrishnarraj/clpeak)
# (peak_flops, peak_bw_bytes)
_DEVICE_PEAKS: dict[str, tuple[int, int]] = {
    "Apple M4 Pro": (int(3.179e12), int(255e9)),
}


def device_peak(device: cl.Device) -> tuple[Optional[int], Optional[int]]:
    return _DEVICE_PEAKS.get(device.name.strip(), (None, None))


def sol_pct(achieved: float, peak: Optional[int], scale: float) -> str:
    if peak is None:
        return "n/a"
    return f"{achieved / (peak / scale) * 100:.1f}%"


def make_profiling_queue() -> cl.CommandQueue:
    platforms = cl.get_platforms()
    gpu_devices = [d for p in platforms for d in p.get_devices(cl.device_type.GPU)]
    if not gpu_devices:
        raise RuntimeError("No GPU device found for OpenCL")
    ctx = cl.Context(devices=[gpu_devices[0]])
    return cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


@pytest.fixture(scope="module")
def queue():
    return make_profiling_queue()


# (B, H, L, S, D) — shapes relevant to LightGlue and typical attention benchmarks
SDPA_BENCH_CASES = [
    (2, 4, 512, 512, 64),  # short prefill
    (2, 4, 1024, 1024, 64),
    (2, 4, 2048, 2048, 64),
    (2, 4, 3840, 3840, 64),  # LightGlue full sequence
]


def flash_v2_sdpa_vload(*args, **kwargs):
    return flash_v2_sdpa(*args, **kwargs, use_vector_load=True)


BENCH_IMPLS = [native_sdpa, flash_v1_sdpa, flash_v2_sdpa, flash_v2_sdpa_vload]


def _print_sol(result: ProfilingResult, device: cl.Device) -> None:
    peak_flops, peak_bw = device_peak(device)
    print(
        f"  {result}\n"
        f"  SOL compute: {sol_pct(result.tflops, peak_flops, 1e12)}  "
        f"SOL bandwidth: {sol_pct(result.bandwidth_gbs, peak_bw, 1e9)}"
    )


@pytest.mark.parametrize("impl", BENCH_IMPLS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("B, H, L, S, D", SDPA_BENCH_CASES)
def test_sdpa_sol(queue, B, H, L, S, D, impl):
    rng = np.random.default_rng(RANDOM_SEED)
    Q = rng.standard_normal((B, H, L, D)).astype(np.float32)
    K = rng.standard_normal((B, H, S, D)).astype(np.float32)
    V = rng.standard_normal((B, H, S, D)).astype(np.float32)

    _, result = impl(queue, Q, K, V, B, H, L, S, D, benchmark=True, warmup=3, iters=10)

    print(
        f"\n{impl.__name__}  B={B} H={H} L={L} S={S} D={D}  device: {queue.device.name.strip()}"
    )
    _print_sol(result, queue.device)
