import os
from typing import Optional

import numpy as np
import pyopencl as cl
import pytest

from functools import partial

from .ops import flash_v2_sdpa, native_sdpa, device_has_subgroups
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


SDPA_BENCH_CASES = [
    (1, 8, 512, 512, 64),
    (1, 8, 1024, 1024, 64),
    (1, 8, 2048, 2048, 64),
    (1, 8, 4096, 4096, 64),
    (1, 8, 8192, 8192, 64),
    (1, 8, 16384, 16384, 64),
]


flash_v2_subgroup_sdpa = partial(flash_v2_sdpa, use_subgroups=True)
flash_v2_subgroup_sdpa.__name__ = "flash_v2_subgroup_sdpa"

BENCH_IMPLS = [native_sdpa, flash_v2_sdpa]


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


def _flops(B, H, L, S, D):
    """Total FLOPs for SDPA: 2 matmuls of 4*B*H*L*S*D."""
    return 4 * B * H * L * S * D


def run_bench_sweep(
    warmup: int = 3,
    iters: int = 15,
    cases: list[tuple[int, int, int, int, int]] | None = None,
    impls: list | None = None,
) -> dict[str, dict[int, ProfilingResult]]:
    """Run all combos using profiling events, return {impl_name: {seq_len: ProfilingResult}}."""
    queue = make_profiling_queue()
    if cases is None:
        cases = SDPA_BENCH_CASES
    if impls is None:
        impls = list(BENCH_IMPLS)
        if device_has_subgroups(queue.device):
            impls.append(flash_v2_subgroup_sdpa)

    results: dict[str, dict[int, ProfilingResult]] = {}
    for impl in impls:
        name = impl.__name__
        results[name] = {}
        for B, H, L, S, D in cases:
            rng = np.random.default_rng(RANDOM_SEED)
            Q = rng.standard_normal((B, H, L, D)).astype(np.float32)
            K = rng.standard_normal((B, H, S, D)).astype(np.float32)
            V = rng.standard_normal((B, H, S, D)).astype(np.float32)

            _, result = impl(
                queue,
                Q,
                K,
                V,
                B,
                H,
                L,
                S,
                D,
                benchmark=True,
                warmup=warmup,
                iters=iters,
            )
            results[name][S] = result
            print(f"{name}  S={S}  {result}")

    return results


def plot_bench(
    results: dict[str, dict[int, ProfilingResult]],
    svg_path: str = "bench_sdpa.svg",
    device_name: str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_lens = sorted({s for r in results.values() for s in r})
    impl_names = list(results.keys())
    n_impls = len(impl_names)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 5))

    positions_base = np.arange(len(seq_lens))
    width = 0.7 / n_impls

    for i, name in enumerate(impl_names):
        gflops = [
            results[name][s].tflops * 10e3 if s in results[name] else 0
            for s in seq_lens
        ]
        pos = positions_base + (i - (n_impls - 1) / 2) * width
        color = colors[i % len(colors)]

        ax.bar(pos, gflops, width=width * 0.85, color=color, alpha=0.75, label=name)

    ax.set_xticks(positions_base)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Sequence length (L = S)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(
        f"SDPA throughput: flash_v2 vs native (D=64)"
        + (f" — {device_name}" if device_name else "")
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"Saved {svg_path}")
    plt.close(fig)


if __name__ == "__main__":
    queue = make_profiling_queue()
    device_name = queue.device.name.strip()
    del queue

    results = run_bench_sweep()
    plot_bench(results, device_name=device_name)
