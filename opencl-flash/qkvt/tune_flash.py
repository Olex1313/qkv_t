"""Sweep flash attention kernel parameters to find optimal config."""

import numpy as np
import pyopencl as cl

from qkvt.ops import flash_v2_sdpa
from qkvt.bench_ops import make_profiling_queue

SEED = 42
WARMUP = 3
ITERS = 5


def run_config(queue, B, H, L, S, D, block_m, block_n, tpr):
    rng = np.random.default_rng(SEED)
    Q = rng.standard_normal((B, H, L, D)).astype(np.float32)
    K = rng.standard_normal((B, H, S, D)).astype(np.float32)
    V = rng.standard_normal((B, H, S, D)).astype(np.float32)

    wg_size = block_m * tpr
    d_slice = D // tpr
    lds_kb = (block_m * D + block_n * D) * 4 / 1024

    # Check constraints
    if D % tpr != 0 or d_slice % 4 != 0:
        return None
    if lds_kb > 32:
        return None
    if wg_size > 1024:
        return None

    try:
        _, result = flash_v2_sdpa(
            queue,
            Q,
            K,
            V,
            B,
            H,
            L,
            S,
            D,
            block_m=block_m,
            block_n=block_n,
            tpr=tpr,
            benchmark=True,
            warmup=WARMUP,
            iters=ITERS,
        )
        return result
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main():
    queue = make_profiling_queue()
    dev = queue.device.name.strip()
    print(f"Device: {dev}\n")

    shapes = [
        (1, 4, 1024, 1024, 64),
        (1, 4, 4096, 4096, 64),
        (1, 4, 2048, 2048, 128),
    ]

    block_ms = [16, 32, 64]
    block_ns = [16, 32, 64]
    tprs = [1, 2, 4, 8]

    for B, H, L, S, D in shapes:
        print(f"=== B={B} H={H} L={L} S={S} D={D} ===")
        results = []

        for bm in block_ms:
            for bn in block_ns:
                for tpr in tprs:
                    wg = bm * tpr
                    lds = (bm * D + bn * D) * 4 / 1024
                    if D % tpr != 0 or (D // tpr) % 4 != 0:
                        continue
                    if lds > 32 or wg > 1024:
                        continue

                    result = run_config(queue, B, H, L, S, D, bm, bn, tpr)
                    if result is None:
                        continue

                    tag = f"M={bm:2d} N={bn:2d} TPR={tpr} WG={wg:3d} LDS={lds:5.1f}KB"
                    results.append((result.median_ms, tag, result))

        results.sort(key=lambda x: x[0])
        print(f"{'Config':<40s}  {'median':>8s}  {'min':>8s}  {'TFLOPS':>8s}")
        print("-" * 72)
        for ms, tag, r in results:
            print(f"{tag:<40s}  {r.median_ms:8.3f}  {r.min_ms:8.3f}  {r.tflops:8.4f}")
        print()


if __name__ == "__main__":
    main()
