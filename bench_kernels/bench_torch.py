"""
PyTorch attention benchmark — same log format as bench_kernels.
Requires CUDA or MPS. Tries flash → efficient → math in order,
or force a specific backend with --backend=flash|efficient|math|cudnn.

Usage: python bench_torch.py [warmup] [iters] [--backend=flash]
Combine with bench_kernels output: python plot_bench.py bench.4060ti.log torch.log
"""

import sys
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[torch] {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[torch] Apple MPS")
else:
    print("No CUDA or MPS device — exiting."); sys.exit(1)

print(f"[torch] torch={torch.__version__}  cuda={torch.version.cuda}")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.lstrip("--").split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a and a.startswith("--")}

WARMUP = int(args[0]) if len(args) > 0 else 3
ITERS  = int(args[1]) if len(args) > 1 else 10

_BACKEND_MAP = {
    "flash":     ("torch-flash",     [SDPBackend.FLASH_ATTENTION]),
    "efficient": ("torch-efficient", [SDPBackend.EFFICIENT_ATTENTION]),
    "math":      ("torch-math",      [SDPBackend.MATH]),
    "cudnn":     ("torch-cudnn",     [SDPBackend.CUDNN_ATTENTION]),
}

forced = flags.get("backend")
if forced:
    if forced not in _BACKEND_MAP:
        print(f"Unknown backend '{forced}'. Choose: {list(_BACKEND_MAP)}"); sys.exit(1)
    BACKENDS = [_BACKEND_MAP[forced]]
    print(f"[torch] forcing backend: {forced}")
else:
    BACKENDS = [_BACKEND_MAP["flash"], _BACKEND_MAP["efficient"], _BACKEND_MAP["math"]]

def sync():
    if device.type == "cuda":   torch.cuda.synchronize()
    elif device.type == "mps":  torch.mps.synchronize()

cases = [
    (B, H, L, D)
    for L in [1024, 2048, 4096, 8192, 16384, 32768]
    for D in [64, 128]
    for H in [4, 8, 16]
    for B in [1, 4]
]

for B, H, L, D in cases:
    q = torch.randn(B, H, L, D, dtype=torch.float32, device=device)
    k = torch.randn(B, H, L, D, dtype=torch.float32, device=device)
    v = torch.randn(B, H, L, D, dtype=torch.float32, device=device)
    scale = D ** -0.5
    for tag, backends in BACKENDS:
        try:
            with sdpa_kernel(backends):
                for _ in range(WARMUP):
                    F.scaled_dot_product_attention(q, k, v, scale=scale)
                sync()
                times = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    F.scaled_dot_product_attention(q, k, v, scale=scale)
                    sync()
                    times.append((time.perf_counter() - t0) * 1e6)
            mn, avg, mx = min(times), sum(times)/len(times), max(times)
            gflops = 4.0 * B * H * L * L * D / (mn / 1e6) / 1e9
            print(f"{tag:<20}  B={B} H={H:2d} L={L:5d} D={D:3d}  "
                  f"min={mn:8.0f}  avg={avg:8.0f}  max={mx:8.0f} us  GFLOPS={gflops:.2f}")
            break
        except Exception:
            continue
