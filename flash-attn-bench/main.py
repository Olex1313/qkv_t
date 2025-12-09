import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import time

N, S, H, E = 4, 256, 12, 64
query = torch.rand(N, H, S, E, dtype=torch.float16, device='cuda')
key = torch.rand(N, H, S, E, dtype=torch.float16, device='cuda')
value = torch.rand(N, H, S, E, dtype=torch.float16, device='cuda')

def run_and_time(backend, n_iter=20):
    with sdpa_kernel(backends=[backend]):
        for _ in range(5):
            F.scaled_dot_product_attention(query, key, value)
    torch.cuda.synchronize()

    times = []
    with sdpa_kernel(backends=[backend]):
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = time.perf_counter()
            F.scaled_dot_product_attention(query, key, value)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    return min(times)*1000

backends = {
    'FLASH_ATTENTION': SDPBackend.FLASH_ATTENTION,
    'EFFICIENT_ATTENTION': SDPBackend.EFFICIENT_ATTENTION,
    'MATH': SDPBackend.MATH
}

print(f"batch_size: {N}, num_heads: {H}, seq_len: {S}, embed_dim: {E}")

for name, backend in backends.items():
    try:
        ms = run_and_time(backend)
        print(f"{name:>20}: {ms:.3f} ms")
    except Exception as e:
        print(f"{name:>20}: Not supported ({e})")
