import math
from pathlib import Path

import numpy as np
import pyopencl as cl

_KERNELS_DIR = Path(__file__).parent / "kernels"
_BMM_TS = 16


def _build(ctx: cl.Context, filename: str, options: list[str] = []) -> cl.Program:
    src = (_KERNELS_DIR / filename).read_text()
    return cl.Program(ctx, src).build(options=options)


# input: float32[B, M, N], other: float32[B, N, K] -> float32[B, M, K] * alpha
def bmm(
    queue: cl.CommandQueue, input: np.ndarray, other: np.ndarray, alpha=1.0
) -> np.ndarray:
    assert input.ndim == 3 and other.ndim == 3
    B, M, N = input.shape
    K = other.shape[2]
    assert other.shape == (B, N, K)

    input = np.ascontiguousarray(input, dtype=np.float32)
    other = np.ascontiguousarray(other, dtype=np.float32)
    output = np.empty((B, M, K), dtype=np.float32)

    ctx = queue.context
    mf = cl.mem_flags
    A_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input)
    B_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other)
    C_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

    prg = _build(ctx, "bmm.cl")

    TILE_SIZE = _BMM_TS
    global_size = (
        int(math.ceil(M / TILE_SIZE) * TILE_SIZE),
        int(math.ceil(K / TILE_SIZE) * TILE_SIZE),
        B,
    )
    local_size = (TILE_SIZE, TILE_SIZE, 1)

    prg.bmm(
        queue,
        global_size,
        local_size,
        np.int32(M),
        np.int32(N),
        np.int32(K),
        A_g,
        B_g,
        C_g,
        np.float32(alpha),
    ).wait()

    cl.enqueue_copy(queue, output, C_g).wait()
    return output


# input - float32[..., S] — softmax applied over last dim
def softmax(queue: cl.CommandQueue, input: np.ndarray) -> np.ndarray:
    input = np.ascontiguousarray(input, dtype=np.float32)
    S = input.shape[-1]
    N_rows = input.size // S
    flat = input.reshape(N_rows, S)
    output = np.empty_like(flat)

    ctx = queue.context
    mf = cl.mem_flags
    I_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
    O_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

    prg = _build(ctx, "scale_softmax.cl")

    WG_SIZE = 64
    prg.scale_softmax(
        queue, (N_rows * WG_SIZE,), (WG_SIZE,), I_g, O_g, np.int32(S)
    ).wait()

    cl.enqueue_copy(queue, output, O_g).wait()
    return output.reshape(input.shape)


def native_sdpa(
    queue: cl.CommandQueue,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    B: int,
    H: int,
    L: int,
    S: int,
    D: int,
    is_causal: bool = False,
):

    # [B*H, L, D] for bmm
    Q_b = Q.reshape(B * H, L, D)
    K_b = K.reshape(B * H, S, D)
    V_b = V.reshape(B * H, S, D)

    scores = bmm(queue, Q_b, K_b.swapaxes(-1, -2), 1.0 / math.sqrt(D))
    applied_softmax = softmax(queue, scores)
    return bmm(queue, applied_softmax, V_b).reshape(B, H, L, D)


_FLASH_BLOCK_SIZE_M = 32
_FLASH_WG_SIZE = 32


# Q: float32[B, H, L, D], K/V: float32[B, H, S, D] -> float32[B, H, L, D]
def flash_v1_sdpa(
    queue: cl.CommandQueue,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    B: int,
    H: int,
    L: int,
    S: int,
    D: int,
    is_causal: bool = False,
) -> np.ndarray:
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)
    output = np.empty_like(Q)

    scale = np.float32(1.0 / math.sqrt(D))

    ctx = queue.context
    mf = cl.mem_flags
    Q_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
    K_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
    V_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
    O_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

    prg = _build(ctx, "flash_attn_v1.cl", options=[f"-D D_HEAD={D}"])

    num_q_blocks = (L + _FLASH_BLOCK_SIZE_M - 1) // _FLASH_BLOCK_SIZE_M
    global_size = (num_q_blocks * _FLASH_WG_SIZE, B, H)
    local_size = (_FLASH_WG_SIZE, 1, 1)

    prg.flash_attention_v1_fwd(
        queue,
        global_size,
        local_size,
        Q_g,
        K_g,
        V_g,
        O_g,
        np.int32(B),
        np.int32(H),
        np.int32(L),
        np.int32(S),
        scale,
        np.int32(is_causal),
    ).wait()

    cl.enqueue_copy(queue, output, O_g).wait()
    return output
