import math
from pathlib import Path

import numpy as np
import pyopencl as cl

from qkvt.profiling import sol_tracked

_KERNELS_DIR = Path(__file__).parent / "kernels"
_BMM_TS = 16


def _build(ctx: cl.Context, filename: str, options: list[str] = []) -> cl.Program:
    src = (_KERNELS_DIR / filename).read_text()
    return cl.Program(ctx, src).build(options=options)


# input: float32[B, M, N], other: float32[B, N, K] -> float32[B, M, K] * alpha
def bmm(
    queue: cl.CommandQueue,
    input: np.ndarray,
    other: np.ndarray,
    alpha=1.0,
    _prof_events: list = None,
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

    event = prg.bmm(
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
    )
    event.wait()
    if _prof_events is not None:
        _prof_events.append(event)

    cl.enqueue_copy(queue, output, C_g).wait()
    return output


# input - float32[..., S] — softmax applied over last dim
def softmax(
    queue: cl.CommandQueue, input: np.ndarray, _prof_events: list = None
) -> np.ndarray:
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
    event = prg.scale_softmax(
        queue, (N_rows * WG_SIZE,), (WG_SIZE,), I_g, O_g, np.int32(S)
    )
    event.wait()
    if _prof_events is not None:
        _prof_events.append(event)

    cl.enqueue_copy(queue, output, O_g).wait()
    return output.reshape(input.shape)


@sol_tracked(
    flop_fn=lambda queue, Q, K, V, B, H, L, S, D, **_: 4 * B * H * L * S * D,
    # materializes QK^T: Q+K read, QK^T written+read, softmax written+read, V read, O written
    bytes_fn=lambda queue, Q, K, V, B, H, L, S, D, **_: 4
    * (2 * B * H * L * D + 2 * B * H * S * D + 4 * B * H * L * S),
)
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
    _prof_events: list = None,
):
    # [B*H, L, D] for bmm
    Q_b = Q.reshape(B * H, L, D)
    K_b = K.reshape(B * H, S, D)
    V_b = V.reshape(B * H, S, D)

    scores = bmm(queue, Q_b, K_b.swapaxes(-1, -2), 1.0 / math.sqrt(D))
    applied_softmax = softmax(queue, scores)
    return bmm(queue, applied_softmax, V_b).reshape(B, H, L, D)


_FLASH_BLOCK_SIZE_M = 64
_FLASH_BLOCK_SIZE_N = 16
_FLASH_THREADS_PER_ROW = 2


# Q: float32[B, H, L, D], K/V: float32[B, H, S, D] -> float32[B, H, L, D]
@sol_tracked(
    flop_fn=lambda queue, Q, K, V, B, H, L, S, D, **_: 4 * B * H * L * S * D,
    bytes_fn=lambda queue, Q, K, V, B, H, L, S, D, **_: 4
    * (B * H * L * D + 2 * B * H * S * D + B * H * L * D),
)
def flash_v2_sdpa(
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
    block_m: int = _FLASH_BLOCK_SIZE_M,
    block_n: int = _FLASH_BLOCK_SIZE_N,
    tpr: int = _FLASH_THREADS_PER_ROW,
    _prof_events: list = None,
) -> np.ndarray:
    # Transpose [B, H, L, D] -> [B, L, H, D] (MNN layout)
    Q_mnn = np.ascontiguousarray(Q.transpose(0, 2, 1, 3), dtype=np.float32)
    K_mnn = np.ascontiguousarray(K.transpose(0, 2, 1, 3), dtype=np.float32)
    V_mnn = np.ascontiguousarray(V.transpose(0, 2, 1, 3), dtype=np.float32)
    output = np.empty_like(Q_mnn)

    scale = np.float32(1.0 / math.sqrt(D))
    wg_size = block_m * tpr

    ctx = queue.context
    mf = cl.mem_flags
    Q_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q_mnn)
    K_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K_mnn)
    V_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V_mnn)
    O_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

    num_q_blocks = (L + block_m - 1) // block_m
    global_size = (num_q_blocks * wg_size, B, H)
    local_size = (wg_size, 1, 1)

    options = [
        f"-D D_HEAD={D}",
        f"-D BLOCK_SIZE_M={block_m}",
        f"-D BLOCK_SIZE_N={block_n}",
        f"-D THREADS_PER_ROW={tpr}",
    ]
    prg = _build(ctx, "flash_attn_v2_mnn.cl", options)

    event = prg.flash_attention_v2_mnn_fwd(
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
    )
    event.wait()
    if _prof_events is not None:
        _prof_events.append(event)

    cl.enqueue_copy(queue, output, O_g).wait()
    # Transpose back [B, L, H, D] -> [B, H, L, D]
    return output.transpose(0, 2, 1, 3)
