import math
import os
import time
import functools

import numpy as np
import pyopencl as cl
import pytest
import torch


def _fmt(v):
    if callable(v) and hasattr(v, "__name__"):
        return v.__name__
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        all_vals = list(args) + list(kwargs.values())
        parts = [f for v in all_vals if (f := _fmt(v)) is not None]
        label = f"{fn.__name__}[{', '.join(parts)}]" if parts else fn.__name__
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n  {label}: {elapsed * 1000:.2f} ms")
        return result

    return wrapper


from .ops import bmm, softmax, native_sdpa, flash_v1_sdpa, flash_v2_sdpa

RANDOM_SEED = 42

os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "1")


def make_queue() -> cl.CommandQueue:
    platforms = cl.get_platforms()
    gpu_devices = [d for p in platforms for d in p.get_devices(cl.device_type.GPU)]
    if not gpu_devices:
        raise RuntimeError("No GPU device found for OpenCL")
    ctx = cl.Context(devices=[gpu_devices[0]])
    return cl.CommandQueue(ctx)


@pytest.fixture(scope="module")
def queue():
    return make_queue()


BMM_CASES = [
    (1, 4, 4, 4),
    (2, 8, 16, 8),
    (4, 32, 64, 32),
    (3, 17, 33, 7),  # non-tile-aligned
    (1, 1, 1, 1),
    (8, 64, 64, 64),
    (2, 128, 256, 64),
]

ALPHA_VALUES = [1.0, 0.5, 2.0, 1.0 / math.sqrt(64)]


@pytest.mark.parametrize("B, M, N, K", BMM_CASES)
def test_bmm_correctness(queue, B, M, N, K):
    rng = np.random.default_rng(RANDOM_SEED)
    A = rng.standard_normal((B, M, N)).astype(np.float32)
    B_ = rng.standard_normal((B, N, K)).astype(np.float32)

    ref = torch.bmm(torch.from_numpy(A), torch.from_numpy(B_)).numpy()
    out = bmm(queue, A, B_)

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("alpha", ALPHA_VALUES)
@pytest.mark.parametrize("B, M, N, K", BMM_CASES)
def test_bmm_alpha(queue, B, M, N, K, alpha):
    rng = np.random.default_rng(RANDOM_SEED)
    A = rng.standard_normal((B, M, N)).astype(np.float32)
    B_ = rng.standard_normal((B, N, K)).astype(np.float32)

    ref = (torch.bmm(torch.from_numpy(A), torch.from_numpy(B_)) * alpha).numpy()
    out = bmm(queue, A, B_, alpha=alpha)

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


SOFTMAX_CASES = [
    (4, 4),
    (8, 64),
    (16, 100),
    (1, 1),
    (1, 200),
    (32, 512),
    (3, 4, 10),
    (2, 8, 256, 64),
]


@pytest.mark.parametrize("shape", SOFTMAX_CASES)
def test_softmax_correctness(queue, shape):
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.standard_normal(shape).astype(np.float32)

    ref = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
    out = softmax(queue, x)

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


SDPA_CASES = [
    # edge cases
    (1, 4, 4, 16, 64, False),
    (1, 4, 10, 100, 64, False),
    # bert/lightglue self-attention
    (4, 8, 256, 256, 64, False),
    (8, 12, 512, 512, 64, False),
    # large batch
    (24, 8, 256, 256, 64, False),
    # cross
    (12, 4, 256, 512, 64, False),
    (12, 4, 256, 2048, 64, False),
    (12, 4, 256, 8192, 64, False),
    # FIXME, disabled is_casual
    # (1, 16, 512, 512, 64, True),
    # (1, 4, 64, 64, 64, True),
]

SPDA_IMPL_CASES = [native_sdpa, flash_v1_sdpa, flash_v2_sdpa]


@timed
@pytest.mark.parametrize("flash_impl", SPDA_IMPL_CASES)
@pytest.mark.parametrize("B, H, L, S, D, is_causal", SDPA_CASES)
def test_native_sdpa_correctness(queue, B, H, L, S, D, is_causal, flash_impl):
    rng = np.random.default_rng(RANDOM_SEED)
    Q = rng.standard_normal((B, H, L, D)).astype(np.float32) * 0.1
    K = rng.standard_normal((B, H, S, D)).astype(np.float32) * 0.1
    V = rng.standard_normal((B, H, S, D)).astype(np.float32) * 0.1

    scale = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(Q),
        torch.from_numpy(K),
        torch.from_numpy(V),
        attn_mask=None,
        is_causal=is_causal,
        scale=scale,
    ).numpy()

    out = flash_impl(queue, Q, K, V, B, H, L, S, D, is_causal=is_causal)

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


@timed
@pytest.mark.parametrize("use_vload", [True, False])
@pytest.mark.parametrize("B, H, L, S, D, is_causal", SDPA_CASES)
def test_mnn_layout_w_fav2(queue, use_vload, B, H, L, S, D, is_causal):
    rng = np.random.default_rng(RANDOM_SEED)
    Q = rng.standard_normal((B, H, L, D)).astype(np.float32) * 0.1
    K = rng.standard_normal((B, H, S, D)).astype(np.float32) * 0.1
    V = rng.standard_normal((B, H, S, D)).astype(np.float32) * 0.1

    scale = 1.0 / math.sqrt(D)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(Q),
        torch.from_numpy(K),
        torch.from_numpy(V),
        attn_mask=None,
        is_causal=is_causal,
        scale=scale,
    ).numpy()

    Q_mnn = Q.transpose(0, 2, 1, 3)
    K_mnn = K.transpose(0, 2, 1, 3)
    V_mnn = V.transpose(0, 2, 1, 3)

    out = flash_v2_sdpa(
        queue,
        Q_mnn,
        K_mnn,
        V_mnn,
        B,
        H,
        L,
        S,
        D,
        is_causal=is_causal,
        mnn_layout=True,
        use_vector_load=use_vload,
    ).transpose(0, 2, 1, 3)

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
