import pytest
import torch
import numpy as np
import pyopencl as cl
import math

KERNEL_FILES = ["kernels/flash_attn.cl", "kernels/flash_attn_v1.cl"]
TORCH_SEED = 42
BLOCK_SIZE_M = 32
WG_SIZE = 64


def execute_opencl_kernel(
    Q_np: np.ndarray,
    K_np: np.ndarray,
    V_np: np.ndarray,
    B: int,
    H: int,
    L: int,
    S: int,
    D: int,
    is_causal: int,
    ocl_kernel: str,
) -> np.ndarray:
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    scale = 1.0 / math.sqrt(D)

    mf = cl.mem_flags
    Q_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q_np)
    K_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K_np)
    V_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V_np)
    O_g = cl.Buffer(ctx, mf.WRITE_ONLY, size=Q_np.nbytes)

    num_q_blocks = (L + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    global_size = (num_q_blocks * WG_SIZE, B, H)
    local_size = (WG_SIZE, 1, 1)

    global_size_aligned = tuple(
        int(math.ceil(g / l) * l) for g, l in zip(global_size, local_size)
    )

    with open(ocl_kernel, "r") as f:
        kernel_code = f.read()

    prg = cl.Program(ctx, kernel_code).build(options=[f"-D D_HEAD={D}"])
    kernel = prg.flash_attention_v1_fwd

    kernel(
        queue,
        global_size_aligned,
        local_size,
        Q_g,
        K_g,
        V_g,
        O_g,
        np.int32(B),
        np.int32(H),
        np.int32(L),
        np.int32(S),
        np.float32(scale),
        np.int32(is_causal),
    ).wait()

    O_opencl_np = np.empty_like(Q_np)
    cl.enqueue_copy(queue, O_opencl_np, O_g).wait()

    return O_opencl_np


TEST_CASES = [
    (1, 4, 4, 16, 64, 0),
    (2, 2, 32, 32, 32, 0),
    (1, 4, 10, 100, 64, 0),
    (1, 4, 100, 10, 64, 0),
    (1, 4, 64, 64, 64, 1),
    (4, 8, 256, 256, 64, 0),
    (1, 16, 512, 512, 64, 1),
]


@pytest.mark.parametrize("ocl_kernel", KERNEL_FILES)
@pytest.mark.parametrize("B, H, L, S, D, is_causal", TEST_CASES)
def test_flash_attention_correctness(B, H, L, S, D, is_causal, ocl_kernel):
    torch.manual_seed(TORCH_SEED)

    Q_pt = torch.randn(B, H, L, D, dtype=torch.float32) * 0.1
    K_pt = torch.randn(B, H, S, D, dtype=torch.float32) * 0.1
    V_pt = torch.randn(B, H, S, D, dtype=torch.float32) * 0.1

    scale = 1.0 / math.sqrt(D)

    with torch.no_grad():
        O_truth_pt = torch.nn.functional.scaled_dot_product_attention(
            Q_pt, K_pt, V_pt, attn_mask=None, is_causal=bool(is_causal), scale=scale
        )

    O_truth_np = O_truth_pt.cpu().numpy().flatten()

    Q_np = Q_pt.numpy().flatten()
    K_np = K_pt.numpy().flatten()
    V_np = V_pt.numpy().flatten()

    O_opencl_np = execute_opencl_kernel(
        Q_np, K_np, V_np, B, H, L, S, D, is_causal, ocl_kernel
    )

    RTOL = 1e-4  # Rel err
    ATOL = 1e-5  # Abs err

    comparison = np.isclose(O_truth_np, O_opencl_np, rtol=RTOL, atol=ATOL)

    if not comparison.all():
        failure_indices = np.where(~comparison)[0]
        max_abs_diff = np.max(np.abs(O_truth_np - O_opencl_np))

        print(
            f"\n--- Test Failure Details (L={L}, S={S}, D={D}, causal={is_causal}) ---"
        )
        print(f"Max Absolute Error: {max_abs_diff:.6e} (Exceeds ATOL={ATOL:.1e})")
        print(f"Number of mismatches: {len(failure_indices)} / {O_truth_np.size}")

        for i in failure_indices[:5]:
            print(
                f"Index {i}: PyTorch={O_truth_np[i]:.6e}, OpenCL={O_opencl_np[i]:.6e}, Diff={(O_truth_np[i] - O_opencl_np[i]):.6e}"
            )

        assert (
            comparison.all()
        ), f"OpenCL output failed comparison with PyTorch (Max Diff: {max_abs_diff:.6e})"
