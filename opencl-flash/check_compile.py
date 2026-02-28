import pyopencl as cl


KERNEL_FILE = (
    "/Users/aleksejlimonov/Documents/github/qkv_t/opencl-flash/kernels/flash_attn.cl"
)
TORCH_SEED = 42
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
WG_SIZE = 32
D = 64

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


with open(KERNEL_FILE, "r") as f:
    kernel_code = f.read()

prg = cl.Program(ctx, kernel_code).build(
    options=[
        f"-D D_HEAD={D}",
        f"-D BLOCK_SIZE_M={BLOCK_SIZE_M}",
        f"-D BLOCK_SIZE_N={BLOCK_SIZE_N}",
    ]
)
kernel = prg.flash_attention_v1_fwd

print("COMPILED")
