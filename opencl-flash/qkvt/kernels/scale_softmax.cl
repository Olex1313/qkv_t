// Row-wise softmax over last dimension
// input/output: [N_rows, S] row-major
#define WG_SIZE 64

__kernel void scale_softmax(
    const __global float* input,
    __global float* output,
    const int S
) {
    const int row = get_group_id(0);
    const int lid = get_local_id(0);

    __local float lmem[WG_SIZE];

    // 1. Find row max
    float lmax = -INFINITY;
    for (int i = lid; i < S; i += WG_SIZE)
        lmax = fmax(lmax, input[row * S + i]);
    lmem[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) lmem[lid] = fmax(lmem[lid], lmem[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = lmem[0];

    // 2. Compute exp(x - max) and accumulate sum
    float lsum = 0.0f;
    for (int i = lid; i < S; i += WG_SIZE) {
        float v = exp(input[row * S + i] - row_max);
        output[row * S + i] = v;
        lsum += v;
    }
    lmem[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) lmem[lid] += lmem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_sum = lmem[0];

    // 3. Normalize
    for (int i = lid; i < S; i += WG_SIZE)
        output[row * S + i] /= row_sum;
}
