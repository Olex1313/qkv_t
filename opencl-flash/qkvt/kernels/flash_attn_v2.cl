#ifndef D_HEAD
#define D_HEAD 128
#endif
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define FLT_MAX 3.402823466e+38F

void load_global_tile(
    __global const float *src,
    __local float *dst,
    const int global_offset,
    const int stride_row,
    const int num_rows,
    const int max_rows_in_tensor,
    const int local_id,
    const int wg_size
) {
    const int total_elements = num_rows * stride_row;
    for (int i = local_id; i < total_elements; i += wg_size) {
        int r = i / stride_row;
        int c = i % stride_row;
        if (r < max_rows_in_tensor) {
            dst[i] = src[global_offset + r * stride_row + c];
        } else {
            dst[i] = 0.0f;
        }
    }
}

__kernel void flash_attention_v2_fwd(
    __global const float *Q,
    __global const float *K,
    __global const float *V,
    __global float *O,
    const int B,
    const int H,
    const int L,
    const int S,
    const float scale,
    const int is_causal
) {
    const int batch_idx = get_global_id(1);
    const int head_idx  = get_global_id(2);
    const int q_block_idx = get_group_id(0);

    if (batch_idx >= B || head_idx >= H) return;

    const int batch_head_offset_Q = (batch_idx * H + head_idx) * L * D_HEAD;
    const int batch_head_offset_K = (batch_idx * H + head_idx) * S * D_HEAD;

    const int tid     = get_local_id(0);
    const int wg_size = get_local_size(0);

    __local float Q_tile[BLOCK_SIZE_M * D_HEAD];
    __local float K_tile[BLOCK_SIZE_N * D_HEAD];
    __local float V_tile[BLOCK_SIZE_N * D_HEAD];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_acc[D_HEAD];
    for (int d = 0; d < D_HEAD; ++d) O_acc[d] = 0.0f;

    const int q_start_row = q_block_idx * BLOCK_SIZE_M;
    const int num_q_rows  = (q_start_row + BLOCK_SIZE_M > L) ? (L - q_start_row) : BLOCK_SIZE_M;

    load_global_tile(
        Q, Q_tile,
        batch_head_offset_Q + q_start_row * D_HEAD,
        D_HEAD, BLOCK_SIZE_M, num_q_rows,
        tid, wg_size
    );
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_kv_blocks = (S + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    for (int k_block = 0; k_block < num_kv_blocks; ++k_block) {
        const int k_start_row = k_block * BLOCK_SIZE_N;
        const int num_k_rows  = (k_start_row + BLOCK_SIZE_N > S) ? (S - k_start_row) : BLOCK_SIZE_N;

        // Causal: skip KV blocks entirely beyond the query block diagonal
        if (is_causal && k_start_row > (q_start_row + BLOCK_SIZE_M - 1)) break;

        load_global_tile(K, K_tile, batch_head_offset_K + k_start_row * D_HEAD, D_HEAD, BLOCK_SIZE_N, num_k_rows, tid, wg_size);
        load_global_tile(V, V_tile, batch_head_offset_K + k_start_row * D_HEAD, D_HEAD, BLOCK_SIZE_N, num_k_rows, tid, wg_size);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
            // Step 1: compute all scores in this KV block and find block max
            float scores[BLOCK_SIZE_N];
            float m_block = -FLT_MAX;
            for (int j = 0; j < num_k_rows; ++j) {
                // Causal: mask future positions within diagonal block
                if (is_causal && (k_start_row + j) > (q_start_row + tid)) {
                    scores[j] = -FLT_MAX;
                    continue;
                }
                float s = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) {
                    s += Q_tile[tid * D_HEAD + d] * K_tile[j * D_HEAD + d];
                }
                scores[j] = s * scale;
                m_block = fmax(m_block, scores[j]);
            }

            // Step 2: single rescale update for the whole block
            float m_new = fmax(m_i, m_block);
            float alpha = exp(m_i - m_new);  // rescale factor for previous O_acc/l_i

            // Rescale old accumulator first, then add new block contributions
            for (int d = 0; d < D_HEAD; ++d) O_acc[d] *= alpha;

            float block_l = 0.0f;
            for (int j = 0; j < num_k_rows; ++j) {
                float e = exp(scores[j] - m_new);
                for (int d = 0; d < D_HEAD; ++d) {
                    O_acc[d] += e * V_tile[j * D_HEAD + d];
                }
                block_l += e;
            }

            l_i = l_i * alpha + block_l;
            m_i = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
        const int global_out_idx = batch_head_offset_Q + (q_start_row + tid) * D_HEAD;
        for (int d = 0; d < D_HEAD; ++d) {
            O[global_out_idx + d] = O_acc[d] / l_i;
        }
    }
}
