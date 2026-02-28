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

__kernel void flash_attention_v1_fwd(
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
    const int head_idx = get_global_id(2);
    const int q_block_idx = get_group_id(0);

    if (batch_idx >= B || head_idx >= H) return;

    const int batch_head_offset_Q = (batch_idx * H + head_idx) * L * D_HEAD;
    const int batch_head_offset_K = (batch_idx * H + head_idx) * S * D_HEAD;
    const int batch_head_offset_V = batch_head_offset_K;

    const int tid = get_local_id(0);
    const int wg_size = get_local_size(0);

    __local float Q_tile[BLOCK_SIZE_M * D_HEAD];
    __local float K_tile[BLOCK_SIZE_N * D_HEAD];
    __local float V_tile[BLOCK_SIZE_N * D_HEAD];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_acc[D_HEAD];
    for (int d = 0; d < D_HEAD; ++d) O_acc[d] = 0.0f;

    const int q_start_row = q_block_idx * BLOCK_SIZE_M;
    const int num_q_rows = (q_start_row + BLOCK_SIZE_M > L) ? (L - q_start_row) : BLOCK_SIZE_M;

    load_global_tile(
        Q, Q_tile,
        batch_head_offset_Q + q_start_row * D_HEAD,
        D_HEAD, BLOCK_SIZE_M, num_q_rows,
        tid, wg_size
    );
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_kv_blocks = (S + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;

    for (int k_block = 0; k_block < num_kv_blocks; ++k_block) {
        int k_start_row = k_block * BLOCK_SIZE_N;
        int num_k_rows = (k_start_row + BLOCK_SIZE_N > S) ? (S - k_start_row) : BLOCK_SIZE_N;

        if (is_causal && k_start_row > (q_start_row + BLOCK_SIZE_M - 1)) break;

        load_global_tile(K, K_tile, batch_head_offset_K + k_start_row * D_HEAD, D_HEAD, BLOCK_SIZE_N, num_k_rows, tid, wg_size);
        load_global_tile(V, V_tile, batch_head_offset_V + k_start_row * D_HEAD, D_HEAD, BLOCK_SIZE_N, num_k_rows, tid, wg_size);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
            for (int k_row = 0; k_row < num_k_rows; ++k_row) {
                if (is_causal && (k_start_row + k_row) > (q_start_row + tid)) continue;

                float score = 0.0f;
                for (int d = 0; d < D_HEAD; ++d) {
                    score += Q_tile[tid * D_HEAD + d] * K_tile[k_row * D_HEAD + d];
                }
                score *= scale;

                float m_prev = m_i;
                m_i = fmax(m_prev, score);

                float exp_score = exp(score - m_i);
                float alpha = exp(m_prev - m_i);

                for (int d = 0; d < D_HEAD; ++d) {
                    O_acc[d] = O_acc[d] * alpha + exp_score * V_tile[k_row * D_HEAD + d];
                }
                l_i = l_i * alpha + exp_score;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < BLOCK_SIZE_M && tid < num_q_rows) {
        int global_out_idx = batch_head_offset_Q + (q_start_row + tid) * D_HEAD;
        for (int d = 0; d < D_HEAD; ++d) {
            O[global_out_idx + d] = O_acc[d] / l_i;
        }
    }
}