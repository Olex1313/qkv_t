// Tiled batched matrix multiplication
// A: [B, M, N] row-major, B: [B, N, K] row-major -> C: [B, M, K] row-major
#define TS 16

__kernel void bmm(
    const int M,
    const int N,
    const int K,
    const __global float* A,
    const __global float* B,
    __global float* C,
    const float alpha
) {
    const int b        = get_global_id(2);
    const int row      = get_local_id(0);
    const int col      = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;

    const int numTiles = (N + TS - 1) / TS;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TS + col;
        Asub[row][col] = (globalRow < M && aCol < N) ? A[b * M * N + globalRow * N + aCol] : 0.0f;

        int bRow = t * TS + row;
        Bsub[row][col] = (bRow < N && globalCol < K) ? B[b * N * K + bRow * K + globalCol] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            acc += Asub[row][k] * Bsub[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < M && globalCol < K) {
        C[b * M * K + globalRow * K + globalCol] = acc * alpha;
    }
}
