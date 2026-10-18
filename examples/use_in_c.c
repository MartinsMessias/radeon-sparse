#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "radeon_sparse.h"

int main(void) {
    printf("======================================================================\n");
    printf("   RADEONSPARSE — C SDK QUICKSTART & VALIDATION\n");
    printf("======================================================================\n");

    rs_context_t ctx = NULL;
    int status = rs_init(&ctx);
    if (status != RS_SUCCESS) {
        fprintf(stderr, "[ERROR] rs_init failed (%d): %s\n", status, rs_status_string(status));
        return 1;
    }

    printf("[GPU] Active Device : %s\n\n", rs_get_device_name(ctx));

    /* Example: 4096 x 4096 Linear Projection from standard FP32 weights */
    const int M = 4096;
    const int K = 4096;

    float* h_W_fp32 = (float*)malloc((size_t)M * K * sizeof(float));
    float* h_X      = (float*)malloc((size_t)K * sizeof(float));
    float* h_Y_gpu  = (float*)malloc((size_t)M * sizeof(float));
    int*   h_idx    = (int*)malloc((size_t)M * sizeof(int));

    for (size_t i = 0; i < (size_t)M * K; i++) {
        h_W_fp32[i] = ((float)((int)(i % 199) - 99)) * 0.01f;
    }
    for (int k = 0; k < K; k++) {
        h_X[k] = ((float)((k % 31) - 15)) * 0.05f;
    }

    /* 1. One-line Automatic FP32 -> INT8 Quantization + VRAM Upload */
    rs_tensor_t d_W = NULL, d_S = NULL;
    status = rs_create_weights_from_fp32(ctx, M, K, h_W_fp32, &d_W, &d_S);
    if (status != RS_SUCCESS) {
        fprintf(stderr, "[ERROR] Weight creation failed: %s\n", rs_status_string(status));
        return 1;
    }

    rs_tensor_t d_X = rs_create_vector_fp32(ctx, K, h_X);
    rs_tensor_t d_Y = rs_create_vector_fp32(ctx, M, NULL);

    /* 2. Dense GEMV */
    rs_gemv_int8(ctx, d_W, d_X, d_S, d_Y);
    rs_read_vector_fp32(ctx, d_Y, h_Y_gpu);

    double dense_bw = 0.0;
    double dense_us = rs_benchmark_gemv(ctx, d_W, d_X, d_S, d_Y, 200, &dense_bw);
    printf("[1] Dense GEMV (%d x %d):\n", M, K);
    printf("    Latency   : %.2f us\n", dense_us);
    printf("    Bandwidth : %.2f GB/s\n", dense_bw);
    printf("    Sample Y  : [%.4f, %.4f, %.4f]\n\n", h_Y_gpu[0], h_Y_gpu[1], h_Y_gpu[2]);

    /* 3. Contextual Sparsity Gather-GEMV (20% active neurons) */
    int num_active = M / 5;
    for (int i = 0; i < num_active; i++) {
        h_idx[i] = i * 5;
    }
    rs_tensor_t d_Idx = rs_create_indices_int32(ctx, num_active, h_idx);

    rs_zero_vector_fp32(ctx, d_Y);
    rs_gather_gemv_int8(ctx, d_W, d_X, d_S, d_Y, d_Idx, num_active);

    double phys_bw = 0.0, eff_bw = 0.0;
    double sparse_us = rs_benchmark_gather_gemv(ctx, d_W, d_X, d_S, d_Y, d_Idx, num_active, 200, &phys_bw, &eff_bw);
    printf("[2] Contextual Sparse Row-Gather GEMV (20%% Active = %d / %d rows):\n", num_active, M);
    printf("    Latency             : %.2f us (%.2fx faster than Dense)\n", sparse_us, dense_us / sparse_us);
    printf("    Physical Bandwidth  : %.2f GB/s\n", phys_bw);
    printf("    Effective Bandwidth : %.2f GB/s\n\n", eff_bw);

    /* 4. Contextual Sparsity Column-Gather GEMV (SwiGLU Down-Projection, 20% active columns) */
    int num_active_cols = K / 5;
    for (int i = 0; i < num_active_cols; i++) {
        h_idx[i] = i * 5;
    }
    rs_tensor_t d_ColIdx = rs_create_indices_int32(ctx, num_active_cols, h_idx);

    rs_zero_vector_fp32(ctx, d_Y);
    rs_column_gather_gemv_int8(ctx, d_W, d_X, d_S, d_Y, d_ColIdx, num_active_cols);

    double col_phys_bw = 0.0, col_eff_bw = 0.0;
    double col_sparse_us = rs_benchmark_column_gather_gemv(ctx, d_W, d_X, d_S, d_Y, d_ColIdx, num_active_cols, 200, &col_phys_bw, &col_eff_bw);
    printf("[3] Contextual Sparse Column-Gather GEMV (20%% Active = %d / %d columns):\n", num_active_cols, K);
    printf("    Latency             : %.2f us (%.2fx faster than Dense)\n", col_sparse_us, dense_us / col_sparse_us);
    printf("    Physical Bandwidth  : %.2f GB/s\n", col_phys_bw);
    printf("    Effective Bandwidth : %.2f GB/s\n", col_eff_bw);
    printf("======================================================================\n");

    rs_free_tensor(ctx, d_ColIdx);

    rs_free_tensor(ctx, d_Idx);
    rs_free_tensor(ctx, d_Y);
    rs_free_tensor(ctx, d_X);
    rs_free_tensor(ctx, d_S);
    rs_free_tensor(ctx, d_W);
    rs_shutdown(ctx);

    free(h_idx);
    free(h_Y_gpu);
    free(h_X);
    free(h_W_fp32);
    return 0;
}
