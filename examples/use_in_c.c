#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdna3_ai_engine.h"

int main(void) {
    printf("======================================================================\n");
    printf("   RDNA 3 / RDNA 4 AI ENGINE — C SDK QUICKSTART & VALIDATION\n");
    printf("======================================================================\n");

    rdna3_context_t ctx = NULL;
    int status = rdna3_init(&ctx);
    if (status != RDNA3_SUCCESS) {
        fprintf(stderr, "[ERROR] rdna3_init failed (%d): %s\n", status, rdna3_status_string(status));
        return 1;
    }

    printf("[GPU] Active Device : %s\n\n", rdna3_get_device_name(ctx));

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
    rdna3_tensor_t d_W = NULL, d_S = NULL;
    status = rdna3_create_weights_from_fp32(ctx, M, K, h_W_fp32, &d_W, &d_S);
    if (status != RDNA3_SUCCESS) {
        fprintf(stderr, "[ERROR] Weight creation failed: %s\n", rdna3_status_string(status));
        return 1;
    }

    rdna3_tensor_t d_X = rdna3_create_vector_fp32(ctx, K, h_X);
    rdna3_tensor_t d_Y = rdna3_create_vector_fp32(ctx, M, NULL);

    /* 2. Dense Wave64 GEMV */
    rdna3_gemv_int8(ctx, d_W, d_X, d_S, d_Y);
    rdna3_read_vector_fp32(ctx, d_Y, h_Y_gpu);

    double dense_bw = 0.0;
    double dense_us = rdna3_benchmark_gemv(ctx, d_W, d_X, d_S, d_Y, 200, &dense_bw);
    printf("[1] Dense Wave64 GEMV (%d x %d):\n", M, K);
    printf("    Latency   : %.2f us\n", dense_us);
    printf("    Bandwidth : %.2f GB/s\n", dense_bw);
    printf("    Sample Y  : [%.4f, %.4f, %.4f]\n\n", h_Y_gpu[0], h_Y_gpu[1], h_Y_gpu[2]);

    /* 3. Contextual Sparsity Gather-GEMV (20% active neurons) */
    int num_active = M / 5;
    for (int i = 0; i < num_active; i++) {
        h_idx[i] = i * 5;
    }
    rdna3_tensor_t d_Idx = rdna3_create_indices_int32(ctx, num_active, h_idx);

    rdna3_zero_vector_fp32(ctx, d_Y);
    rdna3_gather_gemv_int8(ctx, d_W, d_X, d_S, d_Y, d_Idx, num_active);

    double phys_bw = 0.0, eff_bw = 0.0;
    double sparse_us = rdna3_benchmark_gather_gemv(ctx, d_W, d_X, d_S, d_Y, d_Idx, num_active, 200, &phys_bw, &eff_bw);
    printf("[2] Contextual Sparse Gather-GEMV (20%% Active = %d / %d rows):\n", num_active, M);
    printf("    Latency             : %.2f us (%.2fx faster than Dense)\n", sparse_us, dense_us / sparse_us);
    printf("    Physical Bandwidth  : %.2f GB/s\n", phys_bw);
    printf("    Effective Bandwidth : %.2f GB/s\n", eff_bw);
    printf("======================================================================\n");

    rdna3_free_tensor(ctx, d_Idx);
    rdna3_free_tensor(ctx, d_Y);
    rdna3_free_tensor(ctx, d_X);
    rdna3_free_tensor(ctx, d_S);
    rdna3_free_tensor(ctx, d_W);
    rdna3_shutdown(ctx);

    free(h_idx);
    free(h_Y_gpu);
    free(h_X);
    free(h_W_fp32);
    return 0;
}
