#include "rdna3_ai_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("============================================================\n");
    printf(" RDNA 3 AI Engine - Native C Example\n");
    printf(" Targeting AMD Radeon Hardware (Wave64 Infinity-Streaming)\n");
    printf("============================================================\n\n");

    rdna3_context_t ctx = NULL;
    int err = rdna3_init(&ctx);
    if (err != 0) {
        printf("[ERROR] Failed to initialize RDNA 3 Engine (code: %d)\n", err);
        return 1;
    }
    printf("[OK] Hardware pipeline initialized successfully on AMD GPU.\n");

    /* Matrix Dimensions: 4096 x 4096 (standard LLM projection layer) */
    const int M = 4096;
    const int K = 4096;

    signed char* w_host = (signed char*)malloc((size_t)M * K);
    float* x_host = (float*)malloc((size_t)K * sizeof(float));
    float* s_host = (float*)malloc((size_t)M * sizeof(float));
    float* y_host = (float*)malloc((size_t)M * sizeof(float));

    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < (size_t)M * K; i++) w_host[i] = (signed char)((rand() % 255) - 127);
    for (int i = 0; i < K; i++) x_host[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < M; i++) s_host[i] = 0.05f;

    /* Allocate and transfer tensors to VRAM */
    rdna3_tensor_t tW = rdna3_create_weights_int8(ctx, M, K, w_host);
    rdna3_tensor_t tX = rdna3_create_vector_fp32(ctx, K, x_host);
    rdna3_tensor_t tS = rdna3_create_vector_fp32(ctx, M, s_host);
    rdna3_tensor_t tY = rdna3_create_vector_fp32(ctx, M, NULL);

    /* Execute single inference */
    rdna3_gemv_int8(ctx, tW, tX, tS, tY);
    rdna3_read_vector_fp32(ctx, tY, y_host);

    printf("[Inference] First 4 output elements: [%.3f, %.3f, %.3f, %.3f]\n",
           y_host[0], y_host[1], y_host[2], y_host[3]);

    /* Micro-benchmark over 200 iterations */
    double bandwidth_gb_s = 0.0;
    double latency_us = rdna3_benchmark_gemv(ctx, tW, tX, tS, tY, 200, &bandwidth_gb_s);

    printf("\n------------------------------------------------------------\n");
    printf(" Benchmark Results (200 Iterations):\n");
    printf("------------------------------------------------------------\n");
    printf(" Latency:              %.2f microseconds (us)\n", latency_us);
    printf(" Effective Bandwidth:  %.2f GB/s\n", bandwidth_gb_s);
    printf(" Throughput:           %.1f GEMV ops/second\n", 1e6 / latency_us);
    printf("------------------------------------------------------------\n");

    /* Cleanup */
    rdna3_free_tensor(ctx, tW);
    rdna3_free_tensor(ctx, tX);
    rdna3_free_tensor(ctx, tS);
    rdna3_free_tensor(ctx, tY);
    rdna3_shutdown(ctx);

    free(w_host);
    free(x_host);
    free(s_host);
    free(y_host);

    printf("[OK] Resources released. Done!\n");
    return 0;
}
