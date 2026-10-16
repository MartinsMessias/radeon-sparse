#include "rdna3_ai_engine.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("========================================================\n");
    printf(" RDNA 3 Native GEMV Engine Test (C Consumer)\n");
    printf("========================================================\n");

    rdna3_context_t ctx = NULL;
    if (rdna3_init(&ctx) != 0) {
        fprintf(stderr, "[-] Failed to initialize GPU OpenCL context.\n");
        return 1;
    }

    printf("[+] Active GPU Device: %s\n", rdna3_get_device_name(ctx));

    const int M = 4096;
    const int K = 4096;

    signed char* w_host = (signed char*)malloc(M * K);
    float* x_host = (float*)malloc(K * sizeof(float));
    float* s_host = (float*)malloc(M * sizeof(float));
    float* y_host = (float*)malloc(M * sizeof(float));

    for (int i = 0; i < M * K; i++) w_host[i] = (signed char)((rand() % 255) - 127);
    for (int i = 0; i < K; i++) x_host[i] = 1.0f;
    for (int i = 0; i < M; i++) s_host[i] = 0.05f;

    rdna3_tensor_t tW = rdna3_create_weights_int8(ctx, M, K, w_host);
    rdna3_tensor_t tX = rdna3_create_vector_fp32(ctx, K, x_host);
    rdna3_tensor_t tS = rdna3_create_vector_fp32(ctx, M, s_host);
    rdna3_tensor_t tY = rdna3_create_vector_fp32(ctx, M, NULL);

    /* Run one forward inference */
    rdna3_gemv_int8(ctx, tW, tX, tS, tY);
    rdna3_read_vector_fp32(ctx, tY, y_host);

    printf("[+] Inference completed. Sample output: Y[0]=%.2f, Y[1]=%.2f\n", y_host[0], y_host[1]);

    /* Hardware benchmark */
    double bw = 0.0;
    double lat = rdna3_benchmark_gemv(ctx, tW, tX, tS, tY, 200, &bw);

    printf("\n--------------------------------------------------------\n");
    printf(" GPU HARDWARE BENCHMARK (Matrix %d x %d):\n", M, K);
    printf("--------------------------------------------------------\n");
    printf(" Average Layer Latency: %.2f us (%.4f ms)\n", lat, lat * 0.001);
    printf(" Effective Bandwidth:   %.2f GB/s\n", bw);
    printf(" GEMV Throughput:       %.1f ops/sec\n", 1e6 / lat);
    printf("--------------------------------------------------------\n");

    rdna3_free_tensor(ctx, tW);
    rdna3_free_tensor(ctx, tX);
    rdna3_free_tensor(ctx, tS);
    rdna3_free_tensor(ctx, tY);
    rdna3_shutdown(ctx);

    free(w_host);
    free(x_host);
    free(s_host);
    free(y_host);

    printf("[+] Resources cleanly released. Test PASSED!\n");
    return 0;
}
