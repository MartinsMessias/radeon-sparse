#ifndef RDNA3_AI_ENGINE_H
#define RDNA3_AI_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Cross-platform export/import macros */
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef RDNA3_EXPORTS
        #define RDNA3_API __declspec(dllexport)
    #else
        #define RDNA3_API __declspec(dllimport)
    #endif
    #define RDNA3_CALL __cdecl
#else
    #if __GNUC__ >= 4
        #define RDNA3_API __attribute__((visibility("default")))
    #else
        #define RDNA3_API
    #endif
    #define RDNA3_CALL
#endif

/* Opaque handles */
typedef void* rdna3_context_t;
typedef void* rdna3_tensor_t;

/**
 * Initialize GPU OpenCL context and compile JIT RDNA 3 Wave64 compute shaders.
 * Scans available platforms and selects the fastest AMD Radeon device.
 * Returns 0 on success, negative error code on failure.
 */
RDNA3_API int RDNA3_CALL rdna3_init(rdna3_context_t* ctx_out);

/**
 * Returns the name of the active GPU device (e.g., "AMD Radeon RX 7800 XT" or "gfx1101").
 */
RDNA3_API const char* RDNA3_CALL rdna3_get_device_name(rdna3_context_t ctx);

/**
 * Creates an INT8 weight matrix tensor in GPU VRAM.
 * rows: number of output features (M)
 * cols: number of input features (K)
 * host_data: optional pointer to host data (rows * cols bytes) for initial upload.
 */
RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_weights_int8(
    rdna3_context_t ctx,
    int rows,
    int cols,
    const signed char* host_data
);

/**
 * Creates an FP32 vector tensor in GPU VRAM (for inputs, scales, or activations).
 * size: number of float elements
 * host_data: optional pointer to host data (size * sizeof(float)) for initial upload.
 */
RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_vector_fp32(
    rdna3_context_t ctx,
    int size,
    const float* host_data
);

/**
 * Frees a tensor allocated in GPU VRAM.
 */
RDNA3_API void RDNA3_CALL rdna3_free_tensor(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor
);

/**
 * Executes a quantized GEMV: Y = (W * X) * scales
 * W: INT8 weights [M x K]
 * X: FP32 input vector [K]
 * scales: FP32 per-channel quantization scale vector [M]
 * Y: FP32 output vector [M]
 */
RDNA3_API int RDNA3_CALL rdna3_gemv_int8(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y
);

/**
 * Reads an FP32 vector from GPU VRAM back into host memory.
 */
RDNA3_API void RDNA3_CALL rdna3_read_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    float* host_dst
);

/**
 * Measures the average latency and effective hardware bandwidth of GEMV over multiple iterations.
 * Returns average latency in microseconds (us), and writes bandwidth in GB/s to out_bandwidth_gb_s.
 */
RDNA3_API double RDNA3_CALL rdna3_benchmark_gemv(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y,
    int iterations,
    double* out_bandwidth_gb_s
);

/**
 * Shuts down the GPU engine, releasing context, command queue, and compiled kernels.
 */
RDNA3_API void RDNA3_CALL rdna3_shutdown(rdna3_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif /* RDNA3_AI_ENGINE_H */
