#ifndef RDNA3_AI_ENGINE_H
#define RDNA3_AI_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Cross-platform symbol visibility and calling convention macros */
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

/* Status & Error Codes */
typedef enum {
    RDNA3_SUCCESS                  =  0,
    RDNA3_ERR_INVALID_ARG          = -1,
    RDNA3_ERR_OPENCL_LOAD_FAILED   = -2,
    RDNA3_ERR_NO_PLATFORM          = -3,
    RDNA3_ERR_NO_DEVICE            = -4,
    RDNA3_ERR_HOST_ALLOC_FAILED    = -5,
    RDNA3_ERR_CONTEXT_FAILED       = -6,
    RDNA3_ERR_QUEUE_FAILED         = -7,
    RDNA3_ERR_PROGRAM_BUILD_FAILED = -8,
    RDNA3_ERR_KERNEL_FAILED        = -9,
    RDNA3_ERR_DIMENSION_MISMATCH   = -10,
    RDNA3_ERR_EXECUTION_FAILED     = -11,
    RDNA3_ERR_VRAM_ALLOC_FAILED    = -12,
    RDNA3_ERR_TRANSFER_FAILED      = -13
} rdna3_status_t;

/* Opaque handles */
typedef void* rdna3_context_t;
typedef void* rdna3_tensor_t;

/* ============================================================================
 * 1. Lifecycle & Device Management
 * ============================================================================ */

/**
 * Initialize GPU OpenCL context and compile JIT RDNA 3 Wave64 compute shaders.
 * Automatically scans available platforms and selects the primary AMD Radeon GPU.
 * Returns RDNA3_SUCCESS (0) on success, or negative rdna3_status_t error code.
 */
RDNA3_API int RDNA3_CALL rdna3_init(rdna3_context_t* ctx_out);

/**
 * Shuts down the GPU engine, releasing context, command queue, and compiled kernels.
 */
RDNA3_API void RDNA3_CALL rdna3_shutdown(rdna3_context_t ctx);

/**
 * Returns the name of the active GPU device (e.g., "AMD Radeon RX 7800 XT" or "gfx1101").
 */
RDNA3_API const char* RDNA3_CALL rdna3_get_device_name(rdna3_context_t ctx);

/**
 * Returns a human-readable string describing an rdna3_status_t code.
 */
RDNA3_API const char* RDNA3_CALL rdna3_status_string(int status);

/* ============================================================================
 * 2. Tensor Allocation, In-Place Updates & Automatic Quantization
 * ============================================================================ */

/**
 * Creates an INT8 weight matrix tensor in GPU VRAM [rows x cols].
 * host_data: optional pointer to host data (rows * cols bytes) for initial upload.
 */
RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_weights_int8(
    rdna3_context_t ctx,
    int rows,
    int cols,
    const signed char* host_data
);

/**
 * Creates an FP32 vector tensor in GPU VRAM [size].
 * host_data: optional pointer to host data (size * sizeof(float)) for initial upload.
 */
RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_vector_fp32(
    rdna3_context_t ctx,
    int size,
    const float* host_data
);

/**
 * Creates an INT32 index buffer in GPU VRAM [max_indices] for Sparse Gather-GEMV.
 * host_indices: optional pointer to initial active row indices.
 */
RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_indices_int32(
    rdna3_context_t ctx,
    int max_indices,
    const int* host_indices
);

/**
 * High-level helper: Quantizes an FP32 weight matrix [rows x cols] to per-channel
 * symmetric INT8 on the CPU and creates both the INT8 weights tensor and FP32 scales
 * tensor directly in GPU VRAM.
 */
RDNA3_API int RDNA3_CALL rdna3_create_weights_from_fp32(
    rdna3_context_t ctx,
    int rows,
    int cols,
    const float* fp32_weights,
    rdna3_tensor_t* out_weights_int8,
    rdna3_tensor_t* out_scales_fp32
);

/**
 * CPU utility: Quantizes an FP32 matrix [rows x cols] into symmetric per-row INT8
 * weights (-127..127) and per-row FP32 dequantization scales (max_abs / 127.0f).
 */
RDNA3_API int RDNA3_CALL rdna3_quantize_fp32_to_int8(
    const float* fp32_weights,
    int rows,
    int cols,
    signed char* out_int8,
    float* out_scales
);

/**
 * CPU utility: Scans an activation vector [size] and writes indices i where
 * fabs(activations[i]) >= threshold into out_indices. Returns the number of active indices.
 */
RDNA3_API int RDNA3_CALL rdna3_extract_active_indices(
    const float* host_activations,
    int size,
    float threshold,
    int* out_indices,
    int max_indices
);

/**
 * Updates an existing FP32 GPU vector in-place without reallocating VRAM.
 */
RDNA3_API int RDNA3_CALL rdna3_write_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    const float* host_src,
    int count
);

/**
 * Updates an existing INT32 index tensor in-place without reallocating VRAM.
 */
RDNA3_API int RDNA3_CALL rdna3_write_indices_int32(
    rdna3_context_t ctx,
    rdna3_tensor_t indices_tensor,
    const int* host_indices,
    int num_active
);

/**
 * Reads an FP32 vector from GPU VRAM back into host memory.
 */
RDNA3_API int RDNA3_CALL rdna3_read_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    float* host_dst
);

/**
 * Fast GPU zero-fill for an FP32 vector tensor.
 */
RDNA3_API int RDNA3_CALL rdna3_zero_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor
);

/**
 * Frees a tensor allocated in GPU VRAM.
 */
RDNA3_API void RDNA3_CALL rdna3_free_tensor(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor
);

/* ============================================================================
 * 3. RDNA 3 Wave64 Compute Kernels (Dense GEMV, Sparse Gather-GEMV, Gating)
 * ============================================================================ */

/**
 * Executes Dense Quantized GEMV: Y = (W * X) * scales
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
 * Executes Contextual Sparsity Gather-GEMV:
 * Computes ONLY the `num_active` rows specified in `active_indices`!
 * Inactive rows are never fetched from GDDR6 VRAM, multiplying effective bandwidth.
 */
RDNA3_API int RDNA3_CALL rdna3_gather_gemv_int8(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y,
    rdna3_tensor_t active_indices,
    int num_active
);

/**
 * Executes Hardware Activation Gating (SwiGLU / Top-K Magnitude Filter) on GPU:
 * output_h[i] = (fabs(gate[i]) >= threshold) ? (gate[i] * up[i]) : 0.0f
 */
RDNA3_API int RDNA3_CALL rdna3_gate_activations_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t gate_vec,
    rdna3_tensor_t up_vec,
    rdna3_tensor_t output_h,
    float threshold
);

/* ============================================================================
 * 4. Hardware Profiling & Official Benchmarking
 * ============================================================================ */

/**
 * Benchmarks Dense GEMV using hardware event timers (CL_QUEUE_PROFILING_ENABLE).
 * Returns average kernel latency in microseconds (us), and writes physical bandwidth
 * in GB/s to out_bandwidth_gb_s.
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
 * Benchmarks Contextual Sparse Gather-GEMV using hardware event timers.
 * Returns average kernel latency in microseconds (us), writes physical GDDR6
 * bandwidth (bytes actually read / time) to out_physical_bw_gb_s, and writes
 * effective dense-equivalent throughput (total dense matrix bytes / time) to
 * out_effective_bw_gb_s.
 */
RDNA3_API double RDNA3_CALL rdna3_benchmark_gather_gemv(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y,
    rdna3_tensor_t active_indices,
    int num_active,
    int iterations,
    double* out_physical_bw_gb_s,
    double* out_effective_bw_gb_s
);

#ifdef __cplusplus
}
#endif

#endif /* RDNA3_AI_ENGINE_H */
