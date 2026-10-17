#ifndef RADEON_SPARSE_H
#define RADEON_SPARSE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Cross-platform symbol visibility and calling convention macros */
#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef RADEON_SPARSE_EXPORTS
        #define RS_API __declspec(dllexport)
    #else
        #define RS_API __declspec(dllimport)
    #endif
    #define RS_CALL __cdecl
#else
    #if __GNUC__ >= 4
        #define RS_API __attribute__((visibility("default")))
    #else
        #define RS_API
    #endif
    #define RS_CALL
#endif

/* Status & Error Codes */
typedef enum {
    RS_SUCCESS                  =  0,
    RS_ERR_INVALID_ARG          = -1,
    RS_ERR_OPENCL_LOAD_FAILED   = -2,
    RS_ERR_NO_PLATFORM          = -3,
    RS_ERR_NO_DEVICE            = -4,
    RS_ERR_HOST_ALLOC_FAILED    = -5,
    RS_ERR_CONTEXT_FAILED       = -6,
    RS_ERR_QUEUE_FAILED         = -7,
    RS_ERR_PROGRAM_BUILD_FAILED = -8,
    RS_ERR_KERNEL_FAILED        = -9,
    RS_ERR_DIMENSION_MISMATCH   = -10,
    RS_ERR_EXECUTION_FAILED     = -11,
    RS_ERR_VRAM_ALLOC_FAILED    = -12,
    RS_ERR_TRANSFER_FAILED      = -13
} rs_status_t;

/* Opaque handles */
typedef void* rs_context_t;
typedef void* rs_tensor_t;

/* ============================================================================
 * 1. Lifecycle & Device Management
 * ============================================================================ */

/**
 * Initialize GPU OpenCL context and compile JIT Wavefront compute shaders.
 * Automatically scans available platforms and selects the primary AMD Radeon GPU.
 * Returns RS_SUCCESS (0) on success, or negative rs_status_t error code.
 */
RS_API int RS_CALL rs_init(rs_context_t* ctx_out);

/**
 * Shuts down the GPU engine, releasing context, command queue, and compiled kernels.
 */
RS_API void RS_CALL rs_shutdown(rs_context_t ctx);

/**
 * Returns the name/architecture identifier of the active GPU device.
 */
RS_API const char* RS_CALL rs_get_device_name(rs_context_t ctx);

/**
 * Returns a human-readable string describing an rs_status_t code.
 */
RS_API const char* RS_CALL rs_status_string(int status);

/* ============================================================================
 * 2. Tensor Allocation, In-Place Updates & Automatic Quantization
 * ============================================================================ */

/**
 * Creates an INT8 weight matrix tensor in GPU VRAM [rows x cols].
 * host_data: optional pointer to host data (rows * cols bytes) for initial upload.
 */
RS_API rs_tensor_t RS_CALL rs_create_weights_int8(
    rs_context_t ctx,
    int rows,
    int cols,
    const signed char* host_data
);

/**
 * Creates an FP32 vector tensor in GPU VRAM [size].
 * host_data: optional pointer to host data (size * sizeof(float)) for initial upload.
 */
RS_API rs_tensor_t RS_CALL rs_create_vector_fp32(
    rs_context_t ctx,
    int size,
    const float* host_data
);

/**
 * Creates an INT32 index buffer in GPU VRAM [max_indices] for Sparse Gather-GEMV.
 * host_indices: optional pointer to initial active row indices.
 */
RS_API rs_tensor_t RS_CALL rs_create_indices_int32(
    rs_context_t ctx,
    int max_indices,
    const int* host_indices
);

/**
 * High-level helper: Quantizes an FP32 weight matrix [rows x cols] to per-channel
 * symmetric INT8 on the CPU and creates both the INT8 weights tensor and FP32 scales
 * tensor directly in GPU VRAM.
 */
RS_API int RS_CALL rs_create_weights_from_fp32(
    rs_context_t ctx,
    int rows,
    int cols,
    const float* fp32_weights,
    rs_tensor_t* out_weights_int8,
    rs_tensor_t* out_scales_fp32
);

/**
 * CPU utility: Quantizes an FP32 matrix [rows x cols] into symmetric per-row INT8
 * weights (-127..127) and per-row FP32 dequantization scales (max_abs / 127.0f).
 */
RS_API int RS_CALL rs_quantize_fp32_to_int8(
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
RS_API int RS_CALL rs_extract_active_indices(
    const float* host_activations,
    int size,
    float threshold,
    int* out_indices,
    int max_indices
);

/**
 * Updates an existing FP32 GPU vector in-place without reallocating VRAM.
 */
RS_API int RS_CALL rs_write_vector_fp32(
    rs_context_t ctx,
    rs_tensor_t tensor,
    const float* host_src,
    int count
);

/**
 * Updates an existing INT32 index tensor in-place without reallocating VRAM.
 */
RS_API int RS_CALL rs_write_indices_int32(
    rs_context_t ctx,
    rs_tensor_t indices_tensor,
    const int* host_indices,
    int num_active
);

/**
 * Reads an FP32 vector from GPU VRAM back into host memory.
 */
RS_API int RS_CALL rs_read_vector_fp32(
    rs_context_t ctx,
    rs_tensor_t tensor,
    float* host_dst
);

/**
 * Fast GPU zero-fill for an FP32 vector tensor.
 */
RS_API int RS_CALL rs_zero_vector_fp32(
    rs_context_t ctx,
    rs_tensor_t tensor
);

/**
 * Frees a tensor allocated in GPU VRAM.
 */
RS_API void RS_CALL rs_free_tensor(
    rs_context_t ctx,
    rs_tensor_t tensor
);

/* ============================================================================
 * 3. Compute Kernels (Dense GEMV, Sparse Gather-GEMV, Activation Gating)
 * ============================================================================ */

/**
 * Executes Dense Quantized GEMV: Y = (W * X) * scales
 * W: INT8 weights [M x K]
 * X: FP32 input vector [K]
 * scales: FP32 per-channel quantization scale vector [M]
 * Y: FP32 output vector [M]
 */
RS_API int RS_CALL rs_gemv_int8(
    rs_context_t ctx,
    rs_tensor_t weights,
    rs_tensor_t input_x,
    rs_tensor_t scales,
    rs_tensor_t output_y
);

/**
 * Executes Contextual Sparsity Gather-GEMV:
 * Computes ONLY the `num_active` rows specified in `active_indices`.
 * Inactive rows are never fetched from VRAM, multiplying effective bandwidth.
 */
RS_API int RS_CALL rs_gather_gemv_int8(
    rs_context_t ctx,
    rs_tensor_t weights,
    rs_tensor_t input_x,
    rs_tensor_t scales,
    rs_tensor_t output_y,
    rs_tensor_t active_indices,
    int num_active
);

/**
 * Executes Hardware Activation Gating (SwiGLU / Top-K Magnitude Filter) on GPU:
 * output_h[i] = (fabs(gate[i]) >= threshold) ? (gate[i] * up[i]) : 0.0f
 */
RS_API int RS_CALL rs_gate_activations_fp32(
    rs_context_t ctx,
    rs_tensor_t gate_vec,
    rs_tensor_t up_vec,
    rs_tensor_t output_h,
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
RS_API double RS_CALL rs_benchmark_gemv(
    rs_context_t ctx,
    rs_tensor_t weights,
    rs_tensor_t input_x,
    rs_tensor_t scales,
    rs_tensor_t output_y,
    int iterations,
    double* out_bandwidth_gb_s
);

/**
 * Benchmarks Contextual Sparse Gather-GEMV using hardware event timers.
 * Returns average kernel latency in microseconds (us), writes physical VRAM
 * bandwidth (bytes actually read / time) to out_physical_bw_gb_s, and writes
 * effective dense-equivalent throughput (total dense matrix bytes / time) to
 * out_effective_bw_gb_s.
 */
RS_API double RS_CALL rs_benchmark_gather_gemv(
    rs_context_t ctx,
    rs_tensor_t weights,
    rs_tensor_t input_x,
    rs_tensor_t scales,
    rs_tensor_t output_y,
    rs_tensor_t active_indices,
    int num_active,
    int iterations,
    double* out_physical_bw_gb_s,
    double* out_effective_bw_gb_s
);

#ifdef __cplusplus
}
#endif

#endif /* RADEON_SPARSE_H */
