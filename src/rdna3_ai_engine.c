#define _CRT_SECURE_NO_WARNINGS
#define RDNA3_EXPORTS
#include "rdna3_ai_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32) || defined(__CYGWIN__)
    #include <windows.h>
    typedef HMODULE lib_handle_t;
    #define LOAD_LIB(name) LoadLibraryA(name)
    #define GET_SYM(lib, sym) GetProcAddress(lib, sym)
    #define FREE_LIB(lib) FreeLibrary(lib)
    #define OCL_CANDIDATE_1 "C:\\Windows\\System32\\OpenCL.dll"
    #define OCL_CANDIDATE_2 "OpenCL.dll"
    #define OCL_CANDIDATE_3 NULL
    #define CL_CALL_CONV WINAPI
#else
    #include <dlfcn.h>
    typedef void* lib_handle_t;
    #define LOAD_LIB(name) dlopen(name, RTLD_NOW | RTLD_GLOBAL)
    #define GET_SYM(lib, sym) dlsym(lib, sym)
    #define FREE_LIB(lib) dlclose(lib)
    #define OCL_CANDIDATE_1 "libOpenCL.so.1"
    #define OCL_CANDIDATE_2 "libOpenCL.so"
    #define OCL_CANDIDATE_3 "libamdocl64.so"
    #define CL_CALL_CONV
#endif

/* OpenCL 2.0 Base Types and Constants */
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef int cl_int;
typedef size_t cl_size_t;

#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_TYPE_GPU (1ULL << 2)
#define CL_MEM_READ_ONLY (1ULL << 2)
#define CL_MEM_READ_WRITE (1ULL << 0)
#define CL_QUEUE_PROPERTIES 0x1093
#define CL_QUEUE_PROFILING_ENABLE (1ULL << 1)
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_PROGRAM_BUILD_LOG 0x1183

/* OpenCL API Function Pointers */
typedef cl_int (CL_CALL_CONV *pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_CALL_CONV *pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_CALL_CONV *pfn_clGetDeviceInfo)(cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*);
typedef cl_context (CL_CALL_CONV *pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (CL_CALL_CONV *pfn_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_ulong*, cl_int*);
typedef cl_mem (CL_CALL_CONV *pfn_clCreateBuffer)(cl_context, cl_ulong, cl_size_t, void*, cl_int*);
typedef cl_program (CL_CALL_CONV *pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const cl_size_t*, cl_int*);
typedef cl_int (CL_CALL_CONV *pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (CL_CALL_CONV *pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*);
typedef cl_kernel (CL_CALL_CONV *pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (CL_CALL_CONV *pfn_clSetKernelArg)(cl_kernel, cl_uint, cl_size_t, const void*);
typedef cl_int (CL_CALL_CONV *pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const cl_size_t*, const cl_size_t*, const cl_size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_CALL_CONV *pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, cl_size_t, cl_size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_CALL_CONV *pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, cl_size_t, cl_size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_CALL_CONV *pfn_clWaitForEvents)(cl_uint, const cl_event*);
typedef cl_int (CL_CALL_CONV *pfn_clGetEventProfilingInfo)(cl_event, cl_uint, cl_size_t, void*, cl_size_t*);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseEvent)(cl_event);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseProgram)(cl_program);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_CALL_CONV *pfn_clReleaseContext)(cl_context);
typedef cl_int (CL_CALL_CONV *pfn_clFinish)(cl_command_queue);

static lib_handle_t s_hCL = NULL;
static pfn_clGetPlatformIDs clGetPlatformIDs = NULL;
static pfn_clGetDeviceIDs clGetDeviceIDs = NULL;
static pfn_clGetDeviceInfo clGetDeviceInfo = NULL;
static pfn_clCreateContext clCreateContext = NULL;
static pfn_clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties = NULL;
static pfn_clCreateBuffer clCreateBuffer = NULL;
static pfn_clCreateProgramWithSource clCreateProgramWithSource = NULL;
static pfn_clBuildProgram clBuildProgram = NULL;
static pfn_clGetProgramBuildInfo clGetProgramBuildInfo = NULL;
static pfn_clCreateKernel clCreateKernel = NULL;
static pfn_clSetKernelArg clSetKernelArg = NULL;
static pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = NULL;
static pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer = NULL;
static pfn_clEnqueueReadBuffer clEnqueueReadBuffer = NULL;
static pfn_clWaitForEvents clWaitForEvents = NULL;
static pfn_clGetEventProfilingInfo clGetEventProfilingInfo = NULL;
static pfn_clReleaseEvent clReleaseEvent = NULL;
static pfn_clReleaseMemObject clReleaseMemObject = NULL;
static pfn_clReleaseKernel clReleaseKernel = NULL;
static pfn_clReleaseProgram clReleaseProgram = NULL;
static pfn_clReleaseCommandQueue clReleaseCommandQueue = NULL;
static pfn_clReleaseContext clReleaseContext = NULL;
static pfn_clFinish clFinish = NULL;

static int load_opencl(void) {
    if (s_hCL) return 1;

    s_hCL = LOAD_LIB(OCL_CANDIDATE_1);
    if (!s_hCL && OCL_CANDIDATE_2) s_hCL = LOAD_LIB(OCL_CANDIDATE_2);
    if (!s_hCL && OCL_CANDIDATE_3) s_hCL = LOAD_LIB(OCL_CANDIDATE_3);
    if (!s_hCL) return 0;

    #define BIND_SYM(name) do { \
        name = (pfn_##name)GET_SYM(s_hCL, #name); \
        if (!name) { \
            FREE_LIB(s_hCL); \
            s_hCL = NULL; \
            return 0; \
        } \
    } while (0)

    BIND_SYM(clGetPlatformIDs);
    BIND_SYM(clGetDeviceIDs);
    BIND_SYM(clGetDeviceInfo);
    BIND_SYM(clCreateContext);
    BIND_SYM(clCreateCommandQueueWithProperties);
    BIND_SYM(clCreateBuffer);
    BIND_SYM(clCreateProgramWithSource);
    BIND_SYM(clBuildProgram);
    BIND_SYM(clGetProgramBuildInfo);
    BIND_SYM(clCreateKernel);
    BIND_SYM(clSetKernelArg);
    BIND_SYM(clEnqueueNDRangeKernel);
    BIND_SYM(clEnqueueWriteBuffer);
    BIND_SYM(clEnqueueReadBuffer);
    BIND_SYM(clWaitForEvents);
    BIND_SYM(clGetEventProfilingInfo);
    BIND_SYM(clReleaseEvent);
    BIND_SYM(clReleaseMemObject);
    BIND_SYM(clReleaseKernel);
    BIND_SYM(clReleaseProgram);
    BIND_SYM(clReleaseCommandQueue);
    BIND_SYM(clReleaseContext);
    BIND_SYM(clFinish);
    #undef BIND_SYM

    return 1;
}

/* ============================================================================
 * RDNA 3 / RDNA 4 Wave64 Compute Shaders
 * - Uses vectorized 64-bit `vload8` + hardware `convert_float8`
 * - Supports arbitrary K dimensions via scalar tail loop
 * - Uses conflict-free LDS parallel tree reduction
 * ============================================================================ */
static const char* RDNA3_KERNEL_SOURCE =
"/* 1. Dense Wave64 Infinity Stream GEMV */\n"
"__kernel void gemv_rdna3_infinity_stream(\n"
"    __global const char* restrict W,\n"
"    __global const float* restrict X,\n"
"    __global const float* restrict scales,\n"
"    __global float* restrict Y,\n"
"    const int M,\n"
"    const int K\n"
") {\n"
"    int row = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    if (row >= M) return;\n"
"\n"
"    __global const char* rW = W + (size_t)row * K;\n"
"    float8 acc = (float8)(0.0f);\n"
"\n"
"    int num_v8 = K / 8;\n"
"    for (int v = tid; v < num_v8; v += 64) {\n"
"        char8 raw_w = vload8(v, rW);\n"
"        float8 w = convert_float8(raw_w);\n"
"        float8 x = vload8(v, X);\n"
"        acc = mad(w, x, acc);\n"
"    }\n"
"\n"
"    float thread_sum = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;\n"
"\n"
"    int tail_start = num_v8 * 8;\n"
"    for (int k = tail_start + tid; k < K; k += 64) {\n"
"        thread_sum += ((float)rW[k]) * X[k];\n"
"    }\n"
"\n"
"    __local float lds[64];\n"
"    lds[tid] = thread_sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for (int offset = 32; offset > 0; offset >>= 1) {\n"
"        if (tid < offset) {\n"
"            lds[tid] += lds[tid + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    if (tid == 0) {\n"
"        Y[row] = lds[0] * scales[row];\n"
"    }\n"
"}\n"
"\n"
"/* 2. Contextual Sparsity Gather-GEMV (skips inactive rows in GDDR6) */\n"
"__kernel void gemv_rdna3_contextual_gather(\n"
"    __global const char* restrict W,\n"
"    __global const float* restrict X,\n"
"    __global const float* restrict scales,\n"
"    __global float* restrict Y,\n"
"    __global const int* restrict active_rows,\n"
"    const int num_active,\n"
"    const int M,\n"
"    const int K\n"
") {\n"
"    int gid = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    if (gid >= num_active) return;\n"
"\n"
"    int row = active_rows[gid];\n"
"    if (row < 0 || row >= M) return;\n"
"\n"
"    __global const char* rW = W + (size_t)row * K;\n"
"    float8 acc = (float8)(0.0f);\n"
"\n"
"    int num_v8 = K / 8;\n"
"    for (int v = tid; v < num_v8; v += 64) {\n"
"        char8 raw_w = vload8(v, rW);\n"
"        float8 w = convert_float8(raw_w);\n"
"        float8 x = vload8(v, X);\n"
"        acc = mad(w, x, acc);\n"
"    }\n"
"\n"
"    float thread_sum = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;\n"
"\n"
"    int tail_start = num_v8 * 8;\n"
"    for (int k = tail_start + tid; k < K; k += 64) {\n"
"        thread_sum += ((float)rW[k]) * X[k];\n"
"    }\n"
"\n"
"    __local float lds[64];\n"
"    lds[tid] = thread_sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for (int offset = 32; offset > 0; offset >>= 1) {\n"
"        if (tid < offset) {\n"
"            lds[tid] += lds[tid + offset];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    if (tid == 0) {\n"
"        Y[row] = lds[0] * scales[row];\n"
"    }\n"
"}\n"
"\n"
"/* 3. Fast Vector Zeroing */\n"
"__kernel void rdna3_zero_vector(\n"
"    __global float* restrict vec,\n"
"    const int size\n"
") {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < size) {\n"
"        vec[idx] = 0.0f;\n"
"    }\n"
"}\n"
"\n"
"/* 4. Hardware SwiGLU / Top-K Activation Gate */\n"
"__kernel void rdna3_activation_gate(\n"
"    __global const float* restrict gate,\n"
"    __global const float* restrict up,\n"
"    __global float* restrict out_h,\n"
"    const float threshold,\n"
"    const int size\n"
") {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < size) {\n"
"        float g = gate[idx];\n"
"        if (fabs(g) >= threshold) {\n"
"            out_h[idx] = g * up[idx];\n"
"        } else {\n"
"            out_h[idx] = 0.0f;\n"
"        }\n"
"    }\n"
"}\n";

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_dense;
    cl_kernel kernel_gather;
    cl_kernel kernel_zero;
    cl_kernel kernel_gate;
    cl_device_id device;
    char device_name[256];
} rdna3_context_internal;

typedef struct {
    cl_mem buffer;
    size_t size_bytes;
    int rows;
    int cols;
    int type; /* 0 = fp32 vec, 1 = int8 weights, 2 = int32 indices */
} rdna3_tensor_internal;

RDNA3_API const char* RDNA3_CALL rdna3_status_string(int status) {
    switch (status) {
        case RDNA3_SUCCESS:                  return "Success";
        case RDNA3_ERR_INVALID_ARG:          return "Invalid argument (NULL pointer or non-positive dimension)";
        case RDNA3_ERR_OPENCL_LOAD_FAILED:   return "Failed to load OpenCL runtime library (OpenCL.dll / libOpenCL.so)";
        case RDNA3_ERR_NO_PLATFORM:          return "No OpenCL platform found";
        case RDNA3_ERR_NO_DEVICE:            return "No compatible GPU device found";
        case RDNA3_ERR_HOST_ALLOC_FAILED:    return "Host memory allocation failed";
        case RDNA3_ERR_CONTEXT_FAILED:       return "Failed to create OpenCL context";
        case RDNA3_ERR_QUEUE_FAILED:         return "Failed to create OpenCL command queue";
        case RDNA3_ERR_PROGRAM_BUILD_FAILED: return "Failed to compile RDNA 3 OpenCL compute shaders";
        case RDNA3_ERR_KERNEL_FAILED:        return "Failed to create OpenCL kernel object";
        case RDNA3_ERR_DIMENSION_MISMATCH:   return "Tensor dimension mismatch in kernel dispatch";
        case RDNA3_ERR_EXECUTION_FAILED:     return "OpenCL kernel execution failed";
        case RDNA3_ERR_VRAM_ALLOC_FAILED:    return "GPU VRAM buffer allocation failed";
        case RDNA3_ERR_TRANSFER_FAILED:      return "Host-to-GPU or GPU-to-Host memory transfer failed";
        default:                             return "Unknown error code";
    }
}

RDNA3_API int RDNA3_CALL rdna3_init(rdna3_context_t* ctx_out) {
    if (!ctx_out) return RDNA3_ERR_INVALID_ARG;
    *ctx_out = NULL;

    if (!load_opencl()) return RDNA3_ERR_OPENCL_LOAD_FAILED;

    cl_platform_id platforms[8];
    cl_uint num_plats = 0;
    if (clGetPlatformIDs(8, platforms, &num_plats) != 0 || num_plats == 0) return RDNA3_ERR_NO_PLATFORM;

    cl_device_id selected_dev = NULL;
    char dev_name[256] = {0};

    for (cl_uint p = 0; p < num_plats; p++) {
        cl_device_id devs[8];
        cl_uint num_devs = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 8, devs, &num_devs) == 0 && num_devs > 0) {
            for (cl_uint d = 0; d < num_devs; d++) {
                char name[256] = {0};
                clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
                if (strstr(name, "Radeon") || strstr(name, "AMD") || strstr(name, "gfx")) {
                    selected_dev = devs[d];
                    strncpy(dev_name, name, sizeof(dev_name) - 1);
                    break;
                }
                if (!selected_dev) {
                    selected_dev = devs[d];
                    strncpy(dev_name, name, sizeof(dev_name) - 1);
                }
            }
        }
        if (selected_dev && (strstr(dev_name, "Radeon") || strstr(dev_name, "AMD") || strstr(dev_name, "gfx"))) break;
    }

    if (!selected_dev) return RDNA3_ERR_NO_DEVICE;

    rdna3_context_internal* state = (rdna3_context_internal*)calloc(1, sizeof(rdna3_context_internal));
    if (!state) return RDNA3_ERR_HOST_ALLOC_FAILED;

    state->device = selected_dev;
    strncpy(state->device_name, dev_name, sizeof(state->device_name) - 1);

    cl_int err;
    state->ctx = clCreateContext(NULL, 1, &selected_dev, NULL, NULL, &err);
    if (err != 0) { free(state); return RDNA3_ERR_CONTEXT_FAILED; }

    cl_ulong props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    state->queue = clCreateCommandQueueWithProperties(state->ctx, selected_dev, props, &err);
    if (err != 0) {
        clReleaseContext(state->ctx);
        free(state);
        return RDNA3_ERR_QUEUE_FAILED;
    }

    state->program = clCreateProgramWithSource(state->ctx, 1, &RDNA3_KERNEL_SOURCE, NULL, &err);
    const char* opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    if (clBuildProgram(state->program, 1, &selected_dev, opts, NULL, NULL) != 0) {
        char log[4096] = {0};
        clGetProgramBuildInfo(state->program, selected_dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "[RDNA3 AI ENGINE] OpenCL Build Error:\n%s\n", log);
        clReleaseProgram(state->program);
        clReleaseCommandQueue(state->queue);
        clReleaseContext(state->ctx);
        free(state);
        return RDNA3_ERR_PROGRAM_BUILD_FAILED;
    }

    cl_int err1 = 0, err2 = 0, err3 = 0, err4 = 0;
    state->kernel_dense  = clCreateKernel(state->program, "gemv_rdna3_infinity_stream", &err1);
    state->kernel_gather = clCreateKernel(state->program, "gemv_rdna3_contextual_gather", &err2);
    state->kernel_zero   = clCreateKernel(state->program, "rdna3_zero_vector", &err3);
    state->kernel_gate   = clCreateKernel(state->program, "rdna3_activation_gate", &err4);

    if (err1 != 0 || err2 != 0 || err3 != 0 || err4 != 0) {
        rdna3_shutdown((rdna3_context_t)state);
        return RDNA3_ERR_KERNEL_FAILED;
    }

    *ctx_out = (rdna3_context_t)state;
    return RDNA3_SUCCESS;
}

RDNA3_API const char* RDNA3_CALL rdna3_get_device_name(rdna3_context_t ctx) {
    if (!ctx) return "None";
    return ((rdna3_context_internal*)ctx)->device_name;
}

RDNA3_API void RDNA3_CALL rdna3_shutdown(rdna3_context_t ctx) {
    if (!ctx) return;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    if (state->kernel_dense)  clReleaseKernel(state->kernel_dense);
    if (state->kernel_gather) clReleaseKernel(state->kernel_gather);
    if (state->kernel_zero)   clReleaseKernel(state->kernel_zero);
    if (state->kernel_gate)   clReleaseKernel(state->kernel_gate);
    if (state->program)       clReleaseProgram(state->program);
    if (state->queue)         clReleaseCommandQueue(state->queue);
    if (state->ctx)           clReleaseContext(state->ctx);
    free(state);
}

RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_weights_int8(
    rdna3_context_t ctx,
    int rows,
    int cols,
    const signed char* host_data
) {
    if (!ctx || rows <= 0 || cols <= 0) return NULL;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    size_t bytes = (size_t)rows * cols * sizeof(signed char);

    cl_int err;
    cl_mem buf = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != 0) return NULL;

    if (host_data) {
        err = clEnqueueWriteBuffer(state->queue, buf, 1, 0, bytes, host_data, 0, NULL, NULL);
        if (err != 0) {
            clReleaseMemObject(buf);
            return NULL;
        }
    }

    rdna3_tensor_internal* t = (rdna3_tensor_internal*)malloc(sizeof(rdna3_tensor_internal));
    if (!t) { clReleaseMemObject(buf); return NULL; }

    t->buffer = buf;
    t->size_bytes = bytes;
    t->rows = rows;
    t->cols = cols;
    t->type = 1;
    return (rdna3_tensor_t)t;
}

RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_vector_fp32(
    rdna3_context_t ctx,
    int size,
    const float* host_data
) {
    if (!ctx || size <= 0) return NULL;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    size_t bytes = (size_t)size * sizeof(float);

    cl_int err;
    cl_mem buf = clCreateBuffer(state->ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != 0) return NULL;

    if (host_data) {
        err = clEnqueueWriteBuffer(state->queue, buf, 1, 0, bytes, host_data, 0, NULL, NULL);
        if (err != 0) {
            clReleaseMemObject(buf);
            return NULL;
        }
    }

    rdna3_tensor_internal* t = (rdna3_tensor_internal*)malloc(sizeof(rdna3_tensor_internal));
    if (!t) { clReleaseMemObject(buf); return NULL; }

    t->buffer = buf;
    t->size_bytes = bytes;
    t->rows = size;
    t->cols = 1;
    t->type = 0;
    return (rdna3_tensor_t)t;
}

RDNA3_API rdna3_tensor_t RDNA3_CALL rdna3_create_indices_int32(
    rdna3_context_t ctx,
    int max_indices,
    const int* host_indices
) {
    if (!ctx || max_indices <= 0) return NULL;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    size_t bytes = (size_t)max_indices * sizeof(int);

    cl_int err;
    cl_mem buf = clCreateBuffer(state->ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if (err != 0) return NULL;

    if (host_indices) {
        err = clEnqueueWriteBuffer(state->queue, buf, 1, 0, bytes, host_indices, 0, NULL, NULL);
        if (err != 0) {
            clReleaseMemObject(buf);
            return NULL;
        }
    }

    rdna3_tensor_internal* t = (rdna3_tensor_internal*)malloc(sizeof(rdna3_tensor_internal));
    if (!t) { clReleaseMemObject(buf); return NULL; }

    t->buffer = buf;
    t->size_bytes = bytes;
    t->rows = max_indices;
    t->cols = 1;
    t->type = 2;
    return (rdna3_tensor_t)t;
}

RDNA3_API int RDNA3_CALL rdna3_quantize_fp32_to_int8(
    const float* fp32_weights,
    int rows,
    int cols,
    signed char* out_int8,
    float* out_scales
) {
    if (!fp32_weights || !out_int8 || !out_scales || rows <= 0 || cols <= 0) {
        return RDNA3_ERR_INVALID_ARG;
    }

    for (int r = 0; r < rows; r++) {
        const float* row_ptr = fp32_weights + (size_t)r * cols;
        signed char* q_ptr   = out_int8 + (size_t)r * cols;

        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = (float)fabs(row_ptr[c]);
            if (a > max_abs) max_abs = a;
        }

        if (max_abs < 1e-12f) {
            out_scales[r] = 0.0f;
            memset(q_ptr, 0, (size_t)cols * sizeof(signed char));
        } else {
            float scale = max_abs / 127.0f;
            float inv_scale = 127.0f / max_abs;
            out_scales[r] = scale;
            for (int c = 0; c < cols; c++) {
                float v = row_ptr[c] * inv_scale;
                int qi = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                q_ptr[c] = (signed char)qi;
            }
        }
    }
    return RDNA3_SUCCESS;
}

RDNA3_API int RDNA3_CALL rdna3_create_weights_from_fp32(
    rdna3_context_t ctx,
    int rows,
    int cols,
    const float* fp32_weights,
    rdna3_tensor_t* out_weights_int8,
    rdna3_tensor_t* out_scales_fp32
) {
    if (!ctx || !fp32_weights || !out_weights_int8 || !out_scales_fp32 || rows <= 0 || cols <= 0) {
        return RDNA3_ERR_INVALID_ARG;
    }
    *out_weights_int8 = NULL;
    *out_scales_fp32 = NULL;

    signed char* host_q = (signed char*)malloc((size_t)rows * cols * sizeof(signed char));
    float* host_s = (float*)malloc((size_t)rows * sizeof(float));
    if (!host_q || !host_s) {
        free(host_q);
        free(host_s);
        return RDNA3_ERR_HOST_ALLOC_FAILED;
    }

    int q_err = rdna3_quantize_fp32_to_int8(fp32_weights, rows, cols, host_q, host_s);
    if (q_err != RDNA3_SUCCESS) {
        free(host_q);
        free(host_s);
        return q_err;
    }

    rdna3_tensor_t d_w = rdna3_create_weights_int8(ctx, rows, cols, host_q);
    rdna3_tensor_t d_s = rdna3_create_vector_fp32(ctx, rows, host_s);

    free(host_q);
    free(host_s);

    if (!d_w || !d_s) {
        if (d_w) rdna3_free_tensor(ctx, d_w);
        if (d_s) rdna3_free_tensor(ctx, d_s);
        return RDNA3_ERR_VRAM_ALLOC_FAILED;
    }

    *out_weights_int8 = d_w;
    *out_scales_fp32 = d_s;
    return RDNA3_SUCCESS;
}

RDNA3_API int RDNA3_CALL rdna3_extract_active_indices(
    const float* host_activations,
    int size,
    float threshold,
    int* out_indices,
    int max_indices
) {
    if (!host_activations || !out_indices || size <= 0 || max_indices <= 0) return 0;
    int count = 0;
    for (int i = 0; i < size && count < max_indices; i++) {
        if ((float)fabs(host_activations[i]) >= threshold) {
            out_indices[count++] = i;
        }
    }
    return count;
}

RDNA3_API int RDNA3_CALL rdna3_write_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    const float* host_src,
    int count
) {
    if (!ctx || !tensor || !host_src || count <= 0) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)tensor;

    if (count > t->rows) return RDNA3_ERR_DIMENSION_MISMATCH;

    cl_int err = clEnqueueWriteBuffer(state->queue, t->buffer, 1, 0, (size_t)count * sizeof(float), host_src, 0, NULL, NULL);
    return (err == 0) ? RDNA3_SUCCESS : RDNA3_ERR_TRANSFER_FAILED;
}

RDNA3_API int RDNA3_CALL rdna3_write_indices_int32(
    rdna3_context_t ctx,
    rdna3_tensor_t indices_tensor,
    const int* host_indices,
    int num_active
) {
    if (!ctx || !indices_tensor || !host_indices || num_active <= 0) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)indices_tensor;

    if (num_active > t->rows) return RDNA3_ERR_DIMENSION_MISMATCH;

    cl_int err = clEnqueueWriteBuffer(state->queue, t->buffer, 1, 0, (size_t)num_active * sizeof(int), host_indices, 0, NULL, NULL);
    return (err == 0) ? RDNA3_SUCCESS : RDNA3_ERR_TRANSFER_FAILED;
}

RDNA3_API int RDNA3_CALL rdna3_read_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    float* host_dst
) {
    if (!ctx || !tensor || !host_dst) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)tensor;

    cl_int err = clEnqueueReadBuffer(state->queue, t->buffer, 1, 0, t->size_bytes, host_dst, 0, NULL, NULL);
    if (err != 0) return RDNA3_ERR_TRANSFER_FAILED;

    clFinish(state->queue);
    return RDNA3_SUCCESS;
}

RDNA3_API int RDNA3_CALL rdna3_zero_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor
) {
    if (!ctx || !tensor) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)tensor;

    int size = t->rows;
    clSetKernelArg(state->kernel_zero, 0, sizeof(cl_mem), &t->buffer);
    clSetKernelArg(state->kernel_zero, 1, sizeof(int), &size);

    cl_size_t local_size = 256;
    cl_size_t global_size = ((cl_size_t)(size + 255) / 256) * 256;

    cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_zero, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != 0) return RDNA3_ERR_EXECUTION_FAILED;

    clFinish(state->queue);
    return RDNA3_SUCCESS;
}

RDNA3_API void RDNA3_CALL rdna3_free_tensor(rdna3_context_t ctx, rdna3_tensor_t tensor) {
    (void)ctx;
    if (!tensor) return;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)tensor;
    if (t->buffer) clReleaseMemObject(t->buffer);
    free(t);
}

RDNA3_API int RDNA3_CALL rdna3_gemv_int8(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y
) {
    if (!ctx || !weights || !input_x || !scales || !output_y) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tW = (rdna3_tensor_internal*)weights;
    rdna3_tensor_internal* tX = (rdna3_tensor_internal*)input_x;
    rdna3_tensor_internal* tS = (rdna3_tensor_internal*)scales;
    rdna3_tensor_internal* tY = (rdna3_tensor_internal*)output_y;

    if (tW->cols != tX->rows || tW->rows != tS->rows || tW->rows != tY->rows) {
        return RDNA3_ERR_DIMENSION_MISMATCH;
    }

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel_dense, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel_dense, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel_dense, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel_dense, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel_dense, 4, sizeof(int), &M);
    clSetKernelArg(state->kernel_dense, 5, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)M * local_size;

    cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_dense, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != 0) return RDNA3_ERR_EXECUTION_FAILED;

    clFinish(state->queue);
    return RDNA3_SUCCESS;
}

RDNA3_API int RDNA3_CALL rdna3_gather_gemv_int8(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y,
    rdna3_tensor_t active_indices,
    int num_active
) {
    if (!ctx || !weights || !input_x || !scales || !output_y || !active_indices || num_active <= 0) {
        return RDNA3_ERR_INVALID_ARG;
    }
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tW   = (rdna3_tensor_internal*)weights;
    rdna3_tensor_internal* tX   = (rdna3_tensor_internal*)input_x;
    rdna3_tensor_internal* tS   = (rdna3_tensor_internal*)scales;
    rdna3_tensor_internal* tY   = (rdna3_tensor_internal*)output_y;
    rdna3_tensor_internal* tIdx = (rdna3_tensor_internal*)active_indices;

    if (tW->cols != tX->rows || tW->rows != tS->rows || tW->rows != tY->rows || num_active > tIdx->rows) {
        return RDNA3_ERR_DIMENSION_MISMATCH;
    }

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel_gather, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel_gather, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel_gather, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel_gather, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel_gather, 4, sizeof(cl_mem), &tIdx->buffer);
    clSetKernelArg(state->kernel_gather, 5, sizeof(int), &num_active);
    clSetKernelArg(state->kernel_gather, 6, sizeof(int), &M);
    clSetKernelArg(state->kernel_gather, 7, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)num_active * local_size;

    cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_gather, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != 0) return RDNA3_ERR_EXECUTION_FAILED;

    clFinish(state->queue);
    return RDNA3_SUCCESS;
}

RDNA3_API int RDNA3_CALL rdna3_gate_activations_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t gate_vec,
    rdna3_tensor_t up_vec,
    rdna3_tensor_t output_h,
    float threshold
) {
    if (!ctx || !gate_vec || !up_vec || !output_h) return RDNA3_ERR_INVALID_ARG;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tG = (rdna3_tensor_internal*)gate_vec;
    rdna3_tensor_internal* tU = (rdna3_tensor_internal*)up_vec;
    rdna3_tensor_internal* tH = (rdna3_tensor_internal*)output_h;

    if (tG->rows != tU->rows || tG->rows != tH->rows) {
        return RDNA3_ERR_DIMENSION_MISMATCH;
    }

    int size = tG->rows;
    clSetKernelArg(state->kernel_gate, 0, sizeof(cl_mem), &tG->buffer);
    clSetKernelArg(state->kernel_gate, 1, sizeof(cl_mem), &tU->buffer);
    clSetKernelArg(state->kernel_gate, 2, sizeof(cl_mem), &tH->buffer);
    clSetKernelArg(state->kernel_gate, 3, sizeof(float), &threshold);
    clSetKernelArg(state->kernel_gate, 4, sizeof(int), &size);

    cl_size_t local_size = 256;
    cl_size_t global_size = ((cl_size_t)(size + 255) / 256) * 256;

    cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_gate, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != 0) return RDNA3_ERR_EXECUTION_FAILED;

    clFinish(state->queue);
    return RDNA3_SUCCESS;
}

RDNA3_API double RDNA3_CALL rdna3_benchmark_gemv(
    rdna3_context_t ctx,
    rdna3_tensor_t weights,
    rdna3_tensor_t input_x,
    rdna3_tensor_t scales,
    rdna3_tensor_t output_y,
    int iterations,
    double* out_bandwidth_gb_s
) {
    if (!ctx || !weights || !input_x || !scales || !output_y || iterations <= 0) return 0.0;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tW = (rdna3_tensor_internal*)weights;
    rdna3_tensor_internal* tX = (rdna3_tensor_internal*)input_x;
    rdna3_tensor_internal* tS = (rdna3_tensor_internal*)scales;
    rdna3_tensor_internal* tY = (rdna3_tensor_internal*)output_y;

    if (tW->cols != tX->rows || tW->rows != tS->rows || tW->rows != tY->rows) return 0.0;

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel_dense, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel_dense, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel_dense, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel_dense, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel_dense, 4, sizeof(int), &M);
    clSetKernelArg(state->kernel_dense, 5, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)M * local_size;

    for (int w = 0; w < 5; w++) {
        clEnqueueNDRangeKernel(state->queue, state->kernel_dense, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }
    clFinish(state->queue);

    cl_ulong total_time_ns = 0;
    int valid_runs = 0;

    for (int i = 0; i < iterations; i++) {
        cl_event evt = NULL;
        cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_dense, 1, NULL, &global_size, &local_size, 0, NULL, &evt);
        if (err != 0 || !evt) continue;

        clWaitForEvents(1, &evt);

        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        if (end > start) {
            total_time_ns += (end - start);
            valid_runs++;
        }
        clReleaseEvent(evt);
    }

    if (valid_runs == 0) return 0.0;

    double avg_time_us = (double)total_time_ns / (valid_runs * 1000.0);
    double total_bytes = (double)M * K * sizeof(char) + (double)K * sizeof(float) + (double)M * sizeof(float) * 2.0;
    double time_s = (avg_time_us * 1e-6);
    double bw = (total_bytes / time_s) / 1e9;

    if (out_bandwidth_gb_s) *out_bandwidth_gb_s = bw;
    return avg_time_us;
}

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
) {
    if (!ctx || !weights || !input_x || !scales || !output_y || !active_indices || num_active <= 0 || iterations <= 0) {
        return 0.0;
    }
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tW   = (rdna3_tensor_internal*)weights;
    rdna3_tensor_internal* tX   = (rdna3_tensor_internal*)input_x;
    rdna3_tensor_internal* tS   = (rdna3_tensor_internal*)scales;
    rdna3_tensor_internal* tY   = (rdna3_tensor_internal*)output_y;
    rdna3_tensor_internal* tIdx = (rdna3_tensor_internal*)active_indices;

    if (tW->cols != tX->rows || tW->rows != tS->rows || tW->rows != tY->rows || num_active > tIdx->rows) {
        return 0.0;
    }

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel_gather, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel_gather, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel_gather, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel_gather, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel_gather, 4, sizeof(cl_mem), &tIdx->buffer);
    clSetKernelArg(state->kernel_gather, 5, sizeof(int), &num_active);
    clSetKernelArg(state->kernel_gather, 6, sizeof(int), &M);
    clSetKernelArg(state->kernel_gather, 7, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)num_active * local_size;

    for (int w = 0; w < 5; w++) {
        clEnqueueNDRangeKernel(state->queue, state->kernel_gather, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }
    clFinish(state->queue);

    cl_ulong total_time_ns = 0;
    int valid_runs = 0;

    for (int i = 0; i < iterations; i++) {
        cl_event evt = NULL;
        cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel_gather, 1, NULL, &global_size, &local_size, 0, NULL, &evt);
        if (err != 0 || !evt) continue;

        clWaitForEvents(1, &evt);

        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        if (end > start) {
            total_time_ns += (end - start);
            valid_runs++;
        }
        clReleaseEvent(evt);
    }

    if (valid_runs == 0) return 0.0;

    double avg_time_us = (double)total_time_ns / (valid_runs * 1000.0);
    double time_s = (avg_time_us * 1e-6);

    double physical_bytes = (double)num_active * K * sizeof(char) + (double)K * sizeof(float) + (double)num_active * sizeof(float) * 3.0;
    double dense_bytes    = (double)M * K * sizeof(char) + (double)K * sizeof(float) + (double)M * sizeof(float) * 2.0;

    if (out_physical_bw_gb_s)  *out_physical_bw_gb_s  = (physical_bytes / time_s) / 1e9;
    if (out_effective_bw_gb_s) *out_effective_bw_gb_s = (dense_bytes / time_s) / 1e9;

    return avg_time_us;
}
