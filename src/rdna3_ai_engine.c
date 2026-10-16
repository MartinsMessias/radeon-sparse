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
    #define CL_CALL_CONV WINAPI
#else
    #include <dlfcn.h>
    typedef void* lib_handle_t;
    #define LOAD_LIB(name) dlopen(name, RTLD_NOW | RTLD_GLOBAL)
    #define GET_SYM(lib, sym) dlsym(lib, sym)
    #define FREE_LIB(lib) dlclose(lib)
    #define OCL_CANDIDATE_1 "libOpenCL.so.1"
    #define OCL_CANDIDATE_2 "libOpenCL.so"
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
    if (!s_hCL) s_hCL = LOAD_LIB(OCL_CANDIDATE_2);
    if (!s_hCL) return 0;

    #define BIND_SYM(name) do { \
        name = (pfn_##name)GET_SYM(s_hCL, #name); \
        if (!name) return 0; \
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

/* Highly Optimized RDNA 3 Wave64 GEMV Compute Shader */
static const char* RDNA3_KERNEL_SOURCE =
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
"    __global const char* rW = W + row * K;\n"
"    float8 acc = (float8)(0.0f);\n"
"\n"
"    int num_v8 = K / 8;\n"
"    for (int v = tid; v < num_v8; v += 64) {\n"
"        __global const char* ptr = rW + v * 8;\n"
"        float8 w = (float8)(\n"
"            (float)ptr[0], (float)ptr[1], (float)ptr[2], (float)ptr[3],\n"
"            (float)ptr[4], (float)ptr[5], (float)ptr[6], (float)ptr[7]\n"
"        );\n"
"        float8 x = vload8(v, X);\n"
"        acc = mad(w, x, acc);\n"
"    }\n"
"\n"
"    __local float lds[64];\n"
"    lds[tid] = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    if (tid < 32) lds[tid] += lds[tid + 32];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (tid < 16) lds[tid] += lds[tid + 16];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (tid < 8)  lds[tid] += lds[tid + 8];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (tid < 4)  lds[tid] += lds[tid + 4];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    if (tid < 2)  lds[tid] += lds[tid + 2];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    if (tid == 0) {\n"
"        Y[row] = (lds[0] + lds[1]) * scales[row];\n"
"    }\n"
"}\n";

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    char device_name[256];
} rdna3_context_internal;

typedef struct {
    cl_mem buffer;
    size_t size_bytes;
    int rows;
    int cols;
    int is_weight;
} rdna3_tensor_internal;

RDNA3_API int RDNA3_CALL rdna3_init(rdna3_context_t* ctx_out) {
    if (!ctx_out) return -1;
    *ctx_out = NULL;

    if (!load_opencl()) return -2;

    cl_platform_id platforms[8];
    cl_uint num_plats = 0;
    if (clGetPlatformIDs(8, platforms, &num_plats) != 0 || num_plats == 0) return -3;

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
        if (selected_dev && (strstr(dev_name, "Radeon") || strstr(dev_name, "AMD"))) break;
    }

    if (!selected_dev) return -4;

    rdna3_context_internal* state = (rdna3_context_internal*)calloc(1, sizeof(rdna3_context_internal));
    if (!state) return -5;

    state->device = selected_dev;
    strncpy(state->device_name, dev_name, sizeof(state->device_name) - 1);

    cl_int err;
    state->ctx = clCreateContext(NULL, 1, &selected_dev, NULL, NULL, &err);
    if (err != 0) { free(state); return -6; }

    cl_ulong props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    state->queue = clCreateCommandQueueWithProperties(state->ctx, selected_dev, props, &err);
    if (err != 0) {
        clReleaseContext(state->ctx);
        free(state);
        return -7;
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
        return -8;
    }

    state->kernel = clCreateKernel(state->program, "gemv_rdna3_infinity_stream", &err);
    if (err != 0) {
        clReleaseProgram(state->program);
        clReleaseCommandQueue(state->queue);
        clReleaseContext(state->ctx);
        free(state);
        return -9;
    }

    *ctx_out = (rdna3_context_t)state;
    return 0;
}

RDNA3_API const char* RDNA3_CALL rdna3_get_device_name(rdna3_context_t ctx) {
    if (!ctx) return "None";
    return ((rdna3_context_internal*)ctx)->device_name;
}

RDNA3_API void RDNA3_CALL rdna3_shutdown(rdna3_context_t ctx) {
    if (!ctx) return;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    if (state->kernel) clReleaseKernel(state->kernel);
    if (state->program) clReleaseProgram(state->program);
    if (state->queue) clReleaseCommandQueue(state->queue);
    if (state->ctx) clReleaseContext(state->ctx);
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
        clEnqueueWriteBuffer(state->queue, buf, 1, 0, bytes, host_data, 0, NULL, NULL);
    }

    rdna3_tensor_internal* t = (rdna3_tensor_internal*)malloc(sizeof(rdna3_tensor_internal));
    if (!t) { clReleaseMemObject(buf); return NULL; }

    t->buffer = buf;
    t->size_bytes = bytes;
    t->rows = rows;
    t->cols = cols;
    t->is_weight = 1;
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
        clEnqueueWriteBuffer(state->queue, buf, 1, 0, bytes, host_data, 0, NULL, NULL);
    }

    rdna3_tensor_internal* t = (rdna3_tensor_internal*)malloc(sizeof(rdna3_tensor_internal));
    if (!t) { clReleaseMemObject(buf); return NULL; }

    t->buffer = buf;
    t->size_bytes = bytes;
    t->rows = size;
    t->cols = 1;
    t->is_weight = 0;
    return (rdna3_tensor_t)t;
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
    if (!ctx || !weights || !input_x || !scales || !output_y) return -1;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* tW = (rdna3_tensor_internal*)weights;
    rdna3_tensor_internal* tX = (rdna3_tensor_internal*)input_x;
    rdna3_tensor_internal* tS = (rdna3_tensor_internal*)scales;
    rdna3_tensor_internal* tY = (rdna3_tensor_internal*)output_y;

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel, 4, sizeof(int), &M);
    clSetKernelArg(state->kernel, 5, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)M * local_size;

    cl_int err = clEnqueueNDRangeKernel(state->queue, state->kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != 0) return err;

    clFinish(state->queue);
    return 0;
}

RDNA3_API void RDNA3_CALL rdna3_read_vector_fp32(
    rdna3_context_t ctx,
    rdna3_tensor_t tensor,
    float* host_dst
) {
    if (!ctx || !tensor || !host_dst) return;
    rdna3_context_internal* state = (rdna3_context_internal*)ctx;
    rdna3_tensor_internal* t = (rdna3_tensor_internal*)tensor;

    clEnqueueReadBuffer(state->queue, t->buffer, 1, 0, t->size_bytes, host_dst, 0, NULL, NULL);
    clFinish(state->queue);
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

    int M = tW->rows;
    int K = tW->cols;

    clSetKernelArg(state->kernel, 0, sizeof(cl_mem), &tW->buffer);
    clSetKernelArg(state->kernel, 1, sizeof(cl_mem), &tX->buffer);
    clSetKernelArg(state->kernel, 2, sizeof(cl_mem), &tS->buffer);
    clSetKernelArg(state->kernel, 3, sizeof(cl_mem), &tY->buffer);
    clSetKernelArg(state->kernel, 4, sizeof(int), &M);
    clSetKernelArg(state->kernel, 5, sizeof(int), &K);

    cl_size_t local_size = 64;
    cl_size_t global_size = (cl_size_t)M * local_size;

    /* Warmup */
    for (int w = 0; w < 5; w++) {
        clEnqueueNDRangeKernel(state->queue, state->kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }
    clFinish(state->queue);

    cl_ulong total_time_ns = 0;
    for (int i = 0; i < iterations; i++) {
        cl_event evt;
        clEnqueueNDRangeKernel(state->queue, state->kernel, 1, NULL, &global_size, &local_size, 0, NULL, &evt);
        clWaitForEvents(1, &evt);

        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        total_time_ns += (end - start);
        clReleaseEvent(evt);
    }

    double avg_time_us = (double)total_time_ns / (iterations * 1000.0);
    double total_bytes = (double)M * K * sizeof(char) + (double)K * sizeof(float) + (double)M * sizeof(float);
    double time_s = (avg_time_us * 1e-6);
    double bw = (total_bytes / time_s) / 1e9;

    if (out_bandwidth_gb_s) {
        *out_bandwidth_gb_s = bw;
    }
    return avg_time_us;
}
