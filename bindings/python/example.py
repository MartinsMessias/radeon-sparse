"""
Python Bindings and Example for RDNA 3 AI Engine
Hardware-optimized inference on AMD Radeon (Wave64 Coalesced Infinity-Streaming & Contextual Gather-GEMV)
"""

import os
import sys
import ctypes
from ctypes import c_int, c_float, c_double, c_void_p, POINTER, byref, CDLL
import numpy as np

def find_library():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "bin", "rdna3_ai_engine.dll"),
        os.path.join(script_dir, "..", "..", "build", "rdna3_ai_engine.dll"),
        os.path.join(script_dir, "rdna3_ai_engine.dll"),
        "rdna3_ai_engine.dll"
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return candidates[0]

def main():
    lib_path = find_library()
    print("=" * 70)
    print(" RDNA 3 AI Engine - Python Example (ctypes)")
    print(f" Loading DLL from: {lib_path}")
    print("=" * 70)

    if not os.path.exists(lib_path):
        print(f"[ERROR] Could not find {lib_path}. Please build the project first.")
        return

    lib = CDLL(lib_path)

    # Function signatures
    lib.rdna3_init.argtypes = [POINTER(c_void_p)]
    lib.rdna3_init.restype = c_int

    lib.rdna3_shutdown.argtypes = [c_void_p]

    lib.rdna3_create_weights_int8.argtypes = [c_void_p, c_int, c_int, c_void_p]
    lib.rdna3_create_weights_int8.restype = c_void_p

    lib.rdna3_create_vector_fp32.argtypes = [c_void_p, c_int, c_void_p]
    lib.rdna3_create_vector_fp32.restype = c_void_p

    lib.rdna3_create_indices_int32.argtypes = [c_void_p, c_int, c_void_p]
    lib.rdna3_create_indices_int32.restype = c_void_p

    lib.rdna3_zero_vector_fp32.argtypes = [c_void_p, c_void_p]

    lib.rdna3_free_tensor.argtypes = [c_void_p, c_void_p]

    lib.rdna3_gemv_int8.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    lib.rdna3_gemv_int8.restype = c_int

    lib.rdna3_gather_gemv_int8.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_void_p]
    lib.rdna3_gather_gemv_int8.restype = c_int

    lib.rdna3_read_vector_fp32.argtypes = [c_void_p, c_void_p, c_void_p]

    lib.rdna3_benchmark_gemv.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, POINTER(c_double)]
    lib.rdna3_benchmark_gemv.restype = c_double

    lib.rdna3_benchmark_gather_gemv.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_void_p, c_int, POINTER(c_double)]
    lib.rdna3_benchmark_gather_gemv.restype = c_double

    lib.rdna3_gate_activations_fp32.argtypes = [c_void_p, c_void_p, c_float, c_void_p, POINTER(c_int)]
    lib.rdna3_gate_activations_fp32.restype = c_int

    # 1. Initialize GPU
    ctx = c_void_p()
    ret = lib.rdna3_init(byref(ctx))
    if ret != 0:
        raise RuntimeError(f"Failed to initialize RDNA 3 context (code {ret})")

    print("[OK] GPU context initialized successfully on AMD Radeon RX 7800 XT.")

    # 2. Setup Data: Qwen 27B FFN Layer (27648 x 5120)
    M, K = 27648, 5120
    active_count = M // 5  # 20% active neurons
    print(f"[Setup] Allocating Qwen 27B FFN Tensor: {M} x {K} (135.0 MB)...")

    weights = np.random.randint(-127, 127, size=(M, K), dtype=np.int8)
    x_input = np.random.randn(K).astype(np.float32)
    scales = np.full(M, 0.05, dtype=np.float32)
    active_indices = np.arange(0, M, 5, dtype=np.int32)

    tW = lib.rdna3_create_weights_int8(ctx, M, K, weights.ctypes.data_as(c_void_p))
    tX = lib.rdna3_create_vector_fp32(ctx, K, x_input.ctypes.data_as(c_void_p))
    tS = lib.rdna3_create_vector_fp32(ctx, M, scales.ctypes.data_as(c_void_p))
    tY_dense = lib.rdna3_create_vector_fp32(ctx, M, None)
    tY_gather = lib.rdna3_create_vector_fp32(ctx, M, None)
    tIdx = lib.rdna3_create_indices_int32(ctx, active_count, active_indices.ctypes.data_as(c_void_p))

    lib.rdna3_zero_vector_fp32(ctx, tY_gather)

    # 3. Microbenchmark Dense vs Gather
    print("[Benchmarking] Running 100 iterations of Dense vs Gather-GEMV...")
    bw_dense = c_double()
    lat_dense = lib.rdna3_benchmark_gemv(ctx, tW, tX, tS, tY_dense, 100, byref(bw_dense))

    bw_gather = c_double()
    lat_gather = lib.rdna3_benchmark_gather_gemv(ctx, tW, tX, tS, tIdx, active_count, tY_gather, 100, byref(bw_gather))

    print("\n" + "=" * 70)
    print(" HARDWARE BENCHMARK COMPARISON (AMD Radeon RX 7800 XT)")
    print("=" * 70)
    print(f" Dense GEMV Latency:    {lat_dense:.2f} us ({lat_dense/1000.0:.3f} ms)")
    print(f" Gather-GEMV Latency:   {lat_gather:.2f} us ({lat_gather/1000.0:.3f} ms)")
    print(f" Real Hardware Speedup: {lat_dense / lat_gather:.2f}x FASTER")
    print(f" Bandwidth Saved:       {(1.0 - active_count / M) * 100.0:.1f}%")
    print("=" * 70 + "\n")

    # 4. Cleanup
    lib.rdna3_free_tensor(ctx, tW)
    lib.rdna3_free_tensor(ctx, tX)
    lib.rdna3_free_tensor(ctx, tS)
    lib.rdna3_free_tensor(ctx, tY_dense)
    lib.rdna3_free_tensor(ctx, tY_gather)
    lib.rdna3_free_tensor(ctx, tIdx)
    lib.rdna3_shutdown(ctx)
    print("[OK] Done.")

if __name__ == "__main__":
    main()
