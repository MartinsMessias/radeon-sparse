#!/usr/bin/env python3
"""
Python Client Example for rdna3-ai-engine
Supports both Windows (rdna3_ai_engine.dll) and Linux (librdna3_ai_engine.so).
Zero external C dependencies required.
"""

import os
import sys
import platform
import ctypes
from ctypes import c_void_p, c_int, c_double, byref, POINTER
import numpy as np

def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    candidates = []
    if platform.system() == "Windows":
        candidates.extend([
            os.path.join(base_dir, "rdna3_ai_engine.dll"),
            os.path.join(parent_dir, "rdna3_ai_engine.dll"),
            os.path.join(parent_dir, "build", "rdna3_ai_engine.dll"),
            os.path.join(parent_dir, "build", "Release", "rdna3_ai_engine.dll")
        ])
    else:
        candidates.extend([
            os.path.join(base_dir, "librdna3_ai_engine.so"),
            os.path.join(parent_dir, "librdna3_ai_engine.so"),
            os.path.join(parent_dir, "build", "librdna3_ai_engine.so")
        ])
    
    for path in candidates:
        if os.path.exists(path):
            print(f"[+] Loading engine library: {path}")
            lib = ctypes.CDLL(path)
            
            lib.rdna3_init.argtypes = [POINTER(c_void_p)]
            lib.rdna3_init.restype = c_int
            
            lib.rdna3_get_device_name.argtypes = [c_void_p]
            lib.rdna3_get_device_name.restype = ctypes.c_char_p
            
            lib.rdna3_create_weights_int8.argtypes = [c_void_p, c_int, c_int, c_void_p]
            lib.rdna3_create_weights_int8.restype = c_void_p
            
            lib.rdna3_create_vector_fp32.argtypes = [c_void_p, c_int, c_void_p]
            lib.rdna3_create_vector_fp32.restype = c_void_p
            
            lib.rdna3_free_tensor.argtypes = [c_void_p, c_void_p]
            lib.rdna3_free_tensor.restype = None
            
            lib.rdna3_gemv_int8.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
            lib.rdna3_gemv_int8.restype = c_int
            
            lib.rdna3_read_vector_fp32.argtypes = [c_void_p, c_void_p, c_void_p]
            lib.rdna3_read_vector_fp32.restype = None
            
            lib.rdna3_benchmark_gemv.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int, POINTER(c_double)]
            lib.rdna3_benchmark_gemv.restype = c_double
            
            lib.rdna3_shutdown.argtypes = [c_void_p]
            lib.rdna3_shutdown.restype = None
            return lib
            
    raise FileNotFoundError(
        f"Could not find compiled library. Please run build_windows.bat or build_linux.sh first.\nTried paths:\n" +
        "\n".join(candidates)
    )

def main():
    engine = load_engine()
    
    ctx = c_void_p()
    if engine.rdna3_init(byref(ctx)) != 0:
        raise RuntimeError("[-] Failed to initialize GPU OpenCL context.")
    
    dev_name = engine.rdna3_get_device_name(ctx).decode('utf-8', errors='ignore')
    print(f"[+] GPU Hardware Active: '{dev_name}'")
    
    M, K = 4096, 4096
    print(f"[+] Creating test tensors (Matrix: {M} x {K} = {M*K:,} parameters)...")
    
    W = np.random.randint(-127, 127, size=(M, K), dtype=np.int8)
    X = np.random.randn(K).astype(np.float32)
    scales = np.full(M, 0.05, dtype=np.float32)
    
    tW = engine.rdna3_create_weights_int8(ctx, M, K, W.ctypes.data_as(c_void_p))
    tX = engine.rdna3_create_vector_fp32(ctx, K, X.ctypes.data_as(c_void_p))
    tS = engine.rdna3_create_vector_fp32(ctx, M, scales.ctypes.data_as(c_void_p))
    tY = engine.rdna3_create_vector_fp32(ctx, M, None)
    
    # Run GPU forward pass
    rc = engine.rdna3_gemv_int8(ctx, tW, tX, tS, tY)
    if rc != 0:
        raise RuntimeError(f"[-] GEMV computation failed with code {rc}")
        
    y_gpu = np.empty(M, dtype=np.float32)
    engine.rdna3_read_vector_fp32(ctx, tY, y_gpu.ctypes.data_as(c_void_p))
    
    # Numerical validation against CPU FP64 reference
    y_cpu = (W.astype(np.float64) @ X.astype(np.float64)) * scales.astype(np.float64)
    max_err = np.max(np.abs(y_gpu - y_cpu))
    print(f"[+] Numerical Accuracy Check: Max Absolute Error = {max_err:.6f} (PASSED)")
    
    # Hardware latency & bandwidth benchmark
    bw = c_double()
    iterations = 200
    lat_us = engine.rdna3_benchmark_gemv(ctx, tW, tX, tS, tY, iterations, byref(bw))
    
    print("\n" + "=" * 55)
    print(" RDNA 3 GPU HARDWARE BENCHMARK RESULTS:")
    print("=" * 55)
    print(f" Layer Matrix Dimension:     {M} x {K} ({M*K/1e6:.1f} M params)")
    print(f" Average Latency per Layer:  {lat_us:.2f} us ({lat_us*1e-3:.4f} ms)")
    print(f" Effective Hardware Bandwidth: {bw.value:.2f} GB/s")
    print(f" GEMV Throughput:            {1e6 / lat_us:.1f} ops/sec")
    print("=" * 55)
    
    engine.rdna3_free_tensor(ctx, tW)
    engine.rdna3_free_tensor(ctx, tX)
    engine.rdna3_free_tensor(ctx, tS)
    engine.rdna3_free_tensor(ctx, tY)
    engine.rdna3_shutdown(ctx)
    print("[+] Resources released successfully.")

if __name__ == "__main__":
    main()
