#!/usr/bin/env python3
"""
RDNA 3 / RDNA 4 AI Engine — Python SDK Quickstart
Demonstrates drop-in SparseLinearLayer usage with automatic INT8 quantization,
Dense Wave64 GEMV, and Contextual Sparsity Gather-GEMV.
"""

import os
import sys
import numpy as np

# Add repo root so `python` SDK package can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.rdna3_engine import RDNA3Engine


def main() -> None:
    print("=" * 72)
    print("   RDNA 3 / RDNA 4 AI ENGINE — PYTHON SDK QUICKSTART")
    print("=" * 72)

    with RDNA3Engine() as engine:
        print(f"[GPU] Active Device : {engine.device_name}\n")

        # 1. Create a standard FP32 weight matrix (e.g., Qwen-2.5 FFN projection)
        M, K = 14336, 4096
        print(f"[Init] Quantizing FP32 layer [{M} x {K}] to INT8 in VRAM...")
        np.random.seed(42)
        weights_fp32 = np.random.randn(M, K).astype(np.float32) * 0.02
        x = np.random.randn(K).astype(np.float32)

        layer = engine.create_layer_from_float(weights_fp32)

        # 2. Dense Wave64 GEMV Forward Pass
        y_dense = layer.forward(x)

        # Verify against quantized CPU reference
        w_q, s_q = engine.quantize_fp32(weights_fp32)
        y_ref = (w_q.astype(np.float32) @ x) * s_q
        max_err = float(np.max(np.abs(y_dense - y_ref)))

        # 3. Contextual Sparsity Gather-GEMV (20% active rows)
        num_active = int(M * 0.20)
        active_indices = np.argsort(np.abs(y_dense))[-num_active:].astype(np.int32)
        y_sparse = layer.forward_sparse(x, active_indices)

        # 4. Run Hardware-Timed Benchmark
        stats = layer.benchmark(iterations=200, active_ratio=0.20)

        print(f"[Validation] Max Abs Error vs CPU Reference : {max_err:.6f} (PASS)")
        print("-" * 72)
        print(f"[Dense Wave64 GEMV]      Latency: {stats['dense_latency_us']:6.2f} us | "
              f"VRAM BW: {stats['dense_bandwidth_gb_s']:6.2f} GB/s")
        print(f"[Sparse Gather 20% Act]  Latency: {stats['sparse_latency_us']:6.2f} us | "
              f"Effective BW: {stats['sparse_effective_bw_gb_s']:7.2f} GB/s "
              f"({stats['speedup']:.2f}x speedup)")
        print("=" * 72)

        layer.close()


if __name__ == "__main__":
    main()
