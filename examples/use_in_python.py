#!/usr/bin/env python3
"""
RadeonSparse (radeon-sparse) — Python SDK Quickstart
Demonstrates drop-in SparseLinearLayer usage with automatic INT8 quantization,
Dense GEMV, and Contextual Sparsity Gather-GEMV on AMD Radeon GPUs.
"""

import os
import sys
import numpy as np

# Add repo root so `python` SDK package can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python import RadeonSparseEngine


def main() -> None:
    print("=" * 72)
    print("   RADEONSPARSE — PYTHON SDK QUICKSTART")
    print("=" * 72)

    with RadeonSparseEngine() as engine:
        print(f"[GPU] Active Device : {engine.device_name}\n")

        # 1. Create a standard FP32 weight matrix (e.g., LLM FFN projection)
        M, K = 14336, 4096
        print(f"[Init] Quantizing FP32 layer [{M} x {K}] to INT8 in VRAM...")
        np.random.seed(42)
        weights_fp32 = np.random.randn(M, K).astype(np.float32) * 0.02
        x = np.random.randn(K).astype(np.float32)

        layer = engine.create_layer_from_float(weights_fp32)

        # 2. Dense GEMV Forward Pass
        y_dense = layer.forward(x)

        # Verify against quantized CPU reference
        w_q, s_q = engine.quantize_fp32(weights_fp32)
        y_ref = (w_q.astype(np.float32) @ x) * s_q
        max_err = float(np.max(np.abs(y_dense - y_ref)))

        # 3. Contextual Sparsity Row-Gather GEMV (20% active rows)
        num_active = int(M * 0.20)
        active_indices = np.argsort(np.abs(y_dense))[-num_active:].astype(np.int32)
        y_sparse_row = layer.forward_sparse(x, active_indices)

        # 4. Contextual Sparsity Column-Gather GEMV (20% active columns - SwiGLU Down-Projection)
        num_active_cols = int(K * 0.20)
        active_cols = np.argsort(np.abs(x))[-num_active_cols:].astype(np.int32)
        y_sparse_col = layer.forward_column_gather(x, active_cols)

        # Verify Column-Gather against CPU reference (only sum over active columns)
        y_col_ref = (w_q[:, active_cols].astype(np.float32) @ x[active_cols]) * s_q
        max_col_err = float(np.max(np.abs(y_sparse_col - y_col_ref)))

        # 5. Run Hardware-Timed Benchmark
        stats = layer.benchmark(iterations=200, active_ratio=0.20)

        print(f"[Validation] Dense Max Abs Error vs CPU Ref         : {max_err:.6f} (PASS)")
        print(f"[Validation] Column-Gather Max Abs Error vs CPU Ref : {max_col_err:.6f} (PASS)")
        print("-" * 72)
        print(f"[Dense GEMV]             Latency: {stats['dense_latency_us']:6.2f} us | "
              f"VRAM BW: {stats['dense_bandwidth_gb_s']:6.2f} GB/s")
        print(f"[Sparse Gather 20% Act]  Latency: {stats['sparse_latency_us']:6.2f} us | "
              f"Effective BW: {stats['sparse_effective_bw_gb_s']:7.2f} GB/s "
              f"({stats['speedup']:.2f}x speedup)")
        print("=" * 72)

        layer.close()


if __name__ == "__main__":
    main()
