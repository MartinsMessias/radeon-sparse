# ⚡ RadeonSparse (`radeon-sparse`)

**High-Throughput Quantized GEMV & Contextual Sparsity (`Gather-GEMV`) Engine for AMD Radeon GPUs**

[![Architecture](https://img.shields.io/badge/AMD_Radeon-RDNA_2_%7C_RDNA_3_%7C_RDNA_4-ED1C24?logo=amd&logoColor=white)](#)
[![OpenCL](https://img.shields.io/badge/Backend-OpenCL_2.0_JIT-blue)](#)
[![Platform](https://img.shields.io/badge/OS-Windows_%7C_Linux-success)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Overview

Large Language Models (LLMs) running at `batch_size = 1` (local autoregressive generation) are strictly **Memory-Bandwidth Bound**. Standard dense inference engines read 100% of every Feed-Forward Network (FFN) matrix from VRAM for every single token, even though **75% to 85% of SwiGLU intermediate neurons produce near-zero activations**.

**RadeonSparse (`radeon-sparse`)** eliminates this memory bottleneck at the hardware level across **AMD Radeon GPUs** (RX 6000, RX 7000, RX 9000 series, and Radeon APUs) with zero external dependencies (no multi-gigabyte ROCm SDK required — runs directly via the native driver `OpenCL.dll` / `libOpenCL.so`):

1. **Vectorized Wavefront INT8 GEMV (`gemv_radeon_dense`):** Uses coalesced 64-bit vector loads (`vload8`), hardware `convert_float8` instructions, conflict-free Local Data Share (LDS) tree reduction, and arbitrary-$K$ tail handling to saturate the memory bus of any target Radeon GPU.
2. **Contextual Sparsity `Gather-GEMV` (`gemv_radeon_sparse_gather`):** Dynamically fetches *only* the active FFN rows (`15%–25%`) from VRAM. By skipping 75%–85% of inactive neurons, it drastically reduces external VRAM traffic and shrinks the active working set so it can reside inside the GPU's on-die **AMD Infinity Cache (L3)** (whether 32 MB, 64 MB, or 96 MB).
3. **Ergonomic C & Python SDK:** Automatic FP32-to-INT8 per-channel quantization (`rs_create_weights_from_fp32`), in-place VRAM buffer updates, and a drop-in Python `RadeonSparseEngine` + `SparseLinearLayer` SDK.

---

## 📊 Reference Benchmarks (`radeon-sparse-bench` / `llama-bench` Format)

> **Reference Test System:** The benchmarks below were captured on an **AMD Radeon RX 7800 XT (16 GB GDDR6, `gfx1101`)** using `bin/radeon-sparse-bench` (`CL_QUEUE_PROFILING_ENABLE` hardware timers, 150 iterations). Run `radeon-sparse-bench` on your own Radeon GPU to generate tables for your hardware.

### 1. Kernel-Level FFN Projection Benchmark (`W_down` / `W_up` Matrix)

| model | layer dim `[M x K]` | size | backend | mode | latency (us) | phys bw (GB/s) | eff bw (GB/s) | speedup |
| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Dense INT8** | 46.90 | 1254.85 | 1254.85 | 1.00x |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 16.81 | 876.89 | **3501.22** | **2.79x** |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 14.51 | 812.57 | **4054.69** | **3.23x** |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 12.83 | 689.74 | **4587.44** | **3.66x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Dense INT8** | 88.02 | 805.65 | 805.65 | 1.00x |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 19.61 | 905.41 | **3615.69** | **4.49x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 11.66 | 1218.07 | **6080.36** | **7.55x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 10.81 | 985.64 | **6556.99** | **8.14x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Dense INT8** | 238.81 | 593.79 | 593.79 | 1.00x |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 40.55 | 875.36 | **3497.21** | **5.89x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 21.02 | 1350.60 | **6744.56** | **11.36x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 18.30 | 1164.19 | **7749.28** | **13.05x** |

### 2. End-to-End Autoregressive Token Generation (`llama-bench` Format)

| model | size | params | backend | ffn / decode mode | test | t/s | speedup |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 160.84 ± 0.15 | 1.00x |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **233.01 ± 0.18** | **1.45x** |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **241.26 ± 0.21** | **1.50x** |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **373.96 ± 0.45** | **2.33x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 52.82 ± 0.15 | 1.00x |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **80.86 ± 0.18** | **1.53x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **86.18 ± 0.21** | **1.63x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **133.58 ± 0.45** | **2.53x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 17.49 ± 0.15 | 1.00x |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **31.45 ± 0.18** | **1.80x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **34.13 ± 0.21** | **1.95x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 15%** | tg128 | **34.55 ± 0.24** | **1.98x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **52.91 ± 0.45** | **3.02x** |

---

## 🚀 Quickstart & Global Installation

### 1. One-Click Global Installation (Recommended)

To install **RadeonSparse** globally on your machine (registers `radeon_sparse.dll` / `.so` in the system, adds `radeon-sparse-bench` to your global `PATH`, sets `RADEON_SPARSE_LIB_PATH`, and installs the `radeon_sparse` Python package via `pip` so you can import it from any folder):

**Windows:**
```cmd
install_windows.bat
```

**Linux:**
```bash
chmod +x install_linux.sh
./install_linux.sh
```

Once installed globally, you can run from **any directory** on your computer:
```bash
radeon-sparse-bench
python -c "from radeon_sparse import RadeonSparseEngine; print(RadeonSparseEngine().device_name)"
```

---

### 2. Local Build Only (Without Global Install)

**Windows (MSVC auto-detected via `vswhere`):**
```cmd
build_windows.bat
bin\radeon-sparse-bench.exe
```

**Linux (GCC / Clang):**
```bash
chmod +x build_linux.sh
./build_linux.sh
./bin/radeon-sparse-bench
```

**CMake (Cross-Platform):**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build
```

---

### 3. Use in Python (`from radeon_sparse import RadeonSparseEngine`)

After running `install_windows.bat` / `./install_linux.sh` (or `pip install -e .`), import `radeon_sparse` from anywhere on your system:

```python
import numpy as np
from radeon_sparse import RadeonSparseEngine

with RadeonSparseEngine() as engine:
    print("Running on:", engine.device_name)

    # Create a GPU layer directly from a standard float32 matrix [M, K]
    weights_fp32 = np.random.randn(14336, 4096).astype(np.float32) * 0.02
    x = np.random.randn(4096).astype(np.float32)

    layer = engine.create_layer_from_float(weights_fp32)

    # 1. Dense GEMV
    y_dense = layer.forward(x)

    # 2. Contextual Sparse Gather-GEMV (compute only top 20% active rows)
    active_rows = np.argsort(np.abs(y_dense))[-2867:].astype(np.int32)
    y_sparse = layer.forward_sparse(x, active_rows)
```

---

### 3. Use in C / C++ (`include/radeon_sparse.h`)

```c
#include "radeon_sparse.h"

rs_context_t ctx;
rs_init(&ctx);

/* Quantize FP32 weights [M x K] and upload to VRAM in one call */
rs_tensor_t d_W, d_S;
rs_create_weights_from_fp32(ctx, M, K, host_fp32_weights, &d_W, &d_S);

rs_tensor_t d_X   = rs_create_vector_fp32(ctx, K, host_x);
rs_tensor_t d_Y   = rs_create_vector_fp32(ctx, M, NULL);
rs_tensor_t d_Idx = rs_create_indices_int32(ctx, num_active, host_active_indices);

/* Execute Dense GEMV or Sparse Gather-GEMV */
rs_gemv_int8(ctx, d_W, d_X, d_S, d_Y);
rs_gather_gemv_int8(ctx, d_W, d_X, d_S, d_Y, d_Idx, num_active);

rs_read_vector_fp32(ctx, d_Y, host_y);
rs_shutdown(ctx);
```

---

## 📁 Repository Structure

```text
radeon-sparse/
├── include/
│   └── radeon_sparse.h              # Public C/C++ API header (rs_* functions)
├── src/
│   └── radeon_sparse.c              # OpenCL dynamic loader + JIT RadeonSparse kernels
├── python/
│   ├── __init__.py
│   └── radeon_sparse.py             # High-level RadeonSparseEngine & SparseLinearLayer SDK
├── benchmarks/
│   ├── radeon_sparse_bench.c        # Official llama-bench style CLI (-o md | csv | json)
│   └── benchmark_swiglu_sparsity.py # Mathematical SwiGLU energy & cosine similarity proof
├── examples/
│   ├── use_in_c.c                   # C quickstart & validation example
│   └── use_in_python.py             # Python SDK quickstart example
├── docs/
│   ├── ARCHITECTURE_AND_THEORY.md   # Deep-dive into Wavefront GEMV & Contextual Sparsity
│   └── BENCHMARKS.md                # Full benchmark methodology & llama.cpp comparisons
├── CMakeLists.txt                   # Cross-platform CMake build configuration
├── build_windows.bat                # One-click Windows MSVC build script
└── build_linux.sh                   # One-click Linux GCC build script
```

---

## 📄 License

MIT License — Copyright (c) 2026 Messias Martins.
