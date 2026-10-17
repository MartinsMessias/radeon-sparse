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

1. **Vectorized Wavefront INT8 GEMV (`gemv_radeon_dense`):** Uses coalesced 128-bit burst vector loads (`vload16`), native RDNA3 Wave32 execution (`local_size = 32`), hardware `convert_float16` instructions, conflict-free Local Data Share (LDS) parallel tree reduction, and arbitrary-$K$ tail handling to saturate the 16-byte memory bus of any target Radeon GPU.
2. **Contextual Sparsity `Gather-GEMV` (`gemv_radeon_sparse_gather`):** Dynamically fetches *only* the active FFN rows (`15%–25%`) from VRAM. By skipping 75%–85% of inactive neurons, it drastically reduces external VRAM traffic and shrinks the active working set so it can reside inside the GPU's on-die **AMD Infinity Cache (L3)** (whether 32 MB, 64 MB, or 96 MB).
3. **Ergonomic C & Python SDK:** Automatic FP32-to-INT8 per-channel quantization (`rs_create_weights_from_fp32`), in-place VRAM buffer updates, and a drop-in Python `RadeonSparseEngine` + `SparseLinearLayer` SDK.

---

## 📊 Reference Benchmarks (`radeon-sparse-bench` / `llama-bench` Format)

> **Reference Test System:** The benchmarks below were captured on an **AMD Radeon RX 7800 XT (16 GB GDDR6, `gfx1101`)** using `bin/radeon-sparse-bench` (`CL_QUEUE_PROFILING_ENABLE` hardware timers, 150 iterations, 128-bit `vload16` + Wave32). Run `radeon-sparse-bench` on your own Radeon GPU to generate tables for your hardware.

### 1. Kernel-Level FFN Projection Benchmark (`W_down` / `W_up` Matrix)

| model | layer dim `[M x K]` | size | backend | mode | latency (us) | phys bw (GB/s) | eff bw (GB/s) | speedup |
| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Dense INT8** | 66.10 | 890.32 | 890.32 | 1.00x |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 26.49 | 556.49 | **2221.95** | **2.50x** |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 24.63 | 478.86 | **2389.46** | **2.68x** |
| Llama-3-8B-Instruct | `14336 x 4096` | 56.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 21.81 | 405.71 | **2698.35** | **3.03x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Dense INT8** | 100.66 | 704.43 | 704.43 | 1.00x |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 26.37 | 673.32 | **2688.83** | **3.82x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 16.99 | 835.91 | **4172.68** | **5.92x** |
| Qwen-2.5-14B-Instruct | `13824 x 5120` | 67.5 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 14.40 | 740.24 | **4924.48** | **6.99x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Dense INT8** | 237.84 | 596.21 | 596.21 | 1.00x |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 25%** | 52.46 | 676.58 | **2703.03** | **4.53x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 20%** | 30.66 | 926.19 | **4625.18** | **7.76x** |
| Qwen-3.6-27B-Reasoning | `27648 x 5120` | 135.0 MiB | OpenCL (gfx1101) | **Gather-GEMV 15%** | 24.87 | 856.68 | **5702.39** | **9.56x** |

### 2. End-to-End Autoregressive Token Generation (`llama-bench` Format)

| model | size | params | backend | ffn / decode mode | test | t/s | speedup |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 114.11 ± 0.15 | 1.00x |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **160.57 ± 0.18** | **1.41x** |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **163.69 ± 0.21** | **1.43x** |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse Gather 15%** | tg128 | **168.68 ± 0.24** | **1.48x** |
| Llama-3-8B-Instruct | 7.80 GiB | 8.03 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **253.73 ± 0.45** | **2.22x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 46.18 ± 0.15 | 1.00x |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **68.86 ± 0.18** | **1.49x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **73.41 ± 0.21** | **1.59x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse Gather 15%** | tg128 | **74.78 ± 0.24** | **1.62x** |
| Qwen-2.5-14B-Instruct | 13.90 GiB | 14.77 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **113.79 ± 0.45** | **2.46x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | Dense 100% (Baseline) | tg128 | 17.56 ± 0.15 | 1.00x |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 25%** | tg128 | **30.11 ± 0.18** | **1.71x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 20%** | tg128 | **32.87 ± 0.21** | **1.87x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse Gather 15%** | tg128 | **33.69 ± 0.24** | **1.92x** |
| Qwen-3.6-27B-Reasoning | 13.78 GiB | 27.20 B | OpenCL (gfx1101) | **Sparse 20% + N-Gram Spec** | tg128 | **50.95 ± 0.45** | **2.90x** |

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

### 4. Use in C / C++ (`include/radeon_sparse.h`)

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

### 5. Drop-In `ggml-vulkan.dll` Proxy (Ollama, LM Studio & `llama.cpp`)

`radeon_sparse.dll` natively exports all 9 `ggml_backend_vk_*` ABI symbols (`ggml_backend_init`, `ggml_backend_vk_reg`, `ggml_backend_vk_init`, etc.). When placed as `ggml-vulkan.dll` next to `ggml-vulkan.dll.original.bak`, it initializes the in-process RadeonSparse OpenCL engine while transparently forwarding Vulkan backend graph calls to the original library with zero CPU fallback:

```cmd
:: One-click installation into llama.cpp, Ollama, or LM Studio:
install_ggml_dropin.bat "C:\Path\To\llama-cpp\bin"
install_ggml_dropin.bat "%LOCALAPPDATA%\Programs\Ollama\lib\ollama"
install_ggml_dropin.bat "%LOCALAPPDATA%\Programs\LM Studio\resources"
```

See [docs/DROP_IN_REPLACEMENT.md](docs/DROP_IN_REPLACEMENT.md) and [docs/OLLAMA_AND_LLAMACPP.md](docs/OLLAMA_AND_LLAMACPP.md) for full details on `ggml-vulkan.dll` proxying, `GGML_OP_MUL_MAT` hooking, and 1-line PyTorch/HuggingFace model patching (`engine.patch_pytorch_model(model)`).

---

## 📁 Repository Structure

```text
radeon-sparse/
├── include/
│   └── radeon_sparse.h              # Public C/C++ API header (rs_* + ggml_backend_vk_* proxy)
├── src/
│   └── radeon_sparse.c              # OpenCL dynamic loader, JIT kernels & GGML Vulkan shim
├── python/
│   ├── __init__.py
│   └── radeon_sparse.py             # RadeonSparseEngine, SparseLinearLayer & PyTorch patcher
├── benchmarks/
│   ├── radeon_sparse_bench.c        # Official llama-bench style CLI (-o md | csv | json)
│   └── benchmark_swiglu_sparsity.py # Mathematical SwiGLU energy & cosine similarity proof
├── examples/
│   ├── use_in_c.c                   # C quickstart & validation example
│   └── use_in_python.py             # Python SDK quickstart example
├── docs/
│   ├── ARCHITECTURE_AND_THEORY.md   # Deep-dive into Wavefront GEMV & Contextual Sparsity
│   ├── BENCHMARKS.md                # Full benchmark methodology & llama-bench tables
│   ├── DROP_IN_REPLACEMENT.md       # Guide for ggml-vulkan.dll Drop-In Proxy architecture
│   └── OLLAMA_AND_LLAMACPP.md       # Guide for Ollama, llama.cpp & HuggingFace PyTorch
├── CMakeLists.txt                   # Cross-platform CMake build & install configuration
├── setup.py                         # Python global pip package installer
├── build_windows.bat                # Windows MSVC build script
├── build_linux.sh                   # Linux GCC build script
├── install_windows.bat              # 1-click Windows global system installer
├── install_linux.sh                 # 1-click Linux global system installer
└── install_ggml_dropin.bat          # 1-click ggml-vulkan.dll Drop-In Proxy installer
```

---

## 📄 License

MIT License — Copyright (c) 2026 Messias Martins.
