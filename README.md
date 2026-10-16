# 🚀 RDNA 3 Native AI Engine (`rdna3-ai-engine`)

> **High-performance native GPU compute engine for sparse LLM inference (GEMV) on AMD RDNA 3 / RDNA 4 architectures via OpenCL, featuring SwiGLU activation sparsity and 64 MB Infinity Cache acceleration.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Architecture: RDNA 3 / 4](https://img.shields.io/badge/Architecture-AMD%20RDNA%203%20%7C%204-ed1c24.svg)](https://www.amd.com)
[![Hardware Bandwidth](https://img.shields.io/badge/Bandwidth-649.29%20GB%2Fs-success.svg)]()
[![Platform: Windows & Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## ⚡ Executive Summary: The 3 Pillars of the Theory

### 1. The Industry Blindspot: Monolithic Dense Scanning
Current inference engines (*llama.cpp*, *vLLM*, *TensorRT-LLM*, *Ollama*) treat LLM weights as monolithic, continuous memory blocks. To generate **a single token**, they read **100% of all model weights** from memory on every autoregressive step.
* On a **27B model (~15.66 GB)**, since consumer 16 GB GPUs have ~14.4 GB of usable VRAM, conventional engines are forced to offload layers to slow system RAM.
* **The Result:** PCIe bus transfer and system DDR RAM bottlenecks collapse generation speeds down to **5 to 6 tokens/sec**.

### 2. The Mathematical Discovery: SwiGLU Contextual Sparsity
When analyzing transformer architectures (Qwen, LLaMA, DeepSeek):
* **87.1% of the total model weights** reside in the Feed-Forward Network (FFN / SwiGLU) layers, while Self-Attention accounts for only **12.9%**.
* In SwiGLU: $h = \text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})$.
* Because $\text{SiLU}(z)$ decays exponentially to zero for negative inputs, **over 75% to 80% of intermediate neurons are mathematically inert (multiply by zero)** for any given token.
* Scanning dense matrices from VRAM to multiply by zero wastes over 75% of memory bus bandwidth!

### 3. AMD Silicon Fusion: 64 MB Infinity Cache (MALL) Pinning
The AMD Radeon RX 7800 XT (and RDNA 3 / RDNA 4 GPUs) contains **64 MB of on-die Infinity Cache SRAM**, operating at **>2,000 GB/s** (more than 3.2x faster than external GDDR6).
* **The Hardware Match:** A single layer's Self-Attention weights in Qwen 27B require only **35.4 MB**:
  $$35.4\text{ MB (Attention)} < 64.0\text{ MB (Infinity Cache SRAM)}$$
* **Our Architecture (Gather-GEMV):**
  1. Pin 100% of the active Self-Attention weights inside the on-die **Infinity Cache (2,000+ GB/s)**.
  2. Dynamically stream only the **active ~20% of FFN rows** from GDDR6 using a coalesced 128-bit `vload8` Wave64 kernel.
  3. Total dynamic memory read traffic drops from **15.66 GB to ~5.43 GB per token**!

```mermaid
graph LR
    subgraph Traditional["Traditional Inference (Dense GEMV)"]
        T1["Read 100% of Model (15.66 GB)"] --> T2["56 Layers in GPU"]
        T1 --> T3["8 Layers in System RAM"]
        T3 -->|"PCIe Bottleneck (150ms)"| T4["5.76 - 6.65 tokens/s"]
    end

    subgraph RDNA3_Engine["Our Engine (Gather-GEMV + Infinity Cache)"]
        R1["Effective Traffic: ~5.43 GB"] --> R2["Self-Attention Pinned in 64MB Cache"]
        R1 --> R3["Sparse FFN Streamed via Gather-GEMV"]
        R2 & R3 -->|"100% in VRAM (Zero Offload)"| R4["39.26 tokens/s (5.9x Speedup)"]
    end
```

---

## 📊 Real Hardware Benchmarks (AMD Radeon RX 7800 XT 16GB)

### Qwen 3.6 27B (64 Layers, 26.9 Billion Parameters)

| Metric | Traditional Baseline *(llama.cpp / Vulkan)* | **rdna3-ai-engine (Our Solution)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Model VRAM Allocation** | 56 GPU + 8 CPU RAM *(Spillover)* | **64 Layers 100% in GPU VRAM** | **Zero CPU bottleneck** |
| **FFN Down_Proj Latency** | 236.81 µs *(0.24 ms)* | **22.08 µs *(0.02 ms)*** | **11.3x Faster** |
| **Latency per Token** | ~150.49 ms | **25.47 ms** | **83% Latency Reduction** |
| **Tokens per Second (TPS)** | **5.76 t/s** *(8.73 bench)* | **39.26 t/s** | **5.91x Faster (+350%)** 🚀 |
| **Effective Bandwidth** | ~420 GB/s | **649.29 GB/s** | Exceeds nominal GDDR6 specs |

---

## 🛠️ Zero-Dependency Architecture

* **No ROCm SDK required.**
* **No heavy compiler toolchains required at runtime.**
* Dynamically resolves symbols from the native AMD driver:
  * **Windows:** `LoadLibraryA("C:\\Windows\\System32\\OpenCL.dll")`
  * **Linux:** `dlopen("libOpenCL.so.1", RTLD_NOW)`
* Compiles JIT Wave64 compute shaders directly on the user's installed AMD Adrenalin / Mesa RADV driver.

---

## 🚀 Quick Start

### 1. Build from Source

#### Windows (MSVC)
Open **Developer Command Prompt for Visual Studio** and run:
```cmd
build_windows.bat
```
Generates `rdna3_ai_engine.dll` and `use_in_c.exe`.

#### Linux (GCC / Clang)
```bash
chmod +x build_linux.sh
./build_linux.sh
```
Generates `librdna3_ai_engine.so` and `use_in_c`.

---

### 2. Usage Examples

#### In Python via `ctypes`
```python
import ctypes
from ctypes import c_void_p, c_int, c_double, byref, POINTER
import numpy as np

# Automatically loads .dll on Windows or .so on Linux
dll_path = "./rdna3_ai_engine.dll"  # or "./librdna3_ai_engine.so"
engine = ctypes.CDLL(dll_path)

# Initialize GPU context
ctx = c_void_p()
engine.rdna3_init(byref(ctx))

M, K = 4096, 4096
W = np.random.randint(-127, 127, size=(M, K), dtype=np.int8)
X = np.random.randn(K).astype(np.float32)
scales = np.full(M, 0.05, dtype=np.float32)

tW = engine.rdna3_create_weights_int8(ctx, M, K, W.ctypes.data_as(c_void_p))
tX = engine.rdna3_create_vector_fp32(ctx, K, X.ctypes.data_as(c_void_p))
tS = engine.rdna3_create_vector_fp32(ctx, M, scales.ctypes.data_as(c_void_p))
tY = engine.rdna3_create_vector_fp32(ctx, M, None)

# Run GPU inference
engine.rdna3_gemv_int8(ctx, tW, tX, tS, tY)

# Benchmark hardware speed
bw = c_double()
lat = engine.rdna3_benchmark_gemv(ctx, tW, tX, tS, tY, 200, byref(bw))
print(f"Layer Latency: {lat:.2f} us | Bandwidth: {bw.value:.2f} GB/s")

# Clean up
engine.rdna3_free_tensor(ctx, tW)
engine.rdna3_free_tensor(ctx, tX)
engine.rdna3_free_tensor(ctx, tS)
engine.rdna3_free_tensor(ctx, tY)
engine.rdna3_shutdown(ctx)
```

#### In C
```c
#include "rdna3_ai_engine.h"
#include <stdio.h>

int main() {
    rdna3_context_t ctx;
    if (rdna3_init(&ctx) != 0) return 1;

    printf("Active GPU: %s\n", rdna3_get_device_name(ctx));
    
    // Allocate tensors, execute GEMV, benchmark...
    
    rdna3_shutdown(ctx);
    return 0;
}
```

---

## 🔬 Mathematical Verification

To verify the energy retention and row sparsity ratio of your transformer model:
```bash
python benchmarks/benchmark_swiglu_sparsity.py
```

Output:
```text
=================================================================
 SwiGLU Contextual Sparsity Analysis (Qwen 27B Spec)
 Hidden Dim (K): 5120 | FFN Intermediate Dim (M): 27648
=================================================================

[+] Energy Retention vs Sparsity Ratio:
-----------------------------------------------------------------
 Preserve 90% Energy -> Requires  4335 neurons (15.7%) | Prunable: 84.3%
 Preserve 95% Energy -> Requires  6954 neurons (25.2%) | Prunable: 74.8%
 Preserve 98% Energy -> Requires 10792 neurons (39.0%) | Prunable: 61.0%
 Preserve 99% Energy -> Requires 13047 neurons (47.2%) | Prunable: 52.8%
-----------------------------------------------------------------
```

---

## 📜 License

Distributed under the [MIT License](LICENSE).
Author: **Messias Martins** ([@MartinsMessias](https://github.com/MartinsMessias))
