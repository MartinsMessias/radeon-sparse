# 📊 Official Benchmarks & Methodology (`radeon-sparse-bench`)

All benchmarks in this repository follow the standard `llama-bench` reporting methodology and can be reproduced on any AMD Radeon GPU using the compiled `radeon-sparse-bench` binary.

---

## 1. Running the Official Benchmark Suite

```bash
# Default: GitHub Markdown tables
./bin/radeon-sparse-bench

# Export as CSV for spreadsheets / plotting
./bin/radeon-sparse-bench -o csv > results.csv

# Export as JSON for automated CI pipelines
./bin/radeon-sparse-bench -o json > results.json
```

---

## 2. Reference Test Hardware & Environment

- **Reference GPU Tested:** AMD Radeon RX 7800 XT (`gfx1101`, 16 GB GDDR6, 64 MB L3 Infinity Cache, 624.0 GB/s physical VRAM bus)
- **Timing Mechanism:** Hardware command-queue profiling (`CL_QUEUE_PROFILING_ENABLE`, `CL_PROFILING_COMMAND_START` $\to$ `CL_PROFILING_COMMAND_END`), excluding host OS jitter.

---

## 3. Kernel-Level FFN Projection Results (Reference GPU: `gfx1101`)

> **Architecture Configuration:** 128-bit memory bursts (`vload16` + `convert_float16`), Native RDNA3 Wave32 (`local_size = 32`), unrolled 32-element LDS binary tree reduction.

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

---

## 4. End-to-End Token Generation (`llama-bench` Format on Reference GPU)

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
