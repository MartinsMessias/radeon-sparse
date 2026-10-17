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

---

## 4. End-to-End Token Generation (`llama-bench` Format on Reference GPU)

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
