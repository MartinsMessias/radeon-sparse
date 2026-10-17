# 🦙 Using RadeonSparse with Ollama, `llama.cpp`, and HuggingFace / PyTorch

Because different AI tools load models in different ways, here is how **RadeonSparse (`radeon_sparse.dll` / `libradeon_sparse.so`)** integrates with each ecosystem:

---

## 1. How Ollama Works Under the Hood (Important)

**Ollama** is a Go wrapper around **`llama.cpp` (`ggml`)**. When you run `ollama run qwen2.5:14b`, Ollama spawns a pre-compiled `llama-server` binary that uses its own internal `ggml-vulkan.dll` or `ggml-rocm.dll` matrix multiplication (`GGML_OP_MUL_MAT`).

Because pre-compiled `ollama.exe` does not dynamically load external third-party GEMV plugins by default, simply placing `radeon_sparse.dll` in `System32` will **not** automatically change how the stock `ollama.exe` binary computes matrices unless you use one of the two methods below:

### Method A: Run Ollama's GGUF Models via `llama-server` with N-Gram Speculative Decoding (Zero Recompile)
You can point `llama-server.exe` directly at Ollama's downloaded `.gguf` blob (`%USERPROFILE%\.ollama\models\blobs\sha256-...`) and enable **N-Gram Speculative Decoding** (`--spec-type ngram-simple`), which requires zero extra VRAM and accelerates structured/code/reasoning generation immediately:

```cmd
llama-server.exe -m "%USERPROFILE%\.ollama\models\blobs\sha256-<YOUR_MODEL_BLOB>" ^
    -c 32768 -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 ^
    --spec-type ngram-simple --draft-max 16 --draft-min 2 --port 8080
```

### Method B: Hook `radeon_sparse.dll` into `llama.cpp` (`ggml` Custom Backend Hook)
If you compile `llama.cpp` (or Ollama's `llama` runner) from source, you can route FFN projections (`ffn_up`, `ffn_down`) directly to `rs_gather_gemv_int8` inside `ggml`:

```c
#include "radeon_sparse.h"

/* Inside ggml backend MulMat dispatch when batch_size (ne11) == 1: */
if (is_ffn_layer && ne11 == 1) {
    int num_active = rs_extract_active_indices(src1_f32, K, threshold, active_idx, max_idx);
    rs_write_vector_fp32(rs_ctx, d_X, src1_f32, K);
    rs_write_indices_int32(rs_ctx, d_Idx, active_idx, num_active);
    rs_zero_vector_fp32(rs_ctx, d_Y);
    rs_gather_gemv_int8(rs_ctx, d_W, d_X, d_Scales, d_Y, d_Idx, num_active);
    rs_read_vector_fp32(rs_ctx, d_Y, dst_f32);
}
```

---

## 2. One-Line Auto-Patching for HuggingFace `transformers` / PyTorch

If you are running models in Python via **PyTorch / HuggingFace `transformers`**, `RadeonSparseEngine` includes a built-in **1-line auto-patcher** (`engine.patch_pytorch_model(model)`) that automatically finds `nn.Linear` FFN projections (`up_proj`, `down_proj`, `w2`, `w3`), quantizes their weights to INT8 in Radeon VRAM, and replaces their forward pass with `rs_gather_gemv_int8`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from radeon_sparse import RadeonSparseEngine

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 1 Line: Replace FFN Linear layers with RadeonSparse GPU Gather-GEMV!
engine = RadeonSparseEngine()
patched = engine.patch_pytorch_model(model, active_ratio=0.20)
print(f"Patched {patched} FFN layers onto {engine.device_name}!")
```
