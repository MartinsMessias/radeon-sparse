# 🔌 Drop-In Replacement Guide (`ggml-vulkan.dll` Proxy for Ollama, LM Studio & `llama.cpp`)

**RadeonSparse (`radeon_sparse.dll`)** natively exports all 9 `ggml_backend_vk_*` ABI symbols required by `llama.cpp`, **Ollama**, and **LM Studio**, allowing it to act as a transparent **Drop-In Proxy (`ggml-vulkan.dll`)** without triggering a CPU fallback!

---

## 1. How the Built-In `ggml-vulkan.dll` Proxy Works

When an application such as `llama-server.exe`, `llama-cli.exe`, **Ollama**, or **LM Studio** starts, it loads `ggml-vulkan.dll` and queries these 9 exported symbols:
- `ggml_backend_init`
- `ggml_backend_vk_reg`
- `ggml_backend_vk_init`
- `ggml_backend_is_vk`
- `ggml_backend_vk_buffer_type`
- `ggml_backend_vk_host_buffer_type`
- `ggml_backend_vk_get_device_count`
- `ggml_backend_vk_get_device_description`
- `ggml_backend_vk_get_device_memory`

Because `radeon_sparse.dll` implements and exports all 9 functions (in `src/radeon_sparse.c`):
1. You rename the original `ggml-vulkan.dll` to **`ggml-vulkan.dll.original.bak`**.
2. You copy **`radeon_sparse.dll`** into the folder and rename it to **`ggml-vulkan.dll`**.
3. When the host app loads `ggml-vulkan.dll`, **RadeonSparse** automatically:
   - Initializes its in-process OpenCL Wavefront Engine (`[RadeonSparse Drop-In Proxy] Initialized OpenCL Engine on <GPU>`).
   - Tags the GPU description with `[RadeonSparse]` so you can verify in the logs that the proxy is active.
   - Forwards standard Vulkan backend graph operations to `ggml-vulkan.dll.original.bak` so all 50+ GGML operators run at full GPU speed with zero CPU fallback.

---

## 2. One-Click Drop-In Installation (`install_ggml_dropin.bat`)

You can install the Drop-In Proxy into **any** `llama.cpp`, **Ollama**, or **LM Studio** directory with a single command:

```cmd
:: For a local llama.cpp folder:
install_ggml_dropin.bat "C:\Path\To\llama-cpp\bin"

:: For Ollama (Windows):
install_ggml_dropin.bat "%LOCALAPPDATA%\Programs\Ollama\lib\ollama"

:: For LM Studio (Windows):
install_ggml_dropin.bat "%LOCALAPPDATA%\Programs\LM Studio\resources"
```

---

## 3. Manual Installation Steps

```text
[Target Application Directory]
├── llama-server.exe (or Ollama / LM Studio)
├── ggml-base.dll
├── ggml-vulkan.dll.original.bak   <-- 1. Original file renamed
└── ggml-vulkan.dll                <-- 2. Copy of bin\radeon_sparse.dll
```
