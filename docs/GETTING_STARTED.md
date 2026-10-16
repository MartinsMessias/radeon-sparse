# Getting Started with RDNA 3 AI Engine

This guide walks you through system prerequisites, obtaining or building the library, and integrating it into your C/C++ or Python projects on both **Windows** and **Linux**.

---

## 1. Hardware Requirements

* **GPU:** AMD Radeon RX 7000 Series (RDNA 3 architecture) or RDNA 3.5 APU:
  * **Navi 31:** Radeon RX 7900 XTX, RX 7900 XT, RX 7900 GRE (`gfx1100`)
  * **Navi 32:** Radeon RX 7800 XT, RX 7700 XT (`gfx1101`)
  * **Navi 33:** Radeon RX 7600 XT, RX 7600 (`gfx1102`)
  * **Strix Point APU:** AMD Ryzen AI 300 series (`gfx1103`)

---

## 2. Windows Environment

One of the primary benefits of this engine is that **it does not require the ROCm SDK on Windows**. Standard consumer gaming drivers are sufficient.

### Prerequisites:
1. **AMD Adrenalin Driver:** Standard GPU driver installed from AMD's official website (provides `C:\Windows\System32\OpenCL.dll` out of the box).
2. **Visual Studio C++ (for building from source):** Visual Studio 2019, 2022, or 2026 Community with the *"Desktop development with C++"* workload.
3. **Python (optional):** Python 3.8+ with `numpy` installed (`pip install numpy`).

### How to Build on Windows:
1. Clone the repository:
   ```cmd
   git clone https://github.com/SEU_USUARIO/rdna3-ai-engine.git
   cd rdna3-ai-engine
   ```
2. Run the 1-click build script:
   ```cmd
   build.bat
   ```
3. Compiled artifacts will be located in the `bin/` directory:
   * `bin/rdna3_ai_engine.dll` (Shared library)
   * `bin/rdna3_ai_engine.lib` (Import library)
   * `bin/c_example.exe` (Standalone C benchmark)

### Running Examples on Windows:
* **Native C:**
  ```cmd
  bin\c_example.exe
  ```
* **Python (ctypes):**
  ```cmd
  python bindings\python\example.py
  ```

---

## 3. Linux Environment (Ubuntu, Debian, Fedora, Arch)

### Prerequisites:
Install the standard OpenCL runtime provided by AMD or Mesa:

* **Ubuntu / Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential cmake rocm-opencl-runtime
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -S base-devel cmake opencl-amd
  # Or via Mesa Rustic:
  # sudo pacman -S mesa-opencl
  ```
* **Fedora:**
  ```bash
  sudo dnf install -y gcc cmake make rocm-opencl
  ```

### How to Build on Linux:
```bash
git clone https://github.com/SEU_USUARIO/rdna3-ai-engine.git
cd rdna3-ai-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Examples on Linux:
* **Native C:**
  ```bash
  ./c_example
  ```
* **Python:**
  ```bash
  python3 ../bindings/python/example.py
  ```

---

## 4. How to Obtain the Pre-Compiled Binaries

If you do not wish to compile the library locally:
1. Go to the **Releases** page of this repository on GitHub.
2. Download the latest release package: `rdna3-ai-engine-windows-x64.zip` or `rdna3-ai-engine-linux-x64.tar.gz`.
3. Extract `rdna3_ai_engine.dll` (Windows) or `librdna3_ai_engine.so` (Linux).
4. Place the file alongside your application or in your system library path.

---

## 5. Integrating with Projects

### In Python (Zero External Dependencies)
```python
import ctypes
import numpy as np

# Load library
lib = ctypes.CDLL("./rdna3_ai_engine.dll")

# Setup signatures
lib.rdna3_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.rdna3_init.restype = ctypes.c_int

# Initialize context
ctx = ctypes.c_void_p()
lib.rdna3_init(ctypes.byref(ctx))
print("AMD RDNA 3 hardware engine initialized!")
```

### In C / C++
```c
#include "rdna3_ai_engine.h"

int main() {
    rdna3_context_t ctx = NULL;
    if (rdna3_init(&ctx) != 0) {
        return -1;
    }
    
    // Allocate tensors and execute GEMV operations
    // ...
    
    rdna3_shutdown(ctx);
    return 0;
}
```
