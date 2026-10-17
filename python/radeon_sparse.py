#!/usr/bin/env python3
"""
RadeonSparse (radeon-sparse) — High-Level Python SDK
Zero-dependency ctypes wrapper with automatic FP32->INT8 quantization,
VRAM lifecycle management, and drop-in SparseLinearLayer for LLMs on AMD Radeon GPUs.
"""

import os
import shutil
import ctypes
import ctypes.util
from typing import Optional, Tuple
import numpy as np


class RadeonSparseError(RuntimeError):
    """Exception raised when a RadeonSparse C API call returns an error."""
    pass


def _find_library(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    env_path = os.environ.get("RADEON_SPARSE_LIB_PATH")
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)

    lib_name = "radeon_sparse.dll" if os.name == "nt" else "libradeon_sparse.so"
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)

    candidates = [
        os.path.join(pkg_dir, lib_name),
        os.path.join(repo_root, "bin", lib_name),
        os.path.join(repo_root, "build", lib_name),
        os.path.join(repo_root, lib_name),
        os.path.abspath(lib_name),
    ]
    if os.name != "nt":
        candidates.extend([
            f"/usr/local/lib/{lib_name}",
            f"/usr/lib/{lib_name}",
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    # Search in global system PATH or OS dynamic linker cache
    which_lib = shutil.which(lib_name)
    if which_lib:
        return which_lib

    found = ctypes.util.find_library("radeon_sparse")
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find {lib_name}. Run install_windows.bat (Windows) or ./install_linux.sh (Linux) to configure globally."
    )


class RadeonSparseEngine:
    """
    High-level context manager for the RadeonSparse OpenCL GPU Engine.

    Example:
        with RadeonSparseEngine() as engine:
            layer = engine.create_layer_from_float(weights_fp32)
            y_dense = layer.forward(x)
            y_sparse = layer.forward_sparse(x, active_indices)
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = _find_library(lib_path)
        self._lib = ctypes.CDLL(self.lib_path)
        self._bind_signatures()

        self._ctx = ctypes.c_void_p()
        status = self._lib.rs_init(ctypes.byref(self._ctx))
        if status != 0 or not self._ctx:
            msg = self.status_string(status)
            raise RadeonSparseError(f"rs_init failed (code {status}): {msg}")

    def _bind_signatures(self) -> None:
        L = self._lib

        L.rs_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.rs_init.restype = ctypes.c_int

        L.rs_shutdown.argtypes = [ctypes.c_void_p]
        L.rs_shutdown.restype = None

        L.rs_get_device_name.argtypes = [ctypes.c_void_p]
        L.rs_get_device_name.restype = ctypes.c_char_p

        L.rs_status_string.argtypes = [ctypes.c_int]
        L.rs_status_string.restype = ctypes.c_char_p

        L.rs_create_weights_int8.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int8)
        ]
        L.rs_create_weights_int8.restype = ctypes.c_void_p

        L.rs_create_vector_fp32.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]
        L.rs_create_vector_fp32.restype = ctypes.c_void_p

        L.rs_create_indices_int32.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32)
        ]
        L.rs_create_indices_int32.restype = ctypes.c_void_p

        L.rs_create_weights_from_fp32.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        L.rs_create_weights_from_fp32.restype = ctypes.c_int

        L.rs_quantize_fp32_to_int8.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
        ]
        L.rs_quantize_fp32_to_int8.restype = ctypes.c_int

        L.rs_write_vector_fp32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        L.rs_write_vector_fp32.restype = ctypes.c_int

        L.rs_write_indices_int32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int
        ]
        L.rs_write_indices_int32.restype = ctypes.c_int

        L.rs_read_vector_fp32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
        ]
        L.rs_read_vector_fp32.restype = ctypes.c_int

        L.rs_zero_vector_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.rs_zero_vector_fp32.restype = ctypes.c_int

        L.rs_free_tensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.rs_free_tensor.restype = None

        L.rs_gemv_int8.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        L.rs_gemv_int8.restype = ctypes.c_int

        L.rs_gather_gemv_int8.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]
        L.rs_gather_gemv_int8.restype = ctypes.c_int

        L.rs_gate_activations_fp32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float
        ]
        L.rs_gate_activations_fp32.restype = ctypes.c_int

        L.rs_benchmark_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
        ]
        L.rs_benchmark_gemv.restype = ctypes.c_double

        L.rs_benchmark_gather_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
        ]
        L.rs_benchmark_gather_gemv.restype = ctypes.c_double

    @property
    def device_name(self) -> str:
        if not self._ctx:
            return "Closed"
        raw = self._lib.rs_get_device_name(self._ctx)
        return raw.decode("utf-8", errors="replace") if raw else "Unknown"

    def status_string(self, status: int) -> str:
        raw = self._lib.rs_status_string(status)
        return raw.decode("utf-8", errors="replace") if raw else f"Status {status}"

    def quantize_fp32(self, weights_fp32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantizes a 2D FP32 numpy array [M, K] into (weights_int8 [M, K], scales_fp32 [M])."""
        w = np.ascontiguousarray(weights_fp32, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"Expected 2D weight matrix, got {w.ndim}D")
        M, K = w.shape
        out_q = np.empty((M, K), dtype=np.int8)
        out_s = np.empty(M, dtype=np.float32)

        err = self._lib.rs_quantize_fp32_to_int8(
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, K,
            out_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out_s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if err != 0:
            raise RadeonSparseError(self.status_string(err))
        return out_q, out_s

    def create_layer_from_float(self, weights_fp32: np.ndarray) -> "SparseLinearLayer":
        """Creates a GPU-accelerated SparseLinearLayer directly from a float32 weight matrix [M, K]."""
        return SparseLinearLayer.from_float(self, weights_fp32)

    def create_layer_from_int8(self, weights_int8: np.ndarray, scales_fp32: np.ndarray) -> "SparseLinearLayer":
        """Creates a GPU-accelerated SparseLinearLayer from pre-quantized INT8 weights [M, K] and FP32 scales [M]."""
        return SparseLinearLayer.from_int8(self, weights_int8, scales_fp32)

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.rs_shutdown(self._ctx)
            self._ctx = None

    def __enter__(self) -> "RadeonSparseEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class SparseLinearLayer:
    """
    Drop-in GPU INT8 Linear / FFN Layer supporting both Dense GEMV
    and Contextual Sparsity Gather-GEMV with reusable pre-allocated VRAM buffers.
    """

    def __init__(
        self,
        engine: RadeonSparseEngine,
        rows: int,
        cols: int,
        d_w: ctypes.c_void_p,
        d_s: ctypes.c_void_p,
    ):
        self.engine = engine
        self.rows = rows
        self.cols = cols
        self._d_w = d_w
        self._d_s = d_s

        # Pre-allocate reusable input/output/index buffers in VRAM for zero-overhead token loops
        self._d_x = engine._lib.rs_create_vector_fp32(engine._ctx, cols, None)
        self._d_y = engine._lib.rs_create_vector_fp32(engine._ctx, rows, None)
        self._d_idx = engine._lib.rs_create_indices_int32(engine._ctx, rows, None)
        if not self._d_x or not self._d_y or not self._d_idx:
            self.close()
            raise RadeonSparseError("Failed to pre-allocate reusable VRAM vectors for SparseLinearLayer")

    @classmethod
    def from_float(cls, engine: RadeonSparseEngine, weights_fp32: np.ndarray) -> "SparseLinearLayer":
        w = np.ascontiguousarray(weights_fp32, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("weights_fp32 must be a 2D array [rows, cols]")
        rows, cols = w.shape
        d_w = ctypes.c_void_p()
        d_s = ctypes.c_void_p()
        err = engine._lib.rs_create_weights_from_fp32(
            engine._ctx,
            rows,
            cols,
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(d_w),
            ctypes.byref(d_s),
        )
        if err != 0:
            raise RadeonSparseError(engine.status_string(err))
        return cls(engine, rows, cols, d_w, d_s)

    @classmethod
    def from_int8(
        cls, engine: RadeonSparseEngine, weights_int8: np.ndarray, scales_fp32: np.ndarray
    ) -> "SparseLinearLayer":
        w = np.ascontiguousarray(weights_int8, dtype=np.int8)
        s = np.ascontiguousarray(scales_fp32, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("weights_int8 must be a 2D array [rows, cols]")
        rows, cols = w.shape
        if s.shape[0] != rows:
            raise ValueError(f"scales_fp32 length ({s.shape[0]}) must match rows ({rows})")

        d_w = engine._lib.rs_create_weights_int8(
            engine._ctx, rows, cols, w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        )
        d_s = engine._lib.rs_create_vector_fp32(
            engine._ctx, rows, s.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if not d_w or not d_s:
            if d_w:
                engine._lib.rs_free_tensor(engine._ctx, d_w)
            if d_s:
                engine._lib.rs_free_tensor(engine._ctx, d_s)
            raise RadeonSparseError("Failed to allocate weight/scale buffers in VRAM")
        return cls(engine, rows, cols, d_w, d_s)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes Dense Quantized GEMV: Y = (W @ x) * scales."""
        x_arr = np.ascontiguousarray(x, dtype=np.float32)
        if x_arr.size != self.cols:
            raise ValueError(f"Input size ({x_arr.size}) does not match layer cols ({self.cols})")

        L = self.engine._lib
        ctx = self.engine._ctx

        err = L.rs_write_vector_fp32(
            ctx, self._d_x, x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.cols
        )
        if err != 0:
            raise RadeonSparseError(self.engine.status_string(err))

        err = L.rs_gemv_int8(ctx, self._d_w, self._d_x, self._d_s, self._d_y)
        if err != 0:
            raise RadeonSparseError(self.engine.status_string(err))

        out = np.empty(self.rows, dtype=np.float32)
        err = L.rs_read_vector_fp32(
            ctx, self._d_y, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if err != 0:
            raise RadeonSparseError(self.engine.status_string(err))
        return out

    def forward_sparse(
        self,
        x: np.ndarray,
        active_indices: np.ndarray,
        zero_inactive: bool = True,
    ) -> np.ndarray:
        """
        Executes Contextual Sparse Gather-GEMV on ONLY the specified active_indices.
        Inactive rows are never fetched from VRAM!
        """
        x_arr = np.ascontiguousarray(x, dtype=np.float32)
        idx_arr = np.ascontiguousarray(active_indices, dtype=np.int32)
        num_active = idx_arr.size

        if x_arr.size != self.cols:
            raise ValueError(f"Input size ({x_arr.size}) does not match layer cols ({self.cols})")
        if num_active == 0:
            return np.zeros(self.rows, dtype=np.float32)

        L = self.engine._lib
        ctx = self.engine._ctx

        if zero_inactive:
            L.rs_zero_vector_fp32(ctx, self._d_y)

        L.rs_write_vector_fp32(
            ctx, self._d_x, x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.cols
        )
        L.rs_write_indices_int32(
            ctx, self._d_idx, idx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), num_active
        )

        err = L.rs_gather_gemv_int8(
            ctx, self._d_w, self._d_x, self._d_s, self._d_y, self._d_idx, num_active
        )
        if err != 0:
            raise RadeonSparseError(self.engine.status_string(err))

        out = np.empty(self.rows, dtype=np.float32)
        L.rs_read_vector_fp32(
            ctx, self._d_y, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out

    def benchmark(self, iterations: int = 200, active_ratio: float = 0.20) -> dict:
        """Runs hardware-timed Dense and Sparse Gather benchmarks on this layer."""
        L = self.engine._lib
        ctx = self.engine._ctx

        dense_bw = ctypes.c_double(0.0)
        dense_us = L.rs_benchmark_gemv(
            ctx, self._d_w, self._d_x, self._d_s, self._d_y, iterations, ctypes.byref(dense_bw)
        )

        num_active = max(1, int(self.rows * active_ratio))
        idx = np.linspace(0, self.rows - 1, num_active, dtype=np.int32)
        L.rs_write_indices_int32(
            ctx, self._d_idx, idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), num_active
        )

        phys_bw = ctypes.c_double(0.0)
        eff_bw = ctypes.c_double(0.0)
        sparse_us = L.rs_benchmark_gather_gemv(
            ctx, self._d_w, self._d_x, self._d_s, self._d_y,
            self._d_idx, num_active, iterations,
            ctypes.byref(phys_bw), ctypes.byref(eff_bw),
        )

        return {
            "rows": self.rows,
            "cols": self.cols,
            "active_ratio": active_ratio,
            "dense_latency_us": dense_us,
            "dense_bandwidth_gb_s": dense_bw.value,
            "sparse_latency_us": sparse_us,
            "sparse_physical_bw_gb_s": phys_bw.value,
            "sparse_effective_bw_gb_s": eff_bw.value,
            "speedup": (dense_us / sparse_us) if sparse_us > 0 else 0.0,
        }

    def close(self) -> None:
        if getattr(self, "engine", None) and getattr(self.engine, "_ctx", None):
            L = self.engine._lib
            ctx = self.engine._ctx
            for attr in ("_d_w", "_d_s", "_d_x", "_d_y", "_d_idx"):
                ptr = getattr(self, attr, None)
                if ptr:
                    L.rs_free_tensor(ctx, ptr)
                    setattr(self, attr, None)

    def __del__(self) -> None:
        self.close()
