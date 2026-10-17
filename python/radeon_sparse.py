#!/usr/bin/env python3
"""
RadeonSparse (radeon-sparse) — High-Level Python SDK
Provides RadeonSparseEngine and SparseLinearLayer (with backward-compatible alias RDNA3Engine).
"""

from .rdna3_engine import (
    RDNA3Engine as RadeonSparseEngine,
    RDNA3Engine,
    SparseLinearLayer,
    RDNA3Error as RadeonSparseError,
    RDNA3Error,
)

__all__ = [
    "RadeonSparseEngine",
    "SparseLinearLayer",
    "RadeonSparseError",
    "RDNA3Engine",
    "RDNA3Error",
]
