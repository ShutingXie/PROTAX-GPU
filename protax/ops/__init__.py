"""
Custom JAX operations for PROTAX-GPU.

This subpackage provides custom CPU and GPU implementations of operations
that are critical for PROTAX performance, particularly k-nearest neighbor
search on sparse matrices.

The operations are implemented as:
- C++ code for CPU execution (cpu_ops)
- CUDA kernels for GPU execution (gpu_ops)
- JAX primitives with MLIR lowering rules (knn_register)

Main exports:
    knn: K-nearest neighbor search (k=2) with automatic CPU/GPU dispatch
    knn_v2: GPU-only KNN variant with optimized kernel

Note:
    GPU operations are optional and will be unavailable if CUDA is not present.
"""
from . import cpu_ops
try:
    from . import gpu_ops
    is_gpu_available = True
except ImportError:
    is_gpu_available = False

from .knn_register import knn, knn_v2

__all__ = [
    "knn",
    "knn_v2"
]
