# =======================================================
# This file defines the interface between JAX and the
# custom KNN functions that are implemented in C++/CUDA.
# =======================================================
__all__ = ["knn, knn_v2"]

from jax.lib import xla_client
from jax import core
from jax.core import ShapedArray
from jax.interpreters import xla, mlir, batching
from jaxlib.hlo_helpers import (
    custom_call,
)  # NOTE: change to mhlo_helpers in older versions of JAX
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse

# cpp extensions
from . import cpu_ops
try:
    from . import gpu_ops
    is_gpu_available = True
except ImportError:
    is_gpu_available = False


# =======================================================
#                      Helper Functions
# =======================================================
def default_layouts(*shapes):
    """
    Generate default row-major memory layouts for custom call operands/results.

    XLA custom calls require explicit memory layout specifications. This helper
    generates the standard row-major layout (last dimension varies fastest) for
    tensors of given shapes.

    Args:
        *shapes (tuple of tuple): Variable number of shape tuples, e.g.
            (10, 20), (5, 3, 4) for 2D and 3D arrays.

    Returns:
        list of range: List of dimension orderings where each range specifies
            row-major layout (dimensions in reverse order).

    Example:
        >>> default_layouts((3, 4), (5,))
        [range(1, -1, -1), range(0, -1, -1)]
        # For (3,4): dims [1,0] means last dim varies fastest (row-major)
        # For (5,):   dims [0] means single dimension
    """
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


# =======================================================
#                     Primitives
# =======================================================
# primitives exposed to user code

def knn(indptr, indices, matdat, N):
    """
    Compute row-wise k=2 nearest neighbors for a sparse CSR matrix.

    For each row, identifies the two smallest values and returns them. This is
    used in PROTAX to find the k=2 nearest reference sequences per taxonomic node.

    Args:
        indptr (jax.Array): CSR row pointer array, shape (N+1,). Entry i gives
            the start index in `indices`/`matdat` for row i.
        indices (jax.Array): CSR column indices, shape (nnz,).
        matdat (jax.Array): CSR data values (e.g., distances), shape (nnz,).
            Must be float32.
        N (int): Number of rows in the matrix.

    Returns:
        jax.Array: Shape (N, 2) containing the k=2 smallest values per row.
            Rows with fewer than 2 elements are zero-padded.

    Note:
        Automatically dispatches to CPU or GPU implementation based on the
        default backend. For GPU-only execution, use `knn_v2`.

    Example:
        >>> indptr = jnp.array([0, 2, 5])  # 2 rows
        >>> indices = jnp.array([0, 2, 1, 3, 4])
        >>> matdat = jnp.array([0.1, 0.5, 0.2, 0.3, 0.4], dtype=jnp.float32)
        >>> knn(indptr, indices, matdat, 2)
        Array([[0.1, 0.5],
               [0.2, 0.3]], dtype=float32)
    """

    res = jnp.zeros((N, 2))
    return _knn_prim.bind(indptr, indices, matdat, res)


def knn_v2(indptr, indices, matdat, N):
    """
    GPU-only variant of KNN with optimized kernel.

    Similar to `knn` but uses an alternative CUDA kernel implementation that
    may have different performance characteristics. Only works on GPU.

    Args:
        indptr (jax.Array): CSR row pointer array, shape (N+1,).
        indices (jax.Array): CSR column indices, shape (nnz,).
        matdat (jax.Array): CSR data values, shape (nnz,). Must be float32.
        N (int): Number of rows in the matrix.

    Returns:
        jax.Array: Shape (N, 2) containing the k=2 smallest values per row.

    Raises:
        ValueError: If GPU operations are not available.

    Note:
        This function requires CUDA and will fail if GPU ops are not compiled.
    """

    res = jnp.zeros((N, 2))
    return _knn_v2_prim.bind(indptr, indices, matdat, res)


# =======================================================
#             JIT Support for KNN Primitive
# =======================================================
def _knn_abstract_eval(indptr, indices, matdat, res):
    """
    Abstract evaluation for knn primitive (shape and dtype inference).

    JAX's JIT compiler calls this during trace to determine output shapes and
    dtypes without executing the operation. This enables static compilation.

    Args:
        indptr: Abstract value for CSR row pointers.
        indices: Abstract value for CSR column indices.
        matdat: Abstract value for CSR data (determines output dtype).
        res: Abstract value for result template (determines output shape).

    Returns:
        ShapedArray: Abstract array with shape (N, 2) and dtype matching `matdat`.

    Note:
        Signature must match `_knn_prim.bind()` call signature exactly.
    """
    return ShapedArray(res.shape, matdat.dtype)


def _knn_lowering(ctx, indptr, indices, matdat, res, platform="cpu"):
    """
    MLIR lowering rule for knn primitive (compilation to custom call).

    This function is called by JAX's compiler to convert the abstract `knn`
    primitive into concrete MLIR custom_call operations that invoke C++/CUDA
    implementations.

    Args:
        ctx (mlir.LoweringRuleContext): Context containing input abstract values
            and MLIR builder state.
        indptr: MLIR value for CSR row pointers.
        indices: MLIR value for CSR column indices.
        matdat: MLIR value for CSR data.
        res: MLIR value for result template.
        platform (str): Target platform, either "cpu" or "gpu".

    Returns:
        list: Single-element list containing MLIR value for the result tensor.

    Raises:
        NotImplementedError: If dtype is not float32.

    Note:
        GPU version uses an opaque descriptor to pass problem size to CUDA kernel.
        CPU version passes N as a scalar constant operand.
    """

    mat_dtype = ctx.avals_in[2].dtype
    mat_type = mlir.ir.RankedTensorType(matdat.type)
    res_type = mlir.ir.RankedTensorType(res.type)
    ip_type = mlir.ir.RankedTensorType(indptr.type)
    idx_type = mlir.ir.RankedTensorType(indices.type)
    md_type = mlir.ir.RankedTensorType(matdat.type)

    N = res_type.shape[0]

    # dispatch for float32 only
    if mat_dtype != jnp.float32:
        raise NotImplementedError(f"unsupported dtype: {mat_dtype}")
    if platform == "gpu":
        # create opaque descriptor for problem size
        opaque = gpu_ops.build_knn_descriptor(N)
        out = custom_call(
            b"gpu_knn_f32",  # call target name
            result_types=[res_type],
            operands=[indptr, indices, matdat],
            operand_layouts=default_layouts(
                ip_type.shape, idx_type.shape, md_type.shape
            ),
            result_layouts=default_layouts(res_type.shape),  # memory layout
            backend_config=opaque,  # opaque descriptor
        ).results
        # output must be iterable
        return [out]

    # cpu custom call (default to k=2 for now)
    layout = default_layouts(ip_type.shape, idx_type.shape, md_type.shape)
    layout.insert(0, ())
    out = custom_call(
        b"cpu_knn_f32",  # call target name
        result_types=[res_type],
        operands=[mlir.ir_constant(N), indptr, indices, matdat],
        operand_layouts=layout,
        result_layouts=default_layouts(res_type.shape),  # memory layout
    ).results

    return [out]


# =======================================================
#            JIT Support for KNN v2 Primitive
# =======================================================
def _knn_v2_abstract_eval(indptr, indices, matdat, res):
    """
    Abstract evaluation for knn_v2 primitive (GPU variant).

    Identical to `_knn_abstract_eval` but for the knn_v2 primitive.

    Args:
        indptr: Abstract value for CSR row pointers.
        indices: Abstract value for CSR column indices.
        matdat: Abstract value for CSR data (determines output dtype).
        res: Abstract value for result template (determines output shape).

    Returns:
        ShapedArray: Abstract array with shape (N, 2) and dtype matching `matdat`.

    Note:
        Signature must match `_knn_v2_prim.bind()` call signature exactly.
    """
    return ShapedArray(res.shape, matdat.dtype)


def _knn_v2_lowering(ctx, indptr, indices, matdat, res):
    """
    MLIR lowering for knn_v2 primitive (GPU-only variant).

    Similar to `_knn_lowering` but specifically for the GPU-optimized v2 kernel.
    Raises an error if GPU operations are not available.

    Args:
        ctx (mlir.LoweringRuleContext): Context containing input abstract values.
        indptr: MLIR value for CSR row pointers.
        indices: MLIR value for CSR column indices.
        matdat: MLIR value for CSR data.
        res: MLIR value for result template.

    Returns:
        list: Single-element list containing MLIR value for the result tensor.

    Raises:
        NotImplementedError: If dtype is not float32.
        ValueError: If GPU operations are not compiled/available.

    Note:
        This variant calls a different CUDA kernel (gpu_knn_v2_f32) that may
        use alternative optimization strategies.
    """

    mat_dtype = ctx.avals_in[2].dtype
    mat_type = mlir.ir.RankedTensorType(matdat.type)
    res_type = mlir.ir.RankedTensorType(res.type)
    ip_type = mlir.ir.RankedTensorType(indptr.type)
    idx_type = mlir.ir.RankedTensorType(indices.type)
    md_type = mlir.ir.RankedTensorType(matdat.type)

    N = res_type.shape[0]

    # dispatch for float32 only
    if mat_dtype != jnp.float32:
        raise NotImplementedError(f"unsupported dtype: {mat_dtype}")
    try:
        from . import gpu_ops

        if gpu_ops is None:
            raise ValueError("gpu_ops not compiled")

        # create opaque descriptor for problem size
        opaque = gpu_ops.build_knn_descriptor(N)

        out = custom_call(
            b"gpu_knn_v2_f32",  # call target name
            result_types=[res_type],
            operands=[indptr, indices, matdat],
            operand_layouts=default_layouts(
                ip_type.shape, idx_type.shape, md_type.shape
            ),
            result_layouts=default_layouts(res_type.shape),  # memory layout
            backend_config=opaque,  # opaque descriptor
        ).results
    except:
        raise ValueError("gpu_ops not compiled")
    # output must be iterable
    return [out]


# =======================================================
#             Registering KNN Primitive
# =======================================================
# register CPU XLA custom calls
for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# register GPU XLA custom calls
if is_gpu_available:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
# defining KNN primitive for JAX
_knn_prim = core.Primitive("knn")
_knn_prim.def_impl(partial(xla.apply_primitive, _knn_prim))
_knn_prim.def_abstract_eval(_knn_abstract_eval)

# connect XLA translation rules for JIT compilation
if is_gpu_available:
    mlir.register_lowering(
        _knn_prim, partial(_knn_lowering, platform="gpu"), platform="gpu"
    )
mlir.register_lowering(
    _knn_prim, partial(_knn_lowering, platform="cpu"), platform="cpu"
)


if is_gpu_available:
    # defining KNN v2 primitive
    _knn_v2_prim = core.Primitive("knn_v2")
    _knn_v2_prim.def_impl(partial(xla.apply_primitive, _knn_v2_prim))
    _knn_v2_prim.def_abstract_eval(_knn_v2_abstract_eval)
    mlir.register_lowering(_knn_v2_prim, _knn_v2_lowering, platform="gpu")


# testing on a simple example
# TODO: make proper tests for this
if __name__ == "__main__":
    foo = jnp.array(
        [
            [0.012, 0.3, 0.2],
            [0.01, 0, 0.3],
            [0.4, 0.1, 0.5],
        ],
        dtype=jnp.float32,
    )

    foo = sparse.bcsr_fromdense(foo)
    x = knn(foo.indptr, foo.indices, foo.data, 3)
