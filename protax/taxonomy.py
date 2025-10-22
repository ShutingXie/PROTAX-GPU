import jax
from typing import NamedTuple


class CSRWrapper(NamedTuple):
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple

class TaxTree(NamedTuple):
    """
    Container for taxonomy state and reference data used by PROTAX.

    Fields:
        refs: Packed-bit reference sequences, shape [R, D'].
        ok_pos: Packed-bit valid-position masks for references, shape [R, D'].
        segments: Parent segment id per node for segment-wise reductions, [N].
        node2seq: Sparse mapping from nodes to reference indices (CSR-like).
        paths: For each node, a path index per level, used to aggregate probs.
        node_state: Binary indicators per node (e.g., empty-but-known, has-refs).
        prior: Prior probability mass per node.
    """
    refs: jax.Array                  # [R, 5]
    ok_pos: jax.Array                # [R]
    segments: jax.Array              # [N]
    node2seq: CSRWrapper             # [N]
    paths: jax.Array                 # [N]
    node_state: jax.Array            # [N, 2]
    prior: jax.Array


class ProtaxModel(NamedTuple):
    """
    Model parameters and scaling statistics for PROTAX inference/training.

    Fields:
        beta: Per-node parameter matrix aligned with design matrix columns.
        sc_mean: Per-node scaling means for distance features.
        sc_var: Per-node scaling variances for distance features.
    """
    
    beta: jax.Array
    sc_mean: jax.Array
    sc_var: jax.Array
