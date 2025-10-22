import jax
import jax.numpy as jnp
from jax.experimental import sparse
from functools import partial
import numpy as np
from functools import partial
from .ops import knn, knn_v2

# @jax.jit
def seq_dist(q, seqs, ok, ok_query):
    """
    Compute Hamming-like distance between one query and many references.

    Distances are computed as 1 - matches/valid where `valid` counts positions
    that are valid in both the query and each reference. Bits are packed; A/T/G/C
    bits are compared via bitwise operations.

    Args:
        q: Packed bits array for the query sequence bases (A/T/G/C), shape (D',).
        seqs: Packed bits for reference sequences, shape (R, D').
        ok: Packed bits mask of valid positions for references, shape (R, D').
        ok_query: Packed bits mask of valid positions for the query, shape (D',).

    Returns:
        A 1D jax array of distances for each reference, shape (R,).
    """

    # count matches and valid positions
    ok = jnp.bitwise_and(ok_query, ok)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(q, seqs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return 1 - (match_tots / ok)


def seq_dist2(s1, s2):
    """
    Alternative distance using explicit one-hot with validity channel.

    Args:
        s1: One-hot with validity, shape (5, D) where last row is valid mask.
        s2: One-hot with validity, shape (5, D) where last row is valid mask.

    Returns:
        Scalar fraction of matches over valid positions.
    """
    intersect = jnp.bitwise_and(s1, s2)
    matches = jax.lax.population_count(intersect, axis=1)
    
    n_ok = jnp.sum(matches.at[4].get())
    return jnp.sum(matches.at[:4].get()) / n_ok



def get_X(q, ok_q, tree, N, sc_mean, sc_var):
    """
    Construct the PROTAX design matrix using KNN distances and node state.

    The design matrix concatenates binary node state features and KNN-derived
    distance features normalized by per-level scaling parameters.

    Args:
        q: Packed bits for the query sequence bases (A/T/G/C).
        ok_q: Packed bits mask for valid query positions.
        tree: `TaxTree` instance with references, node2seq, and node_state.
        N: Number of taxonomy nodes.
        sc_mean: (N, 2) scaling means per node and distance column.
        sc_var: (N, 2) scaling variances per node and distance column.

    Returns:
        A jax array of shape (N, M) where M = node_state_cols + 2.
    """

    node2seq = tree.node2seq
    dists = seq_dist(q, tree.refs, tree.ok_pos, ok_q)

    # TODO maybe use custom BCSR multiplication kernel
    # or can also be replaced with a take operation
    new_dat = jnp.take(dists, node2seq.indices)

    X = knn(node2seq.indptr, node2seq.indices, new_dat, N)
    X = (((X - sc_mean) / sc_var).T*(tree.node_state[:, 1])).T
    X = jnp.concatenate((tree.node_state, X), axis=1)
    return X


def get_z(X, params):
    """
    Compute per-node linear score z = X · beta.

    Args:
        X: Design matrix of shape (N, M).
        params: `ProtaxModel` containing `beta` with shape (N, M).

    Returns:
        A 1D jax array z of length N.
    """
    z = jnp.sum(jnp.multiply(X, params.beta), axis=1)
    return z


def get_bprobs(z, segments, segnum):
    """
    Compute normalized branch probabilities per parent segment.

    Args:
        z: Nonnegative scores per node, shape (N,).
        segments: Segment id (parent id bucket) per node, shape (N,).
        segnum: Number of unique segment ids.

    Returns:
        A jax array of branch probabilities per node, shape (N,).
    """
    norm_factors = jax.ops.segment_sum(z, segments, num_segments=segnum, indices_are_sorted=True)
    norm_factors = jnp.take(norm_factors, segments, indices_are_sorted=True)
    branch_probs =  jnp.nan_to_num(z / norm_factors)
    branch_probs = branch_probs.at[0].set(1)

    return branch_probs



def get_log_bprobs(z, segments, segnum):
    """
    Compute log-branch probabilities per node in a numerically stable way.

    Useful for training and for accumulating log-probabilities along paths.

    Args:
        z: Real-valued scores per node, shape (N,).
        segments: Segment id (parent id bucket) per node, shape (N,).
        segnum: Number of unique segment ids.

    Returns:
        A jax array of log-branch probabilities per node, shape (N,).
    """

    exp_z = jnp.exp(z)
    norm_factors = jnp.log(jax.ops.segment_sum(exp_z, segments, num_segments=segnum, indices_are_sorted=True))
    norm_factors = jnp.take(norm_factors, segments, indices_are_sorted=True)
    branch_probs =  jnp.nan_to_num(z - norm_factors)
    branch_probs = branch_probs.at[0].set(0)

    return branch_probs


def fill_bprob(X, beta, tree, segnum):
    """
    Compute per-level path-wise branch probabilities across the taxonomy.

    Args:
        X: Design matrix, shape (N, M).
        beta: Parameter matrix, shape (N, M).
        tree: `TaxTree` with segments, paths, node_state, and priors.
        segnum: Number of unique segment ids.

    Returns:
        A jax array of shape (L, P) with per-level branch probabilities along
        each path in `tree.paths` (L levels by P paths).
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum, indices_are_sorted=True)
    max_z = jnp.take(max_z, tree.segments, indices_are_sorted=True)
    exp_z = jnp.exp(z - max_z)*tree.prior
    branch_probs = get_bprobs(exp_z, tree.segments, segnum)

    filled_paths = jnp.take(branch_probs, tree.paths, indices_are_sorted=True,
                          fill_value=1, unique_indices=True)
    return filled_paths


def fill_log_bprob(X, beta, tree, segnum):
    """
    Compute per-level log-branch probabilities across the taxonomy.

    Args:
        X: Design matrix, shape (N, M).
        beta: Parameter matrix, shape (N, M).
        tree: `TaxTree` with segments and paths.
        segnum: Number of unique segment ids.

    Returns:
        A jax array of shape (L, P) with per-level log-branch probabilities
        along each path in `tree.paths`.
    """
    z = jnp.sum(jnp.multiply(X, beta), axis=1)
    max_z = jax.ops.segment_max(z, tree.segments, num_segments=segnum, indices_are_sorted=True)
    max_z = jnp.take(max_z, tree.segments, indices_are_sorted=True)
    z -= max_z
    branch_probs = get_log_bprobs(z, tree.segments, segnum)

    # total probability computation
    filled_paths = jnp.take(branch_probs, tree.paths, indices_are_sorted=True,
                          fill_value=0, unique_indices=True)
    return filled_paths


# @partial(jax.jit, static_argnums=(4, 5))
def get_log_probs(q, ok, tree, params, segnum, N):
    """
    Compute log-probability of each node by summing log-branch probs per path.

    Args:
        q: Packed bits for query bases.
        ok: Packed bits mask for valid query positions.
        tree: `TaxTree` instance.
        params: `ProtaxModel` with beta, sc_mean, sc_var.
        segnum: Number of unique segment ids.
        N: Number of nodes in the taxonomy.

    Returns:
        A 1D jax array of length N with total log-probabilities per node.
    """
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_log_bprob(X, params.beta, tree, segnum)
    return jnp.sum(bprobs, axis=1)

# @partial(jax.jit, static_argnums=(4, 5))
def get_probs(q, ok, tree, params, segnum, N):
    """
    Compute probability of each node by multiplying branch probs per path.

    Args:
        q: Packed bits for query bases.
        ok: Packed bits mask for valid query positions.
        tree: `TaxTree` instance.
        params: `ProtaxModel` with beta, sc_mean, sc_var.
        segnum: Number of unique segment ids.
        N: Number of nodes in the taxonomy.

    Returns:
        A 1D jax array of length N with total probabilities per node.
    """
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_bprob(X, params.beta, tree, segnum)
    return jnp.prod(bprobs, axis=1)

