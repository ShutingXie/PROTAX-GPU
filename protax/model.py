"""
Core probabilistic model implementation for PROTAX-GPU.

This module contains the main computational functions for the PROTAX probabilistic
taxonomic classification model. It implements:
- Sequence distance computation using packed bit operations
- Design matrix construction with KNN-based features
- Branch probability calculations with segment-wise normalization
- Path probability aggregation across taxonomic levels

The implementation uses JAX for automatic differentiation and GPU acceleration,
with custom CUDA kernels for efficient k-nearest neighbor search.
"""
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

    Distances are computed as 1 - (matches / valid_positions) where valid_positions
    counts positions that are valid (not gaps or ambiguous) in both the query and
    each reference. Sequences are stored as packed bits for memory efficiency;
    A/T/G/C are compared via bitwise AND operations followed by population count.

    Args:
        q (jax.Array): Packed bits array for the query sequence bases (A/T/G/C),
            shape (D',) where D' = ceil(4*D/8) for sequence length D.
        seqs (jax.Array): Packed bits for reference sequences, shape (R, D')
            where R is the number of references.
        ok (jax.Array): Packed bits mask of valid positions for references,
            shape (R, D').
        ok_query (jax.Array): Packed bits mask of valid positions for the query,
            shape (D',).

    Returns:
        jax.Array: A 1D array of normalized Hamming distances for each reference,
            shape (R,). Values range from 0 (identical) to 1 (no matches).

    Note:
        This function can be JIT-compiled with `jax.jit` for improved performance.
        The packed bits representation uses 4 bits per nucleotide position (A/T/G/C).
    """

    # count matches and valid positions
    ok = jnp.bitwise_and(ok_query, ok)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(q, seqs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return 1 - (match_tots / ok)


def seq_dist2(s1, s2):
    """
    Alternative distance computation using explicit one-hot encoding with validity.

    This is an alternative implementation that uses a 5-channel representation
    where channels 0-3 are one-hot encoded nucleotides (A/T/G/C) and channel 4
    is a validity mask. Currently unused in favor of `seq_dist`.

    Args:
        s1 (jax.Array): One-hot encoded sequence with validity, shape (5, D)
            where D is sequence length and row 4 is the valid position mask.
        s2 (jax.Array): One-hot encoded sequence with validity, shape (5, D).

    Returns:
        jax.Array: Scalar value representing the fraction of matching positions
            over valid positions. Range [0, 1].

    Note:
        This function is not currently used in the main PROTAX pipeline but may
        be useful for debugging or alternative encoding schemes.
    """
    intersect = jnp.bitwise_and(s1, s2)
    matches = jax.lax.population_count(intersect, axis=1)
    
    n_ok = jnp.sum(matches.at[4].get())
    return jnp.sum(matches.at[:4].get()) / n_ok



def get_X(q, ok_q, tree, N, sc_mean, sc_var):
    """
    Construct the PROTAX design matrix using KNN distances and node state.

    The design matrix is the input to the logistic regression model at each node.
    It combines binary node state indicators (empty-but-known, has-refs) with
    normalized KNN distance features. For each node, the k=2 nearest reference
    sequences are identified and their distances are standardized using per-level
    mean and variance statistics.

    Args:
        q (jax.Array): Packed bits for the query sequence bases (A/T/G/C).
        ok_q (jax.Array): Packed bits mask for valid query positions.
        tree (TaxTree): `TaxTree` instance containing references, node-to-sequence
            mapping (node2seq), and node state indicators.
        N (int): Number of taxonomy nodes.
        sc_mean (jax.Array): Per-node scaling means for distance features,
            shape (N, 2) for k=2 nearest neighbors.
        sc_var (jax.Array): Per-node scaling variances for distance features,
            shape (N, 2).

    Returns:
        jax.Array: Design matrix of shape (N, M) where M = 2 (node_state) + 2 (knn).
            Column 0: empty-but-known indicator
            Column 1: has-references indicator
            Columns 2-3: Standardized distances to k=2 nearest references

    Note:
        Nodes without references will have their KNN features masked to zero
        via multiplication by node_state[:, 1].
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
    Compute per-node linear score z = X · beta (element-wise product then sum).

    This computes the logit for each node in the taxonomic tree by taking the
    element-wise product of the design matrix and parameter matrix, then summing
    across features. This is equivalent to a dot product per row.

    Args:
        X (jax.Array): Design matrix of shape (N, M) where N is number of nodes
            and M is number of features.
        params (ProtaxModel): `ProtaxModel` instance containing `beta` with
            shape (N, M) - per-node regression coefficients.

    Returns:
        jax.Array: A 1D array z of length N containing the linear score for
            each node. These scores are converted to probabilities via softmax
            within each parent segment.
    """
    z = jnp.sum(jnp.multiply(X, params.beta), axis=1)
    return z


def get_bprobs(z, segments, segnum):
    """
    Compute normalized branch probabilities per parent segment using softmax.

    For each parent node, this function normalizes the scores of its children
    to sum to 1, creating a categorical distribution over child nodes. This
    implements the conditional probability P(child | parent) in the PROTAX model.

    Args:
        z (jax.Array): Nonnegative scores per node, shape (N,). Typically
            exp(linear_score) or exp(linear_score) * prior.
        segments (jax.Array): Segment id (parent id bucket) per node, shape (N,).
            Nodes with the same segment id share a parent and compete for probability.
        segnum (int): Number of unique segment ids (number of parents + 1 for root).

    Returns:
        jax.Array: Branch probabilities per node, shape (N,). Values sum to 1
            within each segment. Root node (index 0) is set to probability 1.

    Note:
        NaN values (from 0/0 for empty segments) are replaced with 0.
    """
    norm_factors = jax.ops.segment_sum(z, segments, num_segments=segnum, indices_are_sorted=True)
    norm_factors = jnp.take(norm_factors, segments, indices_are_sorted=True)
    branch_probs =  jnp.nan_to_num(z / norm_factors)
    branch_probs = branch_probs.at[0].set(1)

    return branch_probs



def get_log_bprobs(z, segments, segnum):
    """
    Compute log-branch probabilities per node in a numerically stable way.

    This implements log-softmax normalization within each parent segment, which
    is numerically more stable than computing probabilities then taking logs.
    Useful for training (gradient descent) and for accumulating log-probabilities
    along taxonomic paths.

    Args:
        z (jax.Array): Real-valued scores (logits) per node, shape (N,).
            Can be any real number, not necessarily non-negative.
        segments (jax.Array): Segment id (parent id bucket) per node, shape (N,).
        segnum (int): Number of unique segment ids.

    Returns:
        jax.Array: Log-branch probabilities per node, shape (N,).
            Values are <= 0, and exp(values) sum to 1 within each segment.
            Root node (index 0) is set to log(1) = 0.

    Note:
        This function implements: log(exp(z_i) / sum_j(exp(z_j))) = z_i - log(sum_j(exp(z_j)))
        NaN values from empty segments are replaced with 0.
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

    This function computes the branch probability for each node, then organizes
    them into a matrix indexed by [level, node] for efficient path probability
    computation. The scores are combined with prior probabilities and normalized
    within segments using numerically stable operations.

    Args:
        X (jax.Array): Design matrix, shape (N, M) where N is nodes, M is features.
        beta (jax.Array): Parameter matrix, shape (N, M) containing regression
            coefficients for each node.
        tree (TaxTree): `TaxTree` instance with segments, paths, node_state, and
            prior probability mass per node.
        segnum (int): Number of unique segment ids for segment operations.

    Returns:
        jax.Array: Shape (L, P) with per-level branch probabilities along each
            path in `tree.paths`, where L is number of levels and P is number of
            nodes. Missing entries (invalid paths) are filled with 1.

    Note:
        Uses log-sum-exp trick (max subtraction) for numerical stability before
        exponentiating and normalizing within segments.
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

    Similar to `fill_bprob` but works in log-space for numerical stability.
    This is the preferred version for training and for computing log-likelihood.

    Args:
        X (jax.Array): Design matrix, shape (N, M).
        beta (jax.Array): Parameter matrix, shape (N, M).
        tree (TaxTree): `TaxTree` instance with segments and paths.
        segnum (int): Number of unique segment ids.

    Returns:
        jax.Array: Shape (L, P) with per-level log-branch probabilities along
            each path in `tree.paths`. Missing entries are filled with log(1) = 0.

    Note:
        Log-space arithmetic is used throughout to avoid numerical underflow when
        computing products of many small probabilities along deep taxonomic paths.
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
    Compute log-probability of each node by summing log-branch probs along paths.

    This is the main inference function in log-space. It computes the total
    log-probability of each node in the taxonomy by:
    1. Building the design matrix X from query-reference distances
    2. Computing log-branch probabilities for each node
    3. Summing log-probabilities along paths from root to each node

    Args:
        q (jax.Array): Packed bits for query bases.
        ok (jax.Array): Packed bits mask for valid query positions.
        tree (TaxTree): `TaxTree` instance with taxonomy structure.
        params (ProtaxModel): `ProtaxModel` with beta, sc_mean, sc_var.
        segnum (int): Number of unique segment ids (for JIT compilation).
        N (int): Number of nodes in the taxonomy (for JIT compilation).

    Returns:
        jax.Array: A 1D array of length N with total log-probabilities per node.
            These are log P(node | query), unnormalized across levels.

    Note:
        Can be JIT-compiled by uncommenting the decorator. Static arguments
        (segnum, N) must be known at compile time.
    """
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_log_bprob(X, params.beta, tree, segnum)
    return jnp.sum(bprobs, axis=1)

# @partial(jax.jit, static_argnums=(4, 5))
def get_probs(q, ok, tree, params, segnum, N):
    """
    Compute probability of each node by multiplying branch probs along paths.

    This is the main inference function in probability space (not log-space).
    It computes the total probability of each node in the taxonomy by:
    1. Building the design matrix X from query-reference distances
    2. Computing branch probabilities for each node
    3. Multiplying probabilities along paths from root to each node

    Args:
        q (jax.Array): Packed bits for query bases.
        ok (jax.Array): Packed bits mask for valid query positions.
        tree (TaxTree): `TaxTree` instance with taxonomy structure.
        params (ProtaxModel): `ProtaxModel` with beta, sc_mean, sc_var.
        segnum (int): Number of unique segment ids (for JIT compilation).
        N (int): Number of nodes in the taxonomy (for JIT compilation).

    Returns:
        jax.Array: A 1D array of length N with total probabilities per node.
            These are P(node | query), unnormalized across levels but normalized
            within each parent's children.

    Note:
        For deep taxonomies or many classifications, prefer `get_log_probs` to
        avoid numerical underflow. Can be JIT-compiled by uncommenting the decorator.
    """
    X = get_X(q, ok, tree, N, params.sc_mean, params.sc_var)
    bprobs = fill_bprob(X, params.beta, tree, segnum)
    return jnp.prod(bprobs, axis=1)

