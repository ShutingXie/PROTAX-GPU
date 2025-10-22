"""
Baseline nearest-neighbor classifier for PROTAX-GPU.

This module provides a simple distance-based baseline classifier that assigns
queries to the taxonomic node of their nearest reference sequence. It serves
as a baseline for comparison with the full probabilistic PROTAX model.

Note:
    This implementation currently selects the most DISTANT reference (argmax
    instead of argmin), which appears to be a bug. Use with caution.
"""
import jax
import jax.numpy as jnp
from protax_utils import read_baseline, read_query
import numpy as np
import time
import pandas as pd


@jax.jit
def seq_dist(q, seqs, ok, ok_query):
    """
    Compute distance between one query and many references (baseline NN).

    Distance is defined as 1 - (matches / valid) where `valid` counts positions
    that are valid in both query and reference. Uses packed bits for efficiency.

    Args:
        q (jax.Array): Packed bits for the query bases (A/T/G/C).
        seqs (jax.Array): Packed bits for reference bases, shape (R, D')
            where R is number of references.
        ok (jax.Array): Packed bits mask for valid reference positions, shape (R, D').
        ok_query (jax.Array): Packed bits mask for valid query positions, shape (D',).

    Returns:
        jax.Array: Scalar integer index of the reference with MAXIMUM distance
            (largest dissimilarity).

    Warning:
        This function returns argmax (most distant) instead of argmin (nearest).
        This appears to be a bug and should be corrected for true nearest-neighbor
        classification.
    """

    # count matches and valid positions
    ok = jnp.bitwise_and(ok_query, ok)
    ok = jnp.sum(jax.lax.population_count(ok), axis=1)
    match = jnp.bitwise_and(q, seqs)

    match_tots = jnp.sum(jax.lax.population_count(match), axis=1)
    return jnp.argmax(1 - (match_tots / ok))

def nearest_classifier(q, seqs, ok, ok_query, n2s):
    """
    Return the leaf node id for the reference selected by `seq_dist`.

    This classifier assigns the query to the taxonomic node of the selected
    reference sequence. The reference is chosen by `seq_dist`, which currently
    returns the MOST distant reference due to a bug.

    Args:
        q (jax.Array): Packed query bases (A/T/G/C).
        seqs (jax.Array): Packed reference bases for all references.
        ok (jax.Array): Packed valid positions for references.
        ok_query (jax.Array): Packed valid positions for query.
        n2s (scipy.sparse.csc_matrix): Sparse node-to-sequence mapping in CSC
            format for dereferencing the leaf node id from reference index.

    Returns:
        int: Node id of the selected reference's most specific taxonomic assignment.

    Warning:
        As implemented, `seq_dist` returns the farthest reference (argmax of
        distance), so this does NOT perform true nearest-neighbor classification.
        The last index in n2s[:, closest_r].indices gives the finest taxonomic
        level for that reference.
    """
    closest_r = int(seq_dist(q, seqs, ok, ok_query))
    return n2s[:, closest_r].indices[-1]


def classify_file(qdir, verbose=False):
    """
    Classify queries using a simple nearest-neighbor baseline and save results.

    Reads query sequences from a FASTA-like alignment file, assigns each to
    the taxonomic path of its nearest (or actually farthest, due to bug)
    reference, and writes results to CSV.

    Args:
        qdir (str or Path): Path to query alignment file containing two-line
            records (header, sequence).
        verbose (bool, optional): Unused; reserved for future logging.
            Defaults to False.

    Returns:
        None

    Side Effects:
        Writes `dist_baseline_results.csv` with one row per query containing
        the taxonomic path (node IDs at each level) of the selected reference.
        Prints total classification time to stdout.

    Note:
        Model directory is currently hardcoded to "/home/roy/Documents/PROTAX-dsets/30k_small".
        This should be parameterized for general use.

    Example:
        >>> classify_file("queries.aln")
        finished in 5.67s
    """

    refs, ok_pos, n2s, paths = read_baseline(r"/home/roy/Documents/PROTAX-dsets/30k_small")
    f = open(qdir)

    tot_time = 0
    res = []

    while True:
        curr = f.readline().strip('\n').split('\t')[0]
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        if not seqs:
            break  # EOF

        start = time.time()
        species_id = nearest_classifier(q, refs, ok_pos, ok, n2s)
        end = time.time()
        res.append(paths[species_id])
        tot_time += end-start


    # saving results
    df = pd.DataFrame(np.array(res))
    df.to_csv("dist_baseline_results.csv")
    print(f"finished in {tot_time}s")

if __name__ == "__main__":
    classify_file(r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln")