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

    Distance is defined as 1 - matches/valid where `valid` counts positions
    that are valid in both query and reference.

    Args:
        q: Packed bits for the query bases.
        seqs: Packed bits for reference bases, shape (R, D').
        ok: Packed bits mask for valid reference positions, shape (R, D').
        ok_query: Packed bits mask for valid query positions, shape (D',).

    Returns:
        Index of the reference with maximum distance (largest dissimilarity).
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

    Note: As implemented, `seq_dist` returns the farthest reference (argmax
    of distance), so this does not perform true nearest-neighbor classification.

    Args:
        q: Packed query bases.
        seqs: Packed reference bases.
        ok: Packed valid positions for references.
        ok_query: Packed valid positions for query.
        n2s: Sparse node-to-sequence mapping (CSC) for dereferencing leaf id.

    Returns:
        Integer node id of the selected reference's taxon.
    """
    closest_r = int(seq_dist(q, seqs, ok, ok_query))
    return n2s[:, closest_r].indices[-1]


def classify_file(qdir, verbose=False):
    """
    Classify queries using a simple nearest-neighbor baseline and save results.

    Args:
        qdir: Path to query alignment file (two-line records).
        verbose: Unused; reserved for future logging.

    Side Effects:
        Writes `dist_baseline_results.csv` with predicted leaf paths per query.
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