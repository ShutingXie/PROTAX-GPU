import os

from .protax_utils import read_model_jax, read_query, read_baseline
from .model import get_probs, get_log_probs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
import time
from pathlib import Path


def load_layer(tdir):
    """
    Load per-node taxonomic levels from a taxonomy NPZ file.

    Args:
        tdir: Path-like object or string pointing to a taxonomy `.npz` file
            produced by `convert_taxonomy` (e.g., `models/ref_db/taxonomy*.npz`).

    Returns:
        A 1D numpy array of integers where each entry is the layer index
        (rank depth) for the corresponding node in the taxonomy.
    """
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())

    lvl = tax['node_layer']

    tax = np.load(tax_dir.resolve())

    return lvl

def read_names(tdir):
    """
    Read the taxon names from a PROTAX taxonomy text file.

    The file is expected to be a tab-separated text file where each line
    contains `nid, pid, lvl, name, prior, ...`. Only the `name` field is
    collected.

    Args:
        tdir: Path to the taxonomy `.priors` text file.

    Returns:
        A list of strings containing the taxon names in index order.
    """
    f = open(tdir)
    node_dat = f.readlines()

    names = []
    for l in node_dat:
        l = l.strip("\n")

        # collecting taxon data
        nid, pid, lvl, name, prior, _ = l.split("\t")
        nid, pid, lvl, prior = (int(nid), int(pid), int(lvl), float(prior))
        names.append(name)

    return names

def validate_taxonomy_query(tree, query, ok_query):
    """
    Validate that the taxonomy and query dimensions match.

    Args:
        tree: TaxTree object containing taxonomy information.
        query: Query sequence array.
        ok_query: Boolean array indicating valid positions in the query.

    Raises:
        ValueError: If dimensions are incompatible.
    """
    # Check the number of positions in references and query
    if tree.refs.shape[1] != query.shape[0]:
        raise ValueError(f"Mismatch in sequence lengths: "
                         f"taxonomy={tree.refs.shape[1]} vs query={query.shape[0]}")

    # Check ok_pos and ok_query have the same length
    if tree.ok_pos.shape[1] != ok_query.shape[0]:
        raise ValueError(f"Mismatch in valid positions: "
                         f"taxonomy={tree.ok_pos.shape[1]} vs query={ok_query.shape[0]}")

def classify_file(qdir, par_dir, tax_dir, verbose=False):
    """
    Classify a batch of query sequences using a trained PROTAX model.

    This function streams query sequences from a FASTA-like alignment file
    (two lines per record: header then sequence), computes per-node
    probabilities with the JAX implementation, and writes the per-level
    predictions to `pyprotax_results.csv`.

    Args:
        qdir: Path to the query alignment file (e.g., `refs.aln`).
        par_dir: Path to the model parameters `.npz` file (e.g., `models/params/model.npz`).
        tax_dir: Path to the taxonomy `.npz` file (e.g., `models/ref_db/taxonomy*.npz`).
        verbose: If True, prints per-record timing and optional debug info.

    Side Effects:
        Writes `pyprotax_results.csv` with one row per query containing the
        predicted level indices.
    """

    tree, params, N, segnum = read_model_jax(par_dir, tax_dir)

    n2s_clone = csr_matrix((tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr), shape=(N, tree.refs.shape[0]))
    n2s_clone
    
    f = open(qdir)

    tot_time = 0
    res = []

    while True:
        curr = f.readline().strip('\n')
        curr = curr.replace('|', '\t').split('\t')
        seqs = f.readline().strip('\n')
        q, ok = read_query(seqs)

        validate_taxonomy_query(tree, q, ok)

        curr_name = ''.join(curr[1:])
        if not seqs:
            break  # EOF
            
        start = time.time()
        probs = get_probs(q, ok, tree, params, segnum, N).block_until_ready()
        end = time.time()

        probs = jnp.take(probs, tree.paths, fill_value=-1)


        # TODO argmax at leaf level?
        classified_layer = jnp.argmax(probs, axis=0)
        res.append(classified_layer)

    
        if verbose:
            pass
            # sel_name = names[classified_layer.at[-1].get()]
            # print(f"{curr}: {sel_name}: {end - start}s")
        tot_time += end-start

    # saving results
    df = pd.DataFrame(np.array(res))
    df.to_csv("pyprotax_results.csv")
    print(f"finished in {tot_time}s")



def classify(q, ok, tree, params, segnum, N):
    """
    Classify a single query sequence given a taxonomy and model parameters.

    This is a placeholder for a lower-level API that mirrors `classify_file`.

    Args:
        q: Packed bits representation of the query sequence bases (A/T/G/C).
        ok: Packed bits mask of positions that are valid base calls.
        tree: `TaxTree` containing reference database and topology.
        params: `ProtaxModel` parameters (beta and scaling stats).
        segnum: Number of unique segment ids in `tree.segments`.
        N: Number of nodes in the taxonomy.

    Returns:
        Not implemented yet.
    """
    pass



if __name__ == "__main__":

    # protax_args = sys.argv
    # if len(protax_args) < 4:
    #     print("Usage: python3 classify.py [PATH_TO_TAXONOMY_FILE] [PATH_TO_PARAMETERS] [PATH_TO_QUERY_SEQUENCES]")
    
    # tax_dir, model_dir, query_dir = protax_args[1:4]

    # testing for now
    
    query_dir = r"FinPROTAX/FinPROTAX/modelCOIfull/refs.aln"
    classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz") 
    compute_perplexity(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")

