"""
Classification module for PROTAX-GPU.

This module provides functions for classifying DNA barcode sequences using trained
PROTAX models. It includes utilities for batch classification from FASTA-like files
and helper functions for taxonomy validation.
"""
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

    This function extracts the layer/rank information for each node in the taxonomic
    tree, where layer 0 is typically the root and higher layers represent finer
    taxonomic ranks (e.g., kingdom, phylum, class, order, family, genus, species).

    Args:
        tdir (str or Path): Path-like object or string pointing to a taxonomy `.npz`
            file produced by `convert_taxonomy` (e.g., `models/ref_db/taxonomy*.npz`).

    Returns:
        numpy.ndarray: A 1D numpy array of integers where each entry is the layer
            index (rank depth) for the corresponding node in the taxonomy.

    Example:
        >>> layers = load_layer("models/ref_db/taxonomy37k.npz")
        >>> print(layers[0])  # Root node is typically at layer 0
        0
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
    contains fields in the format: `nid, pid, lvl, name, prior, ...`. Only
    the `name` field is extracted and returned.

    Args:
        tdir (str or Path): Path to the taxonomy `.priors` text file containing
            node information in tab-separated format.

    Returns:
        list of str: A list of strings containing the taxon names in node index
            order, where the index in the list corresponds to the node ID.

    Example:
        >>> names = read_names("models/ref_db/taxonomy.priors")
        >>> print(names[0])  # Root taxon name
        'root'
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

    This function ensures that the reference sequences in the taxonomy and the
    query sequence have compatible dimensions for distance computation. Both the
    packed bits arrays and the validity masks must have matching lengths.

    Args:
        tree (TaxTree): TaxTree object containing taxonomy information including
            reference sequences and validity masks.
        query (jax.Array): Query sequence array (packed bits representation).
        ok_query (jax.Array): Boolean array indicating valid positions in the query
            (packed bits representation).

    Raises:
        ValueError: If the sequence lengths or valid position arrays are incompatible
            between the taxonomy references and the query.

    Example:
        >>> tree, params, N, segnum = read_model_jax("model.npz", "taxonomy.npz")
        >>> q, ok = read_query("ATGC...")
        >>> validate_taxonomy_query(tree, q, ok)  # Raises error if incompatible
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
    predictions to `pyprotax_results.csv`. The classification is based on
    the probabilistic taxonomic model described in the PROTAX paper.

    Args:
        qdir (str or Path): Path to the query alignment file containing sequences
            to classify. Expected format: two lines per record (header, then sequence).
            Example: `refs.aln`.
        par_dir (str or Path): Path to the model parameters `.npz` file containing
            trained beta coefficients and scaling statistics.
            Example: `models/params/model.npz`.
        tax_dir (str or Path): Path to the taxonomy `.npz` file containing the
            reference database, taxonomic tree structure, and node mappings.
            Example: `models/ref_db/taxonomy37k.npz`.
        verbose (bool, optional): If True, prints per-record timing and optional
            debug information. Defaults to False.

    Returns:
        None

    Side Effects:
        Writes `pyprotax_results.csv` in the current directory with one row per
        query containing the predicted node indices at each taxonomic level.
        Prints total classification time to stdout.

    Example:
        >>> classify_file("queries.aln", "models/params/model.npz",
        ...               "models/ref_db/taxonomy37k.npz", verbose=True)
        finished in 12.34s
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

    This is a lower-level API for single-sequence classification that can be
    used programmatically. Currently a placeholder for future implementation.
    Use `classify_file` for batch processing or implement custom logic using
    `get_probs` from the model module.

    Args:
        q (jax.Array): Packed bits representation of the query sequence bases
            (A/T/G/C), with shape (D',) where D' is the packed width.
        ok (jax.Array): Packed bits mask of positions that are valid base calls,
            with shape (D',).
        tree (TaxTree): `TaxTree` instance containing reference database and
            taxonomic topology.
        params (ProtaxModel): `ProtaxModel` instance with beta coefficients and
            scaling statistics (sc_mean, sc_var).
        segnum (int): Number of unique segment ids in `tree.segments` (used for
            JAX segment operations).
        N (int): Total number of nodes in the taxonomy.

    Returns:
        Not implemented yet. Future versions will return predicted node IDs or
        probability distributions.

    Note:
        This function is a placeholder. For single-sequence classification, you
        can use:
        ```python
        probs = get_probs(q, ok, tree, params, segnum, N)
        classified = jnp.argmax(probs)
        ```
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

