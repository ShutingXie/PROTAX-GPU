"""
Training script for PROTAX-GPU using gradient descent.

This script implements training of PROTAX model parameters (beta coefficients)
using stochastic gradient descent on self-classification of reference sequences.
The training process uses leave-one-out cross-validation where each reference
is temporarily removed from the database and used as a query.

The script supports:
- Minibatch gradient descent with configurable batch size
- Cross-entropy loss optimization
- Automatic differentiation via JAX
- Model checkpointing

Usage:
    python scripts/train_gd.py --train_dir PATH_TO_REFS --targ_dir PATH_TO_TARGETS

Arguments:
    --train_dir: Path to reference alignment file (e.g., refs.aln)
    --targ_dir: Path to CSV file with per-reference target node IDs

Configuration:
    Training hyperparameters are defined in the `tc` dictionary:
    - learning_rate: Step size for gradient updates (default: 0.001)
    - batch_size: Number of samples per batch (default: 500)
    - num_epochs: Number of passes through the dataset (default: 30)

Output:
    Trained model is saved to models/params/m2.npz
"""
import protax.model as model
from protax import protax_utils
from protax.taxonomy import CSRWrapper

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

import scipy.sparse as sp
import time
import pandas as pd

from pathlib import Path
import random
import matplotlib.pyplot as plt
from functools import partial
import argparse


def CE_loss(log_probs, y_ind):
    """
    Cross-entropy loss for selected targets from per-node log-probabilities.

    Computes the negative log-likelihood of the target nodes given the model's
    probability distribution. Used as the optimization objective during training.

    Args:
        log_probs (jax.Array): Log-branch probabilities organized as (levels, paths)
            or flattened per-node log-probabilities, shape compatible with indexing
            by y_ind.
        y_ind (jax.Array): Integer indices of target nodes for each sample in the
            batch.

    Returns:
        jax.Array: Scalar negative log-likelihood summed over the batch. Lower
            values indicate better fit to targets.
    """
    return -jnp.sum(jnp.take(log_probs, y_ind, axis=0))


def forward(q, ok, tree, beta, sc_mean, sc_var, N, segnum, y_ind, lvl):
    """
    Forward pass: build design matrix, compute log-branch probs, and CE loss.

    This function executes the full PROTAX forward pass from query sequence to
    loss value. It is the main function that gets differentiated by JAX to compute
    gradients with respect to beta.

    Args:
        q (jax.Array): Packed bits query sequence.
        ok (jax.Array): Packed bits validity mask for query.
        tree (TaxTree): Taxonomy tree structure.
        beta (jax.Array): Parameter matrix per level, shape (num_levels, num_features).
            Will be broadcast to per-node using `lvl`.
        sc_mean (jax.Array): Per-node scaling means.
        sc_var (jax.Array): Per-node scaling variances.
        N (int): Number of nodes in taxonomy.
        segnum (int): Number of unique parent segments.
        y_ind (jax.Array): Target node index for this sample.
        lvl (jax.Array): Per-node level indices for broadcasting beta.

    Returns:
        jax.Array: Scalar cross-entropy loss for this sample.
    """
    beta = jnp.take(beta, lvl, axis=0)
    X = model.get_X(q, ok, tree, N, sc_mean, sc_var)
    log_probs = model.fill_log_bprob(X, beta, tree, segnum)
    return CE_loss(log_probs, y_ind)


f_grad = jax.jit(jax.grad(forward, argnums=(3)), static_argnums=(6, 7))
forward_jit = jax.jit(forward, static_argnums=(6, 7))


def get_targ(target_dir):
    """
    Extract the most specific (lowest-level) node id target for each reference.

    Reads a CSV file where rows are taxonomic levels and columns are references.
    For each reference, finds the deepest level (finest taxonomic assignment) that
    is not -1 (missing), and returns that node ID as the training target.

    Args:
        target_dir (str or Path): Path to CSV file with target node IDs.
            Expected format: rows = levels, columns = references, values = node IDs
            or -1 for missing assignments.

    Returns:
        jax.Array: 1D integer array of length R (number of references) containing
            the target node ID for each reference, representing its finest taxonomic
            assignment.

    Example:
        For a reference assigned to nodes [1, 5, 12, -1, -1, -1, -1] across 7 levels,
        this returns 12 (the last non-missing assignment).
    """
    targ = pd.read_csv(target_dir)
    targ = targ.to_numpy()[:, 1:].T

    res = np.zeros((targ.shape[0],), dtype=np.int32)

    for i in range(len(targ)):
        old = -1
        for j in range(targ.shape[1]):
            if targ[i][j] == -1:
                res[i] = old
            elif j == targ.shape[1] - 1:
                res[i] = targ[i][j]
            old = targ[i][j]

    return jnp.array(res)


def mask_n2s(n2s, node_state, i):
    """
    Remove one reference from the database for leave-one-out cross-validation.

    Masks out reference `i` from the node-to-sequence mapping and updates node
    state indicators to reflect nodes that lost their only reference. This is
    used during training to prevent references from being their own nearest
    neighbors.

    Args:
        n2s (scipy.sparse.csr_matrix): Node-to-sequence mapping, shape (N, R)
            where entry [node, ref] = 1 if ref is assigned to node.
        node_state (numpy.ndarray): Binary node state indicators, shape (N, 1)
            indicating empty-but-known nodes.
        i (int): Index of reference to mask out.

    Returns:
        tuple: (masked_n2s, updated_node_state) where:
            - masked_n2s (CSRWrapper): Updated mapping with reference i removed
            - updated_node_state (jax.Array): Shape (N, 2) with columns:
                [0] = empty-but-known, [1] = has-references

    Note:
        After masking, some nodes may become empty if reference `i` was their only
        assigned sequence. The node_state is updated accordingly.
    """
    ref_mask = np.ones((n2s.shape[1],), dtype=np.int32)
    ref_mask[i] = 0
    n2s = n2s @ sp.diags(ref_mask)

    has_refs = np.array(n2s.sum(axis=1)) > 0
    empty = np.logical_not(has_refs)

    # update empty but known entries
    node_state = np.logical_or(node_state, empty)
    node_state = np.concatenate((node_state, has_refs), axis=1)

    n2s = CSRWrapper(
        data=jnp.array(n2s.data),
        indices=jnp.array(n2s.indices),
        indptr=jnp.array(n2s.indptr),
        shape=n2s.shape,
    )

    return n2s, jnp.array(node_state)


def load_params(pdir, tdir):
    """
    Load raw parameters and layer indices from saved model archives.

    Reads the initial model parameters (beta, scalings) and taxonomy metadata
    (node layers) from .npz files. These serve as initialization for training.

    Args:
        pdir (str or Path): Path to parameter .npz file (e.g., models/params/model.npz).
        tdir (str or Path): Path to taxonomy .npz file (e.g., models/ref_db/taxonomy*.npz).

    Returns:
        tuple: (beta, lvl, sc) where:
            - beta (numpy.ndarray): Initial parameter matrix per level
            - lvl (numpy.ndarray): Layer index for each node
            - sc (numpy.ndarray): Scaling statistics (mean/var) per level

    Note:
        Beta is per-level and must be broadcast to per-node using `lvl` during
        the forward pass.
    """
    par_dir = Path(pdir)
    tax_dir = Path(tdir)

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    beta = par["beta"]
    sc = par["scalings"]
    lvl = tax["node_layer"]

    tax = np.load(tax_dir.resolve())
    par = np.load(par_dir.resolve())

    return beta, lvl, sc


def train(train_config, train_dir, targ_dir):
    """
    Train PROTAX beta parameters via stochastic gradient descent.

    Uses leave-one-out self-classification of reference sequences as training
    signal. Each reference is temporarily removed from the database, classified
    against remaining references, and the model is updated to maximize the
    probability of the correct taxonomic assignment.

    Args:
        train_config (dict): Training configuration with keys:
            - 'learning_rate' (float): Gradient descent step size
            - 'batch_size' (int): Number of samples per gradient update
            - 'num_epochs' (int): Number of passes through the dataset
        train_dir (str or Path): Path to reference alignment file (e.g., refs.aln)
            containing sequences to use for self-supervised training.
        targ_dir (str or Path): Path to CSV file with per-reference target node IDs,
            specifying the ground truth taxonomic assignment for each reference.

    Returns:
        None

    Side Effects:
        - Saves trained model to models/params/m2.npz after each epoch
        - Displays loss curve plot at end of training (via matplotlib)
        - Prints per-batch and per-epoch loss to stdout

    Note:
        The model is initialized with random weights for beta. Scaling statistics
        (sc_mean, sc_var) are loaded from the base model and not trained.
        Training uses JAX's automatic differentiation on the `forward` function.
    """
    tree, params, N, segnum = protax_utils.read_model_jax(
        "models/params/model.npz", "models/ref_db/taxonomy37k.npz"
    )
    pkey = jax.random.PRNGKey(0)
    lr = train_config["learning_rate"]

    beta = jax.random.uniform(pkey, (7, 4))
    n2s = sp.csr_matrix(
        (tree.node2seq.data, tree.node2seq.indices, tree.node2seq.indptr),
        shape=tree.node2seq.shape,
    )
    targ = get_targ(targ_dir)
    seq_list, ok_list = protax_utils.read_refs(train_dir)
    node_state = np.expand_dims(np.array(tree.node_state)[:, 0], 1)

    # params and node lvl
    _, lvl, sc = load_params("models/params/model.npz", "models/ref_db/taxonomy37k.npz")
    loss_hist = []
    for e in range(train_config["num_epochs"]):
        print(f"epoch {e}")

        beta_grad = 0
        loss_sum = 0
        batch_loss = 0

        traversal = list(range(seq_list.shape[0]))
        random.shuffle(traversal)

        # minibatch
        for i in traversal:
            # mask out tree
            q = seq_list[i]
            ok = ok_list[i]

            # tree.node_state = mask_design_mat(tree, num_refs, targ, i)

            # masks out a node2seq column given reference index (~1.2 ms)
            tree.node2seq, tree.node_state = mask_n2s(n2s, node_state, i)
            beta_grad += f_grad(
                q,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
            )
            batch_loss += forward_jit(
                q,
                ok,
                tree,
                beta,
                params.sc_mean,
                params.sc_var,
                N,
                segnum,
                targ.at[i].get(),
                lvl,
            )

            if i % train_config["batch_size"] == 0:
                beta = beta - lr * beta_grad
                loss_sum += batch_loss
                curr_loss = batch_loss / train_config["batch_size"]
                print("batch_loss: ", curr_loss)
                loss_hist.append(curr_loss)
                batch_loss = 0

                # grad norm
                bflat = beta_grad.reshape(beta_grad.shape[0] * beta_grad.shape[1])

        print("loss: ", loss_sum / seq_list.shape[0])

        # save checkpoint
        mf = Path("models/params/m2.npz")
        np.savez_compressed(mf.resolve(), beta=np.array(beta), scalings=sc)
    plt.plot(loss_hist)
    plt.show()


if __name__ == "__main__":
    # parse config from command line
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--train_dir", type=str, help="Path to training data")
    parser.add_argument("--targ_dir", type=str, help="Path to target data")
    args = parser.parse_args()
    train_dir = Path(args.train_dir)
    targ_dir = Path(args.targ_dir)

    # train_dir = r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln"
    # targ_dir = "/home/roy/Documents/PROTAX-dsets/30k_small/30k-targets.csv"

    # training config
    tc = {
        "learning_rate": 0.001,
        "batch_size": 500,
        "num_epochs": 30,
    }

    train(tc, train_dir, targ_dir)  # train the model
