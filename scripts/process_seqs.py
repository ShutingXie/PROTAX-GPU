"""
Command-line script for classifying DNA barcode sequences using PROTAX-GPU.

This script provides a simple command-line interface to the PROTAX-GPU classifier.
It reads query sequences from a FASTA-like alignment file and classifies them using
a pre-trained model and reference database, writing results to CSV.

Usage:
    python scripts/process_seqs.py QUERY_FILE MODEL_FILE TAXONOMY_FILE

Arguments:
    QUERY_FILE: Path to query sequences (e.g., refs.aln) in two-line format
    MODEL_FILE: Path to trained model parameters (e.g., models/params/model.npz)
    TAXONOMY_FILE: Path to taxonomy database (e.g., models/ref_db/taxonomy37k.npz)

Example:
    python scripts/process_seqs.py data/queries.aln models/params/model.npz \\
        models/ref_db/taxonomy37k.npz

Output:
    Results are written to pyprotax_results.csv in the current directory.
"""
from protax.classify import classify_file
import sys

if __name__ == "__main__":
    protax_args = sys.argv
    if len(protax_args) < 4:
        print(
            "Usage: python scripts/process_seqs.py [PATH_TO_QUERY_SEQUENCES] [PATH_TO_PARAMETERS] [PATH_TO_TAXONOMY_FILE]"
        )

    query_dir, model_dir, tax_dir = protax_args[1:4]
    classify_file(query_dir, model_dir, tax_dir)

    # testing
    # query_dir = r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln"
    # classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")
