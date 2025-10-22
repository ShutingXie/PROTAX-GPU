API Reference
=============

This page documents the public API of the PROTAX-GPU package.

PROTAX-GPU provides a GPU-accelerated implementation of the PROTAX probabilistic taxonomic classification algorithm for DNA barcodes. The package is built on JAX and includes custom CUDA kernels for efficient k-nearest neighbor computations.

General usage instructions for PROTAX-GPU are provided in the main README.

.. tip::
    For most users, the main entry points are:
    
    * :func:`~protax.classify.classify_file` - Classify query sequences from a file
    * :func:`~protax.protax_utils.read_model_jax` - Load a trained PROTAX model
    * :func:`~protax.model.get_probs` - Compute classification probabilities for a single query

Core Modules
------------

The PROTAX-GPU package is organized into the following core modules:

* :mod:`protax.taxonomy` - Data structures for taxonomic trees and models
* :mod:`protax.model` - Core probability computations and KNN-based inference
* :mod:`protax.classify` - High-level classification routines
* :mod:`protax.protax_utils` - I/O utilities for models, taxonomies, and sequences
* :mod:`protax.ops` - Low-level CPU/GPU operations (KNN primitives)


Taxonomy and Model Data Structures
-----------------------------------

These classes define the core data structures used throughout PROTAX-GPU.

CSRWrapper
~~~~~~~~~~

.. autoclass:: protax.taxonomy.CSRWrapper
    :members:
    :show-inheritance:

    A lightweight wrapper for Compressed Sparse Row (CSR) matrix format, compatible with JAX arrays.
    
    This structure is used to efficiently represent sparse adjacency relationships in the taxonomic tree,
    such as the mapping from nodes to reference sequences.

TaxTree
~~~~~~~

.. autoclass:: protax.taxonomy.TaxTree
    :members:
    :show-inheritance:

    Represents the complete state of a taxonomic tree used for classification.
    
    This structure contains:
    
    * Reference sequences and their valid positions
    * Node-to-sequence mappings
    * Taxonomic paths and hierarchy information
    * Prior probabilities for each node
    
    **Dimensions:**
    
    * ``N`` - Total number of nodes in the taxonomy
    * ``L`` - Number of non-species nodes (depth < 7)
    * ``R`` - Total number of reference sequences

ProtaxModel
~~~~~~~~~~~

.. autoclass:: protax.taxonomy.ProtaxModel
    :members:
    :show-inheritance:

    Contains the learned parameters for a PROTAX model.
    
    The model parameters include:
    
    * ``beta`` - Weight parameters for the logistic regression at each node
    * ``sc_mean`` - Mean values for feature scaling
    * ``sc_var`` - Variance values for feature scaling


Model Computations
------------------

Core functions for computing sequence distances, design matrices, and classification probabilities.

Sequence Distance Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.model.seq_dist

.. autofunction:: protax.model.seq_dist2


Design Matrix and Feature Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.model.get_X

.. autofunction:: protax.model.get_z


Probability Computations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.model.get_bprobs

.. autofunction:: protax.model.get_log_bprobs

.. autofunction:: protax.model.fill_bprob

.. autofunction:: protax.model.get_probs

.. autofunction:: protax.model.get_log_probs


K-Nearest Neighbor Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.model.knn

.. autofunction:: protax.model.knn_v2


Classification Routines
-----------------------

High-level functions for classifying query sequences.

File-based Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.classify.classify_file

    Process a batch of query sequences from a file using a trained model.
    
    This is the main entry point for running PROTAX-GPU classification on multiple sequences.
    Results are saved to a CSV file with probabilities for each taxonomic level.


Single Query Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: protax.classify.classify

.. autofunction:: protax.classify.validate_taxonomy_query


Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: protax.classify.load_layer

.. autofunction:: protax.classify.read_names


Utilities and I/O
-----------------

Functions for reading and writing models, taxonomies, and sequence data.

Model I/O
~~~~~~~~~

.. autofunction:: protax.protax_utils.read_model_jax

    Load a complete PROTAX model from disk.
    
    This function reads both the model parameters and taxonomy structure,
    returning all components needed for classification.

.. autofunction:: protax.protax_utils.read_params

.. autofunction:: protax.protax_utils.read_scalings

.. autofunction:: protax.protax_utils.read_baseline


Taxonomy I/O
~~~~~~~~~~~~

.. autofunction:: protax.protax_utils.read_taxonomy

.. autofunction:: protax.protax_utils.get_descendants

.. autofunction:: protax.protax_utils.read_refs

.. autofunction:: protax.protax_utils.assign_refs


Sequence I/O
~~~~~~~~~~~~

.. autofunction:: protax.protax_utils.read_query

.. autofunction:: protax.protax_utils.str2batch_query

.. autofunction:: protax.protax_utils.get_seq_bits


Data Conversion
~~~~~~~~~~~~~~~

.. autofunction:: protax.protax_utils.convert_model

.. autofunction:: protax.protax_utils.convert_taxonomy

.. autofunction:: protax.protax_utils.assign_params

.. autofunction:: protax.protax_utils.get_train_targets


Low-level Operations (CPU/GPU KNN)
-----------------------------------

Low-level primitives for k-nearest neighbor computations, with both CPU and GPU implementations.

.. note::
    These functions are typically called internally by higher-level model functions.
    Most users will not need to call these directly.

The :mod:`protax.ops` module provides JAX custom operations for efficient k-nearest neighbor computations:

* **CPU operations**: Compiled C++ extensions for CPU-based KNN
* **GPU operations**: Custom CUDA kernels for GPU-accelerated KNN

These operations are registered with JAX via the :mod:`protax.ops.knn_register` module and are automatically
selected based on the device (CPU or GPU) where the computation is performed.

Key Functions
~~~~~~~~~~~~~

.. autofunction:: protax.ops.knn

.. autofunction:: protax.ops.knn_v2

