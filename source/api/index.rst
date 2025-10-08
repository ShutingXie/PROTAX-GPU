API Reference
=============

This page documents the public API of the PROTAX-GPU package. It follows a
hand-written reStructuredText shell with Sphinx autodoc to render content from
docstrings.

Core namespaces:

- :mod:`protax.taxonomy`: lightweight data structures used across the library
- :mod:`protax.model`: core probability and KNN-based computations
- :mod:`protax.classify`: routines for running classification over queries/files
- :mod:`protax.protax_utils`: I/O and conversion utilities for models and taxonomies
- :mod:`protax.ops`: low-level CPU/GPU KNN primitives exposed to JAX


Taxonomy and Model Structures
-----------------------------

.. autoclass:: protax.taxonomy.CSRWrapper
    :members:
    :show-inheritance:

.. autoclass:: protax.taxonomy.TaxTree
    :members:
    :show-inheritance:

.. autoclass:: protax.taxonomy.ProtaxModel
    :members:
    :show-inheritance:


Model Computations
------------------

.. automodule:: protax.model
    :members:
    :show-inheritance:


Classification Routines
-----------------------

.. automodule:: protax.classify
    :members:
    :show-inheritance:


Utilities and I/O
-----------------

.. automodule:: protax.protax_utils
    :members:
    :show-inheritance:


Low-level Ops (CPU/GPU KNN)
---------------------------

.. automodule:: protax.ops
    :members:

.. automodule:: protax.ops.knn_register
    :members:
    :show-inheritance:


