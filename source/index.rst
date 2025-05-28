.. PROTAX-GPU documentation master file, created by
   sphinx-quickstart on Mon May 26 00:14:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. PROTAX-GPU documentation
.. ========================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

PROTAX-GPU Documentation
========================

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://protax-gpu.readthedocs.io/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/
   :alt: License

**PROTAX-GPU** is a GPU-accelerated JAX-based implementation of PROTAX for fast taxonomic classification of DNA barcode sequences.

.. figure:: ../img/Block_diagram_upd.png
   :width: 600px
   :align: center
   :alt: PROTAX-GPU Architecture

   PROTAX-GPU system architecture showing the GPU acceleration pipeline.

Quick Start
-----------

Install JAX first, then PROTAX-GPU:

.. code-block:: bash

   # For GPU support (Linux)
   pip install "jax[cuda12]"
   
   # Clone and install PROTAX-GPU
   git clone https://github.com/uoguelph-mlrg/PROTAX-GPU.git
   cd PROTAX-GPU
   pip install .

Key Features
------------

* **GPU Acceleration**: Custom CUDA kernels for k-nearest neighbor search
* **JAX Integration**: Leverages JAX for automatic differentiation and JIT compilation  
* **High Performance**: Significant speedups over CPU-only implementations
* **Training Support**: Gradient-based optimization of model parameters
* **Multiple Formats**: Compatible with TSV and PROTAX input formats

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/protax
   api/model
   api/classify
   api/ops

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   
   experiments
   performance
   development

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   
   changelog
   license
   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

