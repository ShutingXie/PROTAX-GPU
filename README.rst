PROTAX-GPU
==========

.. image:: https://readthedocs.org/projects/protax-gpu/badge/?version=latest
   :target: https://protax-gpu.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: img/Block_diagram_upd.png
   :alt: PROTAX-GPU block diagram
   :align: center

A GPU-accelerated JAX-based implementation of `PROTAX <https://pubmed.ncbi.nlm.nih.gov/27296980/>`_.
Contains all code and experiments for PROTAX-GPU.

To reproduce the BOLD 7.8M dataset experiments, PROTAX-GPU requires a NVIDIA GPU with at least 8GB VRAM and CUDA compute capability 6.0 or later.This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, and NVIDIA Hopper™ architecture families.

Functionality
-------------
Estimates the probability of each outcome in a taxonomic tree given a query barcode
sequence compared to reference sequences.

- Uses JAX to accelerate sequence distance and probability decomposition calculations
- Uses custom CUDA kernels to accelerate the calculation of the top-k most similar reference sequences


Features
--------
- CPU and GPU inference
- Gradient-based optimization of model parameters
- Compatible with TSV and PROTAX input format
- Full computation of all probabilities in the taxonomic tree


Repository Organization
-----------------------
::

  experiments/        All experiments for the paper
  lib/                C++/CUDA code for PROTAX-GPU
  scripts/            Scripts for training and inference
  ├-- train_gd.py     Trains a model with gradient descent
  └-- convert.py      Converts .TSV to PROTAX-GPU format
  tests/              Unit tests
  protax/             Main PROTAX-GPU code
  ├-- ops/            JAX bindings for CUDA kernels


Compatibility
-------------
- Linux: CPU (yes), NVIDIA GPU (yes), Apple GPU (n/a)
- Mac x86_64: CPU (yes), NVIDIA GPU (n/a), Apple GPU (no)
- Mac ARM: CPU (yes), NVIDIA GPU (n/a), Apple GPU (no)
- Windows: CPU (experimental), NVIDIA GPU (experimental), Apple GPU (n/a)


Installation
------------
These instructions are for Linux and macOS. Windows support is experimental.

1) Install CUDA (required for GPU support)
   **IMPORTANT:** PROTAX-GPU requires a full local installation of CUDA, including development headers and tools, due to its use of custom CUDA kernels.
   
   - Install CUDA >= 12 from the `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_
   - Install cuDNN >= 8.9 per the `official guide <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_
   - Ensure that your system environment variables are correctly set up to point to your CUDA installation
   
   **NOTE:** While JAX offers an easier CUDA installation via pip wheels for some platforms, this method does not provide the full CUDA toolkit required by PROTAX-GPU. You must perform a local CUDA installation as described above.

2) Create & activate a new environment (recommended)

.. code-block:: bash

   conda create -n [name] python=3.12
   conda activate [name]

3) Install JAX / jaxlib
   - Linux CPU: ``pip install "jax[cpu]"``
   - Linux GPU: ``pip install "jax[cuda12]"`` (ensure CUDA 12.x installed)
   - macOS CPU: ``pip uninstall -y jax jaxlib && conda install -c conda-forge jax=0.4.26 jaxlib=0.4.23``
   - macOS GPU: not yet supported

   **NOTE:** The GPU installation command assumes you have already installed CUDA 12.2 as per step 1.

4) Install PROTAX-GPU

Clone this repository:

.. code-block:: bash 

   git clone https://github.com/uoguelph-mlrg/PROTAX-GPU.git

Build and install PROTAX-GPU (This will install a package called `protax` in your environment)

.. code-block:: bash 

   cd PROTAX-GPU
   pip install .

**NOTE:** CUDA 11.2 is also supported by JAX, but support for it will be dropped in the future. As long as the JAX version supports CUDA 11.2 and is greater than or equal to 0.4.14, this should work. However, ensure that your local CUDA installation matches the version you're using with JAX.

If you are on MacOS and facing installation issues, run the following commands:

.. code-block:: bash

   chmod +x ./scripts/fix_librhash.sh
   ./scripts/fix_librhash.sh


Usage
-----
Instructions for running PROTAX-GPU for inference and training.

Inference
^^^^^^^^^
Once you have a trained model, you can use the classification script to classify query sequences.

.. code-block:: bash

   python scripts/process_seqs.py [PATH_TO_QUERY_SEQUENCES] [PATH_TO_MODEL] [PATH_TO_TAXONOMY]

Example:

.. code-block:: bash

   python scripts/process_seqs.py models/ref_db/test_refs.aln models/params/model.npz models/ref_db/taxonomy37k.npz

Arguments:
- ``PATH_TO_QUERY_SEQUENCES``: File containing the sequences to classify (e.g., FASTA or alignment file)(Can use refs.aln from `FinPROTAX <https://github.com/psomervuo/FinPROTAX/tree/main>`_ for experiment)
- ``PATH_TO_MODEL``: Path to the model (a baseline model is available in ``models/params/model.npz``)
- ``PATH_TO_TAXONOMY``: Path to the taxonomy ``.npz`` file (available in ``models/ref_db/taxonomy37k.npz``)

Results are saved to ``pyprotax_results.csv``.

Training
^^^^^^^^
Run the training script from the command line. You need to specify paths to your training
data and target data using the ``--train_dir`` and ``--targ_dir`` arguments, respectively.

.. code-block:: bash

   python scripts/train_gd.py --train_dir [PATH_TO_TRAINING_DATA] --targ_dir [PATH_TO_TARGET_DATA]

Command Line Arguments
- `--train_dir`: Path to the training data (e.g., refs.aln).
- `--targ_dir`: Path to the target data (e.g., targets.csv).

Training configuration (defaults in the script):
- Learning rate: ``0.001``
- Batch size: ``500``
- Number of epochs: ``30``

These parameters are predefined in the script and can be modified if needed by editing the dictionary `tc` in the code.

The script uses ``models/params/model.npz`` as baseline and saves the trained model at ``models/params/m2.npz``.


Hardware
--------
To reproduce the BOLD dataset experiments, PROTAX-GPU requires an NVIDIA GPU with at least 16GB VRAM and CUDA compute capability 6.0 or later. This corresponds to GPUs in the NVIDIA Pascal, NVIDIA Volta™, NVIDIA Turing™, NVIDIA Ampere architecture, and NVIDIA Hopper™ architecture families.


Datasets
--------
The BOLD 7.8M dataset is available here: `BOLD data release <https://www.boldsystems.org/index.php/datarelease>`_.
The dataset is not included in this repository due to its size.

The smaller FinPROTAX dataset is included in the ``models`` directory, sourced from
`FinPROTAX <https://github.com/psomervuo/FinPROTAX>`_.


License
-------
This project is licensed under the `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_
License — see the ``LICENSE`` file for details.


