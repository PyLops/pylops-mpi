.. _installation:

Installation
############

Dependencies
************
The minimal set of dependencies for the PyLops-MPI project is:

* MPI (Message Passing Interface)
* Python 3.10 or greater
* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org/scipylib/index.html>`_
* `Matplotlib <https://matplotlib.org/>`_
* `MPI4py <https://mpi4py.readthedocs.io/en/stable/>`_
* `PyLops <https://pylops.readthedocs.io/en/stable/>`_

Additionally, to use the NCCL engine, the following additional 
dependencies are required:

* `CuPy <https://cupy.dev/>`_
* `NCCL <https://docs.cupy.dev/en/stable/install.html#additional-cuda-libraries>`__

We highly encourage using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
or its standalone package manager `Conda <https://docs.conda.io/en/latest/index.html>`_. However,
if this is not possible, some of the dependencies must be installed prior to installing PyLops-MPI.

Download and Install MPI
========================
Visit the official MPI website to download an appropriate MPI implementation for your system.
Follow the installation instructions provided by the MPI vendor.

* `Open MPI <https://www.open-mpi.org/software/ompi/v1.10/>`_
* `MPICH <https://www.mpich.org/downloads/>`_
* `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html#gs.10j8fx>`_

Verify MPI Installation
=======================
After installing MPI, verify its installation by opening a terminal and running the following command:

.. code-block:: bash

   >> mpiexec --version

Install NCCL (optional)
=======================
To obtain highly-optimized performance on GPU clusters, PyLops-MPI also supports the Nvidia's collective communication calls
`(NCCL) <https://developer.nvidia.com/nccl>`_. Two additional dependencies are required, CuPy and NCCL, which can be installed
using `pip`:

.. code-block:: bash

   >> pip install cupy-cuda12x nvidia-nccl-cu12

.. note::

   Replace `12x` with your CUDA version (e.g., `11x` for CUDA 11.x).


.. _UserInstall:

Step-by-step installation for users
***********************************

Currently PyLops-MPI can only be installed using ``pip``; simply type the following 
command in your terminal to install the PyPI distribution:

.. code-block:: bash

   >> pip install pylops-mpi

Note that when installing via `pip`, only *required* dependencies are installed.


.. _DevInstall:

Step-by-step installation for developers
****************************************

Fork PyLops-MPI
===============
Fork the `PyLops-MPI repository <https://github.com/PyLops/pylops-mpi>`_ and clone it by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/YOUR-USERNAME/pylops-mpi.git

We recommend installing dependencies into a separate environment.
For that end, we provide a `Makefile` with useful commands for setting up the environment.

Install dependencies
====================

Conda (recommended)
-------------------

For a ``conda`` environment, run

.. code-block:: bash

   >> make dev-install_conda

This will create and activate an environment called ``pylops_mpi``, with all 
required and optional dependencies.

If you want to enable `NCCL <https://developer.nvidia.com/nccl>`_ in PyLops-MPI, run this instead

.. code-block:: bash

   >> make dev-install_conda_nccl

Pip
---
If you prefer a ``pip`` installation, we provide the following command

.. code-block:: bash

   >> make dev-install

Note that, differently from the  ``conda`` command, the above **will not** create a virtual environment.
Make sure you create and activate your environment previously.

Similarly, if you want to enable `NCCL <https://developer.nvidia.com/nccl>`_ but prefer using pip,
you must first check the CUDA version of your system:

.. code-block:: bash

   >> nvidia-smi

The `Makefile` is pre-configured with CUDA 12.x. If you use this version, run

.. code-block:: bash

   >> make dev-install_nccl

Otherwise, you can change the command in `Makefile` to an appropriate CUDA version
i.e., If you use CUDA 11.x, change ``cupy-cuda12x`` and ``nvidia-nccl-cu12`` to 
``cupy-cuda11x`` and ``nvidia-nccl-cu11`` and run the command.

Run tests
=========
To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

   >> make tests

Make sure no tests fail, this guarantees that the installation has been successful.

If PyLops-MPI is installed with NCCL, also run tests:

.. code-block:: bash

   >> make tests_nccl

Run examples and tutorials
==========================
Since the sphinx-gallery creates examples/tutorials using only a single process, it is highly recommended to test the
examples/tutorials using n processes.

run examples:

.. code-block:: bash

   >> make run_examples

run tutorials:

.. code-block:: bash

   >> make run_tutorials

Make sure all the examples and tutorials python scripts are executed without any errors.

Add remote (optional)
=====================
To keep up-to-date on the latest changes while you are developing, you may optionally add
the PyLops-MPI repository as a *remote*.
Run the following command to add the PyLops-MPI repo as a remote named *upstream*:

.. code-block:: bash

   >> git remote add upstream https://github.com/PyLops/pylops-mpi

From then on, you can pull changes (for example, in the main branch) with:

.. code-block:: bash

   >> git pull upstream main

Final steps
===========
PyLops-MPI does not enforce the use of a linter as a pre-commit hook, but we do highly encourage using one before submitting a Pull Request.
A properly configured linter (``flake8``) can be run with:

.. code-block:: bash

   >> make lint

In addition, it is highly encouraged to build the docs prior to submitting a Pull Request.
Apart from ensuring that docstrings are properly formatted, they can aid in catching bugs during development.

Build the docs with:

.. code-block:: bash

   >> make doc

or Update the docs with:

.. code-block:: bash

   >> make docupdate
